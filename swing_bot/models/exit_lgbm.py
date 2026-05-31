"""Supervised exit LightGBM training for valid-fold research.

Exit models are trained on position-state rows generated from OOF entry
predictions.  This module does not create entry candidates and never reads test.
It only consumes an already-built exit dataset and evaluates locked valid folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json
from swing_bot.evaluation.exit_report import exit_fold_metrics, write_exit_report
from swing_bot.models.lgbm_common import (
    ModelTrainingError,
    feature_importance_frame,
    load_lgbm_params,
    make_lgbm_regressor,
    save_lgbm_model,
)

EXIT_TARGET_DEFAULT = "target_exit_hold_delta_bps"

EXIT_META_COLUMNS = {
    "timestamp",
    "entry_time",
    "entry_exec_time",
    "current_time",
    "current_exit_time",
    "future_exit_time",
    "fold",
    "episode_id",
    "side",
    "entry_horizon_minutes",
    "exit_lookahead_minutes",
    "is_entry_score_oof",
    "is_current_score_oof",
}

POSITION_FEATURES = [
    "age_minutes",
    "unrealized_pl_bps",
    "unrealized_pl_after_cost_bps",
    "mfe_since_entry_bps",
    "mae_since_entry_bps",
    "giveback_from_mfe_bps",
]

SCORE_FEATURES = [
    "age_minutes",
    "entry_pred_net_bps",
    "current_pred_net_bps",
    "score_decay_bps",
]

# Known diagnostic/raw execution columns from the exit dataset.  They are useful
# for audit, but should not be direct model features.
EXCLUDED_DATASET_COLUMNS = {
    "entry_price",
    "current_close",
}

FEATURE_SET_CHOICES = ("v0_market_only", "v1_score_decay", "v2_position_aware")


@dataclass(frozen=True)
class ExitTrainingConfig:
    """Config for one exit hold-delta regressor."""

    feature_set: str = "v0_market_only"
    target_col: str = EXIT_TARGET_DEFAULT
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)
    save_models: bool = True

    def __post_init__(self) -> None:
        if self.feature_set not in FEATURE_SET_CHOICES:
            raise ModelTrainingError(
                f"unknown exit feature_set={self.feature_set!r}; choices={', '.join(FEATURE_SET_CHOICES)}"
            )


@dataclass(frozen=True)
class ExitTrainingResult:
    run_dir: Path
    model_dir: Path
    target_col: str
    feature_set: str
    feature_cols: tuple[str, ...]
    fold_metrics: tuple[dict[str, object], ...]
    summary: dict[str, object]


def _required_columns(target_col: str) -> set[str]:
    return {"timestamp", "fold", "episode_id", target_col}


def validate_exit_dataset(df: pd.DataFrame, *, target_col: str = EXIT_TARGET_DEFAULT) -> pd.DataFrame:
    """Validate and normalize an exit position-state dataset."""
    missing = sorted(_required_columns(target_col) - set(df.columns))
    if missing:
        raise ModelTrainingError("exit dataset missing required columns: " + ", ".join(missing))
    out = df.copy()
    for col in ["timestamp", "entry_time", "entry_exec_time", "current_time", "current_exit_time", "future_exit_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            if out[col].isna().any():
                raise ModelTrainingError(f"exit dataset contains unparseable timestamps in {col}")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out["fold"] = out["fold"].astype(str)
    out["episode_id"] = out["episode_id"].astype(str)
    if out.empty:
        raise ModelTrainingError("exit dataset is empty")
    if out["fold"].nunique() < 2:
        raise ModelTrainingError("exit dataset must contain at least two folds for valid-fold training")
    return out.sort_values(["timestamp", "episode_id"]).reset_index(drop=True)


def _is_forbidden_exit_feature(col: str, *, target_col: str) -> bool:
    lower = col.lower()
    if col == target_col:
        return True
    if lower.startswith("target_"):
        return True
    if "future" in lower:
        return True
    if "label" in lower:
        return True
    if lower.startswith("diag_"):
        return True
    if lower in {c.lower() for c in EXCLUDED_DATASET_COLUMNS}:
        return True
    return False


def _market_feature_columns(df: pd.DataFrame, *, target_col: str) -> list[str]:
    reserved = set(EXIT_META_COLUMNS) | set(POSITION_FEATURES) | set(SCORE_FEATURES) | EXCLUDED_DATASET_COLUMNS | {target_col}
    cols: list[str] = []
    for col in df.columns:
        if col in reserved:
            continue
        if _is_forbidden_exit_feature(col, target_col=target_col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def exit_feature_columns(df: pd.DataFrame, *, feature_set: str, target_col: str = EXIT_TARGET_DEFAULT) -> list[str]:
    """Return reviewed feature columns for an exit model feature set."""
    if feature_set not in FEATURE_SET_CHOICES:
        raise ModelTrainingError(f"unknown exit feature_set={feature_set!r}")
    market = _market_feature_columns(df, target_col=target_col)
    if feature_set == "v0_market_only":
        cols = market
    elif feature_set == "v1_score_decay":
        cols = market + [c for c in SCORE_FEATURES if c in df.columns]
    else:
        extra = list(dict.fromkeys(SCORE_FEATURES + POSITION_FEATURES))
        cols = market + [c for c in extra if c in df.columns]

    safe_cols: list[str] = []
    for col in dict.fromkeys(cols):
        if _is_forbidden_exit_feature(col, target_col=target_col):
            raise ModelTrainingError(f"forbidden leakage-like exit feature column selected: {col}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        safe_cols.append(col)
    if not safe_cols:
        raise ModelTrainingError(f"no numeric feature columns for exit feature_set={feature_set}")
    return safe_cols


def _fit_exit_regressor(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
):
    train = train_df.dropna(subset=[target_col]).copy()
    if train.empty:
        raise ModelTrainingError("no non-null target rows for exit training")
    y = pd.to_numeric(train[target_col], errors="coerce")
    ok = y.notna()
    if int(ok.sum()) == 0:
        raise ModelTrainingError("exit target is all NaN after numeric conversion")
    model = make_lgbm_regressor(params)
    model.fit(train.loc[ok, feature_cols], y.loc[ok])
    return model


def _predict_exit(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return np.asarray(model.predict(df[feature_cols]), dtype="float64")


def train_exit_valid_folds(
    *,
    exit_dataset: pd.DataFrame,
    model_config_path: Path | str,
    run_dir: Path,
    model_dir: Path,
    config: ExitTrainingConfig,
    run_metadata: dict[str, Any] | None = None,
) -> ExitTrainingResult:
    """Train/evaluate an exit regressor with fold-level cross fitting.

    For each fold, rows from that fold are held out for evaluation and rows from
    all other folds are used for training.  Episode IDs from the eval fold are
    explicitly excluded from training as an additional safety check.
    """
    data = validate_exit_dataset(exit_dataset, target_col=config.target_col)
    feature_cols = exit_feature_columns(data, feature_set=config.feature_set, target_col=config.target_col)
    params = load_lgbm_params(model_config_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []

    fold_names = sorted(data["fold"].dropna().astype(str).unique().tolist())
    for fold_name in fold_names:
        eval_mask = (data["fold"] == fold_name) & data[config.target_col].notna()
        eval_episode_ids = set(data.loc[eval_mask, "episode_id"].astype(str))
        train_mask = (data["fold"] != fold_name) & data[config.target_col].notna()
        if eval_episode_ids:
            train_mask &= ~data["episode_id"].astype(str).isin(eval_episode_ids)
        train_df = data.loc[train_mask].copy()
        eval_df = data.loc[eval_mask].copy()
        if train_df.empty:
            raise ModelTrainingError(f"{fold_name}: no train rows after excluding eval fold/episodes")
        if eval_df.empty:
            raise ModelTrainingError(f"{fold_name}: no eval rows with non-null target")

        overlap = set(train_df["episode_id"].astype(str)).intersection(set(eval_df["episode_id"].astype(str)))
        if overlap:
            raise ModelTrainingError(f"{fold_name}: train/eval episode_id overlap detected")

        model = _fit_exit_regressor(train_df=train_df, feature_cols=feature_cols, target_col=config.target_col, params=params)
        pred = _predict_exit(model, eval_df, feature_cols)

        keep_cols = [
            c for c in [
                "timestamp",
                "entry_time",
                "current_time",
                "fold",
                "episode_id",
                "side",
                "entry_horizon_minutes",
                "exit_lookahead_minutes",
                "age_minutes",
                config.target_col,
            ]
            if c in eval_df.columns
        ]
        pred_df = eval_df[keep_cols].copy()
        if config.target_col != EXIT_TARGET_DEFAULT and EXIT_TARGET_DEFAULT not in pred_df.columns:
            pred_df[EXIT_TARGET_DEFAULT] = eval_df[config.target_col].to_numpy()
        pred_df["pred_exit_hold_delta_bps"] = pred
        pred_df["is_oof"] = True
        pred_df["exit_feature_set"] = config.feature_set
        predictions.append(pred_df)

        fold_metrics = exit_fold_metrics(
            eval_df[config.target_col],
            pred,
            fold_name=fold_name,
            quantiles=config.quantiles,
        )
        fold_metrics.update({
            "train_count": int(len(train_df)),
            "target_col": config.target_col,
            "feature_set": config.feature_set,
            "feature_count": int(len(feature_cols)),
            "episode_count_eval": int(eval_df["episode_id"].nunique()),
        })
        metrics.append(fold_metrics)
        importances.append(feature_importance_frame(model, feature_cols, fold=fold_name))

        if config.save_models:
            save_lgbm_model(model, model_dir / f"{fold_name}.txt")

    predictions_df = pd.concat(predictions, ignore_index=True).sort_values(["timestamp", "fold", "episode_id"]).reset_index(drop=True)
    summary = write_exit_report(
        predictions=predictions_df,
        fold_metrics=metrics,
        output_dir=run_dir,
        target_col=config.target_col,
        pred_col="pred_exit_hold_delta_bps",
    )

    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    if not imp_df.empty:
        write_frame(imp_df, run_dir / "feature_importance_by_fold.csv")
        agg = imp_df.groupby("feature", as_index=False).agg(
            mean_importance_split=("importance_split", "mean"),
            mean_importance_gain=("importance_gain", "mean"),
        ).sort_values("mean_importance_gain", ascending=False)
        write_frame(agg, run_dir / "feature_importance_mean.csv")

    run_config = {
        "model_role": "exit",
        "feature_set": config.feature_set,
        "target_col": config.target_col,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "lgbm_params": params,
        "folds": fold_names,
        "notes": [
            "Exit dataset must be built from OOF entry candidates.",
            "Each fold is evaluated out-of-fold; rows from the eval fold are not used for training.",
            "Eval episode_id values are explicitly excluded from training.",
            "Test rows are not read by this training function.",
        ],
        "metadata": run_metadata or {},
    }
    write_json(run_config, run_dir / "run_config.json")

    return ExitTrainingResult(
        run_dir=run_dir,
        model_dir=model_dir,
        target_col=config.target_col,
        feature_set=config.feature_set,
        feature_cols=tuple(feature_cols),
        fold_metrics=tuple(metrics),
        summary=summary,
    )

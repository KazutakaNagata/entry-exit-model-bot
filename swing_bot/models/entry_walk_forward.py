"""Walk-forward entry LightGBM training utilities.

This module is intentionally narrow: it retrains the already-selected entry
feature set on progressively updated past data and predicts locked valid folds.
It does not choose features, tune thresholds, or touch test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import gc

import numpy as np
import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json
from swing_bot.evaluation.entry_report import write_entry_report
from swing_bot.evaluation.fold_metrics import regression_fold_metrics
from swing_bot.labels.entry_net_return import entry_target_column
from swing_bot.models.entry_lgbm import merge_features_and_labels
from swing_bot.models.lgbm_common import (
    ModelTrainingError,
    assert_no_forbidden_feature_columns,
    feature_importance_frame,
    load_lgbm_params,
    make_lgbm_regressor,
    numeric_feature_columns,
    predict_regressor,
    save_lgbm_model,
)
from swing_bot.splits.purged_walk_forward import label_interval_end
from swing_bot.splits.split_manifest import FoldSpec, SplitManifest

TrainingMode = Literal["fixed", "expanding", "rolling"]


@dataclass(frozen=True)
class EntryWalkForwardConfig:
    """Configuration for walk-forward entry model evaluation."""

    side: str = "long"
    horizon_minutes: int = 120
    target_col: str | None = None
    training_mode: TrainingMode = "expanding"
    train_window_days: int | None = None
    score_history_days: int = 120
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)
    save_models: bool = True
    save_feature_importance: bool = True
    downcast_float32: bool = True
    max_train_rows: int | None = None
    force_col_wise: bool = True

    @property
    def resolved_target_col(self) -> str:
        return self.target_col or entry_target_column(self.side, self.horizon_minutes)


def _as_utc_series(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True)
    if isinstance(parsed, pd.DatetimeIndex):
        return pd.Series(parsed)
    return parsed.reset_index(drop=True)


def _fold_eval_mask(timestamps: pd.Series | pd.DatetimeIndex, fold: FoldSpec) -> pd.Series:
    ts = _as_utc_series(timestamps)
    return ((ts >= fold.range.start) & (ts <= fold.range.end)).reset_index(drop=True)


def walk_forward_train_mask(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    manifest: SplitManifest,
    fold: FoldSpec,
    training_mode: TrainingMode,
    horizon_minutes: int,
    train_window_days: int | None = None,
) -> pd.Series:
    """Return rows available for training one evaluation fold.

    The mask is past-only relative to ``fold.range.start``.  A row is allowed
    only if its label interval ends strictly before the evaluated fold starts.

    Modes:
    - fixed: initial manifest.train only
    - expanding: manifest.train plus earlier valid folds
    - rolling: fixed number of days immediately before the fold start
    """
    if training_mode not in {"fixed", "expanding", "rolling"}:
        raise ValueError(f"unknown training_mode: {training_mode!r}")
    if training_mode == "rolling" and (train_window_days is None or int(train_window_days) <= 0):
        raise ValueError("train_window_days must be positive for rolling training")

    ts = _as_utc_series(timestamps)
    eval_start = fold.range.start
    if training_mode == "fixed":
        lower = manifest.train.start
        upper = manifest.train.end
        base = (ts >= lower) & (ts <= upper)
    elif training_mode == "expanding":
        lower = manifest.train.start
        base = (ts >= lower) & (ts < eval_start)
    else:
        lower = eval_start - pd.Timedelta(days=int(train_window_days))
        base = (ts >= lower) & (ts < eval_start)

    safe = label_interval_end(ts, int(horizon_minutes)) < eval_start
    return (base & safe).reset_index(drop=True)


def pre_fold_score_history_mask(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    fold: FoldSpec,
    history_days: int,
) -> pd.Series:
    """Return rows used only to warm a fold's rolling score threshold.

    These rows are scored by the fold's model but are not tradable entries in
    the valid backtest.  They are strictly before the fold start.
    """
    if int(history_days) <= 0:
        raise ValueError("history_days must be positive")
    ts = _as_utc_series(timestamps)
    start = fold.range.start - pd.Timedelta(days=int(history_days))
    end = fold.range.start
    return ((ts >= start) & (ts < end)).reset_index(drop=True)




def _downcast_model_frame(df: pd.DataFrame, *, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Reduce memory pressure for walk-forward training.

    The selected long_H120 feature matrix can be large.  Most features are
    rolling numeric values and do not need float64 precision for LightGBM.
    Downcasting before per-fold slicing avoids repeated large float64 copies.
    """
    out = df
    for col in feature_cols:
        if col in out.columns and pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype("float32", copy=False)
        elif col in out.columns and pd.api.types.is_integer_dtype(out[col]):
            # Keep integer identifiers out of feature_cols; if an integer feature
            # exists, float32 is enough for LightGBM and uses less memory than int64.
            out[col] = out[col].astype("float32", copy=False)
    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype("float32")
    return out


def _recent_true_indices(mask: pd.Series, *, timestamps: pd.Series, max_rows: int | None) -> pd.Index:
    """Return true indices, optionally capped to the most recent rows."""
    idx = mask[mask].index
    if max_rows is None or int(max_rows) <= 0 or len(idx) <= int(max_rows):
        return idx
    # Keep the newest rows.  This is a memory safety knob, not the default
    # research protocol; run_config records it when used.
    ordered = timestamps.loc[idx].sort_values().index
    return ordered[-int(max_rows):]


def _fit_regressor_memory_safe(
    *,
    data: pd.DataFrame,
    train_idx: pd.Index,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
):
    """Fit LightGBM without the extra full train_df.copy() used by the generic helper."""
    assert_no_forbidden_feature_columns(feature_cols)
    y = pd.to_numeric(data.loc[train_idx, target_col], errors="coerce")
    ok = y.notna()
    if int(ok.sum()) == 0:
        raise ModelTrainingError("target is all NaN after numeric conversion")
    idx = train_idx[ok.to_numpy()]
    x_train = data.loc[idx, feature_cols]
    y_train = y.loc[idx]
    model = make_lgbm_regressor(params)
    model.fit(x_train, y_train)
    return model

@dataclass(frozen=True)
class EntryWalkForwardResult:
    run_dir: Path
    model_dir: Path
    target_col: str
    feature_cols: tuple[str, ...]
    fold_metrics: tuple[dict[str, object], ...]
    summary: dict[str, object]


def train_entry_walk_forward(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    manifest: SplitManifest,
    model_config_path: Path | str,
    run_dir: Path,
    model_dir: Path,
    config: EntryWalkForwardConfig,
    run_metadata: dict[str, Any] | None = None,
) -> EntryWalkForwardResult:
    """Retrain entry model per valid fold and write OOF predictions/history.

    Test rows are never predicted.  Score history rows are past-only warmup
    rows for rolling quantile thresholds and must not be treated as entries.
    """
    target_col = config.resolved_target_col
    data = merge_features_and_labels(features, labels, target_col=target_col)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    ts = data["timestamp"]

    feature_cols = numeric_feature_columns(data, exclude={target_col})
    if not feature_cols:
        raise ModelTrainingError("no numeric feature columns available")
    if config.downcast_float32:
        data = _downcast_model_frame(data, feature_cols=feature_cols, target_col=target_col)

    params = load_lgbm_params(model_config_path)
    if config.force_col_wise:
        # Avoid LightGBM's per-fold auto thread-layout probing and prefer the
        # memory-safer path suggested by LightGBM when training large local runs.
        params = {**params, "force_col_wise": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[pd.DataFrame] = []
    history_predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []

    for fold in manifest.folds:
        train_mask = walk_forward_train_mask(
            ts,
            manifest=manifest,
            fold=fold,
            training_mode=config.training_mode,
            horizon_minutes=int(config.horizon_minutes),
            train_window_days=config.train_window_days,
        ) & data[target_col].notna()
        eval_mask = _fold_eval_mask(ts, fold) & data[target_col].notna()
        history_mask = pre_fold_score_history_mask(
            ts,
            fold=fold,
            history_days=int(config.score_history_days),
        )

        train_idx = _recent_true_indices(
            train_mask,
            timestamps=data["timestamp"],
            max_rows=config.max_train_rows,
        )
        eval_idx = eval_mask[eval_mask].index
        history_idx = history_mask[history_mask].index
        if len(train_idx) == 0:
            raise ModelTrainingError(f"{fold.name}: no train rows for {config.training_mode} walk-forward")
        if len(eval_idx) == 0:
            raise ModelTrainingError(f"{fold.name}: no eval rows with non-null target")

        model = _fit_regressor_memory_safe(
            data=data,
            train_idx=train_idx,
            feature_cols=feature_cols,
            target_col=target_col,
            params=params,
        )
        eval_view = data.loc[eval_idx, ["timestamp", target_col] + feature_cols]
        pred = predict_regressor(model, eval_view, feature_cols)
        pred_df = pd.DataFrame({
            "timestamp": eval_view["timestamp"].to_numpy(),
            "fold": fold.name,
            "side": config.side,
            "horizon_minutes": int(config.horizon_minutes),
            target_col: eval_view[target_col].to_numpy(),
            "pred_entry_net_bps": pred,
            "is_oof": True,
            "training_mode": config.training_mode,
            "train_window_days": config.train_window_days,
            "train_count": int(len(train_idx)),
            "max_train_rows": config.max_train_rows,
            "force_col_wise": bool(config.force_col_wise),
        })
        predictions.append(pred_df)

        if len(history_idx) > 0:
            history_view = data.loc[history_idx, ["timestamp"] + feature_cols]
            hist_pred = predict_regressor(model, history_view, feature_cols)
            history_predictions.append(pd.DataFrame({
                "timestamp": history_view["timestamp"].to_numpy(),
                "fold": fold.name,
                "side": config.side,
                "horizon_minutes": int(config.horizon_minutes),
                "pred_entry_net_bps": hist_pred,
                "source": "pre_fold_score_history",
                "training_mode": config.training_mode,
                "train_window_days": config.train_window_days,
                "max_train_rows": config.max_train_rows,
                "force_col_wise": bool(config.force_col_wise),
            }))

        fold_metrics = regression_fold_metrics(
            pred_df[target_col],
            pred_df["pred_entry_net_bps"],
            fold_name=fold.name,
            quantiles=config.quantiles,
        )
        fold_metrics.update({
            "train_count": int(len(train_idx)),
            "raw_train_count_before_cap": int(train_mask.sum()),
            "max_train_rows": config.max_train_rows,
            "score_history_count": int(len(history_idx)),
            "target_col": target_col,
            "side": config.side,
            "horizon_minutes": int(config.horizon_minutes),
            "training_mode": config.training_mode,
            "train_window_days": config.train_window_days,
        })
        metrics.append(fold_metrics)
        if config.save_feature_importance:
            importances.append(feature_importance_frame(model, feature_cols, fold=fold.name))

        if config.save_models:
            save_lgbm_model(model, model_dir / f"{fold.name}.txt")

        # Release per-fold objects aggressively.  On local laptops, expanding
        # mode with hundreds of features can otherwise be killed by the OS.
        del model
        if "eval_view" in locals():
            del eval_view
        if "history_view" in locals():
            del history_view
        gc.collect()

    predictions_df = pd.concat(predictions, ignore_index=True).sort_values(["timestamp", "fold"]).reset_index(drop=True)
    summary = write_entry_report(
        predictions=predictions_df,
        fold_metrics=metrics,
        output_dir=run_dir,
        target_col=target_col,
    )
    # Keep the legacy OOF filename expected by downstream backtest scripts.
    write_frame(predictions_df, run_dir / "entry_oof_predictions.parquet")

    if history_predictions:
        history_df_all = pd.concat(history_predictions, ignore_index=True).sort_values(["fold", "timestamp"]).reset_index(drop=True)
    else:
        history_df_all = pd.DataFrame(columns=[
            "timestamp",
            "fold",
            "side",
            "horizon_minutes",
            "pred_entry_net_bps",
            "source",
            "training_mode",
            "train_window_days",
        ])
    write_frame(history_df_all, run_dir / "score_history.parquet")

    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    if not imp_df.empty:
        write_frame(imp_df, run_dir / "feature_importance_by_fold.csv")
        agg = imp_df.groupby("feature", as_index=False).agg(
            mean_importance_split=("importance_split", "mean"),
            mean_importance_gain=("importance_gain", "mean"),
        ).sort_values("mean_importance_gain", ascending=False)
        write_frame(agg, run_dir / "feature_importance_mean.csv")

    run_config = {
        "model_role": "entry_walk_forward",
        "side": config.side,
        "horizon_minutes": int(config.horizon_minutes),
        "target_col": target_col,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "lgbm_params": params,
        "split_version": manifest.split_version,
        "test_usage": manifest.test_usage,
        "training_mode": config.training_mode,
        "train_window_days": config.train_window_days,
        "score_history_days": int(config.score_history_days),
        "score_history_rows": int(len(history_df_all)),
        "downcast_float32": bool(config.downcast_float32),
        "max_train_rows": config.max_train_rows,
        "save_feature_importance": bool(config.save_feature_importance),
        "force_col_wise": bool(config.force_col_wise),
        "notes": [
            "Walk-forward valid evaluation only; test rows are not predicted.",
            "Training rows are past-only relative to each eval fold and label-overlap safe.",
            "score_history.parquet is for rolling threshold warmup only and is not a tradable entry set.",
            "Feature set and policy thresholds must be fixed before using this run for locked test audit.",
        ],
        "metadata": run_metadata or {},
    }
    write_json(run_config, run_dir / "run_config.json")

    return EntryWalkForwardResult(
        run_dir=run_dir,
        model_dir=model_dir,
        target_col=target_col,
        feature_cols=tuple(feature_cols),
        fold_metrics=tuple(metrics),
        summary=summary,
    )

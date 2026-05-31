"""Entry LightGBM training for valid-fold research.

This module trains entry regressors on train rows and evaluates them on the
locked valid folds.  It must not use test rows for fitting, selection, or
reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json
from swing_bot.evaluation.entry_report import write_entry_report
from swing_bot.evaluation.fold_metrics import regression_fold_metrics
from swing_bot.labels.entry_net_return import entry_target_column
from swing_bot.models.lgbm_common import (
    ModelTrainingError,
    feature_importance_frame,
    fit_regressor,
    load_lgbm_params,
    numeric_feature_columns,
    predict_regressor,
    save_lgbm_model,
)
from swing_bot.splits.purged_walk_forward import valid_fold_masks
from swing_bot.splits.split_manifest import SplitManifest


@dataclass(frozen=True)
class EntryTrainingConfig:
    side: str = "long"
    horizon_minutes: int = 60
    target_col: str | None = None
    max_label_minutes: int | None = None
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)
    save_models: bool = True

    @property
    def resolved_target_col(self) -> str:
        return self.target_col or entry_target_column(self.side, self.horizon_minutes)


@dataclass(frozen=True)
class EntryTrainingResult:
    run_dir: Path
    model_dir: Path
    target_col: str
    feature_cols: tuple[str, ...]
    fold_metrics: tuple[dict[str, object], ...]
    summary: dict[str, object]


def merge_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    target_col: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Merge feature matrix with one target column by exact timestamp."""
    if timestamp_col not in features.columns:
        raise ModelTrainingError(f"features missing {timestamp_col!r}")
    if timestamp_col not in labels.columns:
        raise ModelTrainingError(f"labels missing {timestamp_col!r}")
    if target_col not in labels.columns:
        raise ModelTrainingError(f"labels missing target column {target_col!r}")

    left = features.copy()
    right = labels[[timestamp_col, target_col]].copy()
    left[timestamp_col] = pd.to_datetime(left[timestamp_col], utc=True)
    right[timestamp_col] = pd.to_datetime(right[timestamp_col], utc=True)
    if left[timestamp_col].duplicated().any():
        raise ModelTrainingError("features contain duplicate timestamps")
    if right[timestamp_col].duplicated().any():
        raise ModelTrainingError("labels contain duplicate timestamps")
    out = left.merge(right, on=timestamp_col, how="inner", validate="one_to_one")
    if out.empty:
        raise ModelTrainingError("feature/label merge produced zero rows")
    return out.sort_values(timestamp_col).reset_index(drop=True)


def train_entry_valid_folds(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    manifest: SplitManifest,
    model_config_path: Path | str,
    run_dir: Path,
    model_dir: Path,
    config: EntryTrainingConfig,
    run_metadata: dict[str, Any] | None = None,
) -> EntryTrainingResult:
    """Train/evaluate an entry regressor on locked valid folds.

    Training rows come only from ``manifest.train`` after purge.  Evaluation rows
    come only from each valid fold.  Test rows are never predicted or reported.
    """
    target_col = config.resolved_target_col
    data = merge_features_and_labels(features, labels, target_col=target_col)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    feature_cols = numeric_feature_columns(data, exclude={target_col})
    if not feature_cols:
        raise ModelTrainingError("no numeric feature columns available")

    params = load_lgbm_params(model_config_path)
    max_label_minutes = config.max_label_minutes if config.max_label_minutes is not None else int(config.horizon_minutes)
    masks = valid_fold_masks(data["timestamp"], manifest, max_label_minutes=max_label_minutes)

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []

    for fold_name, fold_masks in masks.items():
        train_mask = fold_masks["train"] & data[target_col].notna()
        eval_mask = fold_masks["eval"] & data[target_col].notna()
        train_df = data.loc[train_mask].copy()
        eval_df = data.loc[eval_mask].copy()
        if train_df.empty:
            raise ModelTrainingError(f"{fold_name}: no train rows after purge")
        if eval_df.empty:
            raise ModelTrainingError(f"{fold_name}: no eval rows with non-null target")

        model = fit_regressor(train_df=train_df, feature_cols=feature_cols, target_col=target_col, params=params)
        pred = predict_regressor(model, eval_df, feature_cols)

        pred_df = pd.DataFrame({
            "timestamp": eval_df["timestamp"].to_numpy(),
            "fold": fold_name,
            "side": config.side,
            "horizon_minutes": int(config.horizon_minutes),
            target_col: eval_df[target_col].to_numpy(),
            "pred_entry_net_bps": pred,
            "is_oof": True,
        })
        predictions.append(pred_df)
        fold_metrics = regression_fold_metrics(
            pred_df[target_col],
            pred_df["pred_entry_net_bps"],
            fold_name=fold_name,
            quantiles=config.quantiles,
        )
        fold_metrics.update({
            "train_count": int(len(train_df)),
            "target_col": target_col,
            "side": config.side,
            "horizon_minutes": int(config.horizon_minutes),
        })
        metrics.append(fold_metrics)
        importances.append(feature_importance_frame(model, feature_cols, fold=fold_name))

        if config.save_models:
            save_lgbm_model(model, model_dir / f"{fold_name}.txt")

    predictions_df = pd.concat(predictions, ignore_index=True).sort_values(["timestamp", "fold"]).reset_index(drop=True)
    summary = write_entry_report(
        predictions=predictions_df,
        fold_metrics=metrics,
        output_dir=run_dir,
        target_col=target_col,
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
        "model_role": "entry",
        "side": config.side,
        "horizon_minutes": int(config.horizon_minutes),
        "target_col": target_col,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "lgbm_params": params,
        "split_version": manifest.split_version,
        "test_usage": manifest.test_usage,
        "notes": [
            "Train rows are manifest.train only, purged per valid fold.",
            "Evaluation rows are valid folds only.",
            "Test rows are not predicted or reported by this run.",
        ],
        "metadata": run_metadata or {},
    }
    write_json(run_config, run_dir / "run_config.json")

    return EntryTrainingResult(
        run_dir=run_dir,
        model_dir=model_dir,
        target_col=target_col,
        feature_cols=tuple(feature_cols),
        fold_metrics=tuple(metrics),
        summary=summary,
    )

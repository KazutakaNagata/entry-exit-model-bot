"""Exit model report helpers.

Exit models predict ``hold_delta_bps``: positive values mean that holding for the
configured lookahead was better than exiting immediately.  The most important
checks are therefore ranking checks:

* top predicted rows should have positive realized hold_delta;
* bottom predicted rows should have negative realized hold_delta;
* score deciles should be reasonably monotonic inside each valid fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json
from swing_bot.evaluation.fold_metrics import regression_fold_metrics, summarize_fold_metrics
from swing_bot.evaluation.topk import DEFAULT_TOP_QUANTILES, score_decile_summary, top_quantile_metrics


def bottom_quantile_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_TOP_QUANTILES,
) -> dict[str, float | int]:
    """Return realized target metrics for rows with the lowest predicted scores.

    ``q=0.95`` means the bottom ``5%`` of predictions.  For exit models these
    rows are expected to be exit candidates, so realized hold_delta should often
    be low or negative if the model is useful.
    """
    frame = pd.DataFrame({"target": y_true, "pred": y_pred})
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["pred"] = pd.to_numeric(frame["pred"], errors="coerce")
    frame = frame.dropna(subset=["target", "pred"])

    metrics: dict[str, float | int] = {}
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"bottom quantile must be between 0 and 1: {q}")
        suffix = f"q{int(round(q * 100)):02d}"
        if frame.empty:
            metrics[f"bottom_{suffix}_count"] = 0
            metrics[f"bottom_{suffix}_avg_target_bps"] = float("nan")
            metrics[f"bottom_{suffix}_median_target_bps"] = float("nan")
            metrics[f"bottom_{suffix}_precision_lt0"] = float("nan")
            metrics[f"bottom_{suffix}_pred_threshold"] = float("nan")
            continue
        threshold = frame["pred"].quantile(1.0 - float(q))
        bottom = frame.loc[frame["pred"] <= threshold]
        metrics[f"bottom_{suffix}_count"] = int(len(bottom))
        metrics[f"bottom_{suffix}_avg_target_bps"] = float(bottom["target"].mean()) if len(bottom) else float("nan")
        metrics[f"bottom_{suffix}_median_target_bps"] = float(bottom["target"].median()) if len(bottom) else float("nan")
        metrics[f"bottom_{suffix}_precision_lt0"] = float((bottom["target"] < 0).mean()) if len(bottom) else float("nan")
        metrics[f"bottom_{suffix}_pred_threshold"] = float(threshold)
    return metrics


def exit_fold_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    fold_name: str,
    quantiles: Sequence[float] = DEFAULT_TOP_QUANTILES,
) -> dict[str, float | int | str]:
    """Return fold metrics tailored to supervised exit regression."""
    metrics = regression_fold_metrics(y_true, y_pred, fold_name=fold_name, quantiles=quantiles)
    frame = pd.DataFrame({"target": y_true, "pred": y_pred}).dropna(subset=["target", "pred"])
    if frame.empty:
        metrics.update({
            "spearman_corr": float("nan"),
            "top_bottom_q95_spread_bps": float("nan"),
            "top_bottom_q99_spread_bps": float("nan"),
        })
    else:
        target_rank = frame["target"].rank(method="average")
        pred_rank = frame["pred"].rank(method="average")
        metrics["spearman_corr"] = float(target_rank.corr(pred_rank)) if len(frame) >= 2 else float("nan")
    metrics.update(bottom_quantile_metrics(y_true, y_pred, quantiles=quantiles))
    for q in (0.95, 0.99):
        suffix = f"q{int(round(q * 100)):02d}"
        top_key = f"top_{suffix}_avg_target_bps"
        bottom_key = f"bottom_{suffix}_avg_target_bps"
        spread_key = f"top_bottom_{suffix}_spread_bps"
        top = metrics.get(top_key)
        bottom = metrics.get(bottom_key)
        try:
            metrics[spread_key] = float(top) - float(bottom)  # type: ignore[arg-type]
        except Exception:
            metrics[spread_key] = float("nan")
    return metrics


def write_exit_report(
    *,
    predictions: pd.DataFrame,
    fold_metrics: Sequence[dict[str, object]],
    output_dir: Path,
    target_col: str = "target_exit_hold_delta_bps",
    pred_col: str = "pred_exit_hold_delta_bps",
) -> dict[str, object]:
    """Write exit fold metrics, deciles, predictions, and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(fold_metrics)
    write_frame(metrics_df, output_dir / "fold_metrics.csv")

    decile_rows = []
    for fold_name, fold_df in predictions.groupby("fold", sort=True):
        deciles = score_decile_summary(fold_df[target_col], fold_df[pred_col])
        if not deciles.empty:
            deciles.insert(0, "fold", fold_name)
            decile_rows.append(deciles)
    decile_df = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()
    write_frame(decile_df, output_dir / "score_deciles.csv")
    write_frame(predictions, output_dir / "predictions_valid.parquet")

    summary = summarize_fold_metrics(fold_metrics)
    write_json(summary, output_dir / "summary_metrics.json")
    return summary

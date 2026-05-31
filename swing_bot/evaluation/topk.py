"""Top-score regression metrics.

For entry models, the main question is not generic RMSE.  It is whether the
highest predicted rows actually have positive cost-adjusted future returns.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_TOP_QUANTILES = (0.90, 0.95, 0.97, 0.99)


def _safe_float(value: float | int | np.floating) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def top_quantile_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_TOP_QUANTILES,
) -> dict[str, float | int]:
    """Return realized target metrics for rows with top predicted scores.

    ``q=0.95`` means rows whose prediction is at or above the 95th percentile of
    predictions inside the evaluated fold.
    """
    frame = pd.DataFrame({"target": y_true, "pred": y_pred})
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["pred"] = pd.to_numeric(frame["pred"], errors="coerce")
    frame = frame.dropna(subset=["target", "pred"])

    metrics: dict[str, float | int] = {"eval_count": int(len(frame))}
    if frame.empty:
        for q in quantiles:
            suffix = f"q{int(round(q * 100)):02d}"
            metrics[f"top_{suffix}_count"] = 0
            metrics[f"top_{suffix}_avg_target_bps"] = float("nan")
            metrics[f"top_{suffix}_median_target_bps"] = float("nan")
            metrics[f"top_{suffix}_precision_gt0"] = float("nan")
            metrics[f"top_{suffix}_pred_threshold"] = float("nan")
        return metrics

    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"top quantile must be between 0 and 1: {q}")
        threshold = frame["pred"].quantile(q)
        top = frame.loc[frame["pred"] >= threshold]
        suffix = f"q{int(round(q * 100)):02d}"
        metrics[f"top_{suffix}_count"] = int(len(top))
        metrics[f"top_{suffix}_avg_target_bps"] = _safe_float(top["target"].mean())
        metrics[f"top_{suffix}_median_target_bps"] = _safe_float(top["target"].median())
        metrics[f"top_{suffix}_precision_gt0"] = _safe_float((top["target"] > 0).mean()) if len(top) else float("nan")
        metrics[f"top_{suffix}_pred_threshold"] = _safe_float(threshold)
    return metrics


def score_decile_summary(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return realized target summary by prediction decile.

    Decile 10 is the highest predicted-score bucket.
    """
    frame = pd.DataFrame({"target": y_true, "pred": y_pred}).dropna(subset=["target", "pred"])
    if frame.empty:
        return pd.DataFrame(columns=["score_decile", "count", "avg_target_bps", "median_target_bps", "precision_gt0"])
    ranks = frame["pred"].rank(method="first")
    frame["score_decile"] = pd.qcut(ranks, q=min(n_bins, len(frame)), labels=False, duplicates="drop") + 1
    grouped = frame.groupby("score_decile", observed=True)
    out = grouped.agg(
        count=("target", "size"),
        avg_target_bps=("target", "mean"),
        median_target_bps=("target", "median"),
        precision_gt0=("target", lambda s: float((s > 0).mean())),
        avg_pred=("pred", "mean"),
    ).reset_index()
    out["score_decile"] = out["score_decile"].astype(int)
    return out.sort_values("score_decile").reset_index(drop=True)

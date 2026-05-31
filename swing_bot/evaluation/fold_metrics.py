"""Fold-level regression metrics."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from swing_bot.evaluation.topk import DEFAULT_TOP_QUANTILES, top_quantile_metrics


def regression_fold_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    fold_name: str,
    quantiles: Sequence[float] = DEFAULT_TOP_QUANTILES,
) -> dict[str, float | int | str]:
    """Return generic + top-score metrics for one fold."""
    frame = pd.DataFrame({"target": y_true, "pred": y_pred})
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["pred"] = pd.to_numeric(frame["pred"], errors="coerce")
    frame = frame.dropna(subset=["target", "pred"])

    base: dict[str, float | int | str] = {"fold": fold_name, "eval_count": int(len(frame))}
    if frame.empty:
        base.update({
            "target_mean_bps": float("nan"),
            "pred_mean_bps": float("nan"),
            "mae_bps": float("nan"),
            "rmse_bps": float("nan"),
            "corr": float("nan"),
            "r2": float("nan"),
            "precision_gt0_all": float("nan"),
        })
        base.update(top_quantile_metrics([], [], quantiles=quantiles))
        return base

    target = frame["target"]
    pred = frame["pred"]
    corr = target.corr(pred) if target.nunique(dropna=True) > 1 and pred.nunique(dropna=True) > 1 else float("nan")
    base.update({
        "target_mean_bps": float(target.mean()),
        "target_median_bps": float(target.median()),
        "pred_mean_bps": float(pred.mean()),
        "mae_bps": float(mean_absolute_error(target, pred)),
        "rmse_bps": float(np.sqrt(mean_squared_error(target, pred))),
        "corr": float(corr),
        "r2": float(r2_score(target, pred)) if len(frame) >= 2 else float("nan"),
        "precision_gt0_all": float((target > 0).mean()),
    })
    top = top_quantile_metrics(target, pred, quantiles=quantiles)
    # keep the explicit base eval_count instead of the duplicate returned by top
    top.pop("eval_count", None)
    base.update(top)
    return base


def summarize_fold_metrics(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return mean/worst aggregates for numeric fold metrics."""
    if not metrics:
        return {"fold_count": 0}
    frame = pd.DataFrame(metrics)
    out: dict[str, object] = {"fold_count": int(len(frame))}
    numeric_cols = [c for c in frame.columns if c != "fold" and pd.api.types.is_numeric_dtype(frame[c])]
    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            out[f"mean_{col}"] = float(values.mean())
            # For counts, min is still useful; for returns/precision it is the worst fold.
            out[f"worst_{col}"] = float(values.min())
        else:
            out[f"mean_{col}"] = float("nan")
            out[f"worst_{col}"] = float("nan")
    return out

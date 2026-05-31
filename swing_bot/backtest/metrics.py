"""Episode-level metrics for valid-fold backtests."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = values[values < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def _round_trips_per_day(frame: pd.DataFrame) -> float:
    if frame.empty or "entry_time" not in frame.columns:
        return float("nan")
    dates = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce").dt.date.dropna().unique()
    day_count = max(int(len(dates)), 1)
    return float(len(frame) / day_count)


def episode_fold_metrics(episodes: pd.DataFrame, *, fold_name: str) -> dict[str, object]:
    """Return one fold's episode metrics.

    Metrics are intentionally simple at this MVP stage.  They are for plumbing
    checks first, and for strategy comparison only after stronger features are
    introduced.
    """
    out: dict[str, object] = {"fold": fold_name}
    if episodes.empty:
        out.update({
            "episode_count": 0,
            "gross_pl_bps_sum": 0.0,
            "net_pl_bps_sum": 0.0,
            "avg_net_pl_bps": float("nan"),
            "median_net_pl_bps": float("nan"),
            "win_rate": float("nan"),
            "avg_win_bps": float("nan"),
            "avg_loss_bps": float("nan"),
            "profit_factor": float("nan"),
            "avg_hold_minutes": float("nan"),
            "median_hold_minutes": float("nan"),
            "round_trips_per_day": 0.0,
            "fee_paid_bps": 0.0,
            "avg_mfe_bps": float("nan"),
            "avg_mae_bps": float("nan"),
            "avg_giveback_bps": float("nan"),
        })
        return out

    net = pd.to_numeric(episodes["net_pl_bps"], errors="coerce").dropna()
    gross = pd.to_numeric(episodes["gross_pl_bps"], errors="coerce").dropna()
    hold = pd.to_numeric(episodes.get("hold_minutes", pd.Series(dtype="float64")), errors="coerce").dropna()
    cost = pd.to_numeric(episodes.get("roundtrip_cost_bps", pd.Series(dtype="float64")), errors="coerce").fillna(0.0)

    out.update({
        "episode_count": int(len(episodes)),
        "gross_pl_bps_sum": float(gross.sum()) if len(gross) else float("nan"),
        "net_pl_bps_sum": float(net.sum()) if len(net) else float("nan"),
        "avg_net_pl_bps": float(net.mean()) if len(net) else float("nan"),
        "median_net_pl_bps": float(net.median()) if len(net) else float("nan"),
        "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "avg_win_bps": float(net[net > 0].mean()) if (net > 0).any() else float("nan"),
        "avg_loss_bps": float(net[net < 0].mean()) if (net < 0).any() else float("nan"),
        "profit_factor": _profit_factor(net),
        "avg_hold_minutes": float(hold.mean()) if len(hold) else float("nan"),
        "median_hold_minutes": float(hold.median()) if len(hold) else float("nan"),
        "round_trips_per_day": _round_trips_per_day(episodes),
        "fee_paid_bps": float(cost.sum()),
        "avg_mfe_bps": _safe_float(pd.to_numeric(episodes.get("mfe_bps", pd.Series(dtype="float64")), errors="coerce").mean()),
        "avg_mae_bps": _safe_float(pd.to_numeric(episodes.get("mae_bps", pd.Series(dtype="float64")), errors="coerce").mean()),
        "avg_giveback_bps": _safe_float(pd.to_numeric(episodes.get("giveback_bps", pd.Series(dtype="float64")), errors="coerce").mean()),
    })
    return out


def summarize_episode_metrics(fold_metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return mean/worst summary over fold metrics."""
    if not fold_metrics:
        return {"fold_count": 0}
    df = pd.DataFrame(fold_metrics)
    summary: dict[str, object] = {"fold_count": int(len(df))}
    for col in df.columns:
        if col == "fold" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        summary[f"mean_{col}"] = float(values.mean()) if values.notna().any() else float("nan")
        # For P/L-like metrics, lower is worse.  For cost/episode counts this is still useful as a sanity check.
        summary[f"worst_{col}"] = float(values.min()) if values.notna().any() else float("nan")
    return summary

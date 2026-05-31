"""Cost-aware fixed-hold entry targets.

For decision row `t`, features may only use data up to `t`.  The entry target
therefore assumes execution at `t+1 open` and fixed-horizon exit at
`t+1+H open`.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from swing_bot.labels._time_lookup import price_at_offset_minutes
from swing_bot.labels.costs import net_pl_bps

DEFAULT_ENTRY_HORIZONS_MINUTES = (60, 120, 240)
DEFAULT_SIDES = ("long", "short")


def entry_target_column(side: str, horizon_minutes: int) -> str:
    return f"target_entry_net_bps_{side}_H{int(horizon_minutes)}"


def make_entry_net_return_target(
    df: pd.DataFrame,
    *,
    side: str,
    horizon_minutes: int,
    roundtrip_cost_bps: float = 15.0,
    price_col: str = "open",
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Create one fixed-hold net return target.

    The target is side-normalized.  Positive values mean the entry would have
    made money after roundtrip cost.
    """
    entry_price = price_at_offset_minutes(df, 1, price_col=price_col, timestamp_col=timestamp_col)
    exit_price = price_at_offset_minutes(df, 1 + int(horizon_minutes), price_col=price_col, timestamp_col=timestamp_col)
    target = net_pl_bps(entry_price, exit_price, side, roundtrip_cost_bps=roundtrip_cost_bps)
    return pd.Series(target, index=df.index, name=entry_target_column(side, horizon_minutes))


def add_entry_net_return_targets(
    df: pd.DataFrame,
    *,
    horizons_minutes: Sequence[int] = DEFAULT_ENTRY_HORIZONS_MINUTES,
    sides: Sequence[str] = DEFAULT_SIDES,
    roundtrip_cost_bps: float = 15.0,
    price_col: str = "open",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return a copy of `df` with entry targets appended."""
    out = df.copy()
    for side in sides:
        for horizon in horizons_minutes:
            col = entry_target_column(side, int(horizon))
            out[col] = make_entry_net_return_target(
                out,
                side=side,
                horizon_minutes=int(horizon),
                roundtrip_cost_bps=roundtrip_cost_bps,
                price_col=price_col,
                timestamp_col=timestamp_col,
            )
    return out

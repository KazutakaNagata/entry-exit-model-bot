"""Hold-delta targets for supervised exit models.

For decision row `t`, the initial simple target compares exiting at `t+1 open`
versus exiting at `t+1+K open`.  It does not subtract a fresh roundtrip cost for
every hold decision; fixed costs are handled at episode level.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from swing_bot.labels._time_lookup import price_at_offset_minutes
from swing_bot.labels.costs import side_normalized_return_bps

DEFAULT_EXIT_LOOKAHEADS_MINUTES: dict[str, tuple[int, ...]] = {
    "long": (30, 60),
    "short": (15, 30),
}


def exit_target_column(side: str, lookahead_minutes: int) -> str:
    return f"target_exit_hold_delta_bps_{side}_K{int(lookahead_minutes)}"


def make_exit_hold_delta_target(
    df: pd.DataFrame,
    *,
    side: str,
    lookahead_minutes: int,
    price_col: str = "open",
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Create one hold-delta target for an already-open position."""
    current_exit_price = price_at_offset_minutes(df, 1, price_col=price_col, timestamp_col=timestamp_col)
    future_exit_price = price_at_offset_minutes(df, 1 + int(lookahead_minutes), price_col=price_col, timestamp_col=timestamp_col)
    target = side_normalized_return_bps(current_exit_price, future_exit_price, side)
    return pd.Series(target, index=df.index, name=exit_target_column(side, lookahead_minutes))


def add_exit_hold_delta_targets(
    df: pd.DataFrame,
    *,
    lookaheads_minutes: Mapping[str, Sequence[int]] | None = None,
    price_col: str = "open",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return a copy of `df` with exit hold-delta targets appended."""
    out = df.copy()
    effective = lookaheads_minutes or DEFAULT_EXIT_LOOKAHEADS_MINUTES
    for side, lookaheads in effective.items():
        for lookahead in lookaheads:
            col = exit_target_column(side, int(lookahead))
            out[col] = make_exit_hold_delta_target(
                out,
                side=side,
                lookahead_minutes=int(lookahead),
                price_col=price_col,
                timestamp_col=timestamp_col,
            )
    return out

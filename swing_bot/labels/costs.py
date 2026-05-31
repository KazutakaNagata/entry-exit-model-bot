"""Cost and side-normalized return helpers.

All helpers are deliberately small and deterministic.  Entry targets subtract the
full roundtrip cost.  Exit hold-delta targets normally do not subtract the full
roundtrip cost because they compare exit-now versus exit-later for an already
open position.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["long", "short"]


def side_to_sign(side: str | int | float) -> int:
    """Convert long/short or +/-1 to +1/-1."""
    if isinstance(side, str):
        normalized = side.strip().lower()
        if normalized in {"long", "buy", "+1", "1"}:
            return 1
        if normalized in {"short", "sell", "-1"}:
            return -1
    if side == 1:
        return 1
    if side == -1:
        return -1
    raise ValueError(f"side must be 'long', 'short', +1, or -1; got {side!r}")


def side_normalized_return_bps(
    entry_price: float | pd.Series | np.ndarray,
    exit_price: float | pd.Series | np.ndarray,
    side: str | int | float,
) -> float | pd.Series | np.ndarray:
    """Return log return in bps where positive means favorable for the side."""
    sign = side_to_sign(side)
    result = sign * np.log(np.asarray(exit_price, dtype="float64") / np.asarray(entry_price, dtype="float64")) * 10000.0
    if isinstance(exit_price, pd.Series):
        return pd.Series(result, index=exit_price.index, name=exit_price.name)
    if isinstance(entry_price, pd.Series):
        return pd.Series(result, index=entry_price.index, name=entry_price.name)
    if np.ndim(result) == 0:
        return float(result)
    return result


def net_pl_bps(
    entry_price: float | pd.Series | np.ndarray,
    exit_price: float | pd.Series | np.ndarray,
    side: str | int | float,
    *,
    roundtrip_cost_bps: float = 15.0,
) -> float | pd.Series | np.ndarray:
    """Return side-normalized net P/L in bps after roundtrip cost."""
    gross = side_normalized_return_bps(entry_price, exit_price, side)
    return gross - float(roundtrip_cost_bps)


def subtract_roundtrip_cost(gross_bps: float | pd.Series | np.ndarray, *, roundtrip_cost_bps: float = 15.0):
    """Subtract roundtrip cost from a gross bps value or vector."""
    return gross_bps - float(roundtrip_cost_bps)

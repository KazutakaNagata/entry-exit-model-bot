"""Lightweight past-only pivot-line geometry features.

These are not centered pivots.  They use previous rolling highs/lows and classic
OHLC pivot approximations from past windows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

PIVOT_WINDOWS = (60, 120, 240, 480)
_EPS = 1e-12


def _log_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log((a + _EPS) / (b + _EPS)) * 10000.0


def build_pivot_line_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = PIVOT_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build distances to previous rolling pivot approximations."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        prev_close = close.shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2.0 * pivot - prev_low
        s1 = 2.0 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)

        for level_name, level in (("pivot", pivot), ("r1", r1), ("s1", s1), ("r2", r2), ("s2", s2)):
            name = f"pl_dist_to_{level_name}_{window}m_bps"
            out[name] = _log_bps(close, level)
            specs.append(FeatureSpec(name, "pivot_line", window + 1, description=f"Close distance to previous {window}m {level_name} line."))

        name = f"pl_above_pivot_{window}m"
        out[name] = (close > pivot).astype("float64")
        specs.append(FeatureSpec(name, "pivot_line", window + 1, description=f"Close above previous {window}m pivot line."))

        name = f"pl_between_pivot_r1_{window}m"
        out[name] = ((close > pivot) & (close < r1)).astype("float64")
        specs.append(FeatureSpec(name, "pivot_line", window + 1, description=f"Close between previous {window}m pivot and R1."))

    return out, specs

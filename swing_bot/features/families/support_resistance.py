"""Past-only support/resistance geometry features.

These features intentionally use only finalized bars at or before row ``t``.
Breakout/breakdown features compare ``close[t]`` with support/resistance levels
formed strictly before ``t`` via ``shift(1).rolling(...)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

SR_WINDOWS = (15, 30, 60, 120, 240)
_EPS = 1e-12


def _safe_range_position(close: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
    denom = (high - low).replace(0.0, np.nan)
    return (close - low) / denom


def build_support_resistance_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = SR_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build simple distances to recent support/resistance levels.

    The rolling high/low features include the current finalized bar, which is
    available after ``bar_close``.  The ``prev_*`` breakout features exclude the
    current bar so a current close can be compared with already-known levels.
    """
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    for window in windows:
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()

        name = f"sr_dist_to_support_{window}m_bps"
        out[name] = np.log((close + _EPS) / (rolling_low + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "support_resistance", window, description=f"Close distance above past {window}m rolling low."))

        name = f"sr_dist_to_resistance_{window}m_bps"
        out[name] = np.log((rolling_high + _EPS) / (close + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "support_resistance", window, description=f"Close distance below past {window}m rolling high."))

        name = f"sr_position_in_range_{window}m"
        out[name] = _safe_range_position(close, rolling_low, rolling_high)
        specs.append(FeatureSpec(name, "support_resistance", window, description=f"Close position inside past {window}m high/low range."))

        name = f"sr_break_above_prev_resistance_{window}m_bps"
        out[name] = np.log((close + _EPS) / (prev_high + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "support_resistance", window + 1, description=f"Close versus previous {window}m high, excluding current bar."))

        name = f"sr_break_below_prev_support_{window}m_bps"
        out[name] = np.log((prev_low + _EPS) / (close + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "support_resistance", window + 1, description=f"Previous {window}m low versus close, excluding current bar."))

    return out, specs

"""Past-only market-structure room / void features.

The goal is not to model a perfect order-book void.  These are reviewed,
minimal geometry proxies for upside/downside room relative to previous local
extremes and the age of those extremes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

MSV_WINDOWS = (30, 60, 120, 240)
_EPS = 1e-12


def _bars_since_extreme(values: pd.Series, window: int, *, kind: str) -> pd.Series:
    """Bars since max/min within a rolling window, using only the window ending at t."""
    if kind not in {"max", "min"}:
        raise ValueError("kind must be 'max' or 'min'")

    def calc(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        idx = int(np.argmax(arr) if kind == "max" else np.argmin(arr))
        return float(len(arr) - 1 - idx)

    return values.astype("float64").rolling(window=window, min_periods=window).apply(calc, raw=True)


def build_market_structure_void_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = MSV_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build room-to-extreme and extreme-age features from past OHLCV."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        prev_range = (prev_high - prev_low).replace(0.0, np.nan)

        name = f"msv_upside_room_to_prev_high_{window}m_bps"
        out[name] = np.log((prev_high + _EPS) / (close + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "market_structure_void", window + 1, description=f"Room from close to previous {window}m high; negative after breakout."))

        name = f"msv_downside_room_to_prev_low_{window}m_bps"
        out[name] = np.log((close + _EPS) / (prev_low + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "market_structure_void", window + 1, description=f"Room from close to previous {window}m low; negative after breakdown."))

        name = f"msv_position_vs_prev_range_{window}m"
        out[name] = (close - prev_low) / prev_range
        specs.append(FeatureSpec(name, "market_structure_void", window + 1, description=f"Close position inside previous {window}m high/low range."))

        name = f"msv_breakout_over_prev_high_{window}m_bps"
        out[name] = np.maximum(np.log((close + _EPS) / (prev_high + _EPS)) * 10000.0, 0.0)
        specs.append(FeatureSpec(name, "market_structure_void", window + 1, description=f"Positive close extension above previous {window}m high."))

        name = f"msv_breakdown_under_prev_low_{window}m_bps"
        out[name] = np.maximum(np.log((prev_low + _EPS) / (close + _EPS)) * 10000.0, 0.0)
        specs.append(FeatureSpec(name, "market_structure_void", window + 1, description=f"Positive close extension below previous {window}m low."))

        name = f"msv_bars_since_high_{window}m"
        out[name] = _bars_since_extreme(high, window, kind="max")
        specs.append(FeatureSpec(name, "market_structure_void", window, description=f"Bars elapsed since highest high inside past {window}m window."))

        name = f"msv_bars_since_low_{window}m"
        out[name] = _bars_since_extreme(low, window, kind="min")
        specs.append(FeatureSpec(name, "market_structure_void", window, description=f"Bars elapsed since lowest low inside past {window}m window."))

    return out, specs

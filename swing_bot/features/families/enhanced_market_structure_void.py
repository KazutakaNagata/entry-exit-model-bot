"""Enhanced past-only structure-room / void features.

This family expands the earlier ``market_structure_void`` proxies with normalized
room, asymmetry, and post-break room measures.  It intentionally remains a
price-geometry proxy, not an order-book feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

EMSV_WINDOWS = (60, 120, 240, 480, 720)
_EPS = 1e-12


def _log_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log((a + _EPS) / (b + _EPS)) * 10000.0


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def build_enhanced_market_structure_void_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = EMSV_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build normalized upside/downside room and structure asymmetry features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        prev_range = (prev_high - prev_low).replace(0.0, np.nan)
        atr = tr.rolling(window=max(15, min(window, 120)), min_periods=max(15, min(window, 120))).mean().replace(0.0, np.nan)

        upside_room_bps = _log_bps(prev_high, close)
        downside_room_bps = _log_bps(close, prev_low)

        name = f"emsv_upside_room_norm_range_{window}m"
        out[name] = _safe_div(prev_high - close, prev_range)
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Upside room to previous {window}m high normalized by previous range."))

        name = f"emsv_downside_room_norm_range_{window}m"
        out[name] = _safe_div(close - prev_low, prev_range)
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Downside room to previous {window}m low normalized by previous range."))

        name = f"emsv_upside_room_norm_atr_{window}m"
        out[name] = _safe_div(prev_high - close, atr)
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Upside room normalized by recent ATR proxy for {window}m structure."))

        name = f"emsv_downside_room_norm_atr_{window}m"
        out[name] = _safe_div(close - prev_low, atr)
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Downside room normalized by recent ATR proxy for {window}m structure."))

        name = f"emsv_room_asymmetry_up_minus_down_{window}m"
        out[name] = upside_room_bps - downside_room_bps
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Upside room minus downside room in bps for previous {window}m structure."))

        name = f"emsv_upside_void_after_breakout_{window}m"
        breakout = np.maximum(_log_bps(close, prev_high), 0.0)
        out[name] = breakout * _safe_div(prev_range, atr)
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Breakout extension weighted by previous {window}m range versus ATR."))

        name = f"emsv_downside_void_after_breakdown_{window}m"
        breakdown = np.maximum(_log_bps(prev_low, close), 0.0)
        out[name] = breakdown * _safe_div(prev_range, atr)
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Breakdown extension weighted by previous {window}m range versus ATR."))

        name = f"emsv_close_high_quartile_room_{window}m"
        pos = _safe_div(close - prev_low, prev_range)
        out[name] = (pos >= 0.75).astype("float64") * out[f"emsv_upside_room_norm_atr_{window}m"]
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Upside room when close is in upper quartile of previous {window}m range."))

        name = f"emsv_close_low_quartile_room_{window}m"
        out[name] = (pos <= 0.25).astype("float64") * out[f"emsv_downside_room_norm_atr_{window}m"]
        specs.append(FeatureSpec(name, "enhanced_market_structure_void", window + 1, description=f"Downside room when close is in lower quartile of previous {window}m range."))

    if 120 in windows and 480 in windows:
        out["emsv_upside_room_120m_minus_480m"] = out["emsv_upside_room_norm_range_120m"] - out["emsv_upside_room_norm_range_480m"]
        specs.append(FeatureSpec("emsv_upside_room_120m_minus_480m", "enhanced_market_structure_void", 481, description="Medium-term upside room minus slower upside room."))
    if 120 in windows and 480 in windows:
        out["emsv_room_asymmetry_120m_minus_480m"] = out["emsv_room_asymmetry_up_minus_down_120m"] - out["emsv_room_asymmetry_up_minus_down_480m"]
        specs.append(FeatureSpec("emsv_room_asymmetry_120m_minus_480m", "enhanced_market_structure_void", 481, description="Medium-term room asymmetry minus slower room asymmetry."))

    return out, specs

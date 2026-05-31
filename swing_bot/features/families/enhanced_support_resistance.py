"""Enhanced past-only support/resistance structure features.

These features are broader structural proxies than the lightweight
``support_resistance`` family.  Levels are formed strictly before row ``t`` with
``shift(1).rolling(...)``; the current finalized bar is only used to evaluate how
price behaved relative to those already-known levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

ESR_WINDOWS = (60, 120, 240, 480, 720)
_EPS = 1e-12


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def _log_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log((a + _EPS) / (b + _EPS)) * 10000.0


def build_enhanced_support_resistance_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = ESR_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build long-horizon support/resistance distance and failure proxies."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        prev_mid = (prev_high + prev_low) / 2.0
        prev_range = (prev_high - prev_low).replace(0.0, np.nan)

        name = f"esr_dist_to_prev_resistance_{window}m_bps"
        out[name] = _log_bps(prev_high, close)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Distance from close to previous {window}m resistance."))

        name = f"esr_dist_to_prev_support_{window}m_bps"
        out[name] = _log_bps(close, prev_low)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Distance from previous {window}m support to close."))

        name = f"esr_position_in_prev_range_{window}m"
        out[name] = _safe_div(close - prev_low, prev_range)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Close position in previous {window}m support/resistance range."))

        name = f"esr_dist_from_prev_mid_{window}m_bps"
        out[name] = _log_bps(close, prev_mid)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Close distance from previous {window}m range midpoint."))

        name = f"esr_prev_range_pct_price_{window}m"
        out[name] = prev_range / close.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Previous {window}m high-low range as share of close."))

        name = f"esr_breakout_close_over_prev_high_{window}m_bps"
        out[name] = np.maximum(_log_bps(close, prev_high), 0.0)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Positive close breakout over previous {window}m high."))

        name = f"esr_breakdown_close_under_prev_low_{window}m_bps"
        out[name] = np.maximum(_log_bps(prev_low, close), 0.0)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Positive close breakdown under previous {window}m low."))

        name = f"esr_failed_breakout_wick_{window}m_bps"
        out[name] = np.maximum(_log_bps(high, prev_high), 0.0) * (close < prev_high).astype("float64")
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Current bar pierced previous {window}m high but closed back below it."))

        name = f"esr_failed_breakdown_wick_{window}m_bps"
        out[name] = np.maximum(_log_bps(prev_low, low), 0.0) * (close > prev_low).astype("float64")
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Current bar pierced previous {window}m low but closed back above it."))

        name = f"esr_resistance_vs_support_room_ratio_{window}m"
        dist_res = (prev_high - close).clip(lower=0.0)
        dist_sup = (close - prev_low).clip(lower=0.0)
        out[name] = _safe_div(dist_res, dist_sup + _EPS)
        specs.append(FeatureSpec(name, "enhanced_support_resistance", window + 1, description=f"Upside room versus downside room inside previous {window}m range."))

    if 120 in windows and 720 in windows:
        out["esr_position_120m_minus_720m"] = out["esr_position_in_prev_range_120m"] - out["esr_position_in_prev_range_720m"]
        specs.append(FeatureSpec("esr_position_120m_minus_720m", "enhanced_support_resistance", 721, description="Medium-term range position minus long-term range position."))
    if 240 in windows and 720 in windows:
        out["esr_prev_range_240m_div_720m"] = _safe_div(out["esr_prev_range_pct_price_240m"], out["esr_prev_range_pct_price_720m"])
        specs.append(FeatureSpec("esr_prev_range_240m_div_720m", "enhanced_support_resistance", 721, description="240m structural range size relative to 720m range size."))

    return out, specs

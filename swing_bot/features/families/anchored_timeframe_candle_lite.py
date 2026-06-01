"""Lightweight human-watched anchored timeframe candle features.

This family is intentionally smaller than ``anchored_timeframe_candle`` for
local MacBook experiments.  It keeps the pieces most likely to matter for
human-watched levels: the last closed 5m/15m/60m/240m candle shape and the
current close distance to that candle's high/low/close and recent closed-bar
levels.

All higher-timeframe values use fully closed anchored candles only.  A 60m
feature at 10:37 uses the candle closed at 10:00, not the still-forming
10:00-11:00 candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec
from swing_bot.features.families.timeframe_utils import (
    align_closed_bars_to_1m,
    align_timeframe_series_to_1m,
    log_bps,
    resample_ohlcv_closed,
    safe_div,
    safe_range_position,
    timeframe_label,
)

LITE_TIMEFRAMES = (5, 15, 60, 240)
LEVEL_BARS = 4
FAMILY = "anchored_timeframe_candle_lite"


def _add(out: pd.DataFrame, specs: list[FeatureSpec], name: str, values: pd.Series, lookback: int, description: str) -> None:
    out[name] = values.astype("float64")
    specs.append(FeatureSpec(name, FAMILY, lookback, description=description))


def build_anchored_timeframe_candle_lite_features(
    df: pd.DataFrame,
    *,
    timeframes: tuple[int, ...] = LITE_TIMEFRAMES,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build a compact closed-candle/level feature set."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    current_close = df["close"].astype("float64")

    for tf_minutes in timeframes:
        label = timeframe_label(tf_minutes)
        bars = resample_ohlcv_closed(df, tf_minutes)
        aligned = align_closed_bars_to_1m(df, bars)

        o = aligned["open"].astype("float64")
        h = aligned["high"].astype("float64")
        l = aligned["low"].astype("float64")
        c = aligned["close"].astype("float64")
        rng = h - l
        body = c - o
        upper = h - pd.concat([o, c], axis=1).max(axis=1)
        lower = pd.concat([o, c], axis=1).min(axis=1) - l
        lookback = int(tf_minutes)

        _add(out, specs, f"atcl_{label}_body_bps", log_bps(c, o), lookback, f"Last closed {label} candle body bps.")
        _add(out, specs, f"atcl_{label}_range_bps", safe_div(rng, c.replace(0.0, np.nan), zero_fill=0.0) * 10000.0, lookback, f"Last closed {label} candle range bps.")
        _add(out, specs, f"atcl_{label}_close_position", safe_range_position(c, l, h), lookback, f"Last closed {label} candle close position in range.")
        _add(out, specs, f"atcl_{label}_upper_wick_ratio", safe_div(upper, rng, zero_fill=0.0), lookback, f"Last closed {label} upper wick ratio.")
        _add(out, specs, f"atcl_{label}_lower_wick_ratio", safe_div(lower, rng, zero_fill=0.0), lookback, f"Last closed {label} lower wick ratio.")
        _add(out, specs, f"atcl_{label}_current_vs_high_bps", log_bps(current_close, h), lookback, f"Current close distance to last closed {label} high.")
        _add(out, specs, f"atcl_{label}_current_vs_low_bps", log_bps(current_close, l), lookback, f"Current close distance to last closed {label} low.")
        _add(out, specs, f"atcl_{label}_current_vs_close_bps", log_bps(current_close, c), lookback, f"Current close distance to last closed {label} close.")

        # Recent human-watched closed-bar levels, but only one compact window.
        prev_high = bars["high"].astype("float64").shift(1).rolling(window=LEVEL_BARS, min_periods=LEVEL_BARS).max()
        prev_low = bars["low"].astype("float64").shift(1).rolling(window=LEVEL_BARS, min_periods=LEVEL_BARS).min()
        levels = pd.DataFrame({"prev_high": prev_high, "prev_low": prev_low}, index=bars.index)
        aligned_levels = align_timeframe_series_to_1m(df, levels)
        ph = aligned_levels["prev_high"].astype("float64")
        pl = aligned_levels["prev_low"].astype("float64")
        prng = ph - pl
        level_lookback = int(tf_minutes * (LEVEL_BARS + 1))
        _add(out, specs, f"atcl_{label}_prev4_high_dist_bps", log_bps(current_close, ph), level_lookback, f"Current close distance to previous 4 closed {label} highs.")
        _add(out, specs, f"atcl_{label}_prev4_low_dist_bps", log_bps(current_close, pl), level_lookback, f"Current close distance to previous 4 closed {label} lows.")
        _add(out, specs, f"atcl_{label}_prev4_range_position", safe_range_position(current_close, pl, ph), level_lookback, f"Current close position inside previous 4 closed {label} range.")

    return out, specs

"""Human-watched anchored timeframe candle and level features.

This family uses fully closed 5m/15m/60m/240m anchored candles, plus current
completed 1m candle features.  Higher-timeframe candles are aligned by their
close time so no still-forming higher-timeframe bar leaks into a 1m decision row.
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

TIMEFRAMES = (1, 5, 15, 60, 240)
ROLLING_LEVEL_BARS = (4, 12)
_EPS = 1e-12


def _candle_block(
    *,
    current_close: pd.Series,
    aligned: pd.DataFrame,
    label: str,
    family: str,
    lookback: int,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    out = pd.DataFrame(index=current_close.index)
    specs: list[FeatureSpec] = []
    o = aligned["open"].astype("float64")
    h = aligned["high"].astype("float64")
    l = aligned["low"].astype("float64")
    c = aligned["close"].astype("float64")
    v = aligned["volume"].astype("float64")
    rng = h - l
    body = c - o
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l

    columns: dict[str, pd.Series] = {
        f"atc_{label}_body_bps": log_bps(c, o),
        f"atc_{label}_range_bps": safe_div(rng, c.replace(0.0, np.nan), zero_fill=0.0) * 10000.0,
        f"atc_{label}_body_to_range": safe_div(body.abs(), rng, zero_fill=0.0),
        f"atc_{label}_upper_wick_ratio": safe_div(upper, rng, zero_fill=0.0),
        f"atc_{label}_lower_wick_ratio": safe_div(lower, rng, zero_fill=0.0),
        f"atc_{label}_close_position": safe_range_position(c, l, h),
        f"atc_{label}_is_bull": (c > o).astype("float64"),
        f"atc_{label}_current_close_vs_bar_high_bps": log_bps(current_close, h),
        f"atc_{label}_current_close_vs_bar_low_bps": log_bps(current_close, l),
        f"atc_{label}_current_close_vs_bar_close_bps": log_bps(current_close, c),
        f"atc_{label}_volume_log1p": np.log1p(v),
    }
    for name, values in columns.items():
        out[name] = values
        specs.append(FeatureSpec(name, family, lookback, description=f"Anchored {label} closed-candle feature."))
    return out, specs


def _level_block(
    *,
    df: pd.DataFrame,
    tf_bars: pd.DataFrame,
    label: str,
    tf_minutes: int,
    current_close: pd.Series,
    bars_windows: tuple[int, ...],
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    h = tf_bars["high"].astype("float64")
    l = tf_bars["low"].astype("float64")
    c = tf_bars["close"].astype("float64")
    for bars in bars_windows:
        # Use previous closed higher-timeframe candles only for structural levels.
        prev_high = h.shift(1).rolling(window=bars, min_periods=bars).max()
        prev_low = l.shift(1).rolling(window=bars, min_periods=bars).min()
        prev_close = c.shift(1)
        levels = pd.DataFrame(
            {
                "roll_high": prev_high,
                "roll_low": prev_low,
                "prev_close": prev_close,
            },
            index=tf_bars.index,
        )
        aligned = align_timeframe_series_to_1m(df, levels)
        roll_high = aligned["roll_high"].astype("float64")
        roll_low = aligned["roll_low"].astype("float64")
        prev_close_aligned = aligned["prev_close"].astype("float64")
        rng = roll_high - roll_low
        lookback = int(tf_minutes * (bars + 1))
        cols = {
            f"atc_{label}_prev{bars}_high_dist_bps": log_bps(current_close, roll_high),
            f"atc_{label}_prev{bars}_low_dist_bps": log_bps(current_close, roll_low),
            f"atc_{label}_prev{bars}_range_position": safe_range_position(current_close, roll_low, roll_high),
            f"atc_{label}_prev{bars}_range_bps": safe_div(rng, current_close.replace(0.0, np.nan), zero_fill=0.0) * 10000.0,
            f"atc_{label}_prev{bars}_close_dist_bps": log_bps(current_close, prev_close_aligned),
            f"atc_{label}_break_prev{bars}_high": (current_close > roll_high).astype("float64"),
            f"atc_{label}_break_prev{bars}_low": (current_close < roll_low).astype("float64"),
        }
        for name, values in cols.items():
            out[name] = values
            specs.append(FeatureSpec(name, "anchored_timeframe_candle", lookback, description=f"Anchored {label} prior {bars}-bar level feature."))
    return out, specs


def build_anchored_timeframe_candle_features(
    df: pd.DataFrame,
    *,
    timeframes: tuple[int, ...] = TIMEFRAMES,
    level_bars: tuple[int, ...] = ROLLING_LEVEL_BARS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build closed anchored candle and anchored level features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    current_close = df["close"].astype("float64")

    for tf_minutes in timeframes:
        label = timeframe_label(tf_minutes)
        bars = resample_ohlcv_closed(df, tf_minutes)
        if tf_minutes == 1:
            aligned = pd.DataFrame({
                "open": df["open"].astype("float64"),
                "high": df["high"].astype("float64"),
                "low": df["low"].astype("float64"),
                "close": df["close"].astype("float64"),
                "volume": df["volume"].astype("float64"),
            }, index=df.index)
        else:
            aligned = align_closed_bars_to_1m(df, bars)
        block, block_specs = _candle_block(
            current_close=current_close,
            aligned=aligned,
            label=label,
            family="anchored_timeframe_candle",
            lookback=max(1, tf_minutes),
        )
        out = pd.concat([out, block], axis=1)
        specs.extend(block_specs)
        level_block, level_specs = _level_block(
            df=df,
            tf_bars=bars,
            label=label,
            tf_minutes=tf_minutes,
            current_close=current_close,
            bars_windows=level_bars,
        )
        out = pd.concat([out, level_block], axis=1)
        specs.extend(level_specs)

    return out, specs

"""Human-watched anchored timeframe indicator features.

Adds common chart indicators on closed 1m/5m/15m/60m/240m candles: EMA, ATR,
RSI, Bollinger Bands, and MACD.  Higher-timeframe indicators are computed on
fully closed anchored candles and then as-of aligned to 1m rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec
from swing_bot.features.families.timeframe_utils import (
    align_timeframe_series_to_1m,
    log_bps,
    resample_ohlcv_closed,
    rsi,
    safe_div,
    timeframe_label,
    true_range,
)

TIMEFRAMES = (1, 5, 15, 60, 240)
EMA_SPANS = (10, 20, 50)
ATR_WINDOW = 14
RSI_WINDOW = 14
BB_WINDOW = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def _indicator_frame(bars: pd.DataFrame, tf_minutes: int, label: str) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    ind = pd.DataFrame(index=bars.index)
    specs: list[FeatureSpec] = []
    close = bars["close"].astype("float64")
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")

    ema_values: dict[int, pd.Series] = {}
    for span in EMA_SPANS:
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        ema_values[span] = ema
        name = f"ati_{label}_ema{span}_dist_bps"
        ind[name] = log_bps(close, ema)
        specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * span, description=f"Anchored {label} close distance to EMA{span}."))
        slope_name = f"ati_{label}_ema{span}_slope_bps"
        ind[slope_name] = log_bps(ema, ema.shift(1))
        specs.append(FeatureSpec(slope_name, "anchored_timeframe_indicators", tf_minutes * (span + 1), description=f"Anchored {label} EMA{span} one-bar slope."))

    if 10 in ema_values and 20 in ema_values:
        name = f"ati_{label}_ema10_vs_ema20_bps"
        ind[name] = log_bps(ema_values[10], ema_values[20])
        specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * 20, description=f"Anchored {label} EMA10 vs EMA20."))
    if 20 in ema_values and 50 in ema_values:
        name = f"ati_{label}_ema20_vs_ema50_bps"
        ind[name] = log_bps(ema_values[20], ema_values[50])
        specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * 50, description=f"Anchored {label} EMA20 vs EMA50."))

    tr = true_range(high, low, close)
    atr = tr.rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).mean()
    name = f"ati_{label}_atr14_bps"
    ind[name] = safe_div(atr, close.replace(0.0, np.nan)) * 10000.0
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * (ATR_WINDOW + 1), description=f"Anchored {label} ATR14 normalized by close."))
    name = f"ati_{label}_range_over_atr14"
    ind[name] = safe_div(high - low, atr)
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * (ATR_WINDOW + 1), description=f"Anchored {label} current range over ATR14."))

    rsi_val = rsi(close, window=RSI_WINDOW)
    name = f"ati_{label}_rsi14"
    ind[name] = rsi_val
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * (RSI_WINDOW + 1), description=f"Anchored {label} RSI14."))
    name = f"ati_{label}_rsi14_centered"
    ind[name] = (rsi_val - 50.0) / 50.0
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * (RSI_WINDOW + 1), description=f"Anchored {label} RSI14 centered to [-1,1]-ish."))

    bb_mid = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    bb_std = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
    name = f"ati_{label}_bb20_z"
    ind[name] = safe_div(close - bb_mid, bb_std)
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * BB_WINDOW, description=f"Anchored {label} Bollinger20 z-score."))
    name = f"ati_{label}_bb20_width_bps"
    ind[name] = safe_div(bb_width, close.replace(0.0, np.nan)) * 10000.0
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * BB_WINDOW, description=f"Anchored {label} Bollinger20 width bps."))
    name = f"ati_{label}_bb20_position"
    ind[name] = safe_div(close - bb_lower, bb_width)
    specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * BB_WINDOW, description=f"Anchored {label} Bollinger20 band position."))

    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    macd_hist = macd - macd_signal
    for suffix, values, lookback_bars in (
        ("macd_bps", macd, MACD_SLOW),
        ("macd_signal_bps", macd_signal, MACD_SLOW + MACD_SIGNAL),
        ("macd_hist_bps", macd_hist, MACD_SLOW + MACD_SIGNAL),
    ):
        name = f"ati_{label}_{suffix}"
        ind[name] = safe_div(values, close.replace(0.0, np.nan)) * 10000.0
        specs.append(FeatureSpec(name, "anchored_timeframe_indicators", tf_minutes * lookback_bars, description=f"Anchored {label} MACD-derived feature."))

    return ind, specs


def build_anchored_timeframe_indicator_features(
    df: pd.DataFrame,
    *,
    timeframes: tuple[int, ...] = TIMEFRAMES,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build common anchored-timeframe indicator features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    for tf_minutes in timeframes:
        label = timeframe_label(tf_minutes)
        bars = resample_ohlcv_closed(df, tf_minutes)
        ind, ind_specs = _indicator_frame(bars, tf_minutes, label)
        aligned = align_timeframe_series_to_1m(df, ind)
        out = pd.concat([out, aligned], axis=1)
        specs.extend(ind_specs)
    return out, specs

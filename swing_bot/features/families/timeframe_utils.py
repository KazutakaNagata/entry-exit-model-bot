"""Utilities for anchored higher-timeframe feature families.

The canonical 1m timestamp is the candle open time.  For higher-timeframe
features we conservatively align only fully closed anchored bars to each 1m row:
for a 60m feature at timestamp 10:37, the latest usable 60m candle is the one
that closed at or before 10:37, not the currently forming 10:00-11:00 candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def ensure_utc_timestamp_series(df: pd.DataFrame) -> pd.Series:
    """Return timezone-aware UTC timestamps from a canonical OHLCV frame."""
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    return pd.Series(ts, index=df.index, name="timestamp")


def safe_div(num: pd.Series, den: pd.Series | float, *, zero_fill: float | None = None) -> pd.Series:
    """Divide while guarding zero denominators.

    ``zero_fill`` fills only rows where the denominator is exactly zero. It does
    not fill warmup NaNs, so lookback-related missing values remain visible.
    """
    if isinstance(den, pd.Series):
        zero_den = den.eq(0.0)
        out = num / den.mask(zero_den)
        if zero_fill is not None:
            out = out.mask(zero_den & num.notna(), zero_fill)
        return out.replace([np.inf, -np.inf], np.nan)
    if den == 0:
        return pd.Series(zero_fill, index=num.index, dtype="float64") if zero_fill is not None else num * np.nan
    return (num / den).replace([np.inf, -np.inf], np.nan)


def safe_range_position(value: pd.Series, low: pd.Series, high: pd.Series, *, zero_fill: float = 0.5) -> pd.Series:
    """Position inside [low, high], using a neutral value for zero-width ranges."""
    rng = high - low
    out = (value - low) / rng.mask(rng.eq(0.0))
    return out.mask(rng.eq(0.0) & value.notna(), zero_fill).clip(lower=0.0, upper=1.0)


def log_bps(num: pd.Series, den: pd.Series) -> pd.Series:
    """Log return in basis points with non-positive values guarded."""
    num2 = num.where(num > 0)
    den2 = den.where(den > 0)
    return np.log(num2 / den2) * 10000.0


def timeframe_label(minutes: int) -> str:
    """Stable label used in column names."""
    return f"{minutes}m"


def resample_ohlcv_closed(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Build anchored OHLCV bars and label them by close_time.

    The returned index is ``close_time``.  A row ending at 10:00 represents the
    fully closed interval [09:00, 10:00) for 60m bars.  This makes as-of joins
    straightforward and prevents using a still-forming higher-timeframe candle.
    """
    if timeframe_minutes <= 1:
        ts = ensure_utc_timestamp_series(df)
        out = pd.DataFrame(
            {
                "open": df["open"].astype("float64").to_numpy(),
                "high": df["high"].astype("float64").to_numpy(),
                "low": df["low"].astype("float64").to_numpy(),
                "close": df["close"].astype("float64").to_numpy(),
                "volume": df["volume"].astype("float64").to_numpy(),
            },
            index=ts,
        )
        out.index.name = "close_time"
        return out

    ts = ensure_utc_timestamp_series(df)
    indexed = pd.DataFrame(
        {
            "open": df["open"].astype("float64").to_numpy(),
            "high": df["high"].astype("float64").to_numpy(),
            "low": df["low"].astype("float64").to_numpy(),
            "close": df["close"].astype("float64").to_numpy(),
            "volume": df["volume"].astype("float64").to_numpy(),
        },
        index=ts,
    ).sort_index(kind="mergesort")
    rule = f"{timeframe_minutes}min"
    bars = indexed.resample(rule, label="left", closed="left", origin="epoch").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars.index = bars.index + pd.Timedelta(minutes=timeframe_minutes)
    bars.index.name = "close_time"
    return bars


def align_closed_bars_to_1m(df: pd.DataFrame, bars_by_close_time: pd.DataFrame) -> pd.DataFrame:
    """As-of align higher-timeframe closed bars to canonical 1m rows."""
    ts = ensure_utc_timestamp_series(df)
    left = pd.DataFrame({"timestamp": ts.to_numpy()}, index=df.index).sort_values("timestamp")
    right = bars_by_close_time.reset_index().sort_values("close_time")
    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged.index = left.index
    merged = merged.reindex(df.index)
    return merged.drop(columns=["timestamp", "close_time"], errors="ignore")


def align_timeframe_series_to_1m(df: pd.DataFrame, values_by_close_time: pd.DataFrame) -> pd.DataFrame:
    """As-of align arbitrary higher-timeframe indicator columns to 1m rows."""
    values = values_by_close_time.copy()
    values.index.name = "close_time"
    return align_closed_bars_to_1m(df, values)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Classic true range using previous close."""
    prev_close = close.shift(1)
    parts = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Simple rolling RSI; no future data.

    Flat windows are neutral 50, one-sided up windows are 100, and one-sided
    down windows are 0.  This avoids long NaN stretches in quiet markets.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.mask(avg_loss.eq(0.0))
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.mask(avg_gain.eq(0.0) & avg_loss.eq(0.0), 50.0)
    out = out.mask(avg_gain.gt(0.0) & avg_loss.eq(0.0), 100.0)
    out = out.mask(avg_gain.eq(0.0) & avg_loss.gt(0.0), 0.0)
    return out

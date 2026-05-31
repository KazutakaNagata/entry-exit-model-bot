"""Past-only rejection/reversal candle and failed-break features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

REJECTION_WINDOWS = (15, 30, 60, 120, 240)
_EPS = 1e-12


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0.0, np.nan)


def build_rejection_reversal_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = REJECTION_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build candle rejection and failed-break features.

    Current candle wick/body features are available after the bar closes.  Failed
    breakout/breakdown levels are based on previous rolling highs/lows, excluding
    the current bar via ``shift(1)``.
    """
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    open_ = df["open"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")

    candle_range_bps = np.log((high + _EPS) / (low + _EPS)) * 10000.0
    body_bps = (np.log((close + _EPS) / (open_ + _EPS)) * 10000.0).abs()
    upper_wick_bps = np.log((high + _EPS) / (pd.concat([open_, close], axis=1).max(axis=1) + _EPS)) * 10000.0
    lower_wick_bps = np.log((pd.concat([open_, close], axis=1).min(axis=1) + _EPS) / (low + _EPS)) * 10000.0
    close_position = _safe_div(close - low, high - low)
    body_direction = np.sign(close - open_)

    base_features: list[tuple[str, pd.Series, str]] = [
        ("rr_candle_range_bps", candle_range_bps, "Current finalized candle high-low range in bps."),
        ("rr_body_bps", body_bps, "Current finalized candle body size in bps."),
        ("rr_upper_wick_bps", upper_wick_bps, "Current finalized upper wick size in bps."),
        ("rr_lower_wick_bps", lower_wick_bps, "Current finalized lower wick size in bps."),
        ("rr_upper_wick_ratio", _safe_div(upper_wick_bps, candle_range_bps), "Upper wick divided by candle range."),
        ("rr_lower_wick_ratio", _safe_div(lower_wick_bps, candle_range_bps), "Lower wick divided by candle range."),
        ("rr_close_position_in_candle", close_position, "Close position inside current finalized candle range."),
        ("rr_body_direction", body_direction, "Sign of current candle body."),
    ]
    for name, series, description in base_features:
        out[name] = series
        specs.append(FeatureSpec(name, "rejection_reversal", 1, description=description))

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        upper_mean = upper_wick_bps.shift(1).rolling(window=window, min_periods=window).mean()
        upper_std = upper_wick_bps.shift(1).rolling(window=window, min_periods=window).std(ddof=0)
        lower_mean = lower_wick_bps.shift(1).rolling(window=window, min_periods=window).mean()
        lower_std = lower_wick_bps.shift(1).rolling(window=window, min_periods=window).std(ddof=0)

        failed_up = np.where(
            (high > prev_high) & (close < prev_high),
            np.log((high + _EPS) / (prev_high + _EPS)) * 10000.0,
            0.0,
        )
        failed_down = np.where(
            (low < prev_low) & (close > prev_low),
            np.log((prev_low + _EPS) / (low + _EPS)) * 10000.0,
            0.0,
        )

        name = f"rr_failed_breakout_{window}m_bps"
        out[name] = failed_up
        specs.append(FeatureSpec(name, "rejection_reversal", window + 1, description=f"High pierced previous {window}m high but close returned below it."))

        name = f"rr_failed_breakdown_{window}m_bps"
        out[name] = failed_down
        specs.append(FeatureSpec(name, "rejection_reversal", window + 1, description=f"Low pierced previous {window}m low but close returned above it."))

        name = f"rr_upper_wick_z_{window}m"
        out[name] = (upper_wick_bps - upper_mean) / upper_std.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "rejection_reversal", window + 1, description=f"Current upper wick z-score versus previous {window}m."))

        name = f"rr_lower_wick_z_{window}m"
        out[name] = (lower_wick_bps - lower_mean) / lower_std.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "rejection_reversal", window + 1, description=f"Current lower wick z-score versus previous {window}m."))

        name = f"rr_upper_rejection_score_{window}m"
        out[name] = out[f"rr_upper_wick_z_{window}m"].clip(lower=0.0) * (1.0 - close_position)
        specs.append(FeatureSpec(name, "rejection_reversal", window + 1, description=f"Upper wick surprise times weak close position over previous {window}m context."))

        name = f"rr_lower_rejection_score_{window}m"
        out[name] = out[f"rr_lower_wick_z_{window}m"].clip(lower=0.0) * close_position
        specs.append(FeatureSpec(name, "rejection_reversal", window + 1, description=f"Lower wick surprise times strong close position over previous {window}m context."))

    return out, specs

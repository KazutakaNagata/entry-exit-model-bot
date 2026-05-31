"""Past-only breakout continuation features.

The breakout levels in this module are formed from previous bars via
``shift(1).rolling(...)``.  A row at time ``t`` may compare the finalized
``close[t]`` / ``high[t]`` / ``low[t]`` with those previous levels, but it must
not use bars after ``t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

BREAKOUT_WINDOWS = (15, 30, 60, 120, 240)
_EPS = 1e-12


def _positive_part(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0)


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0.0, np.nan)


def build_breakout_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = BREAKOUT_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build simple breakout and range-expansion features.

    These features are intended to support the old long-continuation hypothesis:
    price breaks a previous range, does so with enough range/volume expansion,
    and has room to continue.  They are deliberately deterministic and
    past-only; no target or future path information is used.
    """
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    true_range_bps = np.log((high + _EPS) / (low + _EPS)) * 10000.0
    log_volume = np.log1p(volume)

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        prev_range = (prev_high - prev_low).replace(0.0, np.nan)
        prev_tr_mean = true_range_bps.shift(1).rolling(window=window, min_periods=window).mean()
        prev_vol_mean = volume.shift(1).rolling(window=window, min_periods=window).mean()
        prev_log_vol_mean = log_volume.shift(1).rolling(window=window, min_periods=window).mean()
        prev_log_vol_std = log_volume.shift(1).rolling(window=window, min_periods=window).std(ddof=0)

        up_dist = np.log((close + _EPS) / (prev_high + _EPS)) * 10000.0
        down_dist = np.log((prev_low + _EPS) / (close + _EPS)) * 10000.0

        name = f"breakout_close_above_prev_high_{window}m_bps"
        out[name] = up_dist
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Close versus previous {window}m high."))

        name = f"breakout_close_below_prev_low_{window}m_bps"
        out[name] = down_dist
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Previous {window}m low versus close."))

        name = f"breakout_up_strength_{window}m_bps"
        out[name] = _positive_part(up_dist)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Positive close breakout distance above previous {window}m high."))

        name = f"breakout_down_strength_{window}m_bps"
        out[name] = _positive_part(down_dist)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Positive close breakdown distance below previous {window}m low."))

        name = f"breakout_high_extension_prev_range_{window}m"
        out[name] = _safe_ratio(high - prev_high, prev_range)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"High extension above previous {window}m range, scaled by range."))

        name = f"breakout_low_extension_prev_range_{window}m"
        out[name] = _safe_ratio(prev_low - low, prev_range)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Low extension below previous {window}m range, scaled by range."))

        name = f"breakout_range_expansion_{window}m"
        out[name] = true_range_bps / prev_tr_mean.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Current true range divided by previous {window}m average range."))

        name = f"breakout_volume_expansion_{window}m"
        out[name] = volume / prev_vol_mean.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Current volume divided by previous {window}m average volume."))

        vol_z = (log_volume - prev_log_vol_mean) / prev_log_vol_std.replace(0.0, np.nan)
        name = f"breakout_up_with_volume_{window}m"
        out[name] = _positive_part(up_dist) * vol_z.clip(lower=0.0)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Positive up breakout distance times positive previous-window log-volume z-score."))

        name = f"breakout_down_with_volume_{window}m"
        out[name] = _positive_part(down_dist) * vol_z.clip(lower=0.0)
        specs.append(FeatureSpec(name, "breakout", window + 1, description=f"Positive down breakout distance times positive previous-window log-volume z-score."))

    return out, specs

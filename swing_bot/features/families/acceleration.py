"""Past-only return acceleration and curvature features.

This family intentionally stays deterministic and uses only finalized bars up to
row t.  It is part of feature factory v3a, whose purpose is to restore a wider
trend / return / activity search space before doing feature selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

RETURN_WINDOWS = (1, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)
PAIR_WINDOWS = ((3, 8), (5, 13), (8, 21), (13, 34), (21, 55), (34, 89), (55, 144), (89, 233))
Z_WINDOWS = (15, 30, 60, 120, 240)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_acceleration_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build past-return acceleration, curvature, and momentum surprise features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0

    returns: dict[int, pd.Series] = {}
    for window in RETURN_WINDOWS:
        returns[window] = np.log(close / close.shift(window)) * 10000.0

    for short, long in PAIR_WINDOWS:
        short_ret = returns[short]
        long_ret = returns[long]
        expected_short = long_ret * (short / long)
        name = f"accel_ret_{short}m_vs_{long}m_expected_bps"
        out[name] = short_ret - expected_short
        specs.append(FeatureSpec(name, "acceleration", long, description=f"{short}m return minus proportional {long}m return."))

        name = f"accel_ret_{short}m_to_{long}m_ratio"
        out[name] = _safe_div(short_ret, expected_short)
        specs.append(FeatureSpec(name, "acceleration", long, description=f"{short}m return divided by proportional {long}m return."))

        name = f"accel_abs_ret_{short}m_to_{long}m_ratio"
        out[name] = _safe_div(short_ret.abs(), long_ret.abs())
        specs.append(FeatureSpec(name, "acceleration", long, description=f"Absolute {short}m return divided by absolute {long}m return."))

    for short, mid, long in ((3, 8, 21), (5, 13, 34), (8, 21, 55), (13, 34, 89), (21, 55, 144), (34, 89, 233)):
        short_rate = returns[short] / short
        mid_rate = returns[mid] / mid
        long_rate = returns[long] / long
        name = f"ret_curvature_{short}_{mid}_{long}m_bps_per_min"
        out[name] = short_rate - 2.0 * mid_rate + long_rate
        specs.append(FeatureSpec(name, "acceleration", long, description="Return-rate curvature across short/mid/long windows."))

        name = f"ret_rate_short_minus_long_{short}_{long}m_bps_per_min"
        out[name] = short_rate - long_rate
        specs.append(FeatureSpec(name, "acceleration", long, description="Short return rate minus long return rate."))

    for window in Z_WINDOWS:
        mean = ret_1m_bps.rolling(window=window, min_periods=window).mean()
        std = ret_1m_bps.rolling(window=window, min_periods=window).std(ddof=0)
        name = f"ret_1m_z_{window}m"
        out[name] = (ret_1m_bps - mean) / std.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "acceleration", window, description=f"Current 1m return z-score over past {window}m."))

        mean_abs = ret_1m_bps.abs().rolling(window=window, min_periods=window).mean()
        name = f"ret_1m_abs_to_mean_abs_{window}m"
        out[name] = _safe_div(ret_1m_bps.abs(), mean_abs)
        specs.append(FeatureSpec(name, "acceleration", window, description=f"Absolute 1m return divided by mean absolute return over {window}m."))

    # Sign persistence / reversal around the current finalized bar.
    sign = np.sign(ret_1m_bps).replace(0.0, np.nan)
    for window in (5, 15, 30, 60, 120):
        name = f"ret_sign_persistence_{window}m"
        out[name] = (sign * sign.shift(1)).rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "acceleration", window + 1, description=f"Mean same-sign tendency of 1m returns over {window}m."))

    return out, specs

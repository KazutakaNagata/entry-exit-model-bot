"""Past-only regime switch and regime change features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

VOL_WINDOWS = (15, 30, 60, 120, 240, 480)
RATIO_PAIRS = ((15, 60), (30, 120), (60, 240), (120, 480))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_regime_switch_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build deterministic indicators of volatility/trend regime changes."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0
    abs_ret = ret_1m_bps.abs()
    bar_range_bps = np.log(high / low.replace(0.0, np.nan)) * 10000.0
    log_volume = np.log1p(volume.clip(lower=0.0))

    vol: dict[int, pd.Series] = {}
    mean_abs: dict[int, pd.Series] = {}
    rng: dict[int, pd.Series] = {}
    vmean: dict[int, pd.Series] = {}
    for window in VOL_WINDOWS:
        vol[window] = ret_1m_bps.rolling(window=window, min_periods=window).std(ddof=0)
        mean_abs[window] = abs_ret.rolling(window=window, min_periods=window).mean()
        rng[window] = bar_range_bps.rolling(window=window, min_periods=window).mean()
        vmean[window] = log_volume.rolling(window=window, min_periods=window).mean()

        name = f"regime_switch_vol_z_{window}m"
        rolling_mean = vol[window].rolling(window=window, min_periods=window).mean()
        rolling_std = vol[window].rolling(window=window, min_periods=window).std(ddof=0)
        out[name] = (vol[window] - rolling_mean) / rolling_std.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "regime_switch", window * 2, description=f"Volatility z-score of {window}m realized vol."))

        name = f"regime_switch_activity_z_{window}m"
        act_mean = mean_abs[window].rolling(window=window, min_periods=window).mean()
        act_std = mean_abs[window].rolling(window=window, min_periods=window).std(ddof=0)
        out[name] = (mean_abs[window] - act_mean) / act_std.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "regime_switch", window * 2, description=f"Activity z-score of {window}m mean absolute return."))

        name = f"regime_switch_volume_z_{window}m"
        vol_mean = vmean[window].rolling(window=window, min_periods=window).mean()
        vol_std = vmean[window].rolling(window=window, min_periods=window).std(ddof=0)
        out[name] = (vmean[window] - vol_mean) / vol_std.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "regime_switch", window * 2, description=f"Log-volume regime z-score over {window}m."))

    for short, long in RATIO_PAIRS:
        name = f"regime_switch_vol_ratio_{short}_{long}m"
        out[name] = _safe_div(vol[short], vol[long])
        specs.append(FeatureSpec(name, "regime_switch", long, description=f"{short}m realized vol divided by {long}m vol."))

        name = f"regime_switch_absret_ratio_{short}_{long}m"
        out[name] = _safe_div(mean_abs[short], mean_abs[long])
        specs.append(FeatureSpec(name, "regime_switch", long, description=f"{short}m mean absolute return divided by {long}m mean absolute return."))

        name = f"regime_switch_range_ratio_{short}_{long}m"
        out[name] = _safe_div(rng[short], rng[long])
        specs.append(FeatureSpec(name, "regime_switch", long, description=f"{short}m mean bar range divided by {long}m mean bar range."))

        name = f"regime_switch_volume_ratio_{short}_{long}m"
        out[name] = _safe_div(vmean[short], vmean[long])
        specs.append(FeatureSpec(name, "regime_switch", long, description=f"{short}m log-volume mean divided by {long}m log-volume mean."))

    # Recent sign flip / regime disagreement features.
    for window in (15, 30, 60, 120, 240):
        fast_ret = np.log(close / close.shift(max(1, window // 4))) * 10000.0
        slow_ret = np.log(close / close.shift(window)) * 10000.0
        name = f"regime_switch_fast_slow_return_disagree_{window}m"
        out[name] = (np.sign(fast_ret) != np.sign(slow_ret)).astype("float64")
        specs.append(FeatureSpec(name, "regime_switch", window, description=f"Fast and slow return signs disagree for {window}m context."))

        name = f"regime_switch_fast_minus_slow_return_{window}m_bps"
        out[name] = fast_ret - slow_ret * (max(1, window // 4) / window)
        specs.append(FeatureSpec(name, "regime_switch", window, description=f"Fast return minus proportional slow return for {window}m context."))

    return out, specs

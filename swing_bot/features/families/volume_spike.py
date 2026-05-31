"""Past-only volume spike and activity burst features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

WINDOWS = (5, 15, 30, 60, 120, 240, 480)
RATIO_PAIRS = ((5, 30), (15, 60), (30, 120), (60, 240), (120, 480))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_volume_spike_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build log-volume z-score, spike, and burst persistence features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    volume = df["volume"].astype("float64").clip(lower=0.0)
    log_volume = np.log1p(volume)
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0
    abs_ret = ret_1m_bps.abs()

    means: dict[int, pd.Series] = {}
    stds: dict[int, pd.Series] = {}
    for window in WINDOWS:
        means[window] = log_volume.rolling(window=window, min_periods=window).mean()
        stds[window] = log_volume.rolling(window=window, min_periods=window).std(ddof=0)
        name = f"volume_spike_logvol_z_{window}m"
        out[name] = (log_volume - means[window]) / stds[window].replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "volume_spike", window, description=f"Current log-volume z-score over {window}m."))

        name = f"volume_spike_logvol_to_mean_{window}m"
        out[name] = _safe_div(log_volume, means[window])
        specs.append(FeatureSpec(name, "volume_spike", window, description=f"Current log-volume divided by {window}m mean log-volume."))

        name = f"volume_spike_top_share_{window}m"
        # Share of bars in the window with log-volume above the previous bar's expanding local mean proxy.
        threshold = means[window] + stds[window]
        out[name] = (log_volume > threshold).astype("float64").rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "volume_spike", window, description=f"Share of recent bars with log-volume above mean+std over {window}m."))

        name = f"volume_spike_absret_weighted_logvol_{window}m"
        out[name] = (log_volume * abs_ret).rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "volume_spike", window, description=f"Mean log-volume times absolute return over {window}m."))

        up_vol = log_volume.where(ret_1m_bps > 0.0, 0.0).rolling(window=window, min_periods=window).sum()
        down_vol = log_volume.where(ret_1m_bps < 0.0, 0.0).rolling(window=window, min_periods=window).sum()
        name = f"volume_spike_up_down_logvol_imbalance_{window}m"
        out[name] = _safe_div(up_vol - down_vol, up_vol + down_vol)
        specs.append(FeatureSpec(name, "volume_spike", window, description=f"Up-vs-down log-volume imbalance over {window}m."))

    for short, long in RATIO_PAIRS:
        name = f"volume_spike_logvol_mean_ratio_{short}_{long}m"
        out[name] = _safe_div(means[short], means[long])
        specs.append(FeatureSpec(name, "volume_spike", long, description=f"{short}m mean log-volume divided by {long}m mean log-volume."))

        name = f"volume_spike_logvol_std_ratio_{short}_{long}m"
        out[name] = _safe_div(stds[short], stds[long])
        specs.append(FeatureSpec(name, "volume_spike", long, description=f"{short}m log-volume std divided by {long}m log-volume std."))

    return out, specs

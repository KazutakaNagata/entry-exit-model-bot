"""Past-only volume/price pressure features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

PRESSURE_WINDOWS = (15, 30, 60, 120, 240)
_EPS = 1e-12


def _weighted_rolling_sum(value: pd.Series, weight: pd.Series, window: int) -> pd.Series:
    return (value * weight).rolling(window=window, min_periods=window).sum()


def build_volume_price_pressure_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = PRESSURE_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build rolling pressure features from returns and volume.

    A row at time ``t`` uses the finalized return from ``t-1`` to ``t`` and the
    finalized volume of row ``t``.  All rolling windows end at ``t``.
    """
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    volume = df["volume"].astype("float64")
    log_volume = np.log1p(volume)
    ret_1m_bps = np.log((close + _EPS) / (close.shift(1) + _EPS)) * 10000.0
    up_ret = ret_1m_bps.clip(lower=0.0)
    down_ret = (-ret_1m_bps.clip(upper=0.0))
    signed_volume = np.sign(ret_1m_bps.fillna(0.0)) * volume
    up_volume = volume.where(ret_1m_bps > 0.0, 0.0)
    down_volume = volume.where(ret_1m_bps < 0.0, 0.0)

    for window in windows:
        vol_sum = volume.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)
        up_vol_sum = up_volume.rolling(window=window, min_periods=window).sum()
        down_vol_sum = down_volume.rolling(window=window, min_periods=window).sum()

        name = f"vpp_signed_volume_imbalance_{window}m"
        out[name] = signed_volume.rolling(window=window, min_periods=window).sum() / vol_sum
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Signed volume imbalance over past {window}m."))

        name = f"vpp_up_volume_share_{window}m"
        out[name] = up_vol_sum / vol_sum
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Share of volume on positive 1m returns over past {window}m."))

        name = f"vpp_down_volume_share_{window}m"
        out[name] = down_vol_sum / vol_sum
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Share of volume on negative 1m returns over past {window}m."))

        name = f"vpp_volume_weighted_return_{window}m_bps"
        out[name] = _weighted_rolling_sum(ret_1m_bps, volume, window) / vol_sum
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Volume-weighted 1m return over past {window}m."))

        name = f"vpp_up_return_pressure_{window}m_bps"
        out[name] = _weighted_rolling_sum(up_ret, volume, window) / vol_sum
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Volume-weighted positive return pressure over past {window}m."))

        name = f"vpp_down_return_pressure_{window}m_bps"
        out[name] = _weighted_rolling_sum(down_ret, volume, window) / vol_sum
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Volume-weighted negative return pressure over past {window}m."))

        name = f"vpp_up_down_volume_ratio_{window}m"
        out[name] = up_vol_sum / down_vol_sum.replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Up-volume divided by down-volume over past {window}m."))

        name = f"vpp_return_volume_corr_{window}m"
        out[name] = ret_1m_bps.rolling(window=window, min_periods=window).corr(log_volume)
        specs.append(FeatureSpec(name, "volume_price_pressure", window, description=f"Rolling correlation of 1m return and log volume over past {window}m."))

    return out, specs

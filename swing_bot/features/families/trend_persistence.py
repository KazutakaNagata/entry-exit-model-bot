"""Past-only trend persistence features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

TREND_WINDOWS = (15, 30, 60, 120, 240)


def _rolling_slope_bps(values: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of log price per minute, expressed in bps/minute."""
    x = np.arange(window, dtype="float64")
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2))

    def slope(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        y = arr - arr.mean()
        return float(np.dot(x_centered, y) / denom * 10000.0)

    return np.log(values.astype("float64")).rolling(window=window, min_periods=window).apply(slope, raw=True)


def build_trend_persistence_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build simple rolling trend features from past closes."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    ret_1m = np.log(close / close.shift(1))

    for window in TREND_WINDOWS:
        ma = close.rolling(window=window, min_periods=window).mean()
        out[f"close_vs_ma_{window}m_bps"] = np.log(close / ma) * 10000.0
        out[f"trend_slope_{window}m_bps_per_min"] = _rolling_slope_bps(close, window)
        out[f"trend_positive_ret_share_{window}m"] = (ret_1m > 0.0).astype("float64").rolling(window=window, min_periods=window).mean()
        specs.extend([
            FeatureSpec(f"close_vs_ma_{window}m_bps", "trend_persistence", window, description=f"Close versus past {window}m simple moving average."),
            FeatureSpec(f"trend_slope_{window}m_bps_per_min", "trend_persistence", window, description=f"Rolling OLS log-price slope over past {window}m."),
            FeatureSpec(f"trend_positive_ret_share_{window}m", "trend_persistence", window, description=f"Share of positive 1m returns over past {window}m."),
        ])

    if 60 in TREND_WINDOWS and 240 in TREND_WINDOWS:
        out["trend_slope_60m_minus_240m_bps_per_min"] = out["trend_slope_60m_bps_per_min"] - out["trend_slope_240m_bps_per_min"]
        specs.append(FeatureSpec("trend_slope_60m_minus_240m_bps_per_min", "trend_persistence", 240, description="60m trend slope minus 240m trend slope."))
    return out, specs

"""Past-only volatility and range features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

VOL_WINDOWS = (15, 30, 60, 120, 240)
RANGE_WINDOWS = (5, 15, 60, 120, 240)


def build_volatility_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build realized volatility and high/low range features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0

    for window in VOL_WINDOWS:
        name = f"realized_vol_{window}m_bps"
        out[name] = ret_1m_bps.rolling(window=window, min_periods=window).std(ddof=0)
        specs.append(FeatureSpec(name, "volatility", window, description=f"Rolling std of 1m log returns over past {window}m."))

    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    eps = 1e-12
    for window in RANGE_WINDOWS:
        name = f"range_{window}m_bps"
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        out[name] = np.log((rolling_high + eps) / (rolling_low + eps)) * 10000.0
        specs.append(FeatureSpec(name, "volatility", window, description=f"Past {window}m rolling high/low range in bps."))

    # Current range relative to recent vol; no future data used.
    if "range_15m_bps" in out and "range_60m_bps" in out:
        out["range_15m_to_60m_ratio"] = out["range_15m_bps"] / out["range_60m_bps"].replace(0.0, np.nan)
        specs.append(FeatureSpec("range_15m_to_60m_ratio", "volatility", 60, description="15m range divided by 60m range."))
    return out, specs

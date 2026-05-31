"""Small, past-only price/candle features from finalized 1-minute OHLCV."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec


def build_price_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build 1-minute candle price features available after bar close."""
    out = pd.DataFrame(index=df.index)
    eps = 1e-12

    out["price_body_1m_bps"] = np.log((df["close"] + eps) / (df["open"] + eps)) * 10000.0
    out["price_upper_wick_1m_bps"] = np.log((df["high"] + eps) / (df[["open", "close"]].max(axis=1) + eps)) * 10000.0
    out["price_lower_wick_1m_bps"] = np.log((df[["open", "close"]].min(axis=1) + eps) / (df["low"] + eps)) * 10000.0
    out["price_range_1m_bps"] = np.log((df["high"] + eps) / (df["low"] + eps)) * 10000.0

    specs = [
        FeatureSpec("price_body_1m_bps", "price", 1, description="1m candle body log return in bps."),
        FeatureSpec("price_upper_wick_1m_bps", "price", 1, description="1m upper wick size in bps."),
        FeatureSpec("price_lower_wick_1m_bps", "price", 1, description="1m lower wick size in bps."),
        FeatureSpec("price_range_1m_bps", "price", 1, description="1m high/low range in bps."),
    ]
    return out, specs

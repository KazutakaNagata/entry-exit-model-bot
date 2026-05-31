"""Past-only volume/activity features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

VOLUME_WINDOWS = (15, 30, 60, 120, 240)


def build_volume_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build simple volume z-score and relative-volume features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    volume = df["volume"].astype("float64")
    log_volume = np.log1p(volume)

    for window in VOLUME_WINDOWS:
        mean = log_volume.rolling(window=window, min_periods=window).mean()
        std = log_volume.rolling(window=window, min_periods=window).std(ddof=0)
        out[f"log_volume_z_{window}m"] = (log_volume - mean) / std.replace(0.0, np.nan)
        out[f"volume_to_ma_{window}m"] = volume / volume.rolling(window=window, min_periods=window).mean().replace(0.0, np.nan)
        specs.extend([
            FeatureSpec(f"log_volume_z_{window}m", "volume", window, description=f"Log1p volume z-score over past {window}m."),
            FeatureSpec(f"volume_to_ma_{window}m", "volume", window, description=f"Volume divided by past {window}m moving average volume."),
        ])
    return out, specs

"""Past-only return path features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

DEFAULT_WINDOWS = (1, 3, 5, 15, 30, 60, 120, 240)


def build_return_path_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build close-to-close log returns over past windows.

    ``ret_60m_bps`` at row t is log(close[t] / close[t-60]), which is available
    only after bar t closes and does not use future bars.
    """
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    for window in windows:
        name = f"ret_{window}m_bps"
        out[name] = np.log(close / close.shift(window)) * 10000.0
        specs.append(FeatureSpec(name, "return_path", window, description=f"Past {window}m close-to-close log return in bps."))

    # Simple acceleration-style deltas are still return_path and use only past returns.
    if 15 in windows and 60 in windows:
        out["ret_15m_minus_60m_avg_bps"] = out["ret_15m_bps"] - out["ret_60m_bps"] / 4.0
        specs.append(FeatureSpec("ret_15m_minus_60m_avg_bps", "return_path", 60, description="15m return minus average 15m segment of 60m return."))
    if 60 in windows and 240 in windows:
        out["ret_60m_minus_240m_avg_bps"] = out["ret_60m_bps"] - out["ret_240m_bps"] / 4.0
        specs.append(FeatureSpec("ret_60m_minus_240m_avg_bps", "return_path", 240, description="60m return minus average 60m segment of 240m return."))
    return out, specs

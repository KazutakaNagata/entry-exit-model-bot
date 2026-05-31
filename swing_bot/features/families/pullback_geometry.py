"""Past-only pullback and recovery geometry features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

PULLBACK_WINDOWS = (30, 60, 120, 240)
_EPS = 1e-12


def build_pullback_geometry_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = PULLBACK_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build reviewed pullback/recovery proxies from past rolling extrema."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    for window in windows:
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        range_abs = (rolling_high - rolling_low).replace(0.0, np.nan)

        name = f"pbg_drawdown_from_high_{window}m_bps"
        out[name] = np.log((close + _EPS) / (rolling_high + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "pullback_geometry", window, description=f"Close drawdown from past {window}m high; usually <= 0."))

        name = f"pbg_bounce_from_low_{window}m_bps"
        out[name] = np.log((close + _EPS) / (rolling_low + _EPS)) * 10000.0
        specs.append(FeatureSpec(name, "pullback_geometry", window, description=f"Close rebound from past {window}m low."))

        name = f"pbg_pullback_depth_in_range_{window}m"
        out[name] = (rolling_high - close) / range_abs
        specs.append(FeatureSpec(name, "pullback_geometry", window, description=f"Pullback depth from high as share of past {window}m range."))

        name = f"pbg_recovery_from_low_in_range_{window}m"
        out[name] = (close - rolling_low) / range_abs
        specs.append(FeatureSpec(name, "pullback_geometry", window, description=f"Recovery from low as share of past {window}m range."))

    if 30 in windows and 120 in windows:
        out["pbg_drawdown_30m_minus_120m_bps"] = out["pbg_drawdown_from_high_30m_bps"] - out["pbg_drawdown_from_high_120m_bps"]
        specs.append(FeatureSpec("pbg_drawdown_30m_minus_120m_bps", "pullback_geometry", 120, description="Short-window drawdown minus medium-window drawdown."))
    if 60 in windows and 240 in windows:
        out["pbg_pullback_depth_60m_minus_240m"] = out["pbg_pullback_depth_in_range_60m"] - out["pbg_pullback_depth_in_range_240m"]
        specs.append(FeatureSpec("pbg_pullback_depth_60m_minus_240m", "pullback_geometry", 240, description="60m pullback depth minus 240m pullback depth."))

    return out, specs

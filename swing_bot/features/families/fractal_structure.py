"""Past-only fractal-ish price structure proxies.

These are deliberately simple, vectorized proxies for higher-high / higher-low
and close-location structure.  They are not pivot labels and do not use future
bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

FRACTAL_WINDOWS = (15, 30, 60, 120, 240)
_EPS = 1e-12


def _log_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log((a + _EPS) / (b + _EPS)) * 10000.0


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def build_fractal_structure_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = FRACTAL_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build past-only structural progression proxies."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    for window in windows:
        half = max(3, window // 2)
        fast_high = high.rolling(window=half, min_periods=half).max()
        slow_high = high.shift(half).rolling(window=half, min_periods=half).max()
        fast_low = low.rolling(window=half, min_periods=half).min()
        slow_low = low.shift(half).rolling(window=half, min_periods=half).min()
        range_now = (fast_high - fast_low).replace(0.0, np.nan)

        name = f"fs_higher_high_strength_{window}m_bps"
        out[name] = _log_bps(fast_high, slow_high)
        specs.append(FeatureSpec(name, "fractal_structure", window, description=f"Recent half-window high versus prior half-window high for {window}m."))

        name = f"fs_higher_low_strength_{window}m_bps"
        out[name] = _log_bps(fast_low, slow_low)
        specs.append(FeatureSpec(name, "fractal_structure", window, description=f"Recent half-window low versus prior half-window low for {window}m."))

        name = f"fs_structure_slope_{window}m_bps"
        out[name] = (out[f"fs_higher_high_strength_{window}m_bps"] + out[f"fs_higher_low_strength_{window}m_bps"]) / 2.0
        specs.append(FeatureSpec(name, "fractal_structure", window, description=f"Average higher-high/higher-low structure slope for {window}m."))

        name = f"fs_structure_width_change_{window}m_bps"
        prev_range = (slow_high - slow_low).replace(0.0, np.nan)
        out[name] = _log_bps(range_now, prev_range)
        specs.append(FeatureSpec(name, "fractal_structure", window, description=f"Recent structural range width versus prior width for {window}m."))

        name = f"fs_close_position_recent_half_{window}m"
        out[name] = _safe_div(close - fast_low, range_now)
        specs.append(FeatureSpec(name, "fractal_structure", window, description=f"Close position in recent half-window range for {window}m."))

        name = f"fs_uptrend_structure_score_{window}m"
        hh = out[f"fs_higher_high_strength_{window}m_bps"].clip(lower=0.0)
        hl = out[f"fs_higher_low_strength_{window}m_bps"].clip(lower=0.0)
        out[name] = (hh + hl) / 200.0 * out[f"fs_close_position_recent_half_{window}m"].clip(0.0, 1.0)
        specs.append(FeatureSpec(name, "fractal_structure", window, description=f"Higher-high/higher-low uptrend score weighted by close position for {window}m."))

    return out, specs

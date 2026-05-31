"""Enhanced past-only pullback quality features for swing entries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

EPB_WINDOWS = (30, 60, 120, 240, 480)
_EPS = 1e-12


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def _log_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log((a + _EPS) / (b + _EPS)) * 10000.0


def build_enhanced_pullback_geometry_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = EPB_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build pullback depth, recovery, and quality proxies."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    ret_1 = close.pct_change()

    for window in windows:
        rh = high.rolling(window=window, min_periods=window).max()
        rl = low.rolling(window=window, min_periods=window).min()
        rr = (rh - rl).replace(0.0, np.nan)
        ma = close.rolling(window=window, min_periods=window).mean()
        slope = _log_bps(close, close.shift(window))
        vol = ret_1.rolling(window=window, min_periods=window).std().replace(0.0, np.nan)

        drawdown_bps = _log_bps(close, rh)
        bounce_bps = _log_bps(close, rl)
        depth = _safe_div(rh - close, rr)
        recovery = _safe_div(close - rl, rr)

        name = f"epb_drawdown_from_high_{window}m_bps"
        out[name] = drawdown_bps
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Close drawdown from rolling {window}m high."))

        name = f"epb_bounce_from_low_{window}m_bps"
        out[name] = bounce_bps
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Close bounce from rolling {window}m low."))

        name = f"epb_pullback_depth_{window}m"
        out[name] = depth
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Pullback depth inside rolling {window}m range."))

        name = f"epb_recovery_ratio_{window}m"
        out[name] = recovery
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Recovery ratio inside rolling {window}m range."))

        name = f"epb_shallow_pullback_trend_score_{window}m"
        out[name] = (slope.clip(lower=0.0) / 100.0) * (1.0 - depth.clip(lower=0.0, upper=1.0))
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Positive trend strength weighted by shallow pullback for {window}m."))

        name = f"epb_recovery_over_depth_{window}m"
        out[name] = _safe_div(recovery, depth.abs() + _EPS)
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Recovery ratio divided by pullback depth for {window}m."))

        name = f"epb_ma_reclaim_{window}m_bps"
        out[name] = _log_bps(close, ma)
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Close versus rolling {window}m mean as reclaim proxy."))

        name = f"epb_drawdown_norm_vol_{window}m"
        out[name] = drawdown_bps / (vol * 10000.0)
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Drawdown from high normalized by realized volatility for {window}m."))

        name = f"epb_bounce_norm_vol_{window}m"
        out[name] = bounce_bps / (vol * 10000.0)
        specs.append(FeatureSpec(name, "enhanced_pullback_geometry", window, description=f"Bounce from low normalized by realized volatility for {window}m."))

    if 60 in windows and 240 in windows:
        out["epb_pullback_depth_60m_minus_240m"] = out["epb_pullback_depth_60m"] - out["epb_pullback_depth_240m"]
        specs.append(FeatureSpec("epb_pullback_depth_60m_minus_240m", "enhanced_pullback_geometry", 240, description="Short pullback depth minus slower pullback depth."))
    if 120 in windows and 480 in windows:
        out["epb_shallow_trend_120m_minus_480m"] = out["epb_shallow_pullback_trend_score_120m"] - out["epb_shallow_pullback_trend_score_480m"]
        specs.append(FeatureSpec("epb_shallow_trend_120m_minus_480m", "enhanced_pullback_geometry", 480, description="Medium shallow-pullback trend score minus slower score."))

    return out, specs

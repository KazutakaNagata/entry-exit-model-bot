"""Past-only trend regime features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

WINDOWS = (15, 30, 60, 120, 240, 480, 720)
SHORT_LONG_PAIRS = ((15, 60), (30, 120), (60, 240), (120, 480), (240, 720))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def _rolling_slope_bps(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype="float64")
    xc = x - x.mean()
    denom = float(np.sum(xc ** 2))

    def slope(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        y = arr - arr.mean()
        return float(np.dot(xc, y) / denom * 10000.0)

    return np.log(close).rolling(window=window, min_periods=window).apply(slope, raw=True)


def build_trend_regime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build trend regime, trend quality, and moving-average stack features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0
    abs_ret = ret_1m_bps.abs()

    ma: dict[int, pd.Series] = {}
    slopes: dict[int, pd.Series] = {}
    for window in WINDOWS:
        ma[window] = close.rolling(window=window, min_periods=window).mean()
        slopes[window] = _rolling_slope_bps(close, window)

        name = f"trend_regime_close_vs_ma_{window}m_bps"
        out[name] = np.log(close / ma[window]) * 10000.0
        specs.append(FeatureSpec(name, "trend_regime", window, description=f"Close distance to {window}m MA."))

        name = f"trend_regime_slope_{window}m_bps_per_min"
        out[name] = slopes[window]
        specs.append(FeatureSpec(name, "trend_regime", window, description=f"Rolling log-price slope over {window}m."))

        name = f"trend_regime_efficiency_{window}m"
        net = np.log(close / close.shift(window)).abs() * 10000.0
        path = abs_ret.rolling(window=window, min_periods=window).sum()
        out[name] = _safe_div(net, path)
        specs.append(FeatureSpec(name, "trend_regime", window, description=f"Directional efficiency over {window}m."))

        name = f"trend_regime_positive_share_{window}m"
        out[name] = (ret_1m_bps > 0.0).astype("float64").rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "trend_regime", window, description=f"Positive return share over {window}m."))

        name = f"trend_regime_signed_absret_share_{window}m"
        out[name] = ret_1m_bps.rolling(window=window, min_periods=window).sum() / abs_ret.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "trend_regime", window, description=f"Signed return sum divided by absolute return sum over {window}m."))

    for short, long in SHORT_LONG_PAIRS:
        name = f"trend_regime_ma_stack_{short}_{long}m_bps"
        out[name] = np.log(ma[short] / ma[long]) * 10000.0
        specs.append(FeatureSpec(name, "trend_regime", long, description=f"{short}m MA versus {long}m MA."))

        name = f"trend_regime_slope_diff_{short}_{long}m_bps_per_min"
        out[name] = slopes[short] - slopes[long]
        specs.append(FeatureSpec(name, "trend_regime", long, description=f"{short}m slope minus {long}m slope."))

        name = f"trend_regime_close_between_ma_{short}_{long}m"
        low_ma = np.minimum(ma[short], ma[long])
        high_ma = np.maximum(ma[short], ma[long])
        out[name] = ((close >= low_ma) & (close <= high_ma)).astype("float64")
        specs.append(FeatureSpec(name, "trend_regime", long, description=f"Close is between {short}m and {long}m moving averages."))

    # Multi-MA stack alignment: positive values mean shorter MAs are above longer MAs.
    stack_pairs = [(15, 30), (30, 60), (60, 120), (120, 240), (240, 480), (480, 720)]
    stack_terms = []
    for short, long in stack_pairs:
        term = np.sign(np.log(ma[short] / ma[long]))
        stack_terms.append(term)
    out["trend_regime_ma_stack_alignment_score"] = pd.concat(stack_terms, axis=1).mean(axis=1)
    specs.append(FeatureSpec("trend_regime_ma_stack_alignment_score", "trend_regime", 720, description="Mean sign of short-vs-long MA stack pairs."))

    return out, specs

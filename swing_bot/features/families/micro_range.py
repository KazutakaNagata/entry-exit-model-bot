"""Past-only micro range, compression, and intrabar structure features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

WINDOWS = (5, 8, 13, 21, 34, 55, 89, 144, 240)
RATIO_PAIRS = ((5, 21), (8, 34), (13, 55), (21, 89), (34, 144), (55, 240))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_micro_range_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build range compression/expansion features from finalized 1m bars."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    open_ = df["open"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    eps = 1e-12

    bar_range_bps = np.log((high + eps) / (low + eps)) * 10000.0
    body_bps = np.log(close / open_.replace(0.0, np.nan)).abs() * 10000.0
    upper_wick_bps = np.log((high + eps) / np.maximum(open_, close).replace(0.0, np.nan)) * 10000.0
    lower_wick_bps = np.log(np.minimum(open_, close).replace(0.0, np.nan) / (low + eps)) * 10000.0
    close_pos = (close - low) / (high - low).replace(0.0, np.nan)

    out["micro_bar_range_1m_bps"] = bar_range_bps
    out["micro_body_to_range_1m"] = _safe_div(body_bps, bar_range_bps)
    out["micro_upper_wick_to_range_1m"] = _safe_div(upper_wick_bps, bar_range_bps)
    out["micro_lower_wick_to_range_1m"] = _safe_div(lower_wick_bps, bar_range_bps)
    out["micro_close_position_1m"] = close_pos
    specs.extend([
        FeatureSpec("micro_bar_range_1m_bps", "micro_range", 1, description="Current finalized 1m high/low range."),
        FeatureSpec("micro_body_to_range_1m", "micro_range", 1, description="Current candle body divided by range."),
        FeatureSpec("micro_upper_wick_to_range_1m", "micro_range", 1, description="Current upper wick divided by range."),
        FeatureSpec("micro_lower_wick_to_range_1m", "micro_range", 1, description="Current lower wick divided by range."),
        FeatureSpec("micro_close_position_1m", "micro_range", 1, description="Close position inside current 1m candle range."),
    ])

    range_mean: dict[int, pd.Series] = {}
    range_std: dict[int, pd.Series] = {}
    for window in WINDOWS:
        range_mean[window] = bar_range_bps.rolling(window=window, min_periods=window).mean()
        range_std[window] = bar_range_bps.rolling(window=window, min_periods=window).std(ddof=0)
        name = f"micro_range_mean_{window}m_bps"
        out[name] = range_mean[window]
        specs.append(FeatureSpec(name, "micro_range", window, description=f"Mean 1m range over past {window}m."))

        name = f"micro_range_z_{window}m"
        out[name] = (bar_range_bps - range_mean[window]) / range_std[window].replace(0.0, np.nan)
        specs.append(FeatureSpec(name, "micro_range", window, description=f"Current 1m range z-score over {window}m."))

        name = f"micro_body_share_mean_{window}m"
        out[name] = _safe_div(body_bps, bar_range_bps).rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "micro_range", window, description=f"Mean body/range share over {window}m."))

        name = f"micro_close_position_mean_{window}m"
        out[name] = close_pos.rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "micro_range", window, description=f"Mean close position inside candles over {window}m."))

        name = f"micro_wick_imbalance_mean_{window}m"
        wick_sum = (upper_wick_bps + lower_wick_bps).replace(0.0, np.nan)
        out[name] = ((lower_wick_bps - upper_wick_bps) / wick_sum).rolling(window=window, min_periods=window).mean()
        specs.append(FeatureSpec(name, "micro_range", window, description=f"Mean lower-minus-upper wick imbalance over {window}m."))

    for short, long in RATIO_PAIRS:
        name = f"micro_range_compression_{short}_{long}m"
        out[name] = _safe_div(range_mean[short], range_mean[long])
        specs.append(FeatureSpec(name, "micro_range", long, description=f"{short}m mean range divided by {long}m mean range."))

        name = f"micro_range_volatility_ratio_{short}_{long}m"
        out[name] = _safe_div(range_std[short], range_std[long])
        specs.append(FeatureSpec(name, "micro_range", long, description=f"{short}m range std divided by {long}m range std."))

    return out, specs

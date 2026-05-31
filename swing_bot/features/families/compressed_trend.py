"""Deterministic compressed trend summary features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_compressed_trend_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build low-dimensional trend summaries across short/mid/long windows."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    windows = (5, 15, 30, 60, 120, 240, 480)
    rets = {w: np.log(close / close.shift(w)) * 10000.0 for w in windows}

    short = pd.concat([rets[5], rets[15], rets[30]], axis=1)
    mid = pd.concat([rets[60], rets[120]], axis=1)
    long = pd.concat([rets[240], rets[480]], axis=1)
    out["compressed_trend_short_mean_bps"] = short.mean(axis=1)
    out["compressed_trend_mid_mean_bps"] = mid.mean(axis=1)
    out["compressed_trend_long_mean_bps"] = long.mean(axis=1)
    out["compressed_trend_short_max_bps"] = short.max(axis=1)
    out["compressed_trend_short_min_bps"] = short.min(axis=1)
    out["compressed_trend_mid_max_bps"] = mid.max(axis=1)
    out["compressed_trend_mid_min_bps"] = mid.min(axis=1)
    out["compressed_trend_long_max_bps"] = long.max(axis=1)
    out["compressed_trend_long_min_bps"] = long.min(axis=1)
    out["compressed_trend_short_minus_mid_bps"] = out["compressed_trend_short_mean_bps"] - out["compressed_trend_mid_mean_bps"]
    out["compressed_trend_mid_minus_long_bps"] = out["compressed_trend_mid_mean_bps"] - out["compressed_trend_long_mean_bps"]
    out["compressed_trend_alignment_score"] = pd.concat([np.sign(rets[w]) for w in windows], axis=1).mean(axis=1)
    out["compressed_trend_positive_window_share"] = pd.concat([(rets[w] > 0.0).astype("float64") for w in windows], axis=1).mean(axis=1)
    out["compressed_trend_abs_short_to_long_ratio"] = _safe_div(out["compressed_trend_short_mean_bps"].abs(), out["compressed_trend_long_mean_bps"].abs())

    ret_matrix = pd.concat([rets[w] for w in windows], axis=1)
    out["compressed_trend_window_std_bps"] = ret_matrix.std(axis=1, ddof=0)
    out["compressed_trend_window_range_bps"] = ret_matrix.max(axis=1) - ret_matrix.min(axis=1)
    out["compressed_trend_best_window_bps"] = ret_matrix.max(axis=1)
    out["compressed_trend_worst_window_bps"] = ret_matrix.min(axis=1)
    out["compressed_trend_short_mid_same_sign"] = (np.sign(out["compressed_trend_short_mean_bps"]) == np.sign(out["compressed_trend_mid_mean_bps"])).astype("float64")
    out["compressed_trend_mid_long_same_sign"] = (np.sign(out["compressed_trend_mid_mean_bps"]) == np.sign(out["compressed_trend_long_mean_bps"])).astype("float64")

    for name in out.columns:
        specs.append(FeatureSpec(name, "compressed_trend", 480, description="Compressed deterministic trend summary across return windows."))
    return out, specs

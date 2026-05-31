"""Deterministic compressed breakout/structure summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

CBO_WINDOWS = (30, 60, 120, 240, 480)
_EPS = 1e-12


def _log_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log((a + _EPS) / (b + _EPS)) * 10000.0


def build_compressed_breakout_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = CBO_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build compact cross-window breakout summaries."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")
    vol_z = (volume - volume.rolling(120, min_periods=120).mean()) / volume.rolling(120, min_periods=120).std().replace(0.0, np.nan)

    breakout_series: list[pd.Series] = []
    breakdown_series: list[pd.Series] = []
    failed_up_series: list[pd.Series] = []
    range_expansion_series: list[pd.Series] = []

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        prev_range = (prev_high - prev_low).replace(0.0, np.nan)
        cur_range = (high - low).replace(0.0, np.nan)
        breakout = np.maximum(_log_bps(close, prev_high), 0.0).rename(str(window))
        breakdown = np.maximum(_log_bps(prev_low, close), 0.0).rename(str(window))
        failed_up = (np.maximum(_log_bps(high, prev_high), 0.0) * (close < prev_high).astype("float64")).rename(str(window))
        range_exp = (cur_range / prev_range).rename(str(window))
        breakout_series.append(breakout)
        breakdown_series.append(breakdown)
        failed_up_series.append(failed_up)
        range_expansion_series.append(range_exp)

    bo = pd.concat(breakout_series, axis=1)
    bd = pd.concat(breakdown_series, axis=1)
    fail = pd.concat(failed_up_series, axis=1)
    rng = pd.concat(range_expansion_series, axis=1)
    max_lb = max(windows) + 1

    features = {
        "cbo_breakout_mean_bps": bo.mean(axis=1),
        "cbo_breakout_max_bps": bo.max(axis=1),
        "cbo_breakout_count_positive": (bo > 0.0).sum(axis=1),
        "cbo_breakdown_mean_bps": bd.mean(axis=1),
        "cbo_breakdown_max_bps": bd.max(axis=1),
        "cbo_failed_breakout_mean_bps": fail.mean(axis=1),
        "cbo_failed_breakout_max_bps": fail.max(axis=1),
        "cbo_range_expansion_mean": rng.mean(axis=1),
        "cbo_range_expansion_max": rng.max(axis=1),
    }
    for name, values in features.items():
        out[name] = values
        specs.append(FeatureSpec(name, "compressed_breakout", max_lb, description="Cross-window deterministic breakout summary."))

    out["cbo_breakout_volume_confirm_score"] = bo.max(axis=1) * vol_z.clip(lower=0.0)
    specs.append(FeatureSpec("cbo_breakout_volume_confirm_score", "compressed_breakout", max(max_lb, 120), description="Max breakout strength weighted by positive volume z-score."))
    out["cbo_breakout_minus_failed_breakout_bps"] = bo.max(axis=1) - fail.max(axis=1)
    specs.append(FeatureSpec("cbo_breakout_minus_failed_breakout_bps", "compressed_breakout", max_lb, description="Max breakout strength minus max failed-breakout wick."))

    return out, specs

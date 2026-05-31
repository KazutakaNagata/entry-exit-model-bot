"""Deterministic compressed market activity features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_compressed_activity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build compact activity summaries from volume, range, and absolute return."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64").clip(lower=0.0)
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0
    abs_ret = ret_1m_bps.abs()
    bar_range = np.log(high / low.replace(0.0, np.nan)) * 10000.0
    log_volume = np.log1p(volume)
    windows = (15, 30, 60, 120, 240)

    absret_means = {w: abs_ret.rolling(w, min_periods=w).mean() for w in windows}
    range_means = {w: bar_range.rolling(w, min_periods=w).mean() for w in windows}
    vol_means = {w: log_volume.rolling(w, min_periods=w).mean() for w in windows}

    out["compressed_activity_absret_short_mean"] = pd.concat([absret_means[15], absret_means[30]], axis=1).mean(axis=1)
    out["compressed_activity_absret_mid_mean"] = pd.concat([absret_means[60], absret_means[120]], axis=1).mean(axis=1)
    out["compressed_activity_absret_long_mean"] = absret_means[240]
    out["compressed_activity_range_short_mean"] = pd.concat([range_means[15], range_means[30]], axis=1).mean(axis=1)
    out["compressed_activity_range_mid_mean"] = pd.concat([range_means[60], range_means[120]], axis=1).mean(axis=1)
    out["compressed_activity_range_long_mean"] = range_means[240]
    out["compressed_activity_volume_short_mean"] = pd.concat([vol_means[15], vol_means[30]], axis=1).mean(axis=1)
    out["compressed_activity_volume_mid_mean"] = pd.concat([vol_means[60], vol_means[120]], axis=1).mean(axis=1)
    out["compressed_activity_volume_long_mean"] = vol_means[240]

    out["compressed_activity_absret_short_to_long"] = _safe_div(out["compressed_activity_absret_short_mean"], out["compressed_activity_absret_long_mean"])
    out["compressed_activity_range_short_to_long"] = _safe_div(out["compressed_activity_range_short_mean"], out["compressed_activity_range_long_mean"])
    out["compressed_activity_volume_short_to_long"] = _safe_div(out["compressed_activity_volume_short_mean"], out["compressed_activity_volume_long_mean"])
    out["compressed_activity_joint_short_score"] = (
        out["compressed_activity_absret_short_to_long"]
        + out["compressed_activity_range_short_to_long"]
        + out["compressed_activity_volume_short_to_long"]
    ) / 3.0
    out["compressed_activity_return_volume_agreement"] = np.sign(ret_1m_bps) * log_volume
    out["compressed_activity_return_volume_agreement_60m"] = out["compressed_activity_return_volume_agreement"].rolling(60, min_periods=60).mean()
    out["compressed_activity_absret_mid_to_long"] = _safe_div(out["compressed_activity_absret_mid_mean"], out["compressed_activity_absret_long_mean"])
    out["compressed_activity_range_mid_to_long"] = _safe_div(out["compressed_activity_range_mid_mean"], out["compressed_activity_range_long_mean"])
    out["compressed_activity_volume_mid_to_long"] = _safe_div(out["compressed_activity_volume_mid_mean"], out["compressed_activity_volume_long_mean"])
    out["compressed_activity_joint_mid_score"] = (
        out["compressed_activity_absret_mid_to_long"]
        + out["compressed_activity_range_mid_to_long"]
        + out["compressed_activity_volume_mid_to_long"]
    ) / 3.0

    for name in out.columns:
        lookback = 240 if not name.endswith("60m") else 60
        specs.append(FeatureSpec(name, "compressed_activity", lookback, description="Compressed deterministic activity summary."))
    return out, specs

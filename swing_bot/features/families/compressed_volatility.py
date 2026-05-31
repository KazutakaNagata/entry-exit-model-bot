"""Deterministic compressed volatility regime features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_compressed_volatility_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build compact volatility summaries and regime ratios."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []
    close = df["close"].astype("float64")
    ret_1m_bps = np.log(close / close.shift(1)) * 10000.0
    windows = (15, 30, 60, 120, 240, 480)
    vols = {w: ret_1m_bps.rolling(w, min_periods=w).std(ddof=0) for w in windows}

    out["compressed_vol_short_mean_bps"] = pd.concat([vols[15], vols[30]], axis=1).mean(axis=1)
    out["compressed_vol_mid_mean_bps"] = pd.concat([vols[60], vols[120]], axis=1).mean(axis=1)
    out["compressed_vol_long_mean_bps"] = pd.concat([vols[240], vols[480]], axis=1).mean(axis=1)
    out["compressed_vol_short_to_mid"] = _safe_div(out["compressed_vol_short_mean_bps"], out["compressed_vol_mid_mean_bps"])
    out["compressed_vol_mid_to_long"] = _safe_div(out["compressed_vol_mid_mean_bps"], out["compressed_vol_long_mean_bps"])
    out["compressed_vol_short_to_long"] = _safe_div(out["compressed_vol_short_mean_bps"], out["compressed_vol_long_mean_bps"])
    out["compressed_vol_expansion_score"] = (out["compressed_vol_short_to_mid"] + out["compressed_vol_mid_to_long"]) / 2.0
    out["compressed_vol_contraction_score"] = _safe_div(1.0, out["compressed_vol_expansion_score"])

    vol_matrix = pd.concat([vols[w] for w in windows], axis=1)
    out["compressed_vol_window_max_bps"] = vol_matrix.max(axis=1)
    out["compressed_vol_window_min_bps"] = vol_matrix.min(axis=1)
    out["compressed_vol_window_spread_bps"] = out["compressed_vol_window_max_bps"] - out["compressed_vol_window_min_bps"]
    out["compressed_vol_argmax_window_rank"] = vol_matrix.values.argmax(axis=1) if len(vol_matrix) else np.nan
    # Replace unreliable argmax rows where any component is NaN with NaN.
    out.loc[vol_matrix.isna().any(axis=1), "compressed_vol_argmax_window_rank"] = np.nan

    vol_of_vol_60 = ret_1m_bps.abs().rolling(60, min_periods=60).std(ddof=0)
    vol_of_vol_240 = ret_1m_bps.abs().rolling(240, min_periods=240).std(ddof=0)
    out["compressed_vol_of_vol_60m"] = vol_of_vol_60
    out["compressed_vol_of_vol_60_to_240"] = _safe_div(vol_of_vol_60, vol_of_vol_240)

    for name in out.columns:
        specs.append(FeatureSpec(name, "compressed_volatility", 480, description="Compressed deterministic volatility summary."))
    return out, specs

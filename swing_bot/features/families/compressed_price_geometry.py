"""Deterministic compressed price-geometry summaries.

No fitted compressor is used.  These summarize support/resistance position,
pullback state, and room across a small window ladder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_bot.features.manifest import FeatureSpec

CPG_WINDOWS = (60, 120, 240, 480, 720)
_EPS = 1e-12


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def build_compressed_price_geometry_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = CPG_WINDOWS,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build compact cross-window geometry features."""
    out = pd.DataFrame(index=df.index)
    specs: list[FeatureSpec] = []

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    positions: list[pd.Series] = []
    upside_rooms: list[pd.Series] = []
    pullback_depths: list[pd.Series] = []
    ranges: list[pd.Series] = []

    for window in windows:
        prev_high = high.shift(1).rolling(window=window, min_periods=window).max()
        prev_low = low.shift(1).rolling(window=window, min_periods=window).min()
        rng = (prev_high - prev_low).replace(0.0, np.nan)
        pos = _safe_div(close - prev_low, rng)
        room = _safe_div(prev_high - close, rng)
        depth = _safe_div(prev_high - close, rng)
        positions.append(pos.rename(str(window)))
        upside_rooms.append(room.rename(str(window)))
        pullback_depths.append(depth.rename(str(window)))
        ranges.append((rng / close.replace(0.0, np.nan)).rename(str(window)))

    pos_df = pd.concat(positions, axis=1)
    room_df = pd.concat(upside_rooms, axis=1)
    depth_df = pd.concat(pullback_depths, axis=1)
    range_df = pd.concat(ranges, axis=1)
    max_lb = max(windows) + 1

    aggregates = {
        "cpg_position_mean": pos_df.mean(axis=1),
        "cpg_position_min": pos_df.min(axis=1),
        "cpg_position_max": pos_df.max(axis=1),
        "cpg_upside_room_mean": room_df.mean(axis=1),
        "cpg_upside_room_max": room_df.max(axis=1),
        "cpg_pullback_depth_mean": depth_df.mean(axis=1),
        "cpg_pullback_depth_min": depth_df.min(axis=1),
        "cpg_range_pct_mean": range_df.mean(axis=1),
        "cpg_range_pct_max": range_df.max(axis=1),
    }

    for name, values in aggregates.items():
        out[name] = values
        specs.append(FeatureSpec(name, "compressed_price_geometry", max_lb, description="Cross-window deterministic price-geometry summary."))

    out["cpg_position_120m_minus_720m"] = pos_df["120"] - pos_df["720"]
    specs.append(FeatureSpec("cpg_position_120m_minus_720m", "compressed_price_geometry", max_lb, description="120m range position minus 720m position."))
    out["cpg_upside_room_120m_minus_720m"] = room_df["120"] - room_df["720"]
    specs.append(FeatureSpec("cpg_upside_room_120m_minus_720m", "compressed_price_geometry", max_lb, description="120m upside room minus 720m upside room."))
    out["cpg_shallow_pullback_room_score"] = (1.0 - depth_df["120"].clip(0.0, 1.0)) * room_df["240"].clip(lower=0.0)
    specs.append(FeatureSpec("cpg_shallow_pullback_room_score", "compressed_price_geometry", max_lb, description="Shallow 120m pullback weighted by 240m upside room."))

    return out, specs

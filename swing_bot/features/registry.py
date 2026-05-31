"""Feature family registry.

Only reviewed, past-only families are registered here.  Future migrated families
should be added one family at a time with tests and a manifest review.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from swing_bot.features.families.acceleration import build_acceleration_features
from swing_bot.features.families.regime_switch import build_regime_switch_features
from swing_bot.features.families.trend_regime import build_trend_regime_features
from swing_bot.features.families.micro_range import build_micro_range_features
from swing_bot.features.families.volume_spike import build_volume_spike_features
from swing_bot.features.families.compressed_trend import build_compressed_trend_features
from swing_bot.features.families.compressed_activity import build_compressed_activity_features
from swing_bot.features.families.compressed_volatility import build_compressed_volatility_features
from swing_bot.features.families.enhanced_support_resistance import build_enhanced_support_resistance_features
from swing_bot.features.families.enhanced_market_structure_void import build_enhanced_market_structure_void_features
from swing_bot.features.families.enhanced_pullback_geometry import build_enhanced_pullback_geometry_features
from swing_bot.features.families.fractal_structure import build_fractal_structure_features
from swing_bot.features.families.pivot_line import build_pivot_line_features
from swing_bot.features.families.compressed_price_geometry import build_compressed_price_geometry_features
from swing_bot.features.families.compressed_breakout import build_compressed_breakout_features

from swing_bot.features.families.breakout import build_breakout_features
from swing_bot.features.families.market_structure_void import build_market_structure_void_features
from swing_bot.features.families.price import build_price_features
from swing_bot.features.families.pullback_geometry import build_pullback_geometry_features
from swing_bot.features.families.rejection_reversal import build_rejection_reversal_features
from swing_bot.features.families.return_path import build_return_path_features
from swing_bot.features.families.support_resistance import build_support_resistance_features
from swing_bot.features.families.trend_persistence import build_trend_persistence_features
from swing_bot.features.families.volume import build_volume_features
from swing_bot.features.families.volume_price_pressure import build_volume_price_pressure_features
from swing_bot.features.families.volatility import build_volatility_features

from swing_bot.features.families.anchored_timeframe_candle import build_anchored_timeframe_candle_features
from swing_bot.features.families.anchored_timeframe_indicators import build_anchored_timeframe_indicator_features
from swing_bot.features.families.timeframe_phase import build_timeframe_phase_features

from swing_bot.features.manifest import FeatureSpec

FeatureBuilder = Callable[[pd.DataFrame], tuple[pd.DataFrame, list[FeatureSpec]]]

FEATURE_BUILDERS: dict[str, FeatureBuilder] = {
    "price": build_price_features,
    "return_path": build_return_path_features,
    "volatility": build_volatility_features,
    "trend_persistence": build_trend_persistence_features,
    "volume": build_volume_features,
    "acceleration": build_acceleration_features,
    "trend_regime": build_trend_regime_features,
    "regime_switch": build_regime_switch_features,
    "micro_range": build_micro_range_features,
    "volume_spike": build_volume_spike_features,
    "compressed_trend": build_compressed_trend_features,
    "compressed_activity": build_compressed_activity_features,
    "compressed_volatility": build_compressed_volatility_features,
    "enhanced_support_resistance": build_enhanced_support_resistance_features,
    "enhanced_market_structure_void": build_enhanced_market_structure_void_features,
    "enhanced_pullback_geometry": build_enhanced_pullback_geometry_features,
    "fractal_structure": build_fractal_structure_features,
    "pivot_line": build_pivot_line_features,
    "compressed_price_geometry": build_compressed_price_geometry_features,
    "compressed_breakout": build_compressed_breakout_features,
    "support_resistance": build_support_resistance_features,
    "market_structure_void": build_market_structure_void_features,
    "breakout": build_breakout_features,
    "volume_price_pressure": build_volume_price_pressure_features,
    "rejection_reversal": build_rejection_reversal_features,
    "pullback_geometry": build_pullback_geometry_features,
    "anchored_timeframe_candle": build_anchored_timeframe_candle_features,
    "anchored_timeframe_indicators": build_anchored_timeframe_indicator_features,
    "timeframe_phase": build_timeframe_phase_features,
}


def available_families() -> list[str]:
    return sorted(FEATURE_BUILDERS.keys())


def get_feature_builder(family: str) -> FeatureBuilder:
    try:
        return FEATURE_BUILDERS[family]
    except KeyError as exc:
        raise KeyError(f"unknown feature family {family!r}; available={available_families()}") from exc

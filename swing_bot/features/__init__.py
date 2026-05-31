"""Feature generation package."""
from swing_bot.features.build_matrix import FeatureBuildError, build_feature_matrix, feature_summary
from swing_bot.features.manifest import FeatureSpec
from swing_bot.features.registry import available_families

__all__ = [
    "FeatureBuildError",
    "FeatureSpec",
    "available_families",
    "build_feature_matrix",
    "feature_summary",
]

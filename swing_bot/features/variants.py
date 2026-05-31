"""Feature-set variant helpers used for family-by-family entry experiments.

The helpers in this module only build YAML configs.  They do not calculate
features, train models, read test data, or decide which variant is best.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

BASE_V1_FAMILIES: tuple[str, ...] = (
    "price",
    "return_path",
    "volatility",
    "trend_persistence",
    "volume",
    "support_resistance",
    "market_structure_void",
    "pullback_geometry",
)

V2_CANDIDATE_FAMILIES: tuple[str, ...] = (
    "breakout",
    "volume_price_pressure",
    "rejection_reversal",
)

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "target_*",
    "future_*",
    "*_label*",
    "*label*",
    "*_y",
    "*hold_delta*",
    "*mfe*",
    "*mae*",
    "diag_*",
)


@dataclass(frozen=True)
class FeatureVariant:
    """One deterministic feature-set variant."""

    name: str
    added_families: tuple[str, ...]
    description: str

    @property
    def include_families(self) -> tuple[str, ...]:
        return BASE_V1_FAMILIES + self.added_families


def default_entry_variants() -> list[FeatureVariant]:
    """Return the reviewed v1/v2 family variants.

    The variants are intentionally small and explicit.  They are designed to
    answer: which of breakout, volume/price pressure, and rejection/reversal
    helped or hurt the v1 baseline?
    """
    return [
        FeatureVariant("v1_baseline", tuple(), "Original v1 baseline families."),
        FeatureVariant("v1_plus_breakout", ("breakout",), "v1 plus breakout continuation family."),
        FeatureVariant(
            "v1_plus_volume_price_pressure",
            ("volume_price_pressure",),
            "v1 plus volume/price pressure family.",
        ),
        FeatureVariant(
            "v1_plus_rejection_reversal",
            ("rejection_reversal",),
            "v1 plus rejection/reversal family.",
        ),
        FeatureVariant(
            "v1_plus_breakout_volume_price_pressure",
            ("breakout", "volume_price_pressure"),
            "v1 plus breakout and volume/price pressure families.",
        ),
        FeatureVariant(
            "v1_plus_breakout_rejection_reversal",
            ("breakout", "rejection_reversal"),
            "v1 plus breakout and rejection/reversal families.",
        ),
        FeatureVariant(
            "v1_plus_volume_price_pressure_rejection_reversal",
            ("volume_price_pressure", "rejection_reversal"),
            "v1 plus volume/price pressure and rejection/reversal families.",
        ),
        FeatureVariant(
            "v2_all",
            V2_CANDIDATE_FAMILIES,
            "v1 plus all reviewed v2 candidate families.",
        ),
    ]


def variant_to_config(variant: FeatureVariant, *, purpose_prefix: str = "entry feature variant") -> dict[str, object]:
    """Convert a :class:`FeatureVariant` to a build_features YAML config dict."""
    return {
        "feature_set_name": f"entry_feature_set_{variant.name}",
        "purpose": f"{purpose_prefix}: {variant.description}",
        "base_feature_set": "entry_feature_set_v1",
        "added_families": list(variant.added_families),
        "include_families": list(variant.include_families),
        "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
        "notes": [
            "Variant configs are for valid-fold feature-family comparison only.",
            "They do not touch test and do not imply production adoption.",
            "Every family must be deterministic and past-only; audit_features.py --strict should pass.",
            "Compare variants on valid folds, with worst-fold metrics emphasized over mean-only gains.",
        ],
    }


def write_variant_configs(
    output_dir: Path | str,
    *,
    variants: Sequence[FeatureVariant] | None = None,
    force: bool = False,
) -> list[Path]:
    """Write feature-set variant YAML configs and return written paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for variant in variants or default_entry_variants():
        path = out_dir / f"entry_feature_set_{variant.name}.yaml"
        if path.exists() and not force:
            written.append(path)
            continue
        config = variant_to_config(variant)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        written.append(path)
    manifest_path = out_dir / "feature_variant_manifest.yaml"
    manifest = {
        "name": "entry_feature_variants_v1_v2",
        "base_families": list(BASE_V1_FAMILIES),
        "candidate_families": list(V2_CANDIDATE_FAMILIES),
        "variants": [
            {
                "name": variant.name,
                "config": f"entry_feature_set_{variant.name}.yaml",
                "added_families": list(variant.added_families),
                "include_families": list(variant.include_families),
                "description": variant.description,
            }
            for variant in (variants or default_entry_variants())
        ],
        "notes": [
            "Use these variants to decompose the v2 all-family result.",
            "Recommended first focus models: long_H120 and short_H240.",
            "Do not use test to choose variants.",
        ],
    }
    if force or not manifest_path.exists():
        with manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return written


def load_variant_manifest(path: Path | str) -> dict[str, object]:
    """Load a generated variant manifest."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def variant_names_from_configs(paths: Iterable[Path | str]) -> list[str]:
    """Infer variant names from config file names."""
    names: list[str] = []
    for item in paths:
        stem = Path(item).stem
        prefix = "entry_feature_set_"
        names.append(stem[len(prefix):] if stem.startswith(prefix) else stem)
    return names

"""Reviewed long_H120 v3b family-addition definitions.

These helpers only write feature-set configs. They do not build features,
train models, tune thresholds, read test data, or select a production config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from swing_bot.features.ablation import DEFAULT_EXCLUDE_PATTERNS, V3A_FAMILIES

# Current long_H120 best candidate from the v3a combined-ablation pass:
# full v3a minus compressed_activity.  Keep breakout / rejection / pressure here
# because the combined ablation showed they interact with compressed_activity
# removal and should not be discarded blindly.
LONG_H120_BASE_V0_FAMILIES: tuple[str, ...] = tuple(
    f for f in V3A_FAMILIES if f != "compressed_activity"
)

V3B_STRUCTURE_FAMILIES: tuple[str, ...] = (
    "enhanced_support_resistance",
    "enhanced_market_structure_void",
    "enhanced_pullback_geometry",
    "fractal_structure",
    "pivot_line",
    "compressed_price_geometry",
    "compressed_breakout",
)


@dataclass(frozen=True)
class V3BAdditionConfig:
    """One deterministic long_H120 v3b addition config."""

    name: str
    include_families: tuple[str, ...]
    description: str
    added_families: tuple[str, ...] = ()


def _append_unique(base: Sequence[str], add: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = list(base)
    seen = set(out)
    for family in add:
        if family not in seen:
            out.append(family)
            seen.add(family)
    return tuple(out)


def default_long_h120_v3b_addition_configs() -> list[V3BAdditionConfig]:
    """Return reviewed v3b family additions for the long_H120 base.

    The previous all-v3b candidate worsened the selected long_H120 base.  This
    experiment asks which structure family, if any, helps when added to the
    current base (v3a minus compressed_activity).  It includes one-at-a-time
    additions plus a few domain-motivated combinations; it intentionally avoids
    broad mechanical unit search.
    """
    variants: list[tuple[str, tuple[str, ...], str]] = [
        (
            "base_v0",
            (),
            "Current long_H120 base: v3a minus compressed_activity; no v3b structure families added.",
        ),
        (
            "base_plus_enhanced_support_resistance",
            ("enhanced_support_resistance",),
            "Adds stronger support/resistance distance and range-position structure.",
        ),
        (
            "base_plus_enhanced_market_structure_void",
            ("enhanced_market_structure_void",),
            "Adds upside room / void and previous high-low geometry context.",
        ),
        (
            "base_plus_enhanced_pullback_geometry",
            ("enhanced_pullback_geometry",),
            "Adds pullback depth, recovery, and trend-context pullback quality.",
        ),
        (
            "base_plus_fractal_structure",
            ("fractal_structure",),
            "Adds deterministic past-only high/low structure proxies; expected to be noisy until proven useful.",
        ),
        (
            "base_plus_pivot_line",
            ("pivot_line",),
            "Adds pivot-line style price-location geometry; included as a cautious structure test.",
        ),
        (
            "base_plus_compressed_price_geometry",
            ("compressed_price_geometry",),
            "Adds deterministic compressed price-position / range-location summaries.",
        ),
        (
            "base_plus_compressed_breakout",
            ("compressed_breakout",),
            "Adds compressed breakout context; risky for H120 but useful to isolate from all-v3b deterioration.",
        ),
        (
            "base_plus_room_pullback",
            ("enhanced_market_structure_void", "enhanced_pullback_geometry"),
            "Domain block: upside room plus pullback quality, without noisy pivot/fractal/breakout additions.",
        ),
        (
            "base_plus_support_room_pullback",
            ("enhanced_support_resistance", "enhanced_market_structure_void", "enhanced_pullback_geometry"),
            "Domain block: support proximity, upside room, and pullback quality.",
        ),
        (
            "base_plus_structure_core",
            (
                "enhanced_support_resistance",
                "enhanced_market_structure_void",
                "enhanced_pullback_geometry",
                "compressed_price_geometry",
            ),
            "Candidate long_H120 structure filter block: support, room, pullback, and compressed geometry.",
        ),
        (
            "base_plus_all_v3b",
            V3B_STRUCTURE_FAMILIES,
            "All v3b structure families; included to reproduce/check the earlier all-v3b candidate effect.",
        ),
    ]
    out: list[V3BAdditionConfig] = []
    for name, added, description in variants:
        out.append(
            V3BAdditionConfig(
                name=name,
                include_families=_append_unique(LONG_H120_BASE_V0_FAMILIES, added),
                added_families=added,
                description=description,
            )
        )
    return out


def v3b_addition_to_feature_config(config: V3BAdditionConfig) -> dict[str, object]:
    """Convert one addition definition into a build_features YAML dict."""
    return {
        "feature_set_name": f"entry_feature_set_long_H120_{config.name}",
        "purpose": "long_H120 v3b family addition: " + config.description,
        "base_feature_set": "entry_feature_set_long_H120_v3a_minus_compressed_activity",
        "side": "long",
        "horizon_minutes": 120,
        "baseline_family_policy": "v3a_minus_compressed_activity",
        "added_families": list(config.added_families),
        "include_families": list(config.include_families),
        "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
        "notes": [
            "Valid-fold addition experiment only; not a production feature set by itself.",
            "Base is long_H120 v3a minus compressed_activity, the current strongest candidate.",
            "Purpose: isolate which v3b structure families help or hurt before building exit datasets.",
            "No test data should be read while evaluating this config.",
            "Every included family must remain deterministic and past-only; audit_features.py --strict should pass.",
        ],
    }


def write_v3b_addition_configs(
    output_dir: Path,
    *,
    configs: Sequence[V3BAdditionConfig] | None = None,
    force: bool = False,
) -> list[Path]:
    """Write reviewed long_H120 v3b addition configs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(configs or default_long_h120_v3b_addition_configs())
    paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for config in configs:
        path = output_dir / f"entry_feature_set_long_H120_{config.name}.yaml"
        if not path.exists() or force:
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(v3b_addition_to_feature_config(config), f, sort_keys=False, allow_unicode=True)
        paths.append(path)
        manifest_rows.append(
            {
                "name": config.name,
                "config": path.name,
                "include_families": list(config.include_families),
                "added_families": list(config.added_families),
                "description": config.description,
                "feature_count_expected": None,
            }
        )

    manifest = {
        "experiment_name": "long_H120_v3b_family_addition",
        "baseline_variant": "base_v0",
        "side": "long",
        "horizon_minutes": 120,
        "base_families": list(LONG_H120_BASE_V0_FAMILIES),
        "v3b_structure_families": list(V3B_STRUCTURE_FAMILIES),
        "configs": manifest_rows,
        "notes": [
            "Addition configs only change feature families; model config and split should stay fixed.",
            "Run on valid folds only; test remains locked-audit only.",
            "Primary question: which v3b structure families improve long_H120 over v3a_minus_compressed_activity?",
        ],
    }
    manifest_path = output_dir / "long_H120_v3b_addition_manifest.yaml"
    if not manifest_path.exists() or force:
        with manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return paths

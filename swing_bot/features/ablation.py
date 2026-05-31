"""Reviewed feature-set ablation definitions.

The helpers here only describe feature-set configs.  They do not build
features, train models, tune thresholds, or read test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

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

V3A_FAMILIES: tuple[str, ...] = (
    "price",
    "return_path",
    "volatility",
    "trend_persistence",
    "volume",
    "support_resistance",
    "market_structure_void",
    "pullback_geometry",
    "breakout",
    "volume_price_pressure",
    "rejection_reversal",
    "acceleration",
    "trend_regime",
    "regime_switch",
    "micro_range",
    "volume_spike",
    "compressed_trend",
    "compressed_activity",
    "compressed_volatility",
)

# These are the families we specifically want to challenge for long_H120 after
# v3a improved the model.  The baseline/core families are intentionally not
# all ablated in this first pass; this experiment is for removing likely noise
# from the v3a expansion, not for rebuilding the whole universe from scratch.
LONG_H120_DEFAULT_DROP_FAMILIES: tuple[str, ...] = (
    "breakout",
    "volume_price_pressure",
    "rejection_reversal",
    "acceleration",
    "trend_regime",
    "regime_switch",
    "micro_range",
    "volume_spike",
    "compressed_trend",
    "compressed_activity",
    "compressed_volatility",
)

LONG_H120_HYPOTHESIS_CORE_FAMILIES: tuple[str, ...] = (
    "price",
    "return_path",
    "volatility",
    "trend_persistence",
    "volume",
    "support_resistance",
    "market_structure_void",
    "pullback_geometry",
    "trend_regime",
    "regime_switch",
    "micro_range",
    "compressed_trend",
    "compressed_volatility",
)


@dataclass(frozen=True)
class FeatureAblationConfig:
    """One deterministic family-ablation feature set."""

    name: str
    include_families: tuple[str, ...]
    description: str
    dropped_families: tuple[str, ...] = ()


def _without(families: Sequence[str], drop: Sequence[str]) -> tuple[str, ...]:
    drop_set = set(drop)
    out = tuple(f for f in families if f not in drop_set)
    missing = sorted(drop_set.difference(families))
    if missing:
        raise ValueError(f"cannot drop unknown families from base: {missing}")
    return out


def default_long_h120_v3a_ablation_configs() -> list[FeatureAblationConfig]:
    """Return reviewed long_H120 v3a ablation configs.

    The first pass asks a narrow question: which v3a/v2 expansion families are
    hurting or helping the already-promising long_H120 model?  It therefore
    includes the full v3a set, leave-one-family-out variants for suspicious or
    important expansion families, two grouped noise removals, and one compact
    domain-hypothesis set.
    """
    configs: list[FeatureAblationConfig] = [
        FeatureAblationConfig(
            name="v3a_full",
            include_families=V3A_FAMILIES,
            description="Full v3a feature factory baseline for long_H120 ablation.",
        )
    ]
    for family in LONG_H120_DEFAULT_DROP_FAMILIES:
        configs.append(
            FeatureAblationConfig(
                name=f"v3a_minus_{family}",
                include_families=_without(V3A_FAMILIES, (family,)),
                dropped_families=(family,),
                description=f"Full v3a minus {family}; tests whether this family hurts long_H120.",
            )
        )

    grouped: list[tuple[str, tuple[str, ...], str]] = [
        (
            "v3a_minus_v2_noise",
            ("breakout", "volume_price_pressure", "rejection_reversal"),
            "Drops all v2 expansion families that may be short/noise oriented for long_H120.",
        ),
        (
            "v3a_minus_short_noise",
            ("volume_price_pressure", "rejection_reversal", "volume_spike"),
            "Drops pressure/rejection/spike families suspected of pulling long_H120 into exhaustion or short-like states.",
        ),
    ]
    for name, drop, description in grouped:
        configs.append(
            FeatureAblationConfig(
                name=name,
                include_families=_without(V3A_FAMILIES, drop),
                dropped_families=drop,
                description=description,
            )
        )

    configs.append(
        FeatureAblationConfig(
            name="hypothesis_core",
            include_families=LONG_H120_HYPOTHESIS_CORE_FAMILIES,
            dropped_families=tuple(f for f in V3A_FAMILIES if f not in LONG_H120_HYPOTHESIS_CORE_FAMILIES),
            description=(
                "Domain-hypothesis long_H120 core: trend/regime, pullback, room, volatility state; "
                "excludes full pressure/rejection/breakout/spike/acceleration noise candidates."
            ),
        )
    )
    return configs


def ablation_to_feature_config(config: FeatureAblationConfig) -> dict[str, object]:
    """Convert one ablation config into a build_features YAML dict."""
    return {
        "feature_set_name": f"entry_feature_set_long_H120_{config.name}",
        "purpose": "long_H120 v3a family ablation: " + config.description,
        "base_feature_set": "entry_feature_set_v3a",
        "side": "long",
        "horizon_minutes": 120,
        "dropped_families": list(config.dropped_families),
        "include_families": list(config.include_families),
        "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
        "notes": [
            "This config is for valid-fold ablation only, not production adoption.",
            "It is designed for long_H120 after v3a improved q95/q97/q99 metrics.",
            "Compare against v3a_full with worst-fold metrics emphasized over mean-only gains.",
            "No test data should be read while evaluating this config.",
            "Every included family must remain deterministic and past-only; audit_features.py --strict should pass.",
        ],
    }


def write_ablation_configs(
    output_dir: Path,
    *,
    configs: Sequence[FeatureAblationConfig] | None = None,
    force: bool = False,
) -> list[Path]:
    """Write reviewed long_H120 v3a ablation configs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(configs or default_long_h120_v3a_ablation_configs())
    paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for config in configs:
        path = output_dir / f"entry_feature_set_long_H120_{config.name}.yaml"
        if path.exists() and not force:
            paths.append(path)
        else:
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(ablation_to_feature_config(config), f, sort_keys=False, allow_unicode=True)
            paths.append(path)
        manifest_rows.append(
            {
                "name": config.name,
                "config": path.name,
                "include_families": list(config.include_families),
                "dropped_families": list(config.dropped_families),
                "description": config.description,
                "feature_count_expected": None,
            }
        )

    manifest = {
        "experiment_name": "long_H120_v3a_family_ablation",
        "baseline_variant": "v3a_full",
        "side": "long",
        "horizon_minutes": 120,
        "configs": manifest_rows,
        "notes": [
            "Ablation configs only change feature families; model config and split should stay fixed.",
            "Primary use: find v3a families that are unnecessary or harmful for long_H120.",
            "Run on valid folds only; test remains locked-audit only.",
        ],
    }
    manifest_path = output_dir / "long_H120_v3a_ablation_manifest.yaml"
    if not manifest_path.exists() or force:
        with manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return paths

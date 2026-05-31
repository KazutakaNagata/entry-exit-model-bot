"""Reviewed combined-ablation definitions for long_H120 v3a.

These helpers only write feature-set configs.  They do not build features,
train models, tune thresholds, read test data, or select a production config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from swing_bot.features.ablation import DEFAULT_EXCLUDE_PATTERNS, V3A_FAMILIES


@dataclass(frozen=True)
class CombinedAblationConfig:
    """One deterministic long_H120 combined family-ablation config."""

    name: str
    include_families: tuple[str, ...]
    description: str
    dropped_families: tuple[str, ...] = ()


def _without(families: Sequence[str], drop: Sequence[str]) -> tuple[str, ...]:
    drop_set = set(drop)
    missing = sorted(drop_set.difference(families))
    if missing:
        raise ValueError(f"cannot drop unknown families from base: {missing}")
    return tuple(f for f in families if f not in drop_set)


def default_long_h120_v3a_combined_ablation_configs() -> list[CombinedAblationConfig]:
    """Return reviewed combined ablations for the promising long_H120 v3a run.

    The previous leave-one-family-out experiment indicated that
    ``compressed_activity``, ``rejection_reversal``, and ``breakout`` were the
    most suspicious families for long_H120.  This second pass tests their
    combinations while keeping ``volume_price_pressure`` mostly intact because
    the first pass suggested it may protect q99 worst-fold behavior even when it
    hurts wider q95/q97 averages.
    """
    variants: list[tuple[str, tuple[str, ...], str]] = [
        (
            "v3a_full",
            (),
            "Full v3a baseline, included for direct comparison with combined removals.",
        ),
        (
            "v3a_minus_compressed_activity",
            ("compressed_activity",),
            "Drops the strongest leave-one-out noise candidate from the first long_H120 ablation.",
        ),
        (
            "v3a_minus_compressed_activity_rejection_reversal",
            ("compressed_activity", "rejection_reversal"),
            "Tests whether removing the activity-compression noise and short-oriented rejection family is additive.",
        ),
        (
            "v3a_minus_compressed_activity_breakout",
            ("compressed_activity", "breakout"),
            "Tests whether long_H120 prefers regime continuation without immediate breakout trigger features.",
        ),
        (
            "v3a_minus_rejection_reversal_breakout",
            ("rejection_reversal", "breakout"),
            "Drops the two non-activity families that improved in leave-one-out, while keeping compressed_activity for interaction check.",
        ),
        (
            "v3a_minus_compressed_activity_rejection_reversal_breakout",
            ("compressed_activity", "rejection_reversal", "breakout"),
            "Main combined-noise candidate: removes activity compression, rejection/failure, and breakout trigger families.",
        ),
        (
            "v3a_minus_compressed_activity_volume_price_pressure",
            ("compressed_activity", "volume_price_pressure"),
            "Cautious test of whether pressure features are still needed after removing compressed_activity.",
        ),
        (
            "v3a_minus_compressed_activity_rejection_reversal_breakout_volume_price_pressure",
            ("compressed_activity", "rejection_reversal", "breakout", "volume_price_pressure"),
            "Aggressive noise-removal candidate; included as a stress test because volume_price_pressure had mixed q95/q99 effects.",
        ),
        (
            "v3a_minus_compressed_activity_rejection_reversal_breakout_volume_spike",
            ("compressed_activity", "rejection_reversal", "breakout", "volume_spike"),
            "Stress test that also removes volume_spike; expected to be risky because leave-one-out suggested volume_spike is important.",
        ),
    ]
    out: list[CombinedAblationConfig] = []
    for name, dropped, description in variants:
        out.append(
            CombinedAblationConfig(
                name=name,
                include_families=_without(V3A_FAMILIES, dropped),
                dropped_families=dropped,
                description=description,
            )
        )
    return out


def combined_ablation_to_feature_config(config: CombinedAblationConfig) -> dict[str, object]:
    """Convert one combined ablation definition into a build_features YAML dict."""
    return {
        "feature_set_name": f"entry_feature_set_long_H120_{config.name}",
        "purpose": "long_H120 v3a combined family ablation: " + config.description,
        "base_feature_set": "entry_feature_set_v3a",
        "side": "long",
        "horizon_minutes": 120,
        "dropped_families": list(config.dropped_families),
        "include_families": list(config.include_families),
        "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
        "notes": [
            "This config is for valid-fold combined ablation only, not production adoption.",
            "It follows the first long_H120 v3a leave-one-family-out ablation.",
            "Primary question: do compressed_activity, rejection_reversal, and breakout hurt additively?",
            "No test data should be read while evaluating this config.",
            "Every included family must remain deterministic and past-only; audit_features.py --strict should pass.",
        ],
    }


def write_combined_ablation_configs(
    output_dir: Path,
    *,
    configs: Sequence[CombinedAblationConfig] | None = None,
    force: bool = False,
) -> list[Path]:
    """Write reviewed long_H120 v3a combined-ablation configs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(configs or default_long_h120_v3a_combined_ablation_configs())
    paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for config in configs:
        path = output_dir / f"entry_feature_set_long_H120_{config.name}.yaml"
        if not path.exists() or force:
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(combined_ablation_to_feature_config(config), f, sort_keys=False, allow_unicode=True)
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
        "experiment_name": "long_H120_v3a_combined_ablation",
        "baseline_variant": "v3a_full",
        "side": "long",
        "horizon_minutes": 120,
        "configs": manifest_rows,
        "notes": [
            "Second-pass combined ablation after the leave-one-family-out long_H120 v3a experiment.",
            "Ablation configs only change feature families; model config and split should stay fixed.",
            "Run on valid folds only; test remains locked-audit only.",
        ],
    }
    manifest_path = output_dir / "long_H120_v3a_combined_ablation_manifest.yaml"
    if not manifest_path.exists() or force:
        with manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return paths

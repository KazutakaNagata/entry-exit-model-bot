"""Reviewed selected feature-set definitions.

These helpers write small YAML configs for feature sets that have passed the
valid-fold entry-selection experiments.  They do not build features, train
models, tune thresholds, read test data, or select a production policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from swing_bot.features.ablation import DEFAULT_EXCLUDE_PATTERNS
from swing_bot.features.v3b_addition import LONG_H120_BASE_V0_FAMILIES


@dataclass(frozen=True)
class SelectedEntryFeatureSet:
    """One reviewed entry feature-set candidate."""

    name: str
    side: str
    horizon_minutes: int
    include_families: tuple[str, ...]
    description: str
    policy_status: str
    added_families: tuple[str, ...] = ()
    dropped_families: tuple[str, ...] = ()


def _append_unique(base: Sequence[str], add: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = list(base)
    seen = set(out)
    for family in add:
        if family not in seen:
            out.append(family)
            seen.add(family)
    return tuple(out)


def default_selected_long_h120_feature_sets() -> list[SelectedEntryFeatureSet]:
    """Return the reviewed long_H120 candidates after v3a/v3b experiments.

    The main candidate is ``v3a - compressed_activity``.  The tail candidate adds
    ``compressed_price_geometry`` because the v3b addition pass improved q99 but
    weakened q97; it should be treated as a high-threshold comparison candidate,
    not as a default replacement.
    """
    base = tuple(LONG_H120_BASE_V0_FAMILIES)
    return [
        SelectedEntryFeatureSet(
            name="long_H120_v0",
            side="long",
            horizon_minutes=120,
            include_families=base,
            dropped_families=("compressed_activity",),
            added_families=(),
            policy_status="primary_candidate",
            description=(
                "Primary long_H120 candidate from valid-fold selection: "
                "feature factory v3a minus compressed_activity."
            ),
        ),
        SelectedEntryFeatureSet(
            name="long_H120_tail_v0",
            side="long",
            horizon_minutes=120,
            include_families=_append_unique(base, ("compressed_price_geometry",)),
            dropped_families=("compressed_activity",),
            added_families=("compressed_price_geometry",),
            policy_status="tail_candidate",
            description=(
                "Tail-oriented long_H120 comparison candidate: primary candidate "
                "plus compressed_price_geometry.  It previously improved q99 but "
                "weakened q97, so evaluate with strict thresholds and episode P/L."
            ),
        ),
    ]


def selected_entry_feature_set_to_config(config: SelectedEntryFeatureSet) -> dict[str, object]:
    """Convert a selected feature set into build_features YAML content."""
    return {
        "feature_set_name": f"entry_feature_set_{config.name}",
        "purpose": config.description,
        "side": config.side,
        "horizon_minutes": config.horizon_minutes,
        "selection_status": config.policy_status,
        "selection_source": "valid_only_long_H120_v3a_combined_and_v3b_addition_experiments",
        "base_feature_set": "entry_feature_set_v3a",
        "dropped_families": list(config.dropped_families),
        "added_families": list(config.added_families),
        "include_families": list(config.include_families),
        "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
        "notes": [
            "Selected-feature config for valid-fold downstream evaluation only.",
            "No test data should be read while building, training, or comparing this feature set.",
            "Entry target remains fixed-hold net_return_bps regression for long_H120.",
            "Use OOF entry predictions from this feature set when building exit datasets.",
            "Every included family must remain deterministic and past-only; audit_features.py --strict should pass.",
        ],
    }


def write_selected_long_h120_feature_sets(output_dir: Path, *, force: bool = False) -> list[Path]:
    """Write selected long_H120 feature-set configs and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = default_selected_long_h120_feature_sets()
    paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for config in configs:
        path = output_dir / f"entry_feature_set_{config.name}.yaml"
        if not path.exists() or force:
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(selected_entry_feature_set_to_config(config), f, sort_keys=False, allow_unicode=True)
        paths.append(path)
        manifest_rows.append(
            {
                "name": config.name,
                "config": path.name,
                "side": config.side,
                "horizon_minutes": config.horizon_minutes,
                "selection_status": config.policy_status,
                "include_families": list(config.include_families),
                "dropped_families": list(config.dropped_families),
                "added_families": list(config.added_families),
                "description": config.description,
            }
        )

    manifest = {
        "experiment_name": "selected_long_H120_entry_feature_sets",
        "side": "long",
        "horizon_minutes": 120,
        "baseline_selected_set": "long_H120_v0",
        "candidates": manifest_rows,
        "notes": [
            "These configs are selected candidates from valid-fold experiments; they are not test-locked policies.",
            "long_H120_v0 is the primary candidate: v3a minus compressed_activity.",
            "long_H120_tail_v0 adds compressed_price_geometry for strict-threshold q99 comparison.",
            "Downstream exit datasets must be regenerated from OOF entry predictions for each candidate.",
        ],
    }
    manifest_path = output_dir / "selected_long_H120_feature_sets.yaml"
    if not manifest_path.exists() or force:
        with manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return paths

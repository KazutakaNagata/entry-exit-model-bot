"""Planning helpers for selected long_H120 entry-to-exit validation.

The selected long_H120 entry feature sets are compared downstream by rebuilding
exit datasets, training exit models, and running valid-fold episode backtests.
This module only constructs deterministic paths and command plans; scripts still
execute each reviewed command explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


DEFAULT_SIDE = "long"
DEFAULT_HORIZON = 120
DEFAULT_ENTRY_OOF_ROOT = Path("outputs/valid/entry_oof")
DEFAULT_EXIT_DATASET_ROOT = Path("outputs/valid/exit_dataset/selected_long_H120")
DEFAULT_EXIT_LGBM_ROOT = Path("outputs/valid/exit_lgbm")
DEFAULT_EPISODE_ROOT = Path("outputs/valid/episode_backtest")


@dataclass(frozen=True)
class SelectedEntrySet:
    """One selected long_H120 entry candidate from the selected manifest."""

    key: str
    config: Path
    feature_output: Path
    grid_run_id: str
    status: str = ""
    description: str = ""

    @property
    def entry_oof_path(self) -> Path:
        return DEFAULT_ENTRY_OOF_ROOT / self.grid_run_id / "entry_oof_predictions.parquet"


@dataclass(frozen=True)
class DownstreamRunSpec:
    """One selected-entry downstream experiment combination."""

    selected_key: str
    entry_oof_path: Path
    features_path: Path
    exit_lookahead_minutes: int
    exit_feature_set: str
    entry_threshold_bps: float
    hold_threshold_bps: float
    exit_dataset_dir: Path
    exit_run_id: str
    exit_predictions_path: Path
    episode_run_id: str
    episode_run_dir: Path


def load_selected_manifest(path: Path) -> list[SelectedEntrySet]:
    """Load selected long_H120 entry candidates from YAML."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rows = data.get("selected_sets") or []
    out: list[SelectedEntrySet] = []
    for row in rows:
        out.append(
            SelectedEntrySet(
                key=str(row["key"]),
                config=Path(row["config"]),
                feature_output=Path(row.get("feature_output") or f"data/processed/binance_japan/BTCJPY/1m/selected_features/features_entry_{row['key']}.parquet"),
                grid_run_id=str(row.get("grid_run_id") or f"entry_grid_{row['key']}"),
                status=str(row.get("status", "")),
                description=str(row.get("description", "")),
            )
        )
    if not out:
        raise ValueError(f"manifest has no selected_sets: {path}")
    return out


def select_entries(entries: Iterable[SelectedEntrySet], keys: list[str] | None) -> list[SelectedEntrySet]:
    """Filter selected entries by key, preserving manifest order."""
    rows = list(entries)
    if not keys:
        return rows
    wanted = set(keys)
    selected = [row for row in rows if row.key in wanted]
    missing = sorted(wanted - {row.key for row in selected})
    if missing:
        raise ValueError("unknown selected set key(s): " + ", ".join(missing))
    return selected


def parse_csv_ints(text: str) -> list[int]:
    """Parse comma-separated positive integers."""
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    if any(x <= 0 for x in vals):
        raise ValueError("integers must be positive")
    return vals


def parse_csv_floats(text: str) -> list[float]:
    """Parse comma-separated floats."""
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one float")
    return vals


def parse_csv_strings(text: str) -> list[str]:
    """Parse comma-separated non-empty strings."""
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one value")
    return vals


def thresholds_for_selected_key(
    key: str,
    *,
    default_thresholds: list[float],
    tail_thresholds: list[float],
) -> list[float]:
    """Return default entry thresholds for a selected candidate key."""
    return tail_thresholds if "tail" in key else default_thresholds


def safe_float_slug(value: float) -> str:
    """Return a stable path-friendly float token."""
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_downstream_specs(
    *,
    selected_entries: list[SelectedEntrySet],
    exit_lookaheads: list[int],
    exit_feature_sets: list[str],
    default_entry_thresholds: list[float],
    tail_entry_thresholds: list[float],
    hold_thresholds: list[float],
    exit_dataset_root: Path = DEFAULT_EXIT_DATASET_ROOT,
    exit_lgbm_root: Path = DEFAULT_EXIT_LGBM_ROOT,
    episode_root: Path = DEFAULT_EPISODE_ROOT,
) -> list[DownstreamRunSpec]:
    """Build deterministic downstream run specs for selected long_H120 candidates."""
    specs: list[DownstreamRunSpec] = []
    for entry in selected_entries:
        entry_thresholds = thresholds_for_selected_key(
            entry.key,
            default_thresholds=default_entry_thresholds,
            tail_thresholds=tail_entry_thresholds,
        )
        for k in exit_lookaheads:
            exit_dataset_dir = exit_dataset_root / f"exit_long_H120_{entry.key}_K{k}"
            for feature_set in exit_feature_sets:
                exit_run_id = f"exit_selected_long_H120_{entry.key}_K{k}_{feature_set}"
                exit_predictions_path = exit_lgbm_root / exit_run_id / "predictions_valid.parquet"
                for entry_thr in entry_thresholds:
                    for hold_thr in hold_thresholds:
                        episode_run_id = (
                            f"episode_selected_long_H120_{entry.key}_K{k}_{feature_set}"
                            f"_entryThr{safe_float_slug(entry_thr)}_holdThr{safe_float_slug(hold_thr)}"
                        )
                        specs.append(
                            DownstreamRunSpec(
                                selected_key=entry.key,
                                entry_oof_path=entry.entry_oof_path,
                                features_path=entry.feature_output,
                                exit_lookahead_minutes=int(k),
                                exit_feature_set=feature_set,
                                entry_threshold_bps=float(entry_thr),
                                hold_threshold_bps=float(hold_thr),
                                exit_dataset_dir=exit_dataset_dir,
                                exit_run_id=exit_run_id,
                                exit_predictions_path=exit_predictions_path,
                                episode_run_id=episode_run_id,
                                episode_run_dir=episode_root / episode_run_id,
                            )
                        )
    return specs


_EPISODE_RE = re.compile(
    r"^episode_selected_long_H120_(?P<key>.+?)_K(?P<k>\d+)_(?P<exit_feature_set>v\d+_[A-Za-z0-9_]+)"
    r"_entryThr(?P<entry_thr>[^_]+)_holdThr(?P<hold_thr>[^_]+)$"
)


def _unslug_float(text: str) -> float:
    return float(text.replace("m", "-").replace("p", "."))


def parse_episode_run_id(run_id: str) -> dict[str, Any]:
    """Parse run metadata from a selected long_H120 episode run id."""
    m = _EPISODE_RE.match(run_id)
    if not m:
        return {"run_id": run_id}
    groups = m.groupdict()
    return {
        "run_id": run_id,
        "selected_key": groups["key"],
        "exit_lookahead_minutes": int(groups["k"]),
        "exit_feature_set": groups["exit_feature_set"],
        "entry_threshold_bps": _unslug_float(groups["entry_thr"]),
        "hold_threshold_bps": _unslug_float(groups["hold_thr"]),
    }

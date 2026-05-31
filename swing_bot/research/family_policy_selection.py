"""Family-level feature policy generation and selector scoring.

This module supports the monthly rolling research workflow where humans define
only the candidate feature universe, then the code mechanically evaluates full
and leave-one-family-out policies inside each rolling cycle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import math
import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json
from swing_bot.features.manifest import read_manifest


@dataclass(frozen=True)
class GeneratedFamilyPolicy:
    """One generated feature policy file."""

    name: str
    features_path: Path
    drop_families: tuple[str, ...]
    feature_count: int
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["features_path"] = str(self.features_path)
        out["drop_families"] = list(self.drop_families)
        return out


def _safe_slug(value: str) -> str:
    allowed = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "policy"


def load_feature_families_from_manifest(manifest_path: Path | str) -> dict[str, list[str]]:
    """Return mapping family -> feature columns from a manifest JSON/CSV."""
    specs = read_manifest(Path(manifest_path))
    mapping: dict[str, list[str]] = {}
    for spec in specs:
        mapping.setdefault(spec.family, []).append(spec.name)
    return {family: sorted(cols) for family, cols in sorted(mapping.items())}


def read_feature_frame(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.name.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_policy_features(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_frame(frame, path)


def build_leave_one_family_out_policies(
    *,
    features_path: Path | str,
    manifest_path: Path | str,
    output_dir: Path | str,
    policy_prefix: str = "family_ablation",
    families: Sequence[str] | None = None,
    exclude_families_from_universe: Sequence[str] | None = None,
    timestamp_col: str = "timestamp",
    output_format: str = "parquet",
    overwrite: bool = False,
) -> tuple[list[GeneratedFamilyPolicy], pd.DataFrame]:
    """Build full and full-minus-one-family feature policy files.

    The function does not use domain knowledge to pre-filter families by default.
    It takes all families present in the manifest unless ``families`` or
    ``exclude_families_from_universe`` is explicitly supplied.
    """
    features_path = Path(features_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    if output_format not in {"parquet", "csv"}:
        raise ValueError("output_format must be 'parquet' or 'csv'")
    suffix = ".parquet" if output_format == "parquet" else ".csv"
    features = read_feature_frame(features_path)
    if timestamp_col not in features.columns:
        raise ValueError(f"features must include timestamp column: {timestamp_col}")
    family_to_cols = load_feature_families_from_manifest(manifest_path)
    all_families = list(family_to_cols)
    if families:
        missing = sorted(set(families) - set(all_families))
        if missing:
            raise ValueError(f"families not present in manifest: {missing}")
        universe = list(families)
    else:
        universe = all_families
    excluded = set(exclude_families_from_universe or [])
    universe = [f for f in universe if f not in excluded]
    if not universe:
        raise ValueError("family universe is empty")

    # Only manifest-backed feature columns are included. This avoids accidental
    # target/future/diagnostic columns even if they exist in the input frame.
    all_feature_cols: list[str] = []
    for family in universe:
        all_feature_cols.extend([c for c in family_to_cols[family] if c in features.columns])
    all_feature_cols = sorted(set(all_feature_cols))
    missing_cols = sorted(set(c for f in universe for c in family_to_cols[f]) - set(features.columns))
    if missing_cols:
        raise ValueError(f"manifest columns missing from features frame, first 20: {missing_cols[:20]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    policies: list[GeneratedFamilyPolicy] = []

    def write_policy(name: str, drop_families: Sequence[str]) -> GeneratedFamilyPolicy:
        drop_set = set(drop_families)
        keep_cols: list[str] = []
        for family in universe:
            if family in drop_set:
                continue
            keep_cols.extend(family_to_cols[family])
        keep_cols = [c for c in sorted(set(keep_cols)) if c in features.columns]
        policy_path = output_dir / f"{_safe_slug(name)}{suffix}"
        if policy_path.exists() and not overwrite:
            # Reuse existing files to avoid rewriting large parquet files when
            # jobs are restarted.
            pass
        else:
            _write_policy_features(features[[timestamp_col] + keep_cols].copy(), policy_path)
        return GeneratedFamilyPolicy(
            name=name,
            features_path=policy_path,
            drop_families=tuple(sorted(drop_set)),
            feature_count=len(keep_cols),
            description=("full universe" if not drop_set else "minus " + ",".join(sorted(drop_set))),
        )

    full_name = f"{policy_prefix}_full"
    policies.append(write_policy(full_name, []))
    for family in universe:
        policies.append(write_policy(f"{policy_prefix}_minus_{_safe_slug(family)}", [family]))

    manifest_rows = []
    for p in policies:
        row = p.to_dict()
        row["family_universe_count"] = len(universe)
        row["family_universe"] = list(universe)
        manifest_rows.append(row)
    manifest_df = pd.DataFrame(manifest_rows)
    write_frame(manifest_df, output_dir / "family_policy_manifest.csv")
    write_json({
        "policy_prefix": policy_prefix,
        "features_path": str(features_path),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "family_universe": universe,
        "policy_count": len(policies),
        "policies": manifest_rows,
        "notes": [
            "Policies are generated from the full family universe plus one leave-one-family-out policy per family.",
            "No family is globally deleted; each rolling monthly cycle selects mechanically from this candidate set.",
        ],
    }, output_dir / "family_policy_manifest.json")
    return policies, manifest_df


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def selector_metrics_from_fold_metrics(
    fold_metrics: pd.DataFrame,
    *,
    min_total_episode_count: int = 10,
    min_active_fold_count: int = 2,
    max_worst_fold_loss_bps: float | None = None,
    worst_weight: float = 0.5,
    median_weight: float = 0.25,
    fail_penalty: float = 1_000_000.0,
) -> dict[str, Any]:
    """Compute the agreed three-fold robust selector score.

    Score:
        total_net + worst_weight * worst_fold_net + median_weight * median_fold_net

    Constraints are applied as large penalties, not silent filtering, so result
    tables still show why a candidate was rejected.
    """
    if fold_metrics.empty:
        return {
            "selector_total_net_pl_bps": 0.0,
            "selector_worst_fold_net_pl_bps": 0.0,
            "selector_median_fold_net_pl_bps": 0.0,
            "selector_total_episode_count": 0,
            "selector_active_fold_count": 0,
            "selector_failed_constraints": "empty_fold_metrics",
            "robust_score": -fail_penalty,
        }
    if "net_pl_bps_sum" not in fold_metrics.columns:
        raise ValueError("fold_metrics must include net_pl_bps_sum")
    net = pd.to_numeric(fold_metrics["net_pl_bps_sum"], errors="coerce").fillna(0.0)
    counts = pd.to_numeric(fold_metrics.get("episode_count", pd.Series([0] * len(fold_metrics))), errors="coerce").fillna(0).astype(int)
    total_net = float(net.sum())
    worst_net = float(net.min()) if len(net) else 0.0
    median_net = float(net.median()) if len(net) else 0.0
    total_count = int(counts.sum())
    active_count = int((counts > 0).sum())
    raw_score = total_net + float(worst_weight) * worst_net + float(median_weight) * median_net
    failures: list[str] = []
    if total_count < int(min_total_episode_count):
        failures.append("min_total_episode_count")
    if active_count < int(min_active_fold_count):
        failures.append("min_active_fold_count")
    if max_worst_fold_loss_bps is not None and worst_net < float(max_worst_fold_loss_bps):
        failures.append("max_worst_fold_loss_bps")
    penalty = fail_penalty * len(failures)
    return {
        "selector_total_net_pl_bps": total_net,
        "selector_worst_fold_net_pl_bps": worst_net,
        "selector_median_fold_net_pl_bps": median_net,
        "selector_total_episode_count": total_count,
        "selector_active_fold_count": active_count,
        "selector_failed_constraints": ",".join(failures),
        "selector_raw_score": raw_score,
        "robust_score": raw_score - penalty,
    }


def family_ablation_deltas(candidate_metrics: pd.DataFrame, *, full_policy_suffix: str = "_full") -> pd.DataFrame:
    """Return per-cycle deltas of minus-family policies against full policy."""
    if candidate_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cycle_id, group in candidate_metrics.groupby("cycle_id", dropna=False):
        full_rows = group[group["policy"].astype(str).str.endswith(full_policy_suffix)]
        if full_rows.empty:
            continue
        full = full_rows.iloc[0]
        full_score = _finite_float(full.get("robust_score"))
        full_net = _finite_float(full.get("selector_total_net_pl_bps", full.get("mean_net_pl_bps_sum")))
        full_worst = _finite_float(full.get("selector_worst_fold_net_pl_bps", full.get("worst_net_pl_bps_sum")))
        for _, row in group.iterrows():
            policy = str(row.get("policy"))
            if policy == str(full.get("policy")):
                continue
            rows.append({
                "cycle_id": cycle_id,
                "policy": policy,
                "delta_score_vs_full": _finite_float(row.get("robust_score")) - full_score,
                "delta_total_net_vs_full": _finite_float(row.get("selector_total_net_pl_bps", row.get("mean_net_pl_bps_sum"))) - full_net,
                "delta_worst_net_vs_full": _finite_float(row.get("selector_worst_fold_net_pl_bps", row.get("worst_net_pl_bps_sum"))) - full_worst,
                "selected": bool(row.get("selected", False)),
            })
    return pd.DataFrame(rows)

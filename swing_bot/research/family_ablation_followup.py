"""Second-stage family-ablation follow-up utilities.

This module salvages an already completed monthly full+LOFO run.  It reads the
existing candidate_valid_metrics.csv, identifies families whose one-family drop
improved the validation score versus full, builds a combined-drop feature policy,
and optionally evaluates that follow-up policy for the same rolling monthly cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import math
import re

import pandas as pd
import yaml



@dataclass(frozen=True)
class FollowupPolicy:
    name: str
    features_path: Path
    drop_families: tuple[str, ...]
    source: str

    def as_config_row(self) -> dict[str, str]:
        return {
            "name": self.name,
            "features": str(self.features_path),
            "description": f"{self.source}; drop={','.join(self.drop_families) if self.drop_families else 'none'}",
        }


def _safe_slug(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if (ch.isalnum() or ch in {"_", "-"}) else "_")
    return "".join(out).strip("_") or "policy"


def read_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"yaml must be a mapping: {path}")
    return data


def write_yaml(data: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_candidate_metrics(source_run_dir: Path | str) -> pd.DataFrame:
    path = Path(source_run_dir) / "candidate_valid_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"candidate_valid_metrics.csv not found: {path}")
    df = pd.read_csv(path)
    required = {"cycle_id", "policy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate_valid_metrics.csv missing columns {sorted(missing)}")
    if df["cycle_id"].nunique() != 1:
        raise ValueError(
            "follow-up expects a single monthly cycle run. "
            "Use per-month source run dirs, not a multi-cycle aggregate."
        )
    return df


def load_source_config(source_run_dir: Path | str) -> dict[str, Any]:
    source_run_dir = Path(source_run_dir)
    for name in ("generated_config.yaml", "run_config.json"):
        p = source_run_dir / name
        if p.exists():
            if p.suffix == ".json":
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"json must be a mapping: {p}")
                return data
            return read_yaml(p)
    raise FileNotFoundError(f"neither generated_config.yaml nor run_config.json found under {source_run_dir}")


def infer_test_months_from_cycles(source_run_dir: Path | str) -> tuple[str, str]:
    path = Path(source_run_dir) / "rolling_cycles.csv"
    if not path.exists():
        raise FileNotFoundError(f"rolling_cycles.csv not found: {path}")
    cycles = pd.read_csv(path)
    if len(cycles) != 1:
        raise ValueError("follow-up expects exactly one row in rolling_cycles.csv")
    start = pd.Timestamp(pd.to_datetime(cycles.iloc[0]["test_start"], utc=True))
    month = f"{start.year:04d}-{start.month:02d}-01"
    return month, month


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _parse_minus_family(policy_name: str) -> str | None:
    marker = "_minus_"
    if marker not in str(policy_name):
        return None
    return str(policy_name).split(marker, 1)[1]


def find_full_policy_row(metrics: pd.DataFrame) -> pd.Series:
    full = metrics[metrics["policy"].astype(str).str.endswith("_full")]
    if full.empty:
        # Fallback: policy exactly full, for custom prefixes.
        full = metrics[metrics["policy"].astype(str).str.lower().eq("full")]
    if full.empty:
        raise ValueError("could not find full policy row; expected policy ending with '_full'")
    return full.iloc[0]


def infer_improving_drop_families(
    metrics: pd.DataFrame,
    *,
    metric_col: str = "selector_raw_score",
    min_delta_bps: float = 0.0,
    ignore_constraint_penalties: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Return families whose single drop improved versus full.

    By default the comparison uses selector_raw_score so hard/soft penalties do
    not hide useful drop candidates.  Use metric_col='robust_score' if you want
    the penalty-adjusted selector score instead.
    """
    if metric_col not in metrics.columns:
        raise ValueError(f"metric_col not found in candidate metrics: {metric_col}")
    full = find_full_policy_row(metrics)
    full_score = _finite_float(full.get(metric_col))
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        policy = str(row.get("policy"))
        family = _parse_minus_family(policy)
        if family is None:
            continue
        score = _finite_float(row.get(metric_col))
        raw_score = _finite_float(row.get("selector_raw_score", score))
        robust_score = _finite_float(row.get("robust_score", score))
        delta = score - full_score
        rows.append({
            "policy": policy,
            "family": family,
            "metric_col": metric_col,
            "full_metric": full_score,
            "policy_metric": score,
            "delta_metric_vs_full": delta,
            "selector_raw_score": raw_score,
            "robust_score": robust_score,
            "selector_failed_constraints": str(row.get("selector_failed_constraints", "") or ""),
            "selector_total_episode_count": row.get("selector_total_episode_count"),
            "selector_active_fold_count": row.get("selector_active_fold_count"),
        })
    df = pd.DataFrame(rows).sort_values("delta_metric_vs_full", ascending=False) if rows else pd.DataFrame()
    families = df.loc[df["delta_metric_vs_full"] > float(min_delta_bps), "family"].tolist() if not df.empty else []
    return families, df


def manifest_family_mapping(manifest_path: Path | str) -> dict[str, list[str]]:
    from swing_bot.features.manifest import read_manifest
    specs = read_manifest(Path(manifest_path))
    mapping: dict[str, list[str]] = {}
    for spec in specs:
        mapping.setdefault(spec.family, []).append(spec.name)
    return {fam: sorted(cols) for fam, cols in sorted(mapping.items())}


def read_features(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_policy_from_drop_families(
    *,
    features: pd.DataFrame,
    family_to_cols: dict[str, list[str]],
    output_path: Path,
    drop_families: Sequence[str],
    timestamp_col: str = "timestamp",
    overwrite: bool = False,
) -> FollowupPolicy:
    drop_set = set(drop_families)
    unknown = sorted(drop_set - set(family_to_cols))
    if unknown:
        raise ValueError(f"drop families not present in manifest: {unknown}")
    keep_cols: list[str] = []
    for family, cols in family_to_cols.items():
        if family in drop_set:
            continue
        keep_cols.extend([c for c in cols if c in features.columns])
    keep_cols = sorted(set(keep_cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        pass
    else:
        frame = features[[timestamp_col] + keep_cols].copy()
        from swing_bot.artifacts.io import write_frame
        write_frame(frame, output_path)
    return FollowupPolicy(
        name=output_path.stem,
        features_path=output_path,
        drop_families=tuple(sorted(drop_set)),
        source="combined_drop_followup",
    )


def _lookup_policy_path_from_generated_config(cfg: dict[str, Any], policy_name: str) -> Path | None:
    for item in cfg.get("feature_policies") or []:
        if str(item.get("name")) == str(policy_name):
            return Path(item.get("features"))
    return None


def prepare_combined_drop_followup(
    *,
    source_run_dir: Path | str,
    output_root: Path | str,
    followup_run_id: str,
    metric_col: str = "selector_raw_score",
    min_delta_bps: float = 0.0,
    include_addback: bool = False,
    include_full: bool = False,
    include_source_selected: bool = True,
    overwrite_features: bool = False,
) -> dict[str, Any]:
    source_run_dir = Path(source_run_dir)
    output_root = Path(output_root)
    metrics = load_candidate_metrics(source_run_dir)
    cfg = load_source_config(source_run_dir)
    cycle_id = str(metrics["cycle_id"].iloc[0])

    fam_cfg = cfg.get("family_ablation") or {}
    base_features = Path(fam_cfg.get("base_features"))
    manifest_path = Path(fam_cfg.get("manifest"))
    policy_prefix = str(fam_cfg.get("policy_prefix") or "lofo")
    output_format = str(fam_cfg.get("output_format") or "parquet")
    suffix = ".parquet" if output_format == "parquet" else ".csv"
    if suffix != ".parquet":
        # The rest of the project can read CSV, but parquet is the practical
        # default for these large feature matrices.
        pass

    drop_families, delta_df = infer_improving_drop_families(metrics, metric_col=metric_col, min_delta_bps=min_delta_bps)
    if not drop_families:
        raise ValueError(
            f"no improving drop families found using {metric_col} with min_delta_bps={min_delta_bps}. "
            "Lower the threshold or inspect candidate_valid_metrics.csv."
        )

    features = read_features(base_features)
    family_to_cols = manifest_family_mapping(manifest_path)
    policy_dir = output_root / followup_run_id / "combined_family_policies"
    policies: list[FollowupPolicy] = []

    combined_name = f"{policy_prefix}_drop_all_improvers_{cycle_id}"
    combined_path = policy_dir / f"{_safe_slug(combined_name)}{suffix}"
    policies.append(build_policy_from_drop_families(
        features=features,
        family_to_cols=family_to_cols,
        output_path=combined_path,
        drop_families=drop_families,
        overwrite=overwrite_features,
    ))

    if include_addback:
        for fam in drop_families:
            reduced = [f for f in drop_families if f != fam]
            name = f"{policy_prefix}_drop_all_improvers_addback_{fam}_{cycle_id}"
            policies.append(build_policy_from_drop_families(
                features=features,
                family_to_cols=family_to_cols,
                output_path=policy_dir / f"{_safe_slug(name)}{suffix}",
                drop_families=reduced,
                overwrite=overwrite_features,
            ))

    # Optional comparators. They will be retrained in the follow-up run, so keep
    # them optional to avoid wasting compute.
    if include_full:
        full_policy = find_full_policy_row(metrics)
        full_name = str(full_policy.get("policy"))
        path = _lookup_policy_path_from_generated_config(cfg, full_name)
        if path is not None:
            policies.append(FollowupPolicy(name=full_name, features_path=path, drop_families=tuple(), source="source_full"))
    if include_source_selected:
        selected_rows = metrics[metrics.get("selected", False).astype(bool)] if "selected" in metrics.columns else pd.DataFrame()
        if not selected_rows.empty:
            selected_name = str(selected_rows.iloc[0].get("policy"))
            selected_path = _lookup_policy_path_from_generated_config(cfg, selected_name)
            if selected_path is not None:
                fam = _parse_minus_family(selected_name)
                policies.append(FollowupPolicy(
                    name=f"source_selected_{selected_name}",
                    features_path=selected_path,
                    drop_families=tuple([fam] if fam else []),
                    source="source_selected",
                ))

    follow_cfg = dict(cfg)
    follow_cfg["feature_policies"] = [p.as_config_row() for p in policies]
    follow_cfg.setdefault("family_ablation_followup", {})
    follow_cfg["family_ablation_followup"].update({
        "source_run_dir": str(source_run_dir),
        "cycle_id": cycle_id,
        "metric_col": metric_col,
        "min_delta_bps": float(min_delta_bps),
        "drop_families": list(drop_families),
        "policy_count": len(policies),
    })
    follow_cfg_path = output_root / followup_run_id / "generated_followup_config.yaml"
    write_yaml(follow_cfg, follow_cfg_path)
    from swing_bot.artifacts.io import write_frame, write_json
    write_frame(delta_df, output_root / followup_run_id / "single_drop_deltas.csv")
    write_json({
        "source_run_dir": str(source_run_dir),
        "cycle_id": cycle_id,
        "metric_col": metric_col,
        "min_delta_bps": float(min_delta_bps),
        "drop_families": list(drop_families),
        "policies": [p.as_config_row() | {"drop_families": list(p.drop_families), "source": p.source} for p in policies],
        "generated_config": str(follow_cfg_path),
        "notes": [
            "This follow-up reuses the completed single-family LOFO metrics to define a combined-drop candidate.",
            "It does not recompute all single-drop candidates, but combined-drop performance still requires retraining/backtesting.",
        ],
    }, output_root / followup_run_id / "followup_plan.json")
    return {
        "source_run_dir": str(source_run_dir),
        "cycle_id": cycle_id,
        "metric_col": metric_col,
        "min_delta_bps": float(min_delta_bps),
        "drop_families": list(drop_families),
        "policy_count": len(policies),
        "generated_config": str(follow_cfg_path),
        "policy_dir": str(policy_dir),
    }


def run_combined_drop_followup(
    *,
    source_run_dir: Path | str,
    output_root: Path | str,
    followup_run_id: str,
    metric_col: str = "selector_raw_score",
    min_delta_bps: float = 0.0,
    include_addback: bool = False,
    include_full: bool = False,
    include_source_selected: bool = True,
    overwrite_features: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = prepare_combined_drop_followup(
        source_run_dir=source_run_dir,
        output_root=output_root,
        followup_run_id=followup_run_id,
        metric_col=metric_col,
        min_delta_bps=min_delta_bps,
        include_addback=include_addback,
        include_full=include_full,
        include_source_selected=include_source_selected,
        overwrite_features=overwrite_features,
    )
    if dry_run:
        plan["dry_run"] = True
        return plan
    first_month, last_month = infer_test_months_from_cycles(source_run_dir)
    from swing_bot.research.monthly_rolling import load_rolling_monthly_config, run_rolling_monthly_research
    cfg = load_rolling_monthly_config(plan["generated_config"])
    result = run_rolling_monthly_research(
        config=cfg,
        first_test_month=first_month,
        last_test_month=last_month,
        output_root=Path(output_root),
        run_id=followup_run_id,
        max_cycles=1,
    )
    plan.update({
        "dry_run": False,
        "run_dir": str(result.run_dir),
        "candidate_valid_metrics": str(result.candidate_csv),
        "rolling_cycles": str(result.cycles_csv),
        "rolling_test_metrics": str(result.test_metrics_csv),
    })
    return plan

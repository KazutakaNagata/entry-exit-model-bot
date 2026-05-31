#!/usr/bin/env python3
"""Run monthly rolling research with mechanically generated family LOFO policies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.run_id import make_run_id, safe_slug  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402
from swing_bot.research.family_policy_selection import build_leave_one_family_out_policies  # noqa: E402
import pandas as pd  # noqa: E402
from swing_bot.research.monthly_rolling import load_rolling_monthly_config, run_rolling_monthly_research  # noqa: E402

DEFAULT_CONFIG = Path("configs/rolling_protocol/long_H120_monthly_family_ablation_v0.yaml")
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "rolling_monthly_research"


def _split_csv(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full+LOFO family policies, then run monthly rolling research.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--first-test-month", type=str, required=True)
    p.add_argument("--last-test-month", type=str, required=True)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite-policies", action="store_true", help="Override config and rebuild policy parquet files.")
    p.add_argument("--skip-policy-build", action="store_true", help="Do not read the base feature matrix; reuse an existing family_policy_manifest.csv in policy_output_dir.")
    p.add_argument("--families", type=str, default=None, help="Optional comma-separated family allow-list; default is all manifest families.")
    p.add_argument("--exclude-families", type=str, default=None, help="Optional comma-separated family deny-list.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = _load_yaml(args.config)
    fam_cfg = raw.get("family_ablation") or {}
    if not fam_cfg:
        raise ValueError("config must include family_ablation section")
    policy_output_dir = Path(fam_cfg.get("policy_output_dir"))
    if args.skip_policy_build:
        manifest_path = policy_output_dir / "family_policy_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"--skip-policy-build requires existing policy manifest: {manifest_path}. "
                "Run scripts/build_family_ablation_policies.py or the generated prebuild_family_policies.sh first."
            )
        manifest_df = pd.read_csv(manifest_path)
        required_cols = {"name", "features_path", "description"}
        missing = required_cols - set(manifest_df.columns)
        if missing:
            raise ValueError(f"policy manifest missing columns {sorted(missing)}: {manifest_path}")
        policies = []
        feature_policy_rows = []
        for _, row in manifest_df.iterrows():
            feature_policy_rows.append({
                "name": str(row["name"]),
                "features": str(row["features_path"]),
                "description": str(row.get("description", "")),
            })
    else:
        policies, manifest_df = build_leave_one_family_out_policies(
            features_path=Path(fam_cfg["base_features"]),
            manifest_path=Path(fam_cfg["manifest"]),
            output_dir=policy_output_dir,
            policy_prefix=str(fam_cfg.get("policy_prefix") or "lofo"),
            families=_split_csv(args.families) or fam_cfg.get("families"),
            exclude_families_from_universe=_split_csv(args.exclude_families) or fam_cfg.get("exclude_families"),
            output_format=str(fam_cfg.get("output_format") or "parquet"),
            overwrite=bool(args.overwrite_policies or fam_cfg.get("overwrite_policies", False)),
        )
        feature_policy_rows = [
            {"name": p.name, "features": str(p.features_path), "description": p.description}
            for p in policies
        ]
    generated = dict(raw)
    generated["feature_policies"] = feature_policy_rows
    run_id = safe_slug(args.run_id) if args.run_id else make_run_id(str(raw.get("name") or "monthly_family_ablation"))
    generated_config_path = args.output_root / run_id / "generated_config.yaml"
    _write_yaml(generated, generated_config_path)
    plan = {
        "run_id": run_id,
        "generated_config": str(generated_config_path),
        "policy_count": len(feature_policy_rows),
        "policy_manifest": str(policy_output_dir / "family_policy_manifest.csv"),
        "families": manifest_df.iloc[0]["family_universe"] if "family_universe" in manifest_df.columns and len(manifest_df) else [],
        "first_test_month": args.first_test_month,
        "last_test_month": args.last_test_month,
        "max_cycles": args.max_cycles,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return 0
    cfg = load_rolling_monthly_config(generated_config_path)
    result = run_rolling_monthly_research(
        config=cfg,
        first_test_month=args.first_test_month,
        last_test_month=args.last_test_month,
        output_root=args.output_root,
        run_id=run_id,
        max_cycles=args.max_cycles,
    )
    print(json.dumps({
        **plan,
        "run_dir": str(result.run_dir),
        "candidate_valid_metrics": str(result.candidate_csv),
        "rolling_cycles": str(result.cycles_csv),
        "rolling_test_metrics": str(result.test_metrics_csv),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

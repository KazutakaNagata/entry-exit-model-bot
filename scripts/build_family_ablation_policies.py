#!/usr/bin/env python3
"""Build full and leave-one-family-out feature policy parquet files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.research.family_policy_selection import build_leave_one_family_out_policies  # noqa: E402


def _split_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full + leave-one-family-out feature policy files from a feature matrix and manifest.")
    p.add_argument("--features", type=Path, required=True, help="Input feature matrix parquet/csv with timestamp + feature columns.")
    p.add_argument("--manifest", type=Path, required=True, help="Feature manifest JSON/CSV containing feature family metadata.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--policy-prefix", type=str, default="family_ablation")
    p.add_argument("--families", type=str, default=None, help="Optional comma-separated family allow-list. Default: all families from manifest.")
    p.add_argument("--exclude-families", type=str, default=None, help="Optional comma-separated family deny-list for the universe.")
    p.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policies, manifest = build_leave_one_family_out_policies(
        features_path=args.features,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        policy_prefix=args.policy_prefix,
        families=_split_csv(args.families),
        exclude_families_from_universe=_split_csv(args.exclude_families),
        output_format=args.output_format,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "policy_count": len(policies),
        "families": manifest.iloc[0]["family_universe"] if len(manifest) else [],
        "manifest_csv": str(args.output_dir / "family_policy_manifest.csv"),
        "manifest_json": str(args.output_dir / "family_policy_manifest.json"),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

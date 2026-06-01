#!/usr/bin/env python3
"""Build/audit/train curated Long_H120 + full human-timeframe feature configs.

This script is intentionally a small orchestration wrapper around the existing
CLI tools. It does not implement new modelling logic.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = {
    "core": {
        "config": "configs/features/model_specific/entry_feature_set_long_H120_curated_core_plus_timeframe_v0.yaml",
        "features": "data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_core_plus_timeframe_v0.parquet",
        "manifest": "outputs/valid/features/entry_feature_manifest_long_H120_curated_core_plus_timeframe_v0.json",
        "audit": "outputs/valid/features/entry_feature_audit_long_H120_curated_core_plus_timeframe_v0.json",
        "grid_id": "entry_grid_long_H120_curated_core_plus_timeframe_v0",
    },
    "broad": {
        "config": "configs/features/model_specific/entry_feature_set_long_H120_curated_broad_plus_timeframe_v0.yaml",
        "features": "data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_broad_plus_timeframe_v0.parquet",
        "manifest": "outputs/valid/features/entry_feature_manifest_long_H120_curated_broad_plus_timeframe_v0.json",
        "audit": "outputs/valid/features/entry_feature_audit_long_H120_curated_broad_plus_timeframe_v0.json",
        "grid_id": "entry_grid_long_H120_curated_broad_plus_timeframe_v0",
    },
    "timeframe_only": {
        "config": "configs/features/model_specific/entry_feature_set_long_H120_curated_timeframe_only_v0.yaml",
        "features": "data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_timeframe_only_v0.parquet",
        "manifest": "outputs/valid/features/entry_feature_manifest_long_H120_curated_timeframe_only_v0.json",
        "audit": "outputs/valid/features/entry_feature_audit_long_H120_curated_timeframe_only_v0.json",
        "grid_id": "entry_grid_long_H120_curated_timeframe_only_v0",
    },
}


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _require_candidates(names: Iterable[str]) -> list[str]:
    result = []
    for name in names:
        if name not in CANDIDATES:
            raise SystemExit(f"unknown candidate {name!r}; choose from {', '.join(CANDIDATES)}")
        result.append(name)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet")
    parser.add_argument("--split", default="configs/splits/btcjpy_1m_split_v1.yaml")
    parser.add_argument("--candidates", default="core,broad", help="Comma-separated: core,broad,timeframe_only")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    py = sys.executable
    candidates = _require_candidates(_csv(args.candidates))

    for name in candidates:
        spec = CANDIDATES[name]
        print(f"\n=== candidate: {name} ===", flush=True)

        if not args.skip_build:
            _run([
                py,
                "scripts/build_features.py",
                "--input", args.input,
                "--config", spec["config"],
                "--output", spec["features"],
                "--manifest-output", spec["manifest"],
                "--report-output", spec["audit"],
            ], dry_run=args.dry_run)

        if not args.skip_audit:
            _run([
                py,
                "scripts/audit_features.py",
                "--features", spec["features"],
                "--manifest", spec["manifest"],
                "--config", spec["config"],
                "--strict",
            ], dry_run=args.dry_run)

        if not args.skip_train:
            _run([
                py,
                "scripts/train_entry_grid.py",
                "--split", args.split,
                "--features", spec["features"],
                "--grid-run-id", spec["grid_id"],
                "--sides", "long",
                "--horizons", "120",
            ], dry_run=args.dry_run)

    print("\nDone. Compare entry_model_comparison.csv under outputs/valid/entry_grid/<grid_id>/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

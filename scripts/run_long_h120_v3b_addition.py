#!/usr/bin/env python3
"""Build features and train long_H120 entry grids for v3b family additions.

Thin orchestration around reviewed CLIs:

- scripts/build_features.py
- scripts/train_entry_grid.py

It trains only long_H120 by default, uses the existing split manifest, and never
reads test data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.features.v3b_addition import write_v3b_addition_configs  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402

DEFAULT_INPUT = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_LABELS = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_CONFIG_DIR = Path("configs/features/additions/long_H120_v3b")
DEFAULT_FEATURE_OUTPUT_DIR = btcjpy_1m_processed_dir() / "feature_additions" / "long_H120_v3b"
DEFAULT_MANIFEST_OUTPUT_DIR = outputs_dir() / "valid" / "features" / "additions" / "long_H120_v3b"


def _parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _addition_name_from_config(path: Path) -> str:
    prefix = "entry_feature_set_long_H120_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def _find_configs(config_dir: Path, names: list[str]) -> list[Path]:
    if not config_dir.exists():
        write_v3b_addition_configs(config_dir, force=False)
    if names:
        paths = []
        for name in names:
            path = config_dir / f"entry_feature_set_long_H120_{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"v3b addition config not found: {path}")
            paths.append(path)
        return paths
    paths = sorted(config_dir.glob("entry_feature_set_long_H120_*.yaml"))
    if not paths:
        write_v3b_addition_configs(config_dir, force=False)
        paths = sorted(config_dir.glob("entry_feature_set_long_H120_*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no v3b addition configs found under {config_dir}")
    return paths


def _load_config_name(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return str(data.get("feature_set_name") or path.stem)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and train long_H120 v3b family-addition entry grids.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Canonical 1m OHLCV file.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--additions", type=str, default=None, help="Comma-separated names, e.g. base_v0,base_plus_enhanced_pullback_geometry")
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--manifest-output-dir", type=Path, default=DEFAULT_MANIFEST_OUTPUT_DIR)
    parser.add_argument("--grid-prefix", type=str, default="long_h120_v3b_addition")
    parser.add_argument("--only-build", action="store_true")
    parser.add_argument("--only-train", action="store_true")
    parser.add_argument("--skip-existing-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.only_build and args.only_train:
        raise ValueError("--only-build and --only-train are mutually exclusive")

    configs = _find_configs(args.config_dir, _parse_csv(args.additions))
    commands: list[dict[str, object]] = []
    for config_path in configs:
        addition = _addition_name_from_config(config_path)
        feature_set_name = _load_config_name(config_path)
        features_path = args.feature_output_dir / f"features_{addition}.parquet"
        manifest_path = args.manifest_output_dir / f"feature_manifest_{addition}.json"
        report_path = args.manifest_output_dir / f"feature_audit_{addition}.json"
        grid_id = f"{args.grid_prefix}_{addition}"

        if not args.only_train:
            build_cmd = [
                sys.executable,
                "scripts/build_features.py",
                "--input", str(args.input),
                "--config", str(config_path),
                "--output", str(features_path),
                "--manifest-output", str(manifest_path),
                "--report-output", str(report_path),
            ]
            if args.skip_existing_features and features_path.exists():
                print(f"Skipping existing features for {addition}: {features_path}", flush=True)
            else:
                _run(build_cmd, dry_run=args.dry_run)
            commands.append({"addition": addition, "feature_set_name": feature_set_name, "stage": "build_features", "cmd": build_cmd})

        if not args.only_build:
            train_cmd = [
                sys.executable,
                "scripts/train_entry_grid.py",
                "--features", str(features_path),
                "--labels", str(args.labels),
                "--split", str(args.split),
                "--sides", "long",
                "--horizons", "120",
                "--grid-run-id", grid_id,
            ]
            _run(train_cmd, dry_run=args.dry_run)
            commands.append({"addition": addition, "feature_set_name": feature_set_name, "stage": "train_entry_grid", "cmd": train_cmd})

    print(json.dumps({
        "addition_count": len(configs),
        "additions": [_addition_name_from_config(p) for p in configs],
        "feature_output_dir": str(args.feature_output_dir),
        "manifest_output_dir": str(args.manifest_output_dir),
        "grid_prefix": args.grid_prefix,
        "side": "long",
        "horizon_minutes": 120,
        "commands": commands,
        "notes": [
            "This script delegates to build_features.py and train_entry_grid.py.",
            "It trains long_H120 only and does not read or evaluate test data.",
            "Compare outputs with scripts/compare_long_h120_v3b_addition.py.",
        ],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

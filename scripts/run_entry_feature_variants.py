#!/usr/bin/env python3
"""Build features and train entry grids for feature-set variants.

This is a thin orchestration script around existing reviewed CLIs:

- scripts/build_features.py
- scripts/train_entry_grid.py

It is intentionally not a new training implementation.  It never reads test and
only passes explicit valid-split inputs through to the existing entry pipeline.
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

from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402

DEFAULT_INPUT = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_LABELS = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_VARIANTS_DIR = Path("configs/features/variants")
DEFAULT_FEATURE_OUTPUT_DIR = btcjpy_1m_processed_dir() / "feature_variants"
DEFAULT_MANIFEST_OUTPUT_DIR = outputs_dir() / "valid" / "features" / "variants"


def _parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _variant_name_from_config(path: Path) -> str:
    stem = path.stem
    prefix = "entry_feature_set_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _find_variant_configs(variants_dir: Path, names: list[str]) -> list[Path]:
    if names:
        paths = []
        for name in names:
            path = variants_dir / f"entry_feature_set_{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"variant config not found: {path}")
            paths.append(path)
        return paths
    paths = sorted(variants_dir.glob("entry_feature_set_*.yaml"))
    if not paths:
        raise FileNotFoundError(
            f"no variant configs found in {variants_dir}; run scripts/build_feature_set_variants.py first"
        )
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
    parser = argparse.ArgumentParser(description="Build and train entry grids for feature variants.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Canonical 1m OHLCV file.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--variants-dir", type=Path, default=DEFAULT_VARIANTS_DIR)
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variant names without entry_feature_set_ prefix.")
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--manifest-output-dir", type=Path, default=DEFAULT_MANIFEST_OUTPUT_DIR)
    parser.add_argument("--grid-prefix", type=str, default="entry_variant", help="Prefix for train_entry_grid --grid-run-id.")
    parser.add_argument("--sides", type=str, default="long,short")
    parser.add_argument("--horizons", type=str, default="60,120,240")
    parser.add_argument("--only-build", action="store_true")
    parser.add_argument("--only-train", action="store_true")
    parser.add_argument("--skip-existing-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.only_build and args.only_train:
        raise ValueError("--only-build and --only-train are mutually exclusive")
    variant_names = _parse_csv(args.variants)
    configs = _find_variant_configs(args.variants_dir, variant_names)

    commands: list[dict[str, object]] = []
    for config_path in configs:
        variant = _variant_name_from_config(config_path)
        feature_set_name = _load_config_name(config_path)
        features_path = args.feature_output_dir / f"features_{variant}.parquet"
        manifest_path = args.manifest_output_dir / f"feature_manifest_{variant}.json"
        report_path = args.manifest_output_dir / f"feature_audit_{variant}.json"
        grid_id = f"{args.grid_prefix}_{variant}"

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
                print(f"Skipping existing features for {variant}: {features_path}", flush=True)
            else:
                _run(build_cmd, dry_run=args.dry_run)
            commands.append({"variant": variant, "stage": "build_features", "cmd": build_cmd})

        if not args.only_build:
            train_cmd = [
                sys.executable,
                "scripts/train_entry_grid.py",
                "--features", str(features_path),
                "--labels", str(args.labels),
                "--split", str(args.split),
                "--sides", args.sides,
                "--horizons", args.horizons,
                "--grid-run-id", grid_id,
            ]
            _run(train_cmd, dry_run=args.dry_run)
            commands.append({"variant": variant, "stage": "train_entry_grid", "cmd": train_cmd})

    print(json.dumps({
        "variant_count": len(configs),
        "variants": [_variant_name_from_config(p) for p in configs],
        "feature_output_dir": str(args.feature_output_dir),
        "manifest_output_dir": str(args.manifest_output_dir),
        "grid_prefix": args.grid_prefix,
        "commands": commands,
        "notes": [
            "This script delegates to existing build_features.py and train_entry_grid.py.",
            "No test data is read or evaluated by the delegated scripts.",
        ],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

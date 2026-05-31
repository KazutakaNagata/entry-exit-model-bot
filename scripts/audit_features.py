#!/usr/bin/env python3
"""Audit a feature matrix and manifest for obvious leakage hazards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.features.leakage_audit import audit_feature_frame  # noqa: E402
from swing_bot.features.manifest import read_manifest  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402

DEFAULT_FEATURES = btcjpy_1m_processed_dir() / "features_entry_v0.parquet"
DEFAULT_MANIFEST = outputs_dir() / "valid" / "features" / "entry_feature_manifest_v0.json"


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_frame(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    raise ValueError("features path must end with .parquet, .pq, .csv, or .csv.gz")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated feature matrix and manifest.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Feature matrix path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Feature manifest path.")
    parser.add_argument("--config", type=Path, default=Path("configs/features/entry_feature_set_v0.yaml"), help="Feature config with exclude_patterns.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when audit violations are found.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    features = _read_frame(args.features)
    specs = read_manifest(args.manifest)
    audit = audit_feature_frame(features, specs, exclude_patterns=cfg.get("exclude_patterns") or [])
    print(json.dumps(audit.to_dict(), indent=2, ensure_ascii=False))
    if args.strict and not audit.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

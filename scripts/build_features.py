#!/usr/bin/env python3
"""Build reviewed past-only features from canonical 1-minute OHLCV.

This script does not build labels, train models, tune thresholds, or touch test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.data.load_ohlcv import load_ohlcv  # noqa: E402
from swing_bot.features.build_matrix import build_feature_matrix, feature_summary  # noqa: E402
from swing_bot.features.leakage_audit import audit_feature_frame  # noqa: E402
from swing_bot.features.manifest import write_manifest  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402

DEFAULT_INPUT = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_OUTPUT = btcjpy_1m_processed_dir() / "features_entry_v0.parquet"
DEFAULT_MANIFEST = outputs_dir() / "valid" / "features" / "entry_feature_manifest_v0.json"
DEFAULT_REPORT = outputs_dir() / "valid" / "features" / "entry_feature_audit_v0.json"


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal reviewed feature matrix.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Canonical OHLCV parquet/csv or directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output feature matrix path (.parquet/.csv).")
    parser.add_argument("--config", type=Path, default=Path("configs/features/entry_feature_set_v0.yaml"), help="Feature set config.")
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST, help="Output feature manifest JSON/CSV path.")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT, help="Output audit report JSON path.")
    parser.add_argument("--families", type=str, default=None, help="Comma-separated family override, e.g. price,return_path.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing feature matrix.")
    return parser.parse_args(argv)


def _parse_families(text: str | None, config: dict) -> list[str]:
    if text:
        return [x.strip() for x in text.split(",") if x.strip()]
    families = config.get("include_families") or config.get("market_feature_families")
    if not families:
        raise ValueError("feature config must define include_families or pass --families")
    return list(families)


def _write_frame(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False)
    elif name.endswith((".csv", ".csv.gz")):
        df.to_csv(path, index=False)
    else:
        raise ValueError("--output must end with .parquet, .pq, .csv, or .csv.gz")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_yaml(args.config)
    families = _parse_families(args.families, config)
    exclude_patterns = config.get("exclude_patterns") or []

    ohlcv = load_ohlcv(args.input)
    features, specs = build_feature_matrix(
        ohlcv,
        include_families=families,
        exclude_patterns=exclude_patterns,
        strict_audit=True,
    )
    audit = audit_feature_frame(features, specs, exclude_patterns=exclude_patterns)
    summary = {
        "feature_set_name": config.get("feature_set_name"),
        "input": str(args.input),
        "families": families,
        "summary": feature_summary(features, specs),
        "audit": audit.to_dict(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.dry_run:
        _write_frame(features, args.output)
        write_manifest(specs, args.manifest_output)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        with args.report_output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote features: {args.output}")
        print(f"Wrote manifest: {args.manifest_output}")
        print(f"Wrote audit report: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

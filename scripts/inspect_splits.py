#!/usr/bin/env python3
"""Inspect a concrete train/valid/test split manifest.

This script does not tune thresholds and does not evaluate test.  It only prints
manifest ranges and optional row counts for a canonical OHLCV file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.data.load_ohlcv import load_ohlcv  # noqa: E402
from swing_bot.splits.split_manifest import SplitManifestError, load_split_manifest, summarize_split_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a concrete BTCJPY 1m split manifest.")
    parser.add_argument("--split", type=Path, required=True, help="Path to concrete split YAML. Do not use the template directly.")
    parser.add_argument("--data", type=Path, default=None, help="Optional canonical OHLCV file or directory for row counts.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of tables.")
    return parser.parse_args(argv)


def _counts_by_split(df: pd.DataFrame, manifest) -> dict[str, int]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    counts = {
        "train": int(manifest.mask(ts, "train").sum()),
        "valid": int(manifest.mask(ts, "valid").sum()),
        "test": int(manifest.mask(ts, "test").sum()),
    }
    for fold in manifest.folds:
        counts[fold.name] = int(manifest.mask(ts, fold.name).sum())
    return counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_split_manifest(args.split)
    except SplitManifestError as exc:
        print(f"Split manifest error: {exc}", file=sys.stderr)
        return 2

    summary = summarize_split_manifest(manifest)
    counts = None
    if args.data is not None:
        df = load_ohlcv(args.data)
        counts = _counts_by_split(df, manifest)
        summary["row_count"] = summary["name"].map(counts)

    if args.json:
        payload = {"manifest": manifest.to_dict(), "row_counts": counts}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(summary.to_string(index=False))
        print(f"\npurge_minutes={manifest.purge_minutes} embargo_minutes={manifest.embargo_minutes}")
        print("test_usage=locked_audit_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

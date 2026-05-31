#!/usr/bin/env python3
"""Audit Binance Japan BTCJPY 1-minute OHLCV data.

Example:
    python scripts/audit_data.py --input data/raw/binance_japan/BTCJPY/1m --write-canonical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.data.load_ohlcv import load_ohlcv, save_normalized_ohlcv  # noqa: E402
from swing_bot.data.quality import audit_ohlcv, write_quality_report  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, btcjpy_1m_raw_dir, outputs_dir  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BTCJPY 1m OHLCV data without repairing it.")
    parser.add_argument("--input", type=Path, default=btcjpy_1m_raw_dir(), help="Raw file or directory. Default: data/raw/binance_japan/BTCJPY/1m")
    parser.add_argument("--output", type=Path, default=outputs_dir() / "valid" / "data_quality" / "binance_japan" / "BTCJPY" / "1m", help="Quality report directory.")
    parser.add_argument("--write-canonical", "--write-normalized", dest="write_canonical", action="store_true", help="Write normalized canonical OHLCV after audit.")
    parser.add_argument("--canonical-output", "--normalized-output", dest="canonical_output", type=Path, default=btcjpy_1m_processed_dir() / "ohlcv.parquet", help="Canonical output path.")
    parser.add_argument("--keep-extra-columns", action="store_true", help="Keep non-canonical columns while loading. Default keeps only OHLCV.")
    parser.add_argument("--fail-on-issues", action="store_true", help="Return non-zero when audit finds data issues.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_ohlcv(args.input, keep_extra_columns=args.keep_extra_columns)
    report = audit_ohlcv(df)
    summary_path = write_quality_report(report, args.output)

    canonical_path = None
    if args.write_canonical:
        canonical_path = save_normalized_ohlcv(df, args.canonical_output)

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nWrote audit summary: {summary_path}")
    if canonical_path is not None:
        print(f"Wrote canonical OHLCV: {canonical_path}")
    return 2 if args.fail_on_issues and not report.summary.passed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

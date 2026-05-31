#!/usr/bin/env python3
"""Collect OOF valid predictions from trained entry runs.

This script is a safety gate before exit-dataset construction.  It refuses any
prediction file where ``is_oof`` is not true for every row and writes one tidy
artifact with exactly one prediction per timestamp/side/horizon.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.models.entry_oof import combine_entry_oof_predictions, summarize_entry_run_dirs  # noqa: E402
from swing_bot.artifacts.io import write_frame  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_oof"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine entry OOF predictions for exit-dataset creation.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Entry run dir. Repeat for each side/horizon.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir. Default: outputs/valid/entry_oof/<first-run>_combined")
    parser.add_argument("--comparison-csv", type=Path, default=None, help="Optional path for compact run comparison CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        first = args.run_dir[0].name
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{first}_combined"
    combined, summary = combine_entry_oof_predictions(args.run_dir, output_dir=args.output_dir)
    comparison = summarize_entry_run_dirs(args.run_dir)
    if args.comparison_csv:
        write_frame(comparison, args.comparison_csv)
    payload = {
        "output_dir": str(args.output_dir),
        "row_count": int(len(combined)),
        "run_count": int(len(args.run_dir)),
        "summary": summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

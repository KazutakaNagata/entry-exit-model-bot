#!/usr/bin/env python3
"""Compare saved entry run summaries without reading test or retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import write_frame  # noqa: E402
from swing_bot.models.entry_oof import summarize_entry_run_dirs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact comparison table from entry run dirs.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Entry run dir. Repeat for each run.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison = summarize_entry_run_dirs(args.run_dir)
    if args.output:
        write_frame(comparison, args.output)
    print(json.dumps(comparison.to_dict(orient="records"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

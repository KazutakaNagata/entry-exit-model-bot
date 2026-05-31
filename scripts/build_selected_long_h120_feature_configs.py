#!/usr/bin/env python3
"""Write reviewed selected long_H120 feature-set configs.

This script only writes YAML configs. It does not build features, train models,
read test data, or tune thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.features.selected import write_selected_long_h120_feature_sets  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("configs/features/selected/long_H120")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write selected long_H120 feature-set configs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing configs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = write_selected_long_h120_feature_sets(args.output_dir, force=args.force)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "files": [str(p) for p in paths],
        "manifest": str(args.output_dir / "selected_long_H120_feature_sets.yaml"),
        "notes": [
            "Config generation only; no training or test access.",
            "Use run_selected_long_h120_entry.py to build features and train long_H120 candidates.",
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Write long_H120 v3a combined-ablation feature-set configs.

This script only writes YAML configs.  It does not build features, train models,
read test data, tune thresholds, or decide which variant is best.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.features.combined_ablation import (  # noqa: E402
    default_long_h120_v3a_combined_ablation_configs,
    write_combined_ablation_configs,
)

DEFAULT_OUTPUT_DIR = Path("configs/features/ablations/long_H120_v3a_combined")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate long_H120 v3a combined-ablation feature-set YAML configs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated configs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = default_long_h120_v3a_combined_ablation_configs()
    paths = write_combined_ablation_configs(args.output_dir, configs=configs, force=args.force)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "config_count": len(configs),
        "configs": [str(p) for p in paths],
        "manifest": str(args.output_dir / "long_H120_v3a_combined_ablation_manifest.yaml"),
        "baseline": "v3a_full",
        "side": "long",
        "horizon_minutes": 120,
        "notes": [
            "Configs are for valid-fold combined ablation only.",
            "Recommended next command: scripts/run_long_h120_v3a_combined_ablation.py --dry-run",
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

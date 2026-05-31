#!/usr/bin/env python3
"""Write reviewed feature-set variant configs for v1/v2 family comparison.

This script only writes YAML configs.  It does not build features, train models,
or touch test data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.features.variants import default_entry_variants, write_variant_configs  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("configs/features/variants")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate entry feature variant YAML configs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated configs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = default_entry_variants()
    paths = write_variant_configs(args.output_dir, variants=variants, force=args.force)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "variant_count": len(variants),
        "configs": [str(p) for p in paths],
        "manifest": str(args.output_dir / "feature_variant_manifest.yaml"),
        "notes": [
            "Configs are for valid-only feature-family comparison.",
            "Recommended first focus: long H120 and short H240.",
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

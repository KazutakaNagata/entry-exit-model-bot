#!/usr/bin/env python3
"""Run monthly rolling train/valid/test research process.

For each rolling test month, this script evaluates a fixed list of feature
policies on the three prior monthly validation folds, selects one policy by a
fixed robust score, then trains on the nine months before the test month and
backtests that month.  It does not touch a separate final locked test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.run_id import make_run_id, safe_slug  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402
from swing_bot.research.monthly_rolling import (  # noqa: E402
    build_monthly_cycles,
    load_rolling_monthly_config,
    run_rolling_monthly_research,
)

DEFAULT_CONFIG = Path("configs/rolling_protocol/long_H120_monthly_v0.yaml")
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "rolling_monthly_research"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly rolling research-process backtest. Valid/test months roll forward; no global final test access.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--first-test-month", type=str, required=True, help="First rolling test month, e.g. 2025-04-01")
    parser.add_argument("--last-test-month", type=str, required=True, help="Last rolling test month inclusive, e.g. 2025-09-01")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-cycles", type=int, default=None, help="Debug knob: only run the first N monthly cycles.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")
    cfg = load_rolling_monthly_config(args.config)
    cycles = build_monthly_cycles(
        first_test_month=args.first_test_month,
        last_test_month=args.last_test_month,
        valid_months=cfg.valid_months,
        test_months=cfg.test_months,
        step_months=cfg.step_months,
    )
    if args.max_cycles is not None:
        cycles = cycles[: int(args.max_cycles)]
    run_id = safe_slug(args.run_id) if args.run_id else make_run_id(f"rolling_monthly_{cfg.name}")
    plan = {
        "run_id": run_id,
        "config": str(args.config),
        "first_test_month": args.first_test_month,
        "last_test_month": args.last_test_month,
        "cycle_count": len(cycles),
        "feature_policies": [p.name for p in cfg.feature_policies],
        "cycles": [
            {
                "cycle_id": c.cycle_id,
                "test_start": c.test_start.isoformat(),
                "test_end": c.test_end.isoformat(),
                "valid_months": [
                    {"name": name, "start": start.isoformat(), "end": end.isoformat()}
                    for name, start, end in c.valid_months
                ],
            }
            for c in cycles
        ],
        "notes": [
            "Feature policy candidates are fixed by config; selection is automatic inside each cycle.",
            "Each test month is evaluated only after selecting on prior valid months.",
            "Do not alter selector rules after inspecting rolling test results without restarting the research process.",
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return 0
    result = run_rolling_monthly_research(
        config=cfg,
        first_test_month=args.first_test_month,
        last_test_month=args.last_test_month,
        output_root=args.output_root,
        run_id=run_id,
        max_cycles=args.max_cycles,
    )
    print(json.dumps({
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "cycles": str(result.cycles_csv),
        "candidate_valid_metrics": str(result.candidate_csv),
        "rolling_test_metrics": str(result.test_metrics_csv),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.research.monthly_parallel import prepare_jobs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prepare one rolling-monthly job script per test month.")
    p.add_argument("--first-test-month", required=True, help="YYYY-MM-01")
    p.add_argument("--last-test-month", required=True, help="YYYY-MM-01, inclusive")
    p.add_argument("--job-root", default="outputs/valid/rolling_monthly_jobs")
    p.add_argument("--batch-id", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to run_rolling_monthly_research.py. Repeat for multiple args, e.g. --extra-arg --config --extra-arg path.yaml",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    batch_dir, jobs = prepare_jobs(
        repo_root=REPO_ROOT,
        job_root=(REPO_ROOT / args.job_root),
        first_test_month=args.first_test_month,
        last_test_month=args.last_test_month,
        python_executable=args.python,
        extra_args=args.extra_arg,
        batch_id=args.batch_id,
    )
    print(
        {
            "batch_dir": str(batch_dir),
            "job_count": len(jobs),
            "manifest": str(batch_dir / "job_manifest.csv"),
            "run_all_serial": str(batch_dir / "run_all_serial.sh"),
            "run_all_parallel": str(batch_dir / "run_all_parallel.sh"),
            "launch_tmux": str(batch_dir / "launch_tmux.sh"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

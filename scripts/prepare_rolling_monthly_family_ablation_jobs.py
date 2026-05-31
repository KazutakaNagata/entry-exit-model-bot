#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.research.monthly_family_ablation_parallel import prepare_family_ablation_jobs  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prepare one rolling-monthly family-ablation job script per test month.")
    p.add_argument("--first-test-month", required=True, help="YYYY-MM-01")
    p.add_argument("--last-test-month", required=True, help="YYYY-MM-01, inclusive")
    p.add_argument("--config", default="configs/rolling_protocol/long_H120_monthly_family_ablation_v0.yaml")
    p.add_argument("--job-root", default="outputs/valid/rolling_monthly_family_ablation_jobs")
    p.add_argument("--batch-id", default=None)
    p.add_argument("--run-id-prefix", default=None, help="Prefix for per-month run_id. Default: batch-id.")
    p.add_argument("--output-root", default=None, help="Optional output root passed to run_rolling_monthly_family_ablation.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to run_rolling_monthly_family_ablation.py. Repeat for multiple args.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prepared = prepare_family_ablation_jobs(
        repo_root=REPO_ROOT,
        job_root=REPO_ROOT / args.job_root,
        config=REPO_ROOT / args.config,
        first_test_month=args.first_test_month,
        last_test_month=args.last_test_month,
        python_executable=args.python,
        batch_id=args.batch_id,
        run_id_prefix=args.run_id_prefix,
        output_root=args.output_root,
        extra_args=args.extra_arg,
    )
    payload = {
        "batch_dir": str(prepared.batch_dir),
        "job_count": len(prepared.jobs),
        "prebuild_family_policies": str(prepared.prebuild_script),
        "manifest": str(prepared.batch_dir / "job_manifest.csv"),
        "run_all_serial": str(prepared.batch_dir / "run_all_serial.sh"),
        "run_all_parallel": str(prepared.batch_dir / "run_all_parallel.sh"),
        "launch_tmux": str(prepared.batch_dir / "launch_tmux.sh"),
        "watch_command": f"python3 scripts/watch_rolling_monthly_family_ablation_jobs.py --batch-dir {prepared.batch_dir} --watch --interval-sec 30",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

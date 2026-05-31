#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(batch_dir: Path) -> list[dict[str, str]]:
    with (batch_dir / "job_manifest.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_status(path: Path) -> dict:
    if not path.exists():
        return {"status": "pending"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}


def tail(path: Path, n: int = 1) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return " | ".join(lines[-n:])


def render(batch_dir: Path, tail_lines: int) -> None:
    rows = load_manifest(batch_dir)
    print(f"batch_dir={batch_dir}")
    print(f"{'job':<14} {'month':<10} {'status':<12} {'elapsed':>8}  last_log")
    for row in rows:
        status = read_status(Path(row["status_path"]))
        elapsed = status.get("elapsed_sec", "")
        print(
            f"{row['job_name']:<14} {row['test_month']:<10} {status.get('status','pending'):<12} {str(elapsed):>8}  {tail(Path(row['log_path']), tail_lines)}"
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Show progress for prepared rolling-monthly jobs.")
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval-sec", type=int, default=30)
    p.add_argument("--tail-lines", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = REPO_ROOT / batch_dir
    while True:
        print("\033c", end="")
        render(batch_dir, args.tail_lines)
        if not args.watch:
            break
        time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize one or more rolling monthly research runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import read_frame, write_frame  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_ROOT = outputs_dir() / "valid" / "rolling_monthly_research"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rolling monthly research runs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--run-prefix", type=str, default="rolling_monthly")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _find(root: Path, prefix: str, explicit: list[Path]) -> list[Path]:
    if explicit:
        return [p.resolve() for p in explicit]
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not runs:
        raise FileNotFoundError(f"no run dirs under {root} matching {prefix}*")
    return runs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for run_dir in _find(args.root, args.run_prefix, args.run_dir):
        metrics_path = run_dir / "rolling_test_metrics.csv"
        cycles_path = run_dir / "rolling_cycles.csv"
        if not metrics_path.exists():
            continue
        df = read_frame(metrics_path)
        if df.empty:
            continue
        row = {
            "run_name": run_dir.name,
            "cycle_count": int(len(df)),
            "selected_policy_counts": df.get("selected_policy", pd.Series(dtype=str)).value_counts().to_dict(),
            "sum_test_mean_net_pl_bps_sum": float(pd.to_numeric(df.get("test_mean_net_pl_bps_sum", 0.0), errors="coerce").sum()),
            "mean_test_mean_net_pl_bps_sum": float(pd.to_numeric(df.get("test_mean_net_pl_bps_sum", 0.0), errors="coerce").mean()),
            "worst_test_mean_net_pl_bps_sum": float(pd.to_numeric(df.get("test_mean_net_pl_bps_sum", 0.0), errors="coerce").min()),
            "sum_test_episode_count": int(pd.to_numeric(df.get("test_episode_count", 0), errors="coerce").fillna(0).sum()),
        }
        if cycles_path.exists():
            cyc = read_frame(cycles_path)
            row["cycles_path"] = str(cycles_path)
            row["candidate_valid_metrics_path"] = str(run_dir / "candidate_valid_metrics.csv")
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("sum_test_mean_net_pl_bps_sum", ascending=False) if rows else pd.DataFrame()
    output = args.output or (args.root / "rolling_monthly_research_comparison.csv")
    write_frame(out, output)
    print(json.dumps({"run_count": int(len(out)), "output": str(output), "preview": out.head(10).to_dict(orient="records")}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

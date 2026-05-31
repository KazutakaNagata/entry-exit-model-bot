#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csvs(root: Path, filename: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob(filename)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["source_file"] = str(path)
        df["source_run_dir"] = str(path.parent)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _read_status(job_root: Path | None) -> pd.DataFrame:
    if job_root is None or not job_root.exists():
        return pd.DataFrame()
    records = []
    for path in sorted(job_root.rglob("status/*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")) | {"status_file": str(path)})
        except Exception as exc:
            records.append({"status_file": str(path), "status": "unreadable", "error": str(exc)})
    return pd.DataFrame(records)


def _summary(df: pd.DataFrame) -> dict:
    out = {"row_count": int(len(df))}
    for col in [
        "net_pl_bps_sum",
        "mean_net_pl_bps_sum",
        "test_net_pl_bps_sum",
        "episode_count",
        "profit_factor",
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_sum"] = float(vals.sum(skipna=True))
            out[f"{col}_mean"] = float(vals.mean(skipna=True))
            out[f"{col}_min"] = float(vals.min(skipna=True))
            out[f"{col}_max"] = float(vals.max(skipna=True))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate rolling monthly job outputs.")
    p.add_argument("--research-root", default="outputs/valid/rolling_monthly_research")
    p.add_argument("--job-root", default="outputs/valid/rolling_monthly_jobs")
    p.add_argument("--output-dir", default="outputs/valid/rolling_monthly_aggregate")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    research_root = REPO_ROOT / args.research_root
    job_root = REPO_ROOT / args.job_root
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = _read_csvs(research_root, "rolling_test_metrics.csv")
    cycles_df = _read_csvs(research_root, "rolling_cycles.csv")
    candidates_df = _read_csvs(research_root, "candidate_valid_metrics.csv")
    status_df = _read_status(job_root)

    if not test_df.empty:
        test_df.to_csv(output_dir / "rolling_test_metrics_all.csv", index=False)
    if not cycles_df.empty:
        cycles_df.to_csv(output_dir / "rolling_cycles_all.csv", index=False)
    if not candidates_df.empty:
        candidates_df.to_csv(output_dir / "candidate_valid_metrics_all.csv", index=False)
    if not status_df.empty:
        status_df.to_csv(output_dir / "job_status_all.csv", index=False)

    summary = {
        "research_root": str(research_root),
        "job_root": str(job_root),
        "output_dir": str(output_dir),
        "test_metrics": _summary(test_df),
        "cycles": _summary(cycles_df),
        "candidate_valid_metrics": _summary(candidates_df),
        "job_status_count": int(len(status_df)),
    }
    (output_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

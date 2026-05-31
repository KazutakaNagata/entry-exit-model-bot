#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csvs(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob(filename)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["source_file"] = str(path)
        df["source_run_dir"] = str(path.parent)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_status(batch_dir: Path | None) -> pd.DataFrame:
    if batch_dir is None or not batch_dir.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(batch_dir.rglob("status/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            record = {"status": "unreadable", "error": str(exc)}
        record["status_file"] = str(path)
        rows.append(record)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> dict:
    out = {"row_count": int(len(df))}
    for col in [
        "net_pl_bps_sum",
        "mean_net_pl_bps_sum",
        "selector_total_net_pl_bps",
        "selector_worst_fold_net_pl_bps",
        "robust_score",
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
    p = argparse.ArgumentParser(description="Aggregate monthly family-ablation job outputs.")
    p.add_argument("--research-root", default="outputs/valid/rolling_monthly_research")
    p.add_argument("--batch-dir", default="outputs/valid/rolling_monthly_family_ablation_jobs")
    p.add_argument("--run-prefix", default=None, help="Optional prefix filter for run directories.")
    p.add_argument("--output-dir", default="outputs/valid/rolling_monthly_family_ablation_aggregate")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    research_root = REPO_ROOT / args.research_root
    if args.run_prefix:
        # Build a temporary combined root list by reading only matching run dirs.
        matching = [p for p in research_root.glob(f"{args.run_prefix}*") if p.is_dir()]
        temp_frames = []
        # Easier: read per filename from each matching run dir.
        test_df = pd.concat([_read_csvs(p, "rolling_test_metrics.csv") for p in matching], ignore_index=True, sort=False) if matching else pd.DataFrame()
        cycles_df = pd.concat([_read_csvs(p, "rolling_cycles.csv") for p in matching], ignore_index=True, sort=False) if matching else pd.DataFrame()
        candidates_df = pd.concat([_read_csvs(p, "candidate_valid_metrics.csv") for p in matching], ignore_index=True, sort=False) if matching else pd.DataFrame()
    else:
        test_df = _read_csvs(research_root, "rolling_test_metrics.csv")
        cycles_df = _read_csvs(research_root, "rolling_cycles.csv")
        candidates_df = _read_csvs(research_root, "candidate_valid_metrics.csv")

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = REPO_ROOT / batch_dir
    status_df = _read_status(batch_dir)

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if not test_df.empty:
        test_df.to_csv(out / "family_rolling_test_metrics_all.csv", index=False)
    if not cycles_df.empty:
        cycles_df.to_csv(out / "family_rolling_cycles_all.csv", index=False)
    if not candidates_df.empty:
        candidates_df.to_csv(out / "family_candidate_valid_metrics_all.csv", index=False)
    if not status_df.empty:
        status_df.to_csv(out / "job_status_all.csv", index=False)

    summary = {
        "research_root": str(research_root),
        "batch_dir": str(batch_dir),
        "run_prefix": args.run_prefix,
        "output_dir": str(out),
        "test_metrics": _summary(test_df),
        "cycles": _summary(cycles_df),
        "candidate_valid_metrics": _summary(candidates_df),
        "job_status_count": int(len(status_df)),
    }
    (out / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

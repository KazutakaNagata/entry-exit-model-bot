#!/usr/bin/env python3
"""Summarize monthly rolling family-ablation results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import write_frame, write_json  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402
from swing_bot.research.family_policy_selection import family_ablation_deltas  # noqa: E402

DEFAULT_ROOT = outputs_dir() / "valid" / "rolling_monthly_research"
DEFAULT_OUTPUT = outputs_dir() / "valid" / "rolling_monthly_family_ablation"


def _find_runs(root: Path, run_prefix: str, run_dirs: list[Path] | None) -> list[Path]:
    if run_dirs:
        out = [Path(p) for p in run_dirs]
    else:
        out = sorted(p for p in root.glob(f"{run_prefix}*") if p.is_dir())
    out = [p for p in out if (p / "candidate_valid_metrics.csv").exists()]
    if not out:
        raise FileNotFoundError(f"no family-ablation run dirs under {root} matching {run_prefix}*")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare rolling monthly family-ablation results.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--run-prefix", type=str, default="long_H120_monthly_family_ablation_v0")
    p.add_argument("--run-dir", type=Path, action="append", default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = _find_runs(args.root, args.run_prefix, args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_frames = []
    cycle_frames = []
    test_frames = []
    for run_dir in runs:
        run_name = run_dir.name
        cand = pd.read_csv(run_dir / "candidate_valid_metrics.csv")
        cand.insert(0, "run_name", run_name)
        candidate_frames.append(cand)
        if (run_dir / "rolling_cycles.csv").exists():
            cycles = pd.read_csv(run_dir / "rolling_cycles.csv")
            cycles.insert(0, "run_name", run_name)
            cycle_frames.append(cycles)
        if (run_dir / "rolling_test_metrics.csv").exists():
            tests = pd.read_csv(run_dir / "rolling_test_metrics.csv")
            tests.insert(0, "run_name", run_name)
            test_frames.append(tests)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    cycles = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    tests = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    deltas = []
    for run_name, frame in candidates.groupby("run_name"):
        d = family_ablation_deltas(frame)
        if not d.empty:
            d.insert(0, "run_name", run_name)
            deltas.append(d)
    delta_df = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()

    selected = candidates[candidates.get("selected", False).astype(bool)].copy() if "selected" in candidates.columns else pd.DataFrame()
    if not selected.empty:
        selected_summary = selected.groupby(["run_name", "policy"], dropna=False).agg(
            selected_count=("cycle_id", "count"),
            mean_robust_score=("robust_score", "mean"),
            mean_total_net=("selector_total_net_pl_bps", "mean"),
            mean_worst_net=("selector_worst_fold_net_pl_bps", "mean"),
        ).reset_index().sort_values(["selected_count", "mean_robust_score"], ascending=[False, False])
    else:
        selected_summary = pd.DataFrame()

    write_frame(candidates, args.output_dir / "family_candidate_valid_metrics_all.csv")
    if not cycles.empty:
        write_frame(cycles, args.output_dir / "family_rolling_cycles_all.csv")
    if not tests.empty:
        write_frame(tests, args.output_dir / "family_rolling_test_metrics_all.csv")
    if not delta_df.empty:
        write_frame(delta_df, args.output_dir / "family_ablation_deltas.csv")
    if not selected_summary.empty:
        write_frame(selected_summary, args.output_dir / "family_selected_policy_summary.csv")
    summary = {
        "run_count": len(runs),
        "runs": [str(r) for r in runs],
        "candidate_rows": int(len(candidates)),
        "cycle_rows": int(len(cycles)),
        "test_rows": int(len(tests)),
        "output_dir": str(args.output_dir),
        "notes": [
            "family_ablation_deltas compares each minus-family policy against the full policy within the same cycle.",
            "selected_policy_summary counts which policy the fixed selector chose across rolling cycles.",
        ],
    }
    write_json(summary, args.output_dir / "family_ablation_summary.json")
    preview = selected_summary.head(20).to_dict(orient="records") if not selected_summary.empty else []
    print(json.dumps({**summary, "selected_preview": preview}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

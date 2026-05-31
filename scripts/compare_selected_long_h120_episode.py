#!/usr/bin/env python3
"""Compare selected long_H120 downstream episode backtest runs.

This script reads existing valid-fold episode summaries.  It does not train,
backtest, tune thresholds, or read test data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.paths import outputs_dir  # noqa: E402
from swing_bot.pipeline.selected_long_h120_downstream import (  # noqa: E402
    DEFAULT_CANDIDATES,
    parse_csv_values,
    summarize_episode_runs,
    write_episode_comparison,
)

DEFAULT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_OUTPUT = outputs_dir() / "valid" / "episode_backtest" / "selected_long_H120_episode_comparison.csv"


def _find_run_dirs(root: Path, prefix: str, explicit: list[Path]) -> list[Path]:
    if explicit:
        return explicit
    dirs = sorted(p for p in root.glob(prefix + "*") if p.is_dir() and (p / "summary_metrics.json").exists())
    if not dirs:
        raise FileNotFoundError(f"no episode run dirs under {root} matching {prefix}*")
    return dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selected long_H120 episode backtest runs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-prefix", type=str, default="selected_long_H120")
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sort-by", type=str, default="mean_net_pl_bps_sum")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = parse_csv_values(args.candidates, default=DEFAULT_CANDIDATES)
    run_dirs = _find_run_dirs(args.root, args.run_prefix, args.run_dir)
    df = summarize_episode_runs(run_dirs, candidates=candidates)
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False).reset_index(drop=True)
    written = write_episode_comparison(df, args.output)

    preview_cols = [
        "candidate",
        "exit_lookahead_minutes",
        "exit_feature_set",
        "entry_selection_mode",
        "entry_threshold_bps",
        "entry_score_floor_bps",
        "rolling_score_quantile",
        "rolling_score_window_days",
        "min_entry_pred_bps",
        "min_score_margin_bps",
        "min_score_ratio",
        "score_history_rows",
        "hold_threshold_bps",
        "episode_count",
        "mean_net_pl_bps_sum",
        "worst_net_pl_bps_sum",
        "mean_avg_net_pl_bps",
        "worst_avg_net_pl_bps",
        "mean_profit_factor",
        "mean_round_trips_per_day",
        "mean_avg_hold_minutes",
        "run_name",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(json.dumps({
        "run_count": len(run_dirs),
        "output": str(written),
        "sort_by": args.sort_by,
        "preview": df[preview_cols].head(20).to_dict(orient="records"),
        "notes": [
            "Episode comparison is valid-fold only.",
            "Do not tune using test; test should be evaluated only from a locked config.",
            "Compare long_H120_v0 and long_H120_tail_v0 with their intended entry thresholds.",
        ],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

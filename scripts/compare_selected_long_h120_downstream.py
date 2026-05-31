#!/usr/bin/env python3
"""Compare selected long_H120 downstream episode backtests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import read_json  # noqa: E402
from swing_bot.experiments.selected_long_h120_downstream import parse_episode_run_id  # noqa: E402

DEFAULT_EPISODE_ROOT = Path("outputs/valid/episode_backtest")
DEFAULT_OUTPUT = DEFAULT_EPISODE_ROOT / "selected_long_H120_downstream_comparison.csv"

SUMMARY_COLS = [
    "episode_count",
    "mean_episode_count",
    "worst_episode_count",
    "mean_gross_pl_bps_sum",
    "worst_gross_pl_bps_sum",
    "mean_net_pl_bps_sum",
    "worst_net_pl_bps_sum",
    "mean_avg_net_pl_bps",
    "worst_avg_net_pl_bps",
    "mean_median_net_pl_bps",
    "worst_median_net_pl_bps",
    "mean_win_rate",
    "worst_win_rate",
    "mean_profit_factor",
    "worst_profit_factor",
    "mean_avg_hold_minutes",
    "worst_avg_hold_minutes",
    "mean_round_trips_per_day",
    "worst_round_trips_per_day",
    "mean_fee_paid_bps",
    "worst_fee_paid_bps",
    "mean_avg_mfe_bps",
    "mean_avg_mae_bps",
    "mean_avg_giveback_bps",
]


def _episode_score(row: pd.Series) -> float:
    """Tiny valid-only ranking aid for episode runs; higher is better."""
    mean_avg = float(row.get("mean_avg_net_pl_bps", 0.0))
    worst_avg = float(row.get("worst_avg_net_pl_bps", 0.0))
    mean_pf = row.get("mean_profit_factor", 0.0)
    try:
        pf_bonus = min(float(mean_pf), 3.0)
    except Exception:
        pf_bonus = 0.0
    return mean_avg + 0.5 * worst_avg + 0.5 * pf_bonus


def _iter_run_dirs(root: Path, explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return explicit
    return sorted(p for p in root.glob("episode_selected_long_H120_*") if p.is_dir())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selected long_H120 episode backtests.")
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", default=None, help="Explicit episode run dir. Can be repeated.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for run_dir in _iter_run_dirs(args.episode_root, args.run_dir):
        summary_path = run_dir / "summary_metrics.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        summary = read_json(summary_path)
        meta = parse_episode_run_id(run_dir.name)
        row: dict[str, object] = {**meta, "run_dir": str(run_dir)}
        for col in SUMMARY_COLS:
            if col in summary:
                row[col] = summary[col]
        skipped = summary.get("skipped") or {}
        if isinstance(skipped, dict):
            for key, val in skipped.items():
                row[f"skipped_{key}"] = val
        rows.append(row)

    if not rows:
        raise FileNotFoundError("no selected long_H120 episode summary files found")
    out = pd.DataFrame(rows)
    out["episode_robust_score"] = out.apply(_episode_score, axis=1)
    sort_cols = ["episode_robust_score", "mean_net_pl_bps_sum", "mean_avg_net_pl_bps"]
    existing_sort_cols = [c for c in sort_cols if c in out.columns]
    out = out.sort_values(existing_sort_cols, ascending=[False] * len(existing_sort_cols))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    best_cols = [c for c in ["run_id", "selected_key", "exit_lookahead_minutes", "exit_feature_set", "entry_threshold_bps", "episode_robust_score", "mean_avg_net_pl_bps", "worst_avg_net_pl_bps", "episode_count"] if c in out.columns]
    best = out.iloc[0][best_cols].to_dict()
    print(json.dumps({"output": str(args.output), "rows": int(len(out)), "missing": missing, "best": best}, indent=2, ensure_ascii=False))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

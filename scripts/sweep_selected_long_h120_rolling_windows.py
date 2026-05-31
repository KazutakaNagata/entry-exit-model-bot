#!/usr/bin/env python3
"""Sweep leak-safe rolling score windows for selected long_H120 fixed-hold runs.

This script deliberately does *not* use fold-wide top quantiles.  Every run it
creates uses ``backtest_fixed_hold_entry_valid.py --entry-selection-mode
rolling_quantile``, where the rolling threshold is computed from past scores
only.  The goal is to test whether shorter score distribution windows work
better than or longer than the earlier 60-day default.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import write_json  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402
from swing_bot.pipeline.rolling_window_sweep import (  # noqa: E402
    parse_csv_floats,
    parse_csv_ints,
    resolve_min_periods,
    threshold_slug,
)
from swing_bot.pipeline.selected_long_h120_downstream import (  # noqa: E402
    parse_csv_values,
    selected_entry_oof_path,
)

DEFAULT_OHLCV = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_ENTRY_OOF_ROOT = outputs_dir() / "valid" / "entry_oof"
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_PLAN_OUTPUT = outputs_dir() / "valid" / "episode_backtest" / "selected_long_H120_window_sweep_plan.json"

DEFAULT_CANDIDATES = ["long_H120_tail_v0"]
DEFAULT_WINDOWS = [14, 30, 45, 60, 90, 120]
DEFAULT_QUANTILES = [0.99, 0.995]
DEFAULT_FLOORS = [30.0, 40.0, 50.0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep live-safe rolling score window lengths for selected long_H120 fixed-hold entry.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--entry-oof-root", type=Path, default=DEFAULT_ENTRY_OOF_ROOT)
    parser.add_argument("--score-history", type=Path, default=None, help="Optional past score history for rolling quantile warmup.")
    parser.add_argument("--entry-grid-prefix", type=str, default="selected_long_H120")
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES), help="Comma-separated candidates. Default: long_H120_tail_v0")
    parser.add_argument("--window-days", type=str, default=",".join(str(x) for x in DEFAULT_WINDOWS), help="Comma-separated rolling windows in days.")
    parser.add_argument("--quantiles", type=str, default=",".join(str(x) for x in DEFAULT_QUANTILES), help="Comma-separated rolling quantiles, e.g. 0.99,0.995")
    parser.add_argument("--floors", type=str, default=",".join(f"{x:g}" for x in DEFAULT_FLOORS), help="Comma-separated absolute score floors in bps.")
    parser.add_argument("--rolling-score-min-periods", type=str, default="auto", help="Integer or 'auto'. Auto uses about half of each window with floor/cap.")
    parser.add_argument("--rolling-history-mode", choices=["fold_local", "continuous_valid", "score_history"], default="fold_local", help="Past-score history source for rolling_quantile mode.")
    parser.add_argument("--fixed-hold-minutes", type=int, default=120)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)
    parser.add_argument("--min-entry-pred-bps", type=float, default=None, help="Optional additional score-strength filter applied to all runs.")
    parser.add_argument("--min-score-margin-bps", type=float, default=None, help="Optional additional margin filter applied to all runs.")
    parser.add_argument("--min-score-ratio", type=float, default=None, help="Optional additional score/effective-threshold ratio filter applied to all runs.")
    parser.add_argument("--run-prefix", type=str, default="selected_long_H120_window_sweep")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists() and not (path.name.endswith(".parquet") and path.with_suffix(".csv").exists()):
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    if args.score_history is not None:
        _require_file(args.score_history, hint="Provide score history for rolling quantile warmup, or omit this option.")

    candidates = parse_csv_values(args.candidates, default=DEFAULT_CANDIDATES)
    windows = parse_csv_ints(args.window_days, default=DEFAULT_WINDOWS)
    quantiles = parse_csv_floats(args.quantiles, default=DEFAULT_QUANTILES)
    floors = parse_csv_floats(args.floors, default=DEFAULT_FLOORS)

    jobs: list[dict[str, object]] = []
    for candidate in candidates:
        entry_oof = selected_entry_oof_path(
            candidate=candidate,
            entry_oof_root=args.entry_oof_root,
            entry_grid_prefix=args.entry_grid_prefix,
        )
        _require_file(entry_oof, hint=f"Run scripts/run_selected_long_h120_entry.py --candidates {candidate} first.")
        for window_days in windows:
            min_periods = resolve_min_periods(args.rolling_score_min_periods, window_days)
            for quantile in quantiles:
                for floor in floors:
                    run_id = (
                        f"{args.run_prefix}_{candidate}_hold{int(args.fixed_hold_minutes)}_"
                        f"rollQ{threshold_slug(quantile)}_win{int(window_days)}d_"
                        f"{args.rolling_history_mode}_minP{int(min_periods)}_floor{threshold_slug(floor)}"
                    )
                    if args.min_entry_pred_bps is not None:
                        run_id += f"_minPred{threshold_slug(args.min_entry_pred_bps)}"
                    if args.min_score_margin_bps is not None:
                        run_id += f"_minMargin{threshold_slug(args.min_score_margin_bps)}"
                    if args.min_score_ratio is not None:
                        run_id += f"_minRatio{threshold_slug(args.min_score_ratio)}"

                    cmd = [
                        sys.executable,
                        "scripts/backtest_fixed_hold_entry_valid.py",
                        "--ohlcv", str(args.ohlcv),
                        "--entry-oof", str(entry_oof),
                        *( ["--score-history", str(args.score_history)] if args.score_history is not None else [] ),
                        "--side", "long",
                        "--entry-horizon", "120",
                        "--fixed-hold-minutes", str(int(args.fixed_hold_minutes)),
                        "--entry-selection-mode", "rolling_quantile",
                        "--rolling-score-window-days", str(int(window_days)),
                        "--rolling-score-quantile", str(float(quantile)),
                        "--rolling-score-min-periods", str(int(min_periods)),
                        "--rolling-history-mode", str(args.rolling_history_mode),
                        "--entry-threshold-bps", str(float(floor)),
                        "--entry-score-floor-bps", str(float(floor)),
                        "--roundtrip-cost-bps", str(float(args.roundtrip_cost_bps)),
                        "--decision-interval-minutes", str(int(args.decision_interval_minutes)),
                        "--cooldown-after-exit-minutes", str(int(args.cooldown_after_exit_minutes)),
                        "--same-side-reentry-block-minutes", str(int(args.same_side_reentry_block_minutes)),
                        "--run-id", run_id,
                        "--output-root", str(args.output_root),
                    ]
                    if args.max_round_trips_per_day is not None:
                        cmd.extend(["--max-round-trips-per-day", str(int(args.max_round_trips_per_day))])
                    if args.min_entry_pred_bps is not None:
                        cmd.extend(["--min-entry-pred-bps", str(float(args.min_entry_pred_bps))])
                    if args.min_score_margin_bps is not None:
                        cmd.extend(["--min-score-margin-bps", str(float(args.min_score_margin_bps))])
                    if args.min_score_ratio is not None:
                        cmd.extend(["--min-score-ratio", str(float(args.min_score_ratio))])

                    jobs.append({
                        "candidate": candidate,
                        "entry_oof": str(entry_oof),
                        "score_history": str(args.score_history) if args.score_history is not None else None,
                        "entry_selection_mode": "rolling_quantile",
                        "rolling_score_window_days": int(window_days),
                        "rolling_score_quantile": float(quantile),
                        "rolling_score_min_periods": int(min_periods),
                        "rolling_history_mode": args.rolling_history_mode,
                        "entry_score_floor_bps": float(floor),
                        "fixed_hold_minutes": int(args.fixed_hold_minutes),
                        "roundtrip_cost_bps": float(args.roundtrip_cost_bps),
                        "min_entry_pred_bps": args.min_entry_pred_bps,
                        "min_score_margin_bps": args.min_score_margin_bps,
                        "min_score_ratio": args.min_score_ratio,
                        "run_id": run_id,
                        "run_dir": str(args.output_root / run_id),
                        "command": cmd,
                    })

    plan = {
        "role": "selected_long_H120_rolling_window_sweep",
        "run_prefix": args.run_prefix,
        "job_count": len(jobs),
        "jobs": jobs,
        "notes": [
            "Valid-only fixed-hold sweep for rolling quantile window length.",
            "No fold-wide top quantile entry rules are used; thresholds are computed from past scores only.",
            "Default candidate is long_H120_tail_v0 because it was the strongest fixed-hold candidate so far.",
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return 0
    write_json(plan, args.plan_output)
    for job in jobs:
        _run(job["command"], dry_run=False)  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

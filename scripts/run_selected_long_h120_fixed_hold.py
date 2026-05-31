#!/usr/bin/env python3
"""Run fixed-hold baselines for selected long_H120 entry candidates."""
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
from swing_bot.pipeline.selected_long_h120_downstream import (  # noqa: E402
    DEFAULT_CANDIDATES,
    parse_csv_values,
    parse_candidate_thresholds,
    selected_entry_oof_path,
    threshold_for_candidate,
)

DEFAULT_OHLCV = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_ENTRY_OOF_ROOT = outputs_dir() / "valid" / "entry_oof"
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_PLAN_OUTPUT = outputs_dir() / "valid" / "episode_backtest" / "selected_long_H120_fixed_hold_plan.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected long_H120 fixed-hold valid baselines.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--entry-oof-root", type=Path, default=DEFAULT_ENTRY_OOF_ROOT)
    parser.add_argument("--score-history", type=Path, default=None, help="Optional score-history file passed to each fixed-hold backtest for rolling quantile warmup.")
    parser.add_argument("--entry-grid-prefix", type=str, default="selected_long_H120")
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--candidate-thresholds", type=str, default=None, help="candidate=value pairs, e.g. long_H120_v0=20,long_H120_tail_v0=25")
    parser.add_argument("--default-entry-threshold-bps", type=float, default=20.0)
    parser.add_argument("--entry-selection-mode", choices=["threshold", "rolling_quantile"], default="threshold")
    parser.add_argument("--rolling-score-window-days", type=int, default=60)
    parser.add_argument("--rolling-score-quantile", type=float, default=0.99)
    parser.add_argument("--rolling-score-min-periods", type=int, default=1000)
    parser.add_argument("--rolling-history-mode", choices=["fold_local", "continuous_valid", "score_history"], default="fold_local", help="Past-score history source for rolling_quantile mode.")
    parser.add_argument("--entry-score-floor-bps", type=float, default=None, help="Floor used only for rolling_quantile mode. Defaults to each candidate threshold.")
    parser.add_argument("--min-entry-pred-bps", type=float, default=None)
    parser.add_argument("--min-score-margin-bps", type=float, default=None)
    parser.add_argument("--min-score-ratio", type=float, default=None)
    parser.add_argument("--fixed-hold-minutes", type=int, default=120)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)
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


def _threshold_slug(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _quantile_slug(value: float) -> str:
    return _threshold_slug(float(value))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    if args.score_history is not None:
        _require_file(args.score_history, hint="Provide score history for rolling quantile warmup, or omit this option.")
    candidates = parse_csv_values(args.candidates, default=DEFAULT_CANDIDATES)
    explicit = parse_candidate_thresholds(args.candidate_thresholds, default_threshold=args.default_entry_threshold_bps)

    jobs: list[dict[str, object]] = []
    for candidate in candidates:
        entry_oof = selected_entry_oof_path(
            candidate=candidate,
            entry_oof_root=args.entry_oof_root,
            entry_grid_prefix=args.entry_grid_prefix,
        )
        _require_file(entry_oof, hint=f"Run scripts/run_selected_long_h120_entry.py --candidates {candidate} first.")
        threshold = threshold_for_candidate(
            candidate,
            explicit_thresholds=explicit,
            default_threshold=args.default_entry_threshold_bps,
        )
        if args.entry_selection_mode == "rolling_quantile":
            floor = threshold if args.entry_score_floor_bps is None else float(args.entry_score_floor_bps)
            run_id = (
                f"selected_long_H120_fixed_hold_{candidate}_hold{int(args.fixed_hold_minutes)}_"
                f"rollQ{_quantile_slug(args.rolling_score_quantile)}_win{int(args.rolling_score_window_days)}d_"
                f"{args.rolling_history_mode}_floor{_threshold_slug(floor)}"
            )
        else:
            run_id = (
                f"selected_long_H120_fixed_hold_{candidate}_hold{int(args.fixed_hold_minutes)}_"
                f"entryThr{_threshold_slug(threshold)}"
            )
        if args.min_entry_pred_bps is not None:
            run_id += f"_minPred{_threshold_slug(args.min_entry_pred_bps)}"
        if args.min_score_margin_bps is not None:
            run_id += f"_minMargin{_threshold_slug(args.min_score_margin_bps)}"
        if args.min_score_ratio is not None:
            run_id += f"_minRatio{_threshold_slug(args.min_score_ratio)}"
        cmd = [
            sys.executable,
            "scripts/backtest_fixed_hold_entry_valid.py",
            "--ohlcv", str(args.ohlcv),
            "--entry-oof", str(entry_oof),
            *(["--score-history", str(args.score_history)] if args.score_history is not None else []),
            "--side", "long",
            "--entry-horizon", "120",
            "--fixed-hold-minutes", str(int(args.fixed_hold_minutes)),
            "--entry-threshold-bps", str(float(threshold)),
            "--entry-selection-mode", str(args.entry_selection_mode),
            "--rolling-score-window-days", str(int(args.rolling_score_window_days)),
            "--rolling-score-quantile", str(float(args.rolling_score_quantile)),
            "--rolling-score-min-periods", str(int(args.rolling_score_min_periods)),
            "--rolling-history-mode", str(args.rolling_history_mode),
            "--roundtrip-cost-bps", str(float(args.roundtrip_cost_bps)),
            "--decision-interval-minutes", str(int(args.decision_interval_minutes)),
            "--cooldown-after-exit-minutes", str(int(args.cooldown_after_exit_minutes)),
            "--same-side-reentry-block-minutes", str(int(args.same_side_reentry_block_minutes)),
            "--run-id", run_id,
            "--output-root", str(args.output_root),
        ]
        if args.entry_score_floor_bps is not None:
            cmd.extend(["--entry-score-floor-bps", str(float(args.entry_score_floor_bps))])
        if args.min_entry_pred_bps is not None:
            cmd.extend(["--min-entry-pred-bps", str(float(args.min_entry_pred_bps))])
        if args.min_score_margin_bps is not None:
            cmd.extend(["--min-score-margin-bps", str(float(args.min_score_margin_bps))])
        if args.min_score_ratio is not None:
            cmd.extend(["--min-score-ratio", str(float(args.min_score_ratio))])
        if args.max_round_trips_per_day is not None:
            cmd.extend(["--max-round-trips-per-day", str(int(args.max_round_trips_per_day))])
        jobs.append({
            "candidate": candidate,
            "entry_oof": str(entry_oof),
            "score_history": str(args.score_history) if args.score_history is not None else None,
            "entry_threshold_bps": threshold,
            "entry_selection_mode": args.entry_selection_mode,
            "rolling_score_window_days": int(args.rolling_score_window_days),
            "rolling_score_quantile": float(args.rolling_score_quantile),
            "rolling_score_min_periods": int(args.rolling_score_min_periods),
            "rolling_history_mode": args.rolling_history_mode,
            "entry_score_floor_bps": args.entry_score_floor_bps,
            "min_entry_pred_bps": args.min_entry_pred_bps,
            "min_score_margin_bps": args.min_score_margin_bps,
            "min_score_ratio": args.min_score_ratio,
            "fixed_hold_minutes": int(args.fixed_hold_minutes),
            "run_id": run_id,
            "run_dir": str(args.output_root / run_id),
            "command": cmd,
        })

    plan = {
        "role": "selected_long_H120_fixed_hold_baseline",
        "candidate_count": len(candidates),
        "jobs": jobs,
        "notes": [
            "Valid-only fixed-hold baseline. No exit model is used.",
            "This checks whether the H120 entry edge survives episode state/cooldown/cost accounting.",
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

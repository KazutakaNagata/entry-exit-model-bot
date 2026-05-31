#!/usr/bin/env python3
"""Sweep live-safe score-strength filters for selected long_H120 fixed-hold entries.

This script deliberately keeps the base entry rule narrow and explicit:

    candidate = long_H120_tail_v0
    entry selection = rolling_quantile
    rolling window = 60d
    rolling quantile = 0.995
    score floor = 40bps
    fixed hold = 120m

It then adds one extra current-score filter at a time:

* min_entry_pred_bps
* min_score_margin_bps = score - effective_threshold
* min_score_ratio = score / effective_threshold

All filters are live-feasible: they use only the current score and the past-only
rolling threshold already computed by the fixed-hold backtester.
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
from swing_bot.pipeline.selected_long_h120_downstream import selected_entry_oof_path  # noqa: E402

DEFAULT_OHLCV = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_ENTRY_OOF_ROOT = outputs_dir() / "valid" / "entry_oof"
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_PLAN_OUTPUT = outputs_dir() / "valid" / "episode_backtest" / "selected_long_H120_score_filter_sweep_plan.json"


def _parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists() and not (path.name.endswith(".parquet") and path.with_suffix(".csv").exists()):
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep score-strength filters for selected long_H120 tail fixed-hold strategy.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--entry-oof-root", type=Path, default=DEFAULT_ENTRY_OOF_ROOT)
    parser.add_argument("--entry-grid-prefix", type=str, default="selected_long_H120")
    parser.add_argument("--candidate", type=str, default="long_H120_tail_v0")
    parser.add_argument("--score-history", type=Path, default=None, help="Optional score history for rolling quantile warmup.")
    parser.add_argument("--fixed-hold-minutes", type=int, default=120)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--rolling-score-window-days", type=int, default=60)
    parser.add_argument("--rolling-score-quantile", type=float, default=0.995)
    parser.add_argument("--rolling-score-min-periods", type=int, default=1000)
    parser.add_argument("--rolling-history-mode", choices=["fold_local", "continuous_valid", "score_history"], default="fold_local", help="Past-score history source for rolling_quantile mode.")
    parser.add_argument("--entry-score-floor-bps", type=float, default=40.0)
    parser.add_argument("--min-entry-pred-values", type=str, default="60,70", help="Comma list. Empty string disables this filter family.")
    parser.add_argument("--min-score-margin-values", type=str, default="10,20,30,40", help="Comma list. Empty string disables this filter family.")
    parser.add_argument("--min-score-ratio-values", type=str, default="1.25,1.5,2.0", help="Comma list. Empty string disables this filter family.")
    parser.add_argument("--include-base", action="store_true", default=True)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _base_cmd(args: argparse.Namespace, entry_oof: Path, run_id: str) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/backtest_fixed_hold_entry_valid.py",
        "--ohlcv", str(args.ohlcv),
        "--entry-oof", str(entry_oof),
        "--side", "long",
        "--entry-horizon", "120",
        "--fixed-hold-minutes", str(int(args.fixed_hold_minutes)),
        "--entry-selection-mode", "rolling_quantile",
        "--entry-threshold-bps", str(float(args.entry_score_floor_bps)),
        "--entry-score-floor-bps", str(float(args.entry_score_floor_bps)),
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
    if args.score_history is not None:
        cmd.extend(["--score-history", str(args.score_history)])
    if args.max_round_trips_per_day is not None:
        cmd.extend(["--max-round-trips-per-day", str(int(args.max_round_trips_per_day))])
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    if args.score_history is not None:
        _require_file(args.score_history, hint="Provide score history for rolling warmup, or omit this option.")
    entry_oof = selected_entry_oof_path(
        candidate=args.candidate,
        entry_oof_root=args.entry_oof_root,
        entry_grid_prefix=args.entry_grid_prefix,
    )
    _require_file(entry_oof, hint=f"Run scripts/run_selected_long_h120_entry.py --candidates {args.candidate} first.")

    jobs: list[dict[str, object]] = []
    qslug = _slug(float(args.rolling_score_quantile))
    floor_slug = _slug(float(args.entry_score_floor_bps))
    prefix = f"selected_long_H120_score_filter_sweep_{args.candidate}_hold{int(args.fixed_hold_minutes)}_rollQ{qslug}_win{int(args.rolling_score_window_days)}d_{args.rolling_history_mode}_floor{floor_slug}"

    if args.include_base:
        run_id = prefix + "_base"
        jobs.append({"kind": "base", "run_id": run_id, "run_dir": str(args.output_root / run_id), "command": _base_cmd(args, entry_oof, run_id)})

    for value in _parse_float_list(args.min_entry_pred_values) if str(args.min_entry_pred_values).strip() else []:
        run_id = prefix + f"_minPred{_slug(value)}"
        cmd = _base_cmd(args, entry_oof, run_id) + ["--min-entry-pred-bps", str(float(value))]
        jobs.append({"kind": "min_entry_pred", "value": float(value), "run_id": run_id, "run_dir": str(args.output_root / run_id), "command": cmd})

    for value in _parse_float_list(args.min_score_margin_values) if str(args.min_score_margin_values).strip() else []:
        run_id = prefix + f"_minMargin{_slug(value)}"
        cmd = _base_cmd(args, entry_oof, run_id) + ["--min-score-margin-bps", str(float(value))]
        jobs.append({"kind": "min_score_margin", "value": float(value), "run_id": run_id, "run_dir": str(args.output_root / run_id), "command": cmd})

    for value in _parse_float_list(args.min_score_ratio_values) if str(args.min_score_ratio_values).strip() else []:
        run_id = prefix + f"_minRatio{_slug(value)}"
        cmd = _base_cmd(args, entry_oof, run_id) + ["--min-score-ratio", str(float(value))]
        jobs.append({"kind": "min_score_ratio", "value": float(value), "run_id": run_id, "run_dir": str(args.output_root / run_id), "command": cmd})

    plan = {
        "role": "selected_long_H120_score_strength_filter_sweep",
        "job_count": len(jobs),
        "candidate": args.candidate,
        "base_rule": {
            "entry_selection_mode": "rolling_quantile",
            "rolling_score_window_days": int(args.rolling_score_window_days),
            "rolling_score_quantile": float(args.rolling_score_quantile),
            "entry_score_floor_bps": float(args.entry_score_floor_bps),
            "fixed_hold_minutes": int(args.fixed_hold_minutes),
        },
        "score_history": str(args.score_history) if args.score_history is not None else None,
        "jobs": jobs,
        "notes": [
            "All filters are live-safe: current score, past-only rolling threshold, or both.",
            "No fold-wide future score quantiles are used.",
            "Valid only; do not inspect test until a locked config exists.",
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

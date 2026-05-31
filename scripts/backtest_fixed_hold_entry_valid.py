#!/usr/bin/env python3
"""Backtest OOF entry predictions with a fixed-hold exit on valid folds only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.config_snapshot import snapshot_paths  # noqa: E402
from swing_bot.artifacts.io import read_frame, write_json  # noqa: E402
from swing_bot.artifacts.run_id import make_run_id, safe_slug  # noqa: E402
from swing_bot.backtest.fixed_hold import FixedHoldBacktestConfig, run_fixed_hold_backtest  # noqa: E402
from swing_bot.backtest.policy import EpisodePolicyConfig  # noqa: E402
from swing_bot.evaluation.episode_report import write_episode_report  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-hold valid-fold entry backtest from OOF predictions.")
    parser.add_argument("--ohlcv", type=Path, required=True, help="Canonical 1m OHLCV parquet/csv.")
    parser.add_argument("--entry-oof", type=Path, required=True, help="Entry OOF predictions parquet/csv.")
    parser.add_argument("--score-history", type=Path, default=None, help="Optional past score history parquet/csv used only to warm rolling quantile thresholds. Rows are not tradable entries.")
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--entry-horizon", type=int, default=120, help="Entry horizon minutes matching OOF predictions.")
    parser.add_argument("--fixed-hold-minutes", type=int, default=None, help="Hold from entry execution open to exit open. Defaults to entry horizon.")
    parser.add_argument("--entry-threshold-bps", type=float, default=20.0, help="Fixed absolute score threshold, or floor when rolling_quantile is used.")
    parser.add_argument("--entry-selection-mode", choices=["threshold", "rolling_quantile"], default="threshold")
    parser.add_argument("--rolling-score-window-days", type=int, default=60, help="Past-only score window for rolling_quantile mode.")
    parser.add_argument("--rolling-score-quantile", type=float, default=0.99, help="Quantile in (0,1) for rolling_quantile mode.")
    parser.add_argument("--rolling-score-min-periods", type=int, default=1000, help="Minimum past scores before rolling_quantile entries are allowed.")
    parser.add_argument("--rolling-history-mode", choices=["fold_local", "continuous_valid", "score_history"], default="fold_local", help="Past-score history source for rolling_quantile mode.")
    parser.add_argument("--entry-score-floor-bps", type=float, default=None, help="Absolute floor for rolling_quantile mode. Defaults to --entry-threshold-bps.")
    parser.add_argument("--min-entry-pred-bps", type=float, default=None, help="Optional live-safe filter: require current entry score to be at least this value.")
    parser.add_argument("--min-score-margin-bps", type=float, default=None, help="Optional live-safe filter: require entry score minus effective threshold to be at least this value.")
    parser.add_argument("--min-score-ratio", type=float, default=None, help="Optional live-safe filter: require entry score / effective threshold to be at least this value.")
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--decision-interval-minutes", type=int, default=5, help="Kept for policy consistency; entry OOF may be denser.")
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists() and not (path.name.endswith(".parquet") and path.with_suffix(".csv").exists()):
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def _default_run_id(args: argparse.Namespace, fixed_hold: int) -> str:
    if args.entry_selection_mode == "rolling_quantile":
        floor = args.entry_threshold_bps if args.entry_score_floor_bps is None else args.entry_score_floor_bps
        prefix = (
            f"fixed_hold_{args.side}_entryH{args.entry_horizon}_hold{fixed_hold}_"
            f"rollQ{args.rolling_score_quantile:g}_win{args.rolling_score_window_days}d_{args.rolling_history_mode}_floor{floor:g}"
        )
    else:
        prefix = f"fixed_hold_{args.side}_entryH{args.entry_horizon}_hold{fixed_hold}_entryThr{args.entry_threshold_bps:g}"
    if args.min_entry_pred_bps is not None:
        prefix += f"_minPred{args.min_entry_pred_bps:g}"
    if args.min_score_margin_bps is not None:
        prefix += f"_minMargin{args.min_score_margin_bps:g}"
    if args.min_score_ratio is not None:
        prefix += f"_minRatio{args.min_score_ratio:g}"
    return make_run_id(safe_slug(prefix))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fixed_hold = int(args.fixed_hold_minutes if args.fixed_hold_minutes is not None else args.entry_horizon)
    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    _require_file(args.entry_oof, hint="Run entry training/grid first to create OOF predictions.")
    if args.score_history is not None:
        _require_file(args.score_history, hint="Provide a score history file with timestamp/side/horizon_minutes/pred_entry_net_bps for rolling warmup.")

    policy = EpisodePolicyConfig(
        entry_threshold_bps=args.entry_threshold_bps,
        hold_threshold_bps=0.0,
        roundtrip_cost_bps=args.roundtrip_cost_bps,
        max_hold_minutes=fixed_hold,
        decision_interval_minutes=args.decision_interval_minutes,
        cooldown_after_exit_minutes=args.cooldown_after_exit_minutes,
        same_side_reentry_block_minutes=args.same_side_reentry_block_minutes,
        max_round_trips_per_day=args.max_round_trips_per_day,
    )
    cfg = FixedHoldBacktestConfig(
        side=args.side,
        entry_horizon_minutes=args.entry_horizon,
        fixed_hold_minutes=fixed_hold,
        policy=policy,
        entry_selection_mode=args.entry_selection_mode,
        rolling_score_window_days=args.rolling_score_window_days,
        rolling_score_quantile=args.rolling_score_quantile,
        rolling_score_min_periods=args.rolling_score_min_periods,
        rolling_history_mode=args.rolling_history_mode,
        entry_score_floor_bps=args.entry_score_floor_bps,
        min_entry_pred_bps=args.min_entry_pred_bps,
        min_score_margin_bps=args.min_score_margin_bps,
        min_score_ratio=args.min_score_ratio,
    )
    input_summary = {
        "ohlcv": str(args.ohlcv),
        "entry_oof": str(args.entry_oof),
        "score_history": str(args.score_history) if args.score_history is not None else None,
        "side": args.side,
        "entry_horizon_minutes": int(args.entry_horizon),
        "fixed_hold_minutes": int(fixed_hold),
        "policy": policy.__dict__,
        "entry_selection_mode": args.entry_selection_mode,
        "rolling_score_window_days": int(args.rolling_score_window_days),
        "rolling_score_quantile": float(args.rolling_score_quantile),
        "rolling_score_min_periods": int(args.rolling_score_min_periods),
        "rolling_history_mode": args.rolling_history_mode,
        "entry_score_floor_bps": args.entry_score_floor_bps,
        "min_entry_pred_bps": args.min_entry_pred_bps,
        "min_score_margin_bps": args.min_score_margin_bps,
        "min_score_ratio": args.min_score_ratio,
        "input_snapshots": snapshot_paths({
            "ohlcv": args.ohlcv,
            "entry_oof": args.entry_oof,
            **({"score_history": args.score_history} if args.score_history is not None else {}),
        }),
    }
    if args.dry_run:
        print(json.dumps(input_summary, indent=2, ensure_ascii=False, default=str))
        return 0

    ohlcv = read_frame(args.ohlcv)
    entry_oof = read_frame(args.entry_oof)
    score_history = read_frame(args.score_history) if args.score_history is not None else None
    episodes, fold_metrics, summary = run_fixed_hold_backtest(
        ohlcv=ohlcv,
        entry_oof=entry_oof,
        config=cfg,
        score_history=score_history,
    )

    run_id = safe_slug(args.run_id) if args.run_id else _default_run_id(args, fixed_hold)
    run_dir = args.output_root / run_id
    write_episode_report(episodes=episodes, fold_metrics=fold_metrics, summary=summary, output_dir=run_dir)
    run_config = {
        "run_id": run_id,
        "model_role": "fixed_hold_entry_backtest",
        "exit_mode": "fixed_hold",
        "notes": [
            "Valid-fold fixed-hold baseline. No exit model predictions are used.",
            "Do not tune on test; compare on valid only and lock before any test audit.",
        ],
        **input_summary,
    }
    write_json(run_config, run_dir / "run_config.json")
    print(json.dumps({
        "run_id": run_id,
        "run_dir": str(run_dir),
        "episode_count": int(len(episodes)),
        "summary": summary,
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

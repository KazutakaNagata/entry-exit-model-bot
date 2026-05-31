#!/usr/bin/env python3
"""Run a valid-fold episode backtest from OOF entry and exit predictions.

This is a plumbing MVP.  It does not train models, does not read test data, and
should not be used for strategy selection before stronger feature sets are in
place.
"""
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
from swing_bot.backtest.episode import EpisodeBacktestConfig, run_episode_backtest  # noqa: E402
from swing_bot.backtest.policy import EpisodePolicyConfig  # noqa: E402
from swing_bot.evaluation.episode_report import write_episode_report  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest entry+exit OOF predictions on valid folds only.")
    parser.add_argument("--ohlcv", type=Path, required=True, help="Canonical 1m OHLCV parquet/csv.")
    parser.add_argument("--entry-oof", type=Path, required=True, help="Aggregated entry OOF predictions parquet/csv.")
    parser.add_argument("--exit-predictions", type=Path, required=True, help="Exit model OOF predictions parquet/csv.")
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--entry-horizon", type=int, default=60, help="Entry horizon minutes matching the OOF predictions.")
    parser.add_argument("--entry-threshold-bps", type=float, default=20.0)
    parser.add_argument("--hold-threshold-bps", type=float, default=0.0)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--max-hold-minutes", type=int, default=240)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--exit-confirm-bars", type=int, default=1)
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--hard-stop-bps", type=float, default=None)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Load inputs and print config only; do not write artifacts.")
    return parser.parse_args(argv)


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def _default_run_id(args: argparse.Namespace) -> str:
    prefix = f"episode_{args.side}_entryH{args.entry_horizon}_entryThr{args.entry_threshold_bps:g}_holdThr{args.hold_threshold_bps:g}"
    return make_run_id(safe_slug(prefix))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    _require_file(args.entry_oof, hint="Run scripts/train_entry_grid.py or collect_entry_oof_predictions.py first.")
    _require_file(args.exit_predictions, hint="Run scripts/train_exit_lgbm.py first and pass predictions_valid.parquet/csv.")

    policy = EpisodePolicyConfig(
        entry_threshold_bps=args.entry_threshold_bps,
        hold_threshold_bps=args.hold_threshold_bps,
        roundtrip_cost_bps=args.roundtrip_cost_bps,
        max_hold_minutes=args.max_hold_minutes,
        decision_interval_minutes=args.decision_interval_minutes,
        exit_confirm_bars=args.exit_confirm_bars,
        cooldown_after_exit_minutes=args.cooldown_after_exit_minutes,
        same_side_reentry_block_minutes=args.same_side_reentry_block_minutes,
        hard_stop_bps=args.hard_stop_bps,
        max_round_trips_per_day=args.max_round_trips_per_day,
    )
    cfg = EpisodeBacktestConfig(
        side=args.side,
        entry_horizon_minutes=args.entry_horizon,
        policy=policy,
    )
    input_summary = {
        "ohlcv": str(args.ohlcv),
        "entry_oof": str(args.entry_oof),
        "exit_predictions": str(args.exit_predictions),
        "side": args.side,
        "entry_horizon_minutes": int(args.entry_horizon),
        "policy": policy.__dict__,
        "input_snapshots": snapshot_paths({
            "ohlcv": args.ohlcv,
            "entry_oof": args.entry_oof,
            "exit_predictions": args.exit_predictions,
        }),
    }
    if args.dry_run:
        print(json.dumps(input_summary, indent=2, ensure_ascii=False, default=str))
        return 0

    ohlcv = read_frame(args.ohlcv)
    entry_oof = read_frame(args.entry_oof)
    exit_predictions = read_frame(args.exit_predictions)
    episodes, fold_metrics, summary = run_episode_backtest(
        ohlcv=ohlcv,
        entry_oof=entry_oof,
        exit_predictions=exit_predictions,
        config=cfg,
    )

    run_id = safe_slug(args.run_id) if args.run_id else _default_run_id(args)
    run_dir = args.output_root / run_id
    write_episode_report(episodes=episodes, fold_metrics=fold_metrics, summary=summary, output_dir=run_dir)
    run_config = {
        "run_id": run_id,
        "model_role": "episode_backtest",
        "notes": [
            "Valid-fold plumbing backtest only; do not interpret performance before feature expansion.",
            "Inputs must be OOF entry/exit predictions.",
            "Test data is not read by this script.",
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

#!/usr/bin/env python3
"""Run downstream validation for selected long_H120 entry candidates.

This script orchestrates existing reviewed CLIs:

1. make_exit_dataset.py
2. train_exit_lgbm.py
3. backtest_episode_valid.py

It does not build features, train entry models, tune thresholds, or read test
periods.  Entry OOF predictions must already exist from
``scripts/run_selected_long_h120_entry.py``.
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

from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402
from swing_bot.pipeline.selected_long_h120_downstream import (  # noqa: E402
    DEFAULT_CANDIDATES,
    DEFAULT_EXIT_FEATURE_SETS,
    DEFAULT_EXIT_LOOKAHEADS,
    existing_table_path,
    make_plans,
    parse_candidate_thresholds,
    parse_csv_values,
    parse_int_csv_values,
)

DEFAULT_OHLCV = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_FEATURE_ROOT = btcjpy_1m_processed_dir() / "feature_selected" / "long_H120"
DEFAULT_ENTRY_OOF_ROOT = outputs_dir() / "valid" / "entry_oof"
DEFAULT_EXIT_DATASET_ROOT = outputs_dir() / "valid" / "exit_dataset"
DEFAULT_EXIT_OUTPUT_ROOT = outputs_dir() / "valid" / "exit_lgbm"
DEFAULT_EPISODE_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_MODEL_CONFIG = Path("configs/models/lgbm_exit_v0.yaml")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _require_file(path: Path, *, hint: str) -> None:
    if not existing_table_path(path).exists():
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected long_H120 entry candidates through exit/episode plumbing.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--entry-oof-root", type=Path, default=DEFAULT_ENTRY_OOF_ROOT)
    parser.add_argument("--entry-grid-prefix", type=str, default="selected_long_H120")
    parser.add_argument("--exit-dataset-root", type=Path, default=DEFAULT_EXIT_DATASET_ROOT)
    parser.add_argument("--exit-output-root", type=Path, default=DEFAULT_EXIT_OUTPUT_ROOT)
    parser.add_argument("--episode-output-root", type=Path, default=DEFAULT_EPISODE_OUTPUT_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)

    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--exit-lookaheads", type=str, default=",".join(str(x) for x in DEFAULT_EXIT_LOOKAHEADS))
    parser.add_argument("--exit-feature-sets", type=str, default=",".join(DEFAULT_EXIT_FEATURE_SETS))
    parser.add_argument(
        "--exit-market-features",
        choices=["none", "all"],
        default="none",
        help=(
            "Market features to materialize into the exit dataset. "
            "Default 'none' uses only score/position-state features and avoids "
            "expanding hundreds of entry features across every position row."
        ),
    )
    parser.add_argument("--candidate-entry-thresholds", type=str, default="long_H120_v0=20,long_H120_tail_v0=25")
    parser.add_argument("--default-entry-threshold-bps", type=float, default=20.0)
    parser.add_argument("--hold-threshold-bps", type=float, default=0.0)

    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--max-hold-minutes", type=int, default=240)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--candidate-interval-minutes", type=int, default=5)
    parser.add_argument("--min-entry-pred-bps", type=float, default=0.0)
    parser.add_argument("--top-entry-quantile", type=float, default=None)
    parser.add_argument("--exit-confirm-bars", type=int, default=1)
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--hard-stop-bps", type=float, default=None)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)

    parser.add_argument("--skip-exit-dataset", action="store_true")
    parser.add_argument("--skip-exit-train", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = parse_csv_values(args.candidates, default=DEFAULT_CANDIDATES)
    exit_lookaheads = parse_int_csv_values(args.exit_lookaheads, default=DEFAULT_EXIT_LOOKAHEADS)
    exit_feature_sets = parse_csv_values(args.exit_feature_sets, default=DEFAULT_EXIT_FEATURE_SETS)
    candidate_thresholds = parse_candidate_thresholds(
        args.candidate_entry_thresholds,
        default_threshold=args.default_entry_threshold_bps,
    )

    plans = make_plans(
        candidates=candidates,
        exit_lookaheads=exit_lookaheads,
        exit_feature_sets=exit_feature_sets,
        candidate_thresholds=candidate_thresholds,
        default_entry_threshold_bps=args.default_entry_threshold_bps,
        hold_threshold_bps=args.hold_threshold_bps,
        entry_oof_root=args.entry_oof_root,
        entry_grid_prefix=args.entry_grid_prefix,
        feature_root=args.features_root,
        exit_dataset_root=args.exit_dataset_root,
        exit_output_root=args.exit_output_root,
        episode_output_root=args.episode_output_root,
    )

    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    _require_file(args.split, hint="Create a concrete split manifest first.")
    _require_file(args.model_config, hint="Exit model config should exist under configs/models/.")

    executed: list[dict[str, object]] = []
    for plan in plans:
        _require_file(plan.entry_oof_path, hint="Run scripts/run_selected_long_h120_entry.py for this candidate first.")
        if args.exit_market_features == "all":
            _require_file(plan.features_path, hint="Run scripts/run_selected_long_h120_entry.py to build selected candidate features first.")

        if not args.skip_exit_dataset:
            make_exit_cmd = [
                sys.executable,
                "scripts/make_exit_dataset.py",
                "--ohlcv", str(args.ohlcv),
                "--features", str(plan.features_path),
                "--entry-oof", str(plan.entry_oof_path),
                "--split", str(args.split),
                "--side", "long",
                "--entry-horizon", "120",
                "--exit-lookahead", str(plan.exit_lookahead_minutes),
                "--max-hold-minutes", str(args.max_hold_minutes),
                "--decision-interval-minutes", str(args.decision_interval_minutes),
                "--candidate-interval-minutes", str(args.candidate_interval_minutes),
                "--min-entry-pred-bps", str(args.min_entry_pred_bps),
                "--roundtrip-cost-bps", str(args.roundtrip_cost_bps),
                "--output-dir", str(plan.exit_dataset_dir),
            ]
            if args.exit_market_features == "none":
                make_exit_cmd += ["--no-market-features"]
            if args.top_entry_quantile is not None:
                make_exit_cmd += ["--top-entry-quantile", str(args.top_entry_quantile)]
            _run(make_exit_cmd, dry_run=args.dry_run)
            executed.append({"candidate": plan.candidate, "stage": "make_exit_dataset", "cmd": make_exit_cmd})

        if not args.skip_exit_train:
            _require_file(plan.exit_dataset_path, hint="Run make_exit_dataset stage first or pass --skip-exit-dataset only if it already exists.")
            train_exit_cmd = [
                sys.executable,
                "scripts/train_exit_lgbm.py",
                "--exit-dataset", str(plan.exit_dataset_path),
                "--model-config", str(args.model_config),
                "--feature-set", plan.exit_feature_set,
                "--run-id", plan.exit_run_id,
                "--output-root", str(args.exit_output_root),
            ]
            _run(train_exit_cmd, dry_run=args.dry_run)
            executed.append({"candidate": plan.candidate, "stage": "train_exit_lgbm", "cmd": train_exit_cmd})

        if not args.skip_backtest:
            exit_pred = existing_table_path(plan.exit_run_dir / "predictions_valid.parquet")
            _require_file(exit_pred, hint="Run train_exit_lgbm stage first or pass --skip-exit-train only if predictions already exist.")
            backtest_cmd = [
                sys.executable,
                "scripts/backtest_episode_valid.py",
                "--ohlcv", str(args.ohlcv),
                "--entry-oof", str(plan.entry_oof_path),
                "--exit-predictions", str(exit_pred),
                "--side", "long",
                "--entry-horizon", "120",
                "--entry-threshold-bps", str(plan.entry_threshold_bps),
                "--hold-threshold-bps", str(plan.hold_threshold_bps),
                "--roundtrip-cost-bps", str(args.roundtrip_cost_bps),
                "--max-hold-minutes", str(args.max_hold_minutes),
                "--decision-interval-minutes", str(args.decision_interval_minutes),
                "--exit-confirm-bars", str(args.exit_confirm_bars),
                "--cooldown-after-exit-minutes", str(args.cooldown_after_exit_minutes),
                "--same-side-reentry-block-minutes", str(args.same_side_reentry_block_minutes),
                "--run-id", plan.episode_run_id,
                "--output-root", str(args.episode_output_root),
            ]
            if args.hard_stop_bps is not None:
                backtest_cmd += ["--hard-stop-bps", str(args.hard_stop_bps)]
            if args.max_round_trips_per_day is not None:
                backtest_cmd += ["--max-round-trips-per-day", str(args.max_round_trips_per_day)]
            _run(backtest_cmd, dry_run=args.dry_run)
            executed.append({"candidate": plan.candidate, "stage": "backtest_episode_valid", "cmd": backtest_cmd})

    print(json.dumps({
        "plan_count": len(plans),
        "candidates": candidates,
        "exit_lookaheads": exit_lookaheads,
        "exit_feature_sets": exit_feature_sets,
        "executed": executed,
        "notes": [
            "This script only orchestrates valid-fold OOF downstream validation.",
            "It never builds selected entry features or trains selected entry models; run_selected_long_h120_entry.py must run first.",
            "It does not read test data or tune thresholds from test results.",
        ],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

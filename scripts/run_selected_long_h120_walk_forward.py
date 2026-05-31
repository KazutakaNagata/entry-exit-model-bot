#!/usr/bin/env python3
"""Run selected long_H120_tail_v0 walk-forward entry and fixed-hold backtest.

This is a thin orchestration script.  It keeps the selected feature set and
entry policy fixed, and varies only the model retraining schedule.
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

DEFAULT_OHLCV = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_FEATURES = btcjpy_1m_processed_dir() / "feature_selected" / "long_H120" / "features_long_H120_tail_v0.parquet"
DEFAULT_LABELS = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/models/lgbm_entry_v0.yaml")


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_modes(text: str) -> list[str]:
    out = [x.strip() for x in text.split(",") if x.strip()]
    valid = {"expanding", "rolling"}
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"unknown modes: {bad}; valid={sorted(valid)}")
    return out


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected long_H120_tail_v0 walk-forward experiments.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--modes", type=str, default="rolling", help="Comma-separated: expanding,rolling. Default is rolling to avoid memory-heavy expanding runs.")
    parser.add_argument("--rolling-train-windows", type=str, default="365", help="Days for rolling mode. Use 365,730 for a heavier comparison.")
    parser.add_argument("--score-history-days", type=int, default=120)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap: keep the most recent N train rows per fold to avoid SIGKILL on laptops.")
    parser.add_argument("--write-feature-importance", action="store_true", help="Write feature importance. Disabled by default to save memory.")
    parser.add_argument("--no-downcast-float32", action="store_true", help="Do not downcast features to float32 before training.")
    parser.add_argument("--no-force-col-wise", action="store_true", help="Do not inject LightGBM force_col_wise=true. Default keeps local memory usage lower.")
    parser.add_argument("--rolling-score-window-days", type=int, default=60)
    parser.add_argument("--rolling-score-quantile", type=float, default=0.995)
    parser.add_argument("--rolling-score-min-periods", type=int, default=30000)
    parser.add_argument("--entry-score-floor-bps", type=float, default=40.0)
    parser.add_argument("--entry-threshold-bps", type=float, default=40.0)
    parser.add_argument("--fixed-hold-minutes", type=int, default=120)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--grid-prefix", type=str, default="selected_long_H120_walk_forward")
    parser.add_argument("--only-train", action="store_true")
    parser.add_argument("--only-backtest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _wf_run_id(prefix: str, mode: str, train_window: int | None) -> str:
    if mode == "rolling":
        return f"{prefix}_rolling{int(train_window)}d"
    return f"{prefix}_expanding"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.only_train and args.only_backtest:
        raise ValueError("--only-train and --only-backtest are mutually exclusive")
    modes = _parse_modes(args.modes)
    rolling_windows = _parse_csv_ints(args.rolling_train_windows)

    commands: list[dict[str, object]] = []
    for mode in modes:
        windows = rolling_windows if mode == "rolling" else [None]
        for train_window in windows:
            run_id = _wf_run_id(args.grid_prefix, mode, train_window)
            wf_run_dir = outputs_dir() / "valid" / "entry_walk_forward" / run_id
            if not args.only_backtest:
                train_cmd = [
                    sys.executable,
                    "scripts/train_entry_walk_forward.py",
                    "--features", str(args.features),
                    "--labels", str(args.labels),
                    "--split", str(args.split),
                    "--model-config", str(args.model_config),
                    "--side", "long",
                    "--horizon", "120",
                    "--training-mode", mode,
                    "--score-history-days", str(args.score_history_days),
                    "--run-id", run_id,
                ]
                if args.max_train_rows is not None:
                    train_cmd += ["--max-train-rows", str(int(args.max_train_rows))]
                if not args.write_feature_importance:
                    train_cmd += ["--skip-feature-importance"]
                if args.no_downcast_float32:
                    train_cmd += ["--no-downcast-float32"]
                if args.no_force_col_wise:
                    train_cmd += ["--no-force-col-wise"]
                if mode == "rolling":
                    train_cmd += ["--train-window-days", str(int(train_window))]
                _run(train_cmd, dry_run=args.dry_run)
                commands.append({"stage": "train_entry_walk_forward", "run_id": run_id, "cmd": train_cmd})

            if not args.only_train:
                bt_run_id = (
                    f"{run_id}_hold{int(args.fixed_hold_minutes)}_"
                    f"rollQ{args.rolling_score_quantile:g}_win{int(args.rolling_score_window_days)}d_"
                    f"scoreHistory_floor{args.entry_score_floor_bps:g}"
                )
                backtest_cmd = [
                    sys.executable,
                    "scripts/backtest_fixed_hold_entry_valid.py",
                    "--ohlcv", str(args.ohlcv),
                    "--entry-oof", str(wf_run_dir / "entry_oof_predictions.parquet"),
                    "--score-history", str(wf_run_dir / "score_history.parquet"),
                    "--side", "long",
                    "--entry-horizon", "120",
                    "--fixed-hold-minutes", str(int(args.fixed_hold_minutes)),
                    "--entry-selection-mode", "rolling_quantile",
                    "--rolling-history-mode", "score_history",
                    "--rolling-score-window-days", str(int(args.rolling_score_window_days)),
                    "--rolling-score-quantile", str(float(args.rolling_score_quantile)),
                    "--rolling-score-min-periods", str(int(args.rolling_score_min_periods)),
                    "--entry-score-floor-bps", str(float(args.entry_score_floor_bps)),
                    "--entry-threshold-bps", str(float(args.entry_threshold_bps)),
                    "--roundtrip-cost-bps", str(float(args.roundtrip_cost_bps)),
                    "--run-id", bt_run_id,
                ]
                _run(backtest_cmd, dry_run=args.dry_run)
                commands.append({"stage": "backtest_fixed_hold", "run_id": bt_run_id, "cmd": backtest_cmd})

    print(json.dumps({
        "experiment": args.grid_prefix,
        "modes": modes,
        "rolling_train_windows": rolling_windows,
        "max_train_rows": args.max_train_rows,
        "write_feature_importance": bool(args.write_feature_importance),
        "downcast_float32": not args.no_downcast_float32,
        "force_col_wise": not args.no_force_col_wise,
        "feature_set": "long_H120_tail_v0",
        "entry_policy": {
            "mode": "rolling_quantile",
            "history_mode": "score_history",
            "window_days": args.rolling_score_window_days,
            "quantile": args.rolling_score_quantile,
            "floor_bps": args.entry_score_floor_bps,
        },
        "exit": {"mode": "fixed_hold", "hold_minutes": args.fixed_hold_minutes},
        "commands": commands,
        "notes": [
            "Feature set and entry policy are fixed; only model retraining schedule changes.",
            "Default run is rolling365 only. Use --modes expanding,rolling --rolling-train-windows 365,730 for a heavier comparison.",
            "Backtest uses score_history generated by each fold model for pre-fold rolling-threshold warmup.",
            "Test is not read or evaluated.",
        ],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

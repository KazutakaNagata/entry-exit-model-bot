#!/usr/bin/env python3
"""Train selected entry model with walk-forward retraining on valid folds only."""
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
from swing_bot.artifacts.run_id import safe_slug  # noqa: E402
from swing_bot.models.entry_walk_forward import EntryWalkForwardConfig, train_entry_walk_forward  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, models_dir, outputs_dir  # noqa: E402
from swing_bot.splits.split_manifest import load_split_manifest  # noqa: E402

DEFAULT_FEATURES = btcjpy_1m_processed_dir() / "feature_selected" / "long_H120" / "features_long_H120_tail_v0.parquet"
DEFAULT_LABELS = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/models/lgbm_entry_v0.yaml")
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_walk_forward"
DEFAULT_MODEL_ROOT = models_dir() / "entry_walk_forward"


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists() and not (path.name.endswith(".parquet") and path.with_suffix(".csv").exists()):
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward entry LGBM training. Valid folds only; no test access.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--training-mode", choices=["fixed", "expanding", "rolling"], default="expanding")
    parser.add_argument("--train-window-days", type=int, default=None, help="Required for --training-mode rolling.")
    parser.add_argument("--score-history-days", type=int, default=120, help="Past days scored by each fold model for rolling-threshold warmup.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--skip-feature-importance", action="store_true", help="Do not write feature importance. Saves memory during walk-forward smoke tests.")
    parser.add_argument("--no-downcast-float32", action="store_true", help="Keep original numeric dtypes instead of downcasting feature matrix to float32.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional memory safety cap: keep only the most recent N training rows per fold.")
    parser.add_argument("--no-force-col-wise", action="store_true", help="Do not inject LightGBM force_col_wise=true. Default keeps local memory usage lower.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _default_run_id(args: argparse.Namespace) -> str:
    suffix = args.training_mode
    if args.training_mode == "rolling":
        suffix += f"{int(args.train_window_days or 0)}d"
    return safe_slug(f"entry_wf_{args.side}_H{int(args.horizon)}_{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.training_mode == "rolling" and (args.train_window_days is None or args.train_window_days <= 0):
        raise ValueError("--train-window-days must be positive when --training-mode rolling")
    _require_file(args.features, hint="Run selected feature generation first.")
    _require_file(args.labels, hint="Run scripts/build_labels.py first.")
    _require_file(args.split, hint="Provide concrete split manifest. Do not use the template.")
    _require_file(args.model_config, hint="Model config should exist under configs/models/.")

    manifest = load_split_manifest(args.split)
    features = read_frame(args.features)
    labels = read_frame(args.labels)
    run_id = safe_slug(args.run_id) if args.run_id else _default_run_id(args)
    run_dir = args.output_root / run_id
    model_dir = args.model_root / run_id
    target_col = args.target_col or f"target_entry_net_bps_{args.side}_H{int(args.horizon)}"

    summary = {
        "run_id": run_id,
        "features_path": str(args.features),
        "labels_path": str(args.labels),
        "split": manifest.split_version,
        "side": args.side,
        "horizon_minutes": int(args.horizon),
        "target_col": target_col,
        "training_mode": args.training_mode,
        "train_window_days": args.train_window_days,
        "score_history_days": int(args.score_history_days),
        "max_train_rows": args.max_train_rows,
        "downcast_float32": not args.no_downcast_float32,
        "skip_feature_importance": bool(args.skip_feature_importance),
        "force_col_wise": not args.no_force_col_wise,
        "feature_rows": int(len(features)),
        "label_rows": int(len(labels)),
        "test_usage": manifest.test_usage,
        "notes": [
            "Walk-forward entry run only; test is untouched.",
            "Feature selection and entry thresholds are not tuned here.",
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return 0

    cfg = EntryWalkForwardConfig(
        side=args.side,
        horizon_minutes=int(args.horizon),
        target_col=args.target_col,
        training_mode=args.training_mode,
        train_window_days=args.train_window_days,
        score_history_days=int(args.score_history_days),
        save_models=not args.no_save_models,
        save_feature_importance=not args.skip_feature_importance,
        downcast_float32=not args.no_downcast_float32,
        max_train_rows=args.max_train_rows,
        force_col_wise=not args.no_force_col_wise,
    )
    metadata = {
        **summary,
        "input_snapshots": snapshot_paths({
            "features": args.features,
            "labels": args.labels,
            "split": args.split,
            "model_config": args.model_config,
        }),
    }
    result = train_entry_walk_forward(
        features=features,
        labels=labels,
        manifest=manifest,
        model_config_path=args.model_config,
        run_dir=run_dir,
        model_dir=model_dir,
        config=cfg,
        run_metadata=metadata,
    )
    write_json(metadata, run_dir / "input_snapshot.json")
    print(json.dumps({
        "run_id": run_id,
        "run_dir": str(result.run_dir),
        "model_dir": str(result.model_dir),
        "predictions": str(result.run_dir / "entry_oof_predictions.parquet"),
        "score_history": str(result.run_dir / "score_history.parquet"),
        "target_col": result.target_col,
        "feature_count": len(result.feature_cols),
        "summary": result.summary,
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

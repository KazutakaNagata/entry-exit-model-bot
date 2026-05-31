#!/usr/bin/env python3
"""Train a supervised exit LightGBM regressor on a position-state dataset.

This script consumes an exit dataset built by ``scripts/make_exit_dataset.py``.
It does not create entry candidates and it does not read test data.
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
from swing_bot.models.exit_lgbm import ExitTrainingConfig, exit_feature_columns, train_exit_valid_folds, validate_exit_dataset  # noqa: E402
from swing_bot.paths import models_dir, outputs_dir  # noqa: E402

DEFAULT_MODEL_CONFIG = Path("configs/models/lgbm_exit_v0.yaml")
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "exit_lgbm"
DEFAULT_MODEL_ROOT = models_dir() / "exit"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exit LGBM regressor and evaluate valid folds only.")
    parser.add_argument("--exit-dataset", type=Path, required=True, help="Exit position-state dataset parquet/csv.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="LightGBM exit model config YAML.")
    parser.add_argument(
        "--feature-set",
        choices=["v0_market_only", "v1_score_decay", "v2_position_aware"],
        default="v0_market_only",
        help="Reviewed exit feature set to train.",
    )
    parser.add_argument("--target-col", type=str, default="target_exit_hold_delta_bps", help="Exit regression target column.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output dir for valid artifacts.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT, help="Root model dir.")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save fold LightGBM model text files.")
    parser.add_argument("--dry-run", action="store_true", help="Load dataset and print selected feature columns without training.")
    return parser.parse_args(argv)


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def _infer_run_suffix(dataset: Path, feature_set: str) -> str:
    parent = dataset.parent.name if dataset.parent.name else dataset.stem
    return safe_slug(f"exit_{parent}_{feature_set}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.exit_dataset, hint="Run scripts/make_exit_dataset.py first.")
    _require_file(args.model_config, hint="Model config should exist under configs/models/.")

    raw = read_frame(args.exit_dataset)
    data = validate_exit_dataset(raw, target_col=args.target_col)
    feature_cols = exit_feature_columns(data, feature_set=args.feature_set, target_col=args.target_col)
    dataset_summary = {
        "exit_dataset": str(args.exit_dataset),
        "rows": int(len(data)),
        "folds": sorted(data["fold"].astype(str).unique().tolist()),
        "episodes": int(data["episode_id"].nunique()),
        "target_col": args.target_col,
        "feature_set": args.feature_set,
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
    }
    if args.dry_run:
        print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
        return 0

    run_id = safe_slug(args.run_id) if args.run_id else make_run_id(_infer_run_suffix(args.exit_dataset, args.feature_set))
    run_dir = args.output_root / run_id
    model_dir = args.model_root / run_id
    metadata = {
        **dataset_summary,
        "input_snapshots": snapshot_paths({
            "exit_dataset": args.exit_dataset,
            "model_config": args.model_config,
        }),
    }
    cfg = ExitTrainingConfig(
        feature_set=args.feature_set,
        target_col=args.target_col,
        save_models=not args.no_save_models,
    )
    result = train_exit_valid_folds(
        exit_dataset=data,
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
        "target_col": result.target_col,
        "feature_set": result.feature_set,
        "feature_count": len(result.feature_cols),
        "summary": result.summary,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

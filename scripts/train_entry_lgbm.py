#!/usr/bin/env python3
"""Train a cost-aware entry LightGBM regressor on valid folds.

Default target is long H60 because old research suggested long 60m had the most
stable signal.  This script never evaluates test; it only trains on train rows
and predicts locked valid folds from the supplied split manifest.
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
from swing_bot.models.entry_lgbm import EntryTrainingConfig, train_entry_valid_folds  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, models_dir, outputs_dir  # noqa: E402
from swing_bot.splits.split_manifest import load_split_manifest  # noqa: E402

DEFAULT_FEATURES = btcjpy_1m_processed_dir() / "features_entry_v0.parquet"
DEFAULT_LABELS = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/models/lgbm_entry_v0.yaml")
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_lgbm"
DEFAULT_MODEL_ROOT = models_dir() / "entry"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train entry LGBM regressor and evaluate valid folds only.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Feature matrix parquet/csv from build_features.py.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Label parquet/csv from build_labels.py.")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="Concrete split manifest YAML. Do not pass the template.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="LightGBM model config YAML.")
    parser.add_argument("--side", choices=["long", "short"], default="long", help="Entry side to train.")
    parser.add_argument("--horizon", type=int, default=60, help="Entry fixed-hold horizon in minutes.")
    parser.add_argument("--target-col", type=str, default=None, help="Override target column name.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output dir for valid artifacts.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT, help="Root model dir.")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save fold LightGBM model text files.")
    parser.add_argument("--dry-run", action="store_true", help="Load inputs and print row counts without training.")
    return parser.parse_args(argv)


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.features, hint="Run scripts/build_features.py first.")
    _require_file(args.labels, hint="Run scripts/build_labels.py first.")
    _require_file(args.split, hint="Copy configs/splits/btcjpy_1m_split_v1.template.yaml to btcjpy_1m_split_v1.yaml and fill real UTC dates.")
    _require_file(args.model_config, hint="Model config should exist under configs/models/.")

    manifest = load_split_manifest(args.split)
    features = read_frame(args.features)
    labels = read_frame(args.labels)
    target_col = args.target_col or f"target_entry_net_bps_{args.side}_H{int(args.horizon)}"

    train_mask = manifest.mask(features["timestamp"], "train") if "timestamp" in features else []
    valid_mask = manifest.mask(features["timestamp"], "valid") if "timestamp" in features else []
    summary = {
        "features_path": str(args.features),
        "labels_path": str(args.labels),
        "split": manifest.split_version,
        "side": args.side,
        "horizon_minutes": int(args.horizon),
        "target_col": target_col,
        "feature_rows": int(len(features)),
        "label_rows": int(len(labels)),
        "features_train_rows_by_split": int(sum(train_mask)) if len(features) else 0,
        "features_valid_rows_by_split": int(sum(valid_mask)) if len(features) else 0,
        "test_usage": manifest.test_usage,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    run_id = safe_slug(args.run_id) if args.run_id else make_run_id("entry", side=args.side, horizon_minutes=args.horizon)
    run_dir = args.output_root / run_id
    model_dir = args.model_root / run_id

    cfg = EntryTrainingConfig(
        side=args.side,
        horizon_minutes=int(args.horizon),
        target_col=args.target_col,
        save_models=not args.no_save_models,
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
    result = train_entry_valid_folds(
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
        "target_col": result.target_col,
        "feature_count": len(result.feature_cols),
        "summary": result.summary,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Train a small grid of entry LGBM regressors and collect OOF predictions.

Default grid is the current research baseline:

- long H60/H120/H240
- short H60/H120/H240

This script still follows the same rule as train_entry_lgbm.py: fit on train,
evaluate only valid folds, and do not touch test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.config_snapshot import snapshot_paths  # noqa: E402
from swing_bot.artifacts.io import read_frame, write_frame, write_json  # noqa: E402
from swing_bot.artifacts.run_id import make_run_id, safe_slug  # noqa: E402
from swing_bot.labels.entry_net_return import entry_target_column  # noqa: E402
from swing_bot.models.entry_lgbm import EntryTrainingConfig, train_entry_valid_folds  # noqa: E402
from swing_bot.models.entry_oof import combine_entry_oof_predictions, summarize_entry_run_dirs  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, models_dir, outputs_dir  # noqa: E402
from swing_bot.splits.split_manifest import load_split_manifest  # noqa: E402

DEFAULT_FEATURES = btcjpy_1m_processed_dir() / "features_entry_v0.parquet"
DEFAULT_LABELS = outputs_dir() / "valid" / "labels" / "btcjpy_1m_labels.parquet"
DEFAULT_SPLIT = Path("configs/splits/btcjpy_1m_split_v1.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/models/lgbm_entry_v0.yaml")
DEFAULT_ENTRY_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_lgbm"
DEFAULT_GRID_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_grid"
DEFAULT_OOF_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_oof"
DEFAULT_MODEL_ROOT = models_dir() / "entry"


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_sides(text: str) -> list[str]:
    sides = [x.strip() for x in text.split(",") if x.strip()]
    bad = [s for s in sides if s not in {"long", "short"}]
    if bad:
        raise ValueError("invalid sides: " + ", ".join(bad))
    return sides


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train entry LGBM grid and combine OOF valid predictions.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--sides", type=str, default="long,short", help="Comma-separated sides: long,short")
    parser.add_argument("--horizons", type=str, default="60,120,240", help="Comma-separated horizons in minutes.")
    parser.add_argument("--grid-run-id", type=str, default=None, help="Optional explicit grid id.")
    parser.add_argument("--entry-output-root", type=Path, default=DEFAULT_ENTRY_OUTPUT_ROOT)
    parser.add_argument("--grid-output-root", type=Path, default=DEFAULT_GRID_OUTPUT_ROOT)
    parser.add_argument("--oof-output-root", type=Path, default=DEFAULT_OOF_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--skip-missing-targets", action="store_true", help="Skip side/horizon combos whose target column is absent.")
    parser.add_argument("--dry-run", action="store_true", help="Print target availability and exit without training.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.features, hint="Run scripts/build_features.py first.")
    _require_file(args.labels, hint="Run scripts/build_labels.py first.")
    _require_file(args.split, hint="Create concrete split yaml from the template before training.")
    _require_file(args.model_config, hint="Model config should exist under configs/models/.")

    sides = _parse_csv_sides(args.sides)
    horizons = _parse_csv_ints(args.horizons)
    grid_id = safe_slug(args.grid_run_id) if args.grid_run_id else make_run_id("entry_grid")

    manifest = load_split_manifest(args.split)
    features = read_frame(args.features)
    labels = read_frame(args.labels)

    target_status = []
    combos: list[tuple[str, int, str]] = []
    for side in sides:
        for horizon in horizons:
            target_col = entry_target_column(side, horizon)
            exists = target_col in labels.columns
            target_status.append({"side": side, "horizon_minutes": horizon, "target_col": target_col, "exists": bool(exists)})
            if exists:
                combos.append((side, horizon, target_col))
            elif not args.skip_missing_targets:
                raise KeyError(f"labels missing target column {target_col!r}; rebuild labels or pass --skip-missing-targets")

    if args.dry_run:
        print(json.dumps({
            "grid_run_id": grid_id,
            "split": manifest.split_version,
            "target_status": target_status,
            "will_train": [{"side": s, "horizon_minutes": h, "target_col": c} for s, h, c in combos],
            "test_usage": manifest.test_usage,
        }, indent=2, ensure_ascii=False))
        return 0

    if not combos:
        raise ValueError("no trainable side/horizon combinations")

    grid_dir = args.grid_output_root / grid_id
    grid_dir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    run_rows: list[dict[str, object]] = []
    for side, horizon, target_col in combos:
        run_id = safe_slug(f"{grid_id}_{side}_H{horizon}")
        run_dir = args.entry_output_root / run_id
        model_dir = args.model_root / run_id
        cfg = EntryTrainingConfig(
            side=side,
            horizon_minutes=int(horizon),
            save_models=not args.no_save_models,
        )
        metadata = {
            "grid_run_id": grid_id,
            "grid_member": {"side": side, "horizon_minutes": horizon, "target_col": target_col},
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
        run_dirs.append(result.run_dir)
        run_rows.append({
            "entry_run_id": run_id,
            "run_dir": str(result.run_dir),
            "model_dir": str(result.model_dir),
            "side": side,
            "horizon_minutes": int(horizon),
            "target_col": result.target_col,
            "feature_count": len(result.feature_cols),
            **{f"summary_{k}": v for k, v in result.summary.items() if isinstance(v, (int, float))},
        })

    combined, oof_summary = combine_entry_oof_predictions(run_dirs, output_dir=args.oof_output_root / grid_id)
    comparison = summarize_entry_run_dirs(run_dirs)
    run_table = pd.DataFrame(run_rows)
    write_frame(run_table, grid_dir / "grid_runs.csv")
    write_frame(comparison, grid_dir / "entry_model_comparison.csv")
    write_frame(oof_summary, grid_dir / "entry_oof_summary.csv")
    write_json({
        "grid_run_id": grid_id,
        "split_version": manifest.split_version,
        "sides": sides,
        "horizons_minutes": horizons,
        "trained_run_dirs": [str(p) for p in run_dirs],
        "oof_output_dir": str(args.oof_output_root / grid_id),
        "combined_oof_rows": int(len(combined)),
        "target_status": target_status,
        "notes": [
            "Each member run fits on train only and predicts valid folds only.",
            "The combined OOF artifact is intended for future exit dataset construction.",
            "No test rows are read or evaluated by this script.",
        ],
    }, grid_dir / "grid_manifest.json")

    print(json.dumps({
        "grid_run_id": grid_id,
        "grid_dir": str(grid_dir),
        "oof_output_dir": str(args.oof_output_root / grid_id),
        "trained_runs": [str(p) for p in run_dirs],
        "combined_oof_rows": int(len(combined)),
        "comparison": comparison.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

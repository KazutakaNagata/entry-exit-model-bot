#!/usr/bin/env python3
"""Recompute entry valid metrics from saved prediction files.

This is a lightweight audit script.  It does not train, select thresholds, or
read test.  Use it when reviewing a run artifact.
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

from swing_bot.artifacts.io import read_frame, write_frame, write_json  # noqa: E402
from swing_bot.evaluation.fold_metrics import regression_fold_metrics, summarize_fold_metrics  # noqa: E402
from swing_bot.evaluation.topk import score_decile_summary  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved entry valid predictions.")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions_valid.parquet/csv from train_entry_lgbm.py.")
    parser.add_argument("--target-col", type=str, required=True, help="Target column to evaluate.")
    parser.add_argument("--pred-col", type=str, default="pred_entry_net_bps", help="Prediction column.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to write refreshed metrics.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = read_frame(args.predictions)
    required = {"fold", args.target_col, args.pred_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("predictions missing required columns: " + ", ".join(missing))

    metrics = []
    decile_rows = []
    for fold, fold_df in df.groupby("fold", sort=True):
        metrics.append(regression_fold_metrics(fold_df[args.target_col], fold_df[args.pred_col], fold_name=str(fold)))
        deciles = score_decile_summary(fold_df[args.target_col], fold_df[args.pred_col])
        if not deciles.empty:
            deciles.insert(0, "fold", fold)
            decile_rows.append(deciles)
    summary = summarize_fold_metrics(metrics)
    payload = {"metrics": metrics, "summary": summary}
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_frame(pd.DataFrame(metrics), args.output_dir / "fold_metrics.csv")
        if decile_rows:
            write_frame(pd.concat(decile_rows, ignore_index=True), args.output_dir / "score_deciles.csv")
        write_json(summary, args.output_dir / "summary_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

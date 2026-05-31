#!/usr/bin/env python3
"""Compare long_H120 v3b family-addition entry-grid results.

The script reads existing entry_model_comparison.csv files. It does not train,
tune thresholds, read test data, or select production settings.
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

from swing_bot.artifacts.io import write_frame  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_GRID_ROOT = outputs_dir() / "valid" / "entry_grid"
DEFAULT_OUTPUT = outputs_dir() / "valid" / "entry_grid" / "long_H120_v3b_addition_comparison.csv"
DEFAULT_GRID_PREFIX = "long_h120_v3b_addition"
DEFAULT_BASELINE = "base_v0"

METRIC_COLUMNS = [
    "mean_top_q95_avg_target_bps",
    "worst_top_q95_avg_target_bps",
    "mean_top_q97_avg_target_bps",
    "worst_top_q97_avg_target_bps",
    "mean_top_q99_avg_target_bps",
    "worst_top_q99_avg_target_bps",
    "mean_top_q95_precision_gt0",
    "worst_top_q95_precision_gt0",
    "mean_top_q99_precision_gt0",
    "worst_top_q99_precision_gt0",
]

ROBUST_SCORE_WEIGHTS = {
    "mean_top_q95_avg_target_bps": 0.20,
    "worst_top_q95_avg_target_bps": 0.20,
    "mean_top_q97_avg_target_bps": 0.20,
    "worst_top_q97_avg_target_bps": 0.15,
    "mean_top_q99_avg_target_bps": 0.15,
    "worst_top_q99_avg_target_bps": 0.10,
}


def _addition_from_grid_id(grid_id: str, prefix: str) -> str:
    if grid_id.startswith(prefix + "_"):
        return grid_id[len(prefix) + 1:]
    return grid_id


def _find_grid_dirs(grid_root: Path, prefix: str, explicit: list[Path]) -> list[Path]:
    if explicit:
        return explicit
    dirs = sorted(p for p in grid_root.glob(prefix + "_*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no grid dirs found under {grid_root} matching {prefix}_*")
    return dirs


def _read_one(grid_dir: Path, prefix: str) -> pd.DataFrame:
    path = grid_dir / "entry_model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing comparison file: {path}")
    df = pd.read_csv(path)
    df = df[(df["side"] == "long") & (df["horizon_minutes"].astype(int) == 120)].copy()
    if df.empty:
        raise ValueError(f"comparison has no long_H120 row: {path}")
    df.insert(0, "grid_id", grid_dir.name)
    df.insert(1, "addition", _addition_from_grid_id(grid_dir.name, prefix))
    return df


def _add_robust_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    for col, weight in ROBUST_SCORE_WEIGHTS.items():
        if col in out.columns:
            score += pd.to_numeric(out[col], errors="coerce") * weight
    out["long_h120_robust_score"] = score
    return out


def _add_baseline_deltas(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    bdf = df[df["addition"] == baseline]
    if bdf.empty:
        return df
    brow = bdf.iloc[0]
    out = df.copy()
    cols = [c for c in METRIC_COLUMNS + ["long_h120_robust_score", "feature_count"] if c in out.columns]
    for col in cols:
        out[f"delta_vs_{baseline}_{col}"] = pd.to_numeric(out[col], errors="coerce") - float(brow[col])
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare long_H120 v3b family-addition results.")
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument("--grid-prefix", type=str, default=DEFAULT_GRID_PREFIX)
    parser.add_argument("--grid-dir", type=Path, action="append", default=[], help="Explicit grid dir; may repeat.")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sort-by", type=str, default="long_h120_robust_score")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grid_dirs = _find_grid_dirs(args.grid_root, args.grid_prefix, args.grid_dir)
    combined = pd.concat([_read_one(path, args.grid_prefix) for path in grid_dirs], ignore_index=True, sort=False)
    combined = _add_robust_score(combined)
    combined = _add_baseline_deltas(combined, args.baseline)
    if args.sort_by in combined.columns:
        combined = combined.sort_values(args.sort_by, ascending=False).reset_index(drop=True)
    written = write_frame(combined, args.output)

    preview_cols = [
        "addition",
        "feature_count",
        "long_h120_robust_score",
        f"delta_vs_{args.baseline}_long_h120_robust_score",
        "mean_top_q95_avg_target_bps",
        "worst_top_q95_avg_target_bps",
        "mean_top_q97_avg_target_bps",
        "worst_top_q97_avg_target_bps",
        "mean_top_q99_avg_target_bps",
        "worst_top_q99_avg_target_bps",
        "mean_top_q99_precision_gt0",
    ]
    preview_cols = [c for c in preview_cols if c in combined.columns]
    print(json.dumps({
        "grid_count": len(grid_dirs),
        "rows": int(len(combined)),
        "output": str(written),
        "baseline": args.baseline,
        "sort_by": args.sort_by,
        "preview": combined[preview_cols].head(20).to_dict(orient="records"),
        "notes": [
            "Robust score is only a valid-fold comparison helper; it is not a production selector.",
            "Prefer additions that improve q95/q97/worst metrics while preserving q99 strength.",
            "Test must remain untouched until a locked config exists.",
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

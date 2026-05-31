#!/usr/bin/env python3
"""Compare entry grid results across feature-set variants.

The script reads each variant's existing ``entry_model_comparison.csv`` and
produces one long-form comparison plus optional focus slices.  It does not train
models, tune thresholds, or read test data.
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
DEFAULT_OUTPUT = outputs_dir() / "valid" / "entry_grid" / "feature_variant_comparison.csv"
DEFAULT_FOCUS_OUTPUT = outputs_dir() / "valid" / "entry_grid" / "feature_variant_focus_comparison.csv"

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


def _parse_focus(items: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"focus must be side:horizon, got {item!r}")
        side, horizon = item.split(":", 1)
        side = side.strip()
        if side not in {"long", "short"}:
            raise ValueError(f"invalid focus side: {side!r}")
        out.append((side, int(horizon)))
    return out


def _variant_from_grid_id(grid_id: str, prefix: str) -> str:
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


def _read_grid_comparison(grid_dir: Path, prefix: str) -> pd.DataFrame:
    path = grid_dir / "entry_model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing comparison file: {path}")
    df = pd.read_csv(path)
    df.insert(0, "grid_id", grid_dir.name)
    df.insert(1, "variant", _variant_from_grid_id(grid_dir.name, prefix))
    return df


def _add_baseline_deltas(df: pd.DataFrame, baseline_variant: str) -> pd.DataFrame:
    key_cols = ["side", "horizon_minutes"]
    baseline = df[df["variant"] == baseline_variant][key_cols + [c for c in METRIC_COLUMNS if c in df.columns]].copy()
    if baseline.empty:
        return df
    baseline = baseline.rename(columns={c: f"baseline_{c}" for c in baseline.columns if c not in key_cols})
    out = df.merge(baseline, on=key_cols, how="left")
    for c in METRIC_COLUMNS:
        b = f"baseline_{c}"
        if c in out.columns and b in out.columns:
            out[f"delta_vs_{baseline_variant}_{c}"] = pd.to_numeric(out[c], errors="coerce") - pd.to_numeric(out[b], errors="coerce")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare entry grid results across feature variants.")
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT)
    parser.add_argument("--grid-prefix", type=str, default="entry_variant")
    parser.add_argument("--grid-dir", type=Path, action="append", default=[], help="Explicit grid dir; may repeat.")
    parser.add_argument("--baseline-variant", type=str, default="v1_baseline")
    parser.add_argument("--focus", action="append", default=None, help="Focus side:horizon; may repeat. Default: long:120 and short:240.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focus-output", type=Path, default=DEFAULT_FOCUS_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grid_dirs = _find_grid_dirs(args.grid_root, args.grid_prefix, args.grid_dir)
    frames = [_read_grid_comparison(path, args.grid_prefix) for path in grid_dirs]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _add_baseline_deltas(combined, args.baseline_variant)
    sort_cols = [c for c in ["side", "horizon_minutes", "variant"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    focus_pairs = _parse_focus(args.focus or ["long:120", "short:240"])
    focus_mask = pd.Series(False, index=combined.index)
    for side, horizon in focus_pairs:
        focus_mask |= (combined["side"] == side) & (combined["horizon_minutes"].astype(int) == horizon)
    focus = combined[focus_mask].copy()

    written = write_frame(combined, args.output)
    focus_written = write_frame(focus, args.focus_output)
    preview_cols = [
        "variant", "side", "horizon_minutes",
        "mean_top_q95_avg_target_bps", "worst_top_q95_avg_target_bps",
        "mean_top_q99_avg_target_bps", "worst_top_q99_avg_target_bps",
    ]
    preview_cols = [c for c in preview_cols if c in focus.columns]
    print(json.dumps({
        "grid_count": len(grid_dirs),
        "rows": int(len(combined)),
        "output": str(written),
        "focus_output": str(focus_written),
        "baseline_variant": args.baseline_variant,
        "focus": [f"{s}:{h}" for s, h in focus_pairs],
        "focus_preview": focus[preview_cols].to_dict(orient="records") if not focus.empty else [],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

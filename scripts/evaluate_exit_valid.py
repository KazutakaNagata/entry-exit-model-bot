#!/usr/bin/env python3
"""Inspect one or more exit LGBM valid runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import read_frame, read_json, write_frame  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_OUTPUT = outputs_dir() / "valid" / "exit_lgbm" / "exit_run_comparison.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Exit run directory. Repeatable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Optional comparison CSV output.")
    return parser.parse_args(argv)


def _load_run_summary(run_dir: Path) -> dict[str, object]:
    run_cfg = read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    summary = read_json(run_dir / "summary_metrics.json") if (run_dir / "summary_metrics.json").exists() else {}
    row: dict[str, object] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "feature_set": run_cfg.get("feature_set"),
        "target_col": run_cfg.get("target_col"),
        "feature_count": run_cfg.get("feature_count"),
    }
    for key, value in summary.items():
        if isinstance(value, (int, float, str)) or value is None:
            row[key] = value
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for run_dir in args.run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"missing run dir: {run_dir}")
        rows.append(_load_run_summary(run_dir))
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        preferred = [
            "run_name",
            "feature_set",
            "feature_count",
            "mean_spearman_corr",
            "worst_spearman_corr",
            "mean_top_q95_avg_target_bps",
            "worst_top_q95_avg_target_bps",
            "mean_bottom_q95_avg_target_bps",
            "worst_bottom_q95_avg_target_bps",
            "mean_top_bottom_q95_spread_bps",
            "worst_top_bottom_q95_spread_bps",
            "mean_top_q99_avg_target_bps",
            "worst_top_q99_avg_target_bps",
            "mean_bottom_q99_avg_target_bps",
            "worst_bottom_q99_avg_target_bps",
            "mean_top_bottom_q99_spread_bps",
            "worst_top_bottom_q99_spread_bps",
            "run_dir",
        ]
        cols = [c for c in preferred if c in comparison.columns] + [c for c in comparison.columns if c not in preferred]
        comparison = comparison[cols]
    out_path = write_frame(comparison, args.output)
    print(json.dumps({"output": str(out_path), "runs": len(comparison)}, indent=2, ensure_ascii=False))
    if not comparison.empty:
        print(comparison.to_string(index=False, max_cols=20, max_colwidth=80))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

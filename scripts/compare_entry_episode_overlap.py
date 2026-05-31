#!/usr/bin/env python3
"""Compare exact entry overlap and P/L between two episode backtest runs.

Typical use:

python3 scripts/compare_entry_episode_overlap.py \
  --left-run-dir outputs/valid/episode_backtest/<fixed_train_best_run> \
  --right-run-dir outputs/valid/episode_backtest/<rolling180d_run> \
  --left-label fixed_train \
  --right-label rolling180d

This script is diagnostic-only.  It does not train, tune, or read test data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.evaluation.entry_overlap import compare_episode_entries, read_episode_frame, write_frame  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402

DEFAULT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "entry_overlap"


def _resolve_run_path(root: Path, run_dir: Path | None, run_name: str | None) -> Path:
    if run_dir is not None:
        return run_dir
    if not run_name:
        raise ValueError("provide either --run-dir or --run-name")
    path = root / run_name
    if not path.exists():
        matches = sorted(root.glob(run_name + "*"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"multiple runs match {run_name!r}: " + ", ".join(p.name for p in matches[:10]))
        raise FileNotFoundError(f"run does not exist under {root}: {run_name}")
    return path


def _safe_slug(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare entry overlap and P/L between two episode runs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Episode backtest root directory.")
    parser.add_argument("--left-run-dir", type=Path, default=None, help="Left episode run directory or episodes file.")
    parser.add_argument("--right-run-dir", type=Path, default=None, help="Right episode run directory or episodes file.")
    parser.add_argument("--left-run-name", type=str, default=None, help="Left run directory name under --root.")
    parser.add_argument("--right-run-name", type=str, default=None, help="Right run directory name under --root.")
    parser.add_argument("--left-label", type=str, default="fixed_train")
    parser.add_argument("--right-label", type=str, default="rolling")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--print-worst", type=int, default=10, help="Number of worst right-only/left-only entries to print.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    left_path = _resolve_run_path(args.root, args.left_run_dir, args.left_run_name)
    right_path = _resolve_run_path(args.root, args.right_run_dir, args.right_run_name)
    left_label = _safe_slug(args.left_label)
    right_label = _safe_slug(args.right_label)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / f"{left_label}_vs_{right_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    left = read_episode_frame(left_path)
    right = read_episode_frame(right_path)
    result = compare_episode_entries(
        left_episodes=left,
        right_episodes=right,
        left_label=left_label,
        right_label=right_label,
    )

    summary_path = output_dir / "entry_overlap_summary.json"
    by_group_path = output_dir / "entry_overlap_by_group.csv"
    by_fold_path = output_dir / "entry_overlap_by_fold.csv"
    episodes_path = output_dir / "entry_overlap_episodes.parquet"

    summary = dict(result.summary)
    summary.update({
        "left_path": str(left_path),
        "right_path": str(right_path),
        "output_dir": str(output_dir),
    })
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_frame(result.by_group, by_group_path)
    write_frame(result.by_fold, by_fold_path)
    written_episodes = write_frame(result.episodes, episodes_path)

    preview_cols = [
        "fold",
        "overlap_group",
        "episode_count",
        "net_pl_bps_sum",
        "avg_net_pl_bps",
        "win_rate",
        "profit_factor",
        "avg_mfe_bps",
        "avg_mae_bps",
    ]
    preview_cols = [c for c in preview_cols if c in result.by_fold.columns]

    worst_cols = [
        "fold",
        "side",
        "entry_time",
        "overlap_group",
        "left_net_pl_bps",
        "right_net_pl_bps",
        "canonical_net_pl_bps",
        "left_entry_pred_net_bps",
        "right_entry_pred_net_bps",
    ]
    worst_cols = [c for c in worst_cols if c in result.episodes.columns]
    worst = result.episodes.sort_values("canonical_net_pl_bps", ascending=True).head(max(0, int(args.print_worst)))

    print(json.dumps({
        "summary_path": str(summary_path),
        "by_group_path": str(by_group_path),
        "by_fold_path": str(by_fold_path),
        "episodes_path": str(written_episodes),
        "summary": summary,
        "by_group": result.by_group.to_dict(orient="records"),
        "by_fold_preview": result.by_fold[preview_cols].head(20).to_dict(orient="records"),
        "worst_entries_preview": worst[worst_cols].to_dict(orient="records") if worst_cols else [],
        "notes": [
            "Exact overlap key is fold + side + entry_time.",
            "Compare left_only and right_only P/L to see whether rolling retrain misses good entries or adds bad entries.",
            "This reads valid episode outputs only and does not touch test data.",
        ],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

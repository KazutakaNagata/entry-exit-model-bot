#!/usr/bin/env python3
"""Run a second-stage combined-drop follow-up from an existing monthly LOFO run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.paths import outputs_dir  # noqa: E402
from swing_bot.artifacts.run_id import safe_slug  # noqa: E402
from swing_bot.research.family_ablation_followup import run_combined_drop_followup  # noqa: E402

DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "rolling_monthly_research"


def _resolve_source_run_dir(args: argparse.Namespace) -> Path:
    if args.source_run_dir:
        return Path(args.source_run_dir)
    if not args.source_run_name:
        raise ValueError("provide --source-run-dir or --source-run-name")
    candidates = [
        Path(args.source_root) / args.source_run_name,
        outputs_dir() / "valid" / "rolling_monthly_research" / args.source_run_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"could not resolve source run name {args.source_run_name!r}; tried {candidates}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run combined-drop family ablation follow-up without rerunning all single drops.")
    p.add_argument("--source-run-dir", type=Path, default=None, help="Existing single-month LOFO run directory.")
    p.add_argument("--source-run-name", type=str, default=None, help="Existing run name under --source-root.")
    p.add_argument("--source-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--metric-col", type=str, default="selector_raw_score", choices=["selector_raw_score", "robust_score", "selector_total_net_pl_bps", "mean_net_pl_bps_sum"])
    p.add_argument("--min-delta-bps", type=float, default=0.0, help="Family is dropped if single-drop metric improves full by more than this.")
    p.add_argument("--include-addback", action="store_true", help="Also evaluate combined-drop plus each dropped family added back.")
    p.add_argument("--include-full", action="store_true", help="Also re-evaluate the source full policy.")
    p.add_argument("--no-include-source-selected", action="store_true", help="Do not include the original selected single-drop policy as comparator.")
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_run_dir = _resolve_source_run_dir(args)
    run_id = safe_slug(args.run_id) if args.run_id else safe_slug(f"{source_run_dir.name}_combined_drop_followup")
    result = run_combined_drop_followup(
        source_run_dir=source_run_dir,
        output_root=args.output_root,
        followup_run_id=run_id,
        metric_col=args.metric_col,
        min_delta_bps=args.min_delta_bps,
        include_addback=args.include_addback,
        include_full=args.include_full,
        include_source_selected=not args.no_include_source_selected,
        overwrite_features=args.overwrite_features,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

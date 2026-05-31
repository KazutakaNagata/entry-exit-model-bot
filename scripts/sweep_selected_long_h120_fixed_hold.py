#!/usr/bin/env python3
"""Run a small, live-feasible fixed-hold threshold sweep for selected long_H120.

This script intentionally avoids fold-wide top quantile selection.  It can run:

* absolute threshold sweeps, e.g. pred > 25 / 30 / 35
* rolling quantile sweeps, where the threshold at time t is computed only from
  scores before t in the same fold

It is valid-fold research only and does not touch test data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import write_json  # noqa: E402
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir  # noqa: E402
from swing_bot.pipeline.selected_long_h120_downstream import (  # noqa: E402
    DEFAULT_CANDIDATES,
    parse_csv_values,
    selected_entry_oof_path,
)

DEFAULT_OHLCV = btcjpy_1m_processed_dir() / "ohlcv.parquet"
DEFAULT_ENTRY_OOF_ROOT = outputs_dir() / "valid" / "entry_oof"
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "episode_backtest"
DEFAULT_PLAN_OUTPUT = outputs_dir() / "valid" / "episode_backtest" / "selected_long_H120_fixed_hold_sweep_plan.json"
DEFAULT_THRESHOLD_GRIDS = {
    "long_H120_v0": [20.0, 25.0, 30.0, 35.0],
    "long_H120_tail_v0": [25.0, 30.0, 35.0, 40.0, 50.0],
}


def _parse_float_list(text: str) -> list[float]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("empty float list")
    return [float(v) for v in values]


def _parse_candidate_grids(text: str | None) -> dict[str, list[float]]:
    if text is None or str(text).strip() == "":
        return {k: list(v) for k, v in DEFAULT_THRESHOLD_GRIDS.items()}
    out: dict[str, list[float]] = {}
    for item in str(text).split(";"):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("threshold grid must be candidate=v1,v2;candidate2=v3")
        key, values = part.split("=", 1)
        out[key.strip()] = _parse_float_list(values)
    if not out:
        raise ValueError("no candidate threshold grids parsed")
    return out


def _slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists() and not (path.name.endswith(".parquet") and path.with_suffix(".csv").exists()):
        raise FileNotFoundError(f"missing file: {path}\n{hint}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep selected long_H120 fixed-hold entry rules without score-distribution leakage.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--entry-oof-root", type=Path, default=DEFAULT_ENTRY_OOF_ROOT)
    parser.add_argument("--score-history", type=Path, default=None, help="Optional score-history file passed to each fixed-hold backtest for rolling quantile warmup.")
    parser.add_argument("--entry-grid-prefix", type=str, default="selected_long_H120")
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--modes", type=str, default="threshold,rolling_quantile", help="Comma list: threshold,rolling_quantile")
    parser.add_argument("--threshold-grids", type=str, default=None, help="candidate=v1,v2;candidate2=v3. Used for threshold mode and rolling floors.")
    parser.add_argument("--rolling-score-window-days", type=int, default=60)
    parser.add_argument("--rolling-score-quantiles", type=str, default="0.98,0.99,0.995")
    parser.add_argument("--rolling-score-min-periods", type=int, default=1000)
    parser.add_argument("--rolling-history-mode", choices=["fold_local", "continuous_valid", "score_history"], default="fold_local", help="Past-score history source for rolling_quantile mode.")
    parser.add_argument("--fixed-hold-minutes", type=int, default=120)
    parser.add_argument("--roundtrip-cost-bps", type=float, default=15.0)
    parser.add_argument("--decision-interval-minutes", type=int, default=5)
    parser.add_argument("--cooldown-after-exit-minutes", type=int, default=15)
    parser.add_argument("--same-side-reentry-block-minutes", type=int, default=30)
    parser.add_argument("--max-round-trips-per-day", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_file(args.ohlcv, hint="Run scripts/audit_data.py --write-canonical first.")
    if args.score_history is not None:
        _require_file(args.score_history, hint="Provide score history for rolling quantile warmup, or omit this option.")
    candidates = parse_csv_values(args.candidates, default=DEFAULT_CANDIDATES)
    modes = parse_csv_values(args.modes, default=("threshold", "rolling_quantile"))
    allowed = {"threshold", "rolling_quantile"}
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise ValueError(f"invalid modes: {invalid}")
    threshold_grids = _parse_candidate_grids(args.threshold_grids)
    rolling_quantiles = _parse_float_list(args.rolling_score_quantiles)

    jobs: list[dict[str, object]] = []
    for candidate in candidates:
        entry_oof = selected_entry_oof_path(
            candidate=candidate,
            entry_oof_root=args.entry_oof_root,
            entry_grid_prefix=args.entry_grid_prefix,
        )
        _require_file(entry_oof, hint=f"Run scripts/run_selected_long_h120_entry.py --candidates {candidate} first.")
        thresholds = threshold_grids.get(candidate)
        if not thresholds:
            raise ValueError(f"no thresholds configured for candidate {candidate!r}")

        if "threshold" in modes:
            for threshold in thresholds:
                run_id = (
                    f"selected_long_H120_fixed_hold_sweep_{candidate}_hold{int(args.fixed_hold_minutes)}_"
                    f"entryThr{_slug_float(threshold)}"
                )
                cmd = [
                    sys.executable,
                    "scripts/backtest_fixed_hold_entry_valid.py",
                    "--ohlcv", str(args.ohlcv),
                    "--entry-oof", str(entry_oof),
                    *(["--score-history", str(args.score_history)] if args.score_history is not None else []),
                    "--side", "long",
                    "--entry-horizon", "120",
                    "--fixed-hold-minutes", str(int(args.fixed_hold_minutes)),
                    "--entry-selection-mode", "threshold",
                    "--entry-threshold-bps", str(float(threshold)),
                    "--roundtrip-cost-bps", str(float(args.roundtrip_cost_bps)),
                    "--decision-interval-minutes", str(int(args.decision_interval_minutes)),
                    "--cooldown-after-exit-minutes", str(int(args.cooldown_after_exit_minutes)),
                    "--same-side-reentry-block-minutes", str(int(args.same_side_reentry_block_minutes)),
                    "--run-id", run_id,
                    "--output-root", str(args.output_root),
                ]
                if args.max_round_trips_per_day is not None:
                    cmd.extend(["--max-round-trips-per-day", str(int(args.max_round_trips_per_day))])
                jobs.append({
                    "candidate": candidate,
                    "mode": "threshold",
                    "entry_threshold_bps": float(threshold),
                    "run_id": run_id,
                    "run_dir": str(args.output_root / run_id),
                    "score_history": str(args.score_history) if args.score_history is not None else None,
                    "command": cmd,
                })

        if "rolling_quantile" in modes:
            for floor in thresholds:
                for quantile in rolling_quantiles:
                    run_id = (
                        f"selected_long_H120_fixed_hold_sweep_{candidate}_hold{int(args.fixed_hold_minutes)}_"
                        f"rollQ{_slug_float(quantile)}_win{int(args.rolling_score_window_days)}d_{args.rolling_history_mode}_floor{_slug_float(floor)}"
                    )
                    cmd = [
                        sys.executable,
                        "scripts/backtest_fixed_hold_entry_valid.py",
                        "--ohlcv", str(args.ohlcv),
                        "--entry-oof", str(entry_oof),
                        *(["--score-history", str(args.score_history)] if args.score_history is not None else []),
                        "--side", "long",
                        "--entry-horizon", "120",
                        "--fixed-hold-minutes", str(int(args.fixed_hold_minutes)),
                        "--entry-selection-mode", "rolling_quantile",
                        "--entry-threshold-bps", str(float(floor)),
                        "--entry-score-floor-bps", str(float(floor)),
                        "--rolling-score-window-days", str(int(args.rolling_score_window_days)),
                        "--rolling-score-quantile", str(float(quantile)),
                        "--rolling-score-min-periods", str(int(args.rolling_score_min_periods)),
                        "--rolling-history-mode", str(args.rolling_history_mode),
                        "--roundtrip-cost-bps", str(float(args.roundtrip_cost_bps)),
                        "--decision-interval-minutes", str(int(args.decision_interval_minutes)),
                        "--cooldown-after-exit-minutes", str(int(args.cooldown_after_exit_minutes)),
                        "--same-side-reentry-block-minutes", str(int(args.same_side_reentry_block_minutes)),
                        "--run-id", run_id,
                        "--output-root", str(args.output_root),
                    ]
                    if args.max_round_trips_per_day is not None:
                        cmd.extend(["--max-round-trips-per-day", str(int(args.max_round_trips_per_day))])
                    jobs.append({
                        "candidate": candidate,
                        "mode": "rolling_quantile",
                        "entry_score_floor_bps": float(floor),
                        "rolling_score_window_days": int(args.rolling_score_window_days),
                        "rolling_score_quantile": float(quantile),
                        "rolling_score_min_periods": int(args.rolling_score_min_periods),
                        "rolling_history_mode": args.rolling_history_mode,
                        "run_id": run_id,
                        "run_dir": str(args.output_root / run_id),
                        "command": cmd,
                    })

    plan = {
        "role": "selected_long_H120_fixed_hold_leak_safe_threshold_sweep",
        "job_count": len(jobs),
        "jobs": jobs,
        "notes": [
            "No fold-wide top quantile entry rules are used.",
            "rolling_quantile thresholds use past scores only inside the fixed-hold backtester.",
            "Valid only; test must remain untouched until a locked config exists.",
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return 0
    write_json(plan, args.plan_output)
    for job in jobs:
        _run(job["command"], dry_run=False)  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

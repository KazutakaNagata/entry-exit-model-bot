#!/usr/bin/env python3
"""Replay existing monthly-family-ablation predictions with pre-period score quantile thresholds.

This is a lightweight diagnostic/replay path.  It does *not* retrain LGBM
models.  It reads existing candidate ``entry_oof_predictions.parquet`` and
``score_history.parquet`` files produced by monthly family ablation runs, then
recomputes fixed-hold backtests using:

    threshold = max(quantile(score_history.pred_entry_net_bps, q), floor_bps)

per valid/test fold.  The score history is strictly pre-period data generated
for that fold/model, so evaluation-month score distribution is not used.

If the replay-selected policy was not the policy originally evaluated on the
source test month, test predictions for that policy will usually be missing;
this script reports that explicitly instead of fabricating a test result.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swing_bot.artifacts.io import write_frame, write_json  # noqa: E402
from swing_bot.evaluation.episode_report import write_episode_report  # noqa: E402
from swing_bot.paths import outputs_dir  # noqa: E402
from swing_bot.research.family_policy_selection import selector_metrics_from_fold_metrics  # noqa: E402
from swing_bot.research.monthly_rolling import _backtest, load_rolling_monthly_config  # noqa: E402

DEFAULT_CONFIG = Path("configs/rolling_protocol/long_H120_monthly_family_ablation_score_history_quantile_v0.yaml")
DEFAULT_SOURCE_ROOT = outputs_dir() / "valid" / "rolling_monthly_research"
DEFAULT_OUTPUT_ROOT = outputs_dir() / "valid" / "rolling_monthly_score_quantile_replay"


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_source_run(source_root: Path, source_run_name: str | None, source_run_dir: Path | None) -> Path:
    if source_run_dir is not None:
        path = source_run_dir
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not source_run_name:
        raise ValueError("either --source-run-name or --source-run-dir is required")
    path = source_root / source_run_name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _valid_candidate_dirs(cycle_dir: Path) -> list[Path]:
    base = cycle_dir / "valid_candidates"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _test_policy_dir(cycle_dir: Path, policy: str) -> Path | None:
    path = cycle_dir / "test" / policy
    if path.exists():
        return path
    return None


def _cycle_dirs(source_run_dir: Path) -> list[Path]:
    return sorted([p for p in source_run_dir.iterdir() if p.is_dir() and p.name.startswith("test_")])


def _scalar_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool)) or v is None}


def _evaluate_prediction_dir(
    *,
    pred_dir: Path,
    ohlcv: pd.DataFrame,
    cfg,
    output_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preds = _read_frame(pred_dir / "entry_oof_predictions.parquet")
    hist_path = pred_dir / "score_history.parquet"
    if not hist_path.exists():
        hist_path = pred_dir / "score_history.csv"
    score_history = _read_frame(hist_path)
    episodes, fold_metrics, summary = _backtest(ohlcv=ohlcv, predictions=preds, score_history=score_history, cfg=cfg)
    if output_dir is not None:
        write_episode_report(episodes=episodes, fold_metrics=fold_metrics.to_dict("records") if isinstance(fold_metrics, pd.DataFrame) else fold_metrics, summary=summary, output_dir=output_dir)
        # Keep replay-specific threshold details in summary_metrics.json.  No extra files needed.
    return episodes, pd.DataFrame(fold_metrics), summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay existing monthly family-ablation predictions with score-history quantile thresholds.")
    p.add_argument("--source-run-name", type=str, default=None)
    p.add_argument("--source-run-dir", type=Path, default=None)
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--score-quantile", type=float, default=None, help="Override config entry_selection.score_quantile")
    p.add_argument("--score-floor-bps", type=float, default=None, help="Override config entry_selection.score_floor_bps")
    p.add_argument("--score-quantile-min-periods", type=int, default=None)
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--write-candidate-reports", action="store_true", help="Write per-candidate replay episodes/fold metrics. This can create many files.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_root if args.source_root.is_absolute() else REPO_ROOT / args.source_root
    source_run_dir = _find_source_run(source_root, args.source_run_name, args.source_run_dir)
    run_id = args.run_id or f"{source_run_dir.name}_score_history_q{str(args.score_quantile or 'cfg').replace('.', 'p')}_replay"
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    run_dir = output_root / run_id
    cfg = load_rolling_monthly_config(args.config)
    cfg = replace(
        cfg,
        entry_selection_mode="score_history_quantile",
        score_quantile=float(args.score_quantile if args.score_quantile is not None else cfg.score_quantile),
        score_floor_bps=float(args.score_floor_bps if args.score_floor_bps is not None else cfg.score_floor_bps),
        score_quantile_min_periods=int(args.score_quantile_min_periods if args.score_quantile_min_periods is not None else cfg.score_quantile_min_periods),
    )
    cycles = _cycle_dirs(source_run_dir)
    if args.max_cycles is not None:
        cycles = cycles[: int(args.max_cycles)]
    plan = {
        "source_run_dir": str(source_run_dir),
        "run_dir": str(run_dir),
        "cycle_count": len(cycles),
        "entry_selection_mode": cfg.entry_selection_mode,
        "score_quantile": cfg.score_quantile,
        "score_floor_bps": cfg.score_floor_bps,
        "score_quantile_min_periods": cfg.score_quantile_min_periods,
        "write_candidate_reports": bool(args.write_candidate_reports),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    ohlcv = _read_frame(cfg.ohlcv_path)
    if "timestamp" in ohlcv.columns:
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)

    candidate_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for cycle_dir in cycles:
        cycle_id = cycle_dir.name
        replay_cycle_dir = run_dir / cycle_id
        replay_cycle_dir.mkdir(parents=True, exist_ok=True)
        best_policy = None
        best_score = -float("inf")
        best_summary = None
        candidate_dirs = _valid_candidate_dirs(cycle_dir)
        if not candidate_dirs:
            raise FileNotFoundError(f"no valid_candidates under {cycle_dir}")
        for cand_dir in candidate_dirs:
            out_dir = replay_cycle_dir / "valid_candidates" / cand_dir.name if args.write_candidate_reports else None
            _, fold_metrics, summary = _evaluate_prediction_dir(pred_dir=cand_dir, ohlcv=ohlcv, cfg=cfg, output_dir=out_dir)
            selector_metrics = selector_metrics_from_fold_metrics(
                pd.DataFrame(fold_metrics),
                min_total_episode_count=int(cfg.min_valid_episode_count),
                min_active_fold_count=int(cfg.min_active_valid_folds),
                max_worst_fold_loss_bps=cfg.max_worst_valid_loss_bps,
                worst_weight=float(cfg.robust_worst_weight),
                median_weight=float(cfg.robust_median_weight),
                fail_penalty=float(cfg.selector_fail_penalty),
            )
            score = float(selector_metrics["robust_score"])
            row = {
                "cycle_id": cycle_id,
                "policy": cand_dir.name,
                "selected": False,
                "robust_score": score,
                **selector_metrics,
                **_scalar_summary(summary),
            }
            candidate_rows.append(row)
            if score > best_score:
                best_score = score
                best_policy = cand_dir.name
                best_summary = summary
        assert best_policy is not None
        for row in candidate_rows:
            if row["cycle_id"] == cycle_id and row["policy"] == best_policy:
                row["selected"] = True
        test_status = "missing_selected_test_predictions"
        test_summary: dict[str, Any] = {}
        test_dir = _test_policy_dir(cycle_dir, best_policy)
        if test_dir is not None:
            _, test_fm, test_summary = _evaluate_prediction_dir(pred_dir=test_dir, ohlcv=ohlcv, cfg=cfg, output_dir=replay_cycle_dir / "test" / best_policy)
            test_status = "success"
        test_rows.append({
            "cycle_id": cycle_id,
            "selected_policy": best_policy,
            "valid_robust_score": best_score,
            "test_replay_status": test_status,
            **{f"test_{k}": v for k, v in _scalar_summary(test_summary).items()},
        })
        cycle_rows.append({
            "cycle_id": cycle_id,
            "selected_policy": best_policy,
            "valid_robust_score": best_score,
            "valid_mean_net_pl_bps_sum": None if best_summary is None else best_summary.get("mean_net_pl_bps_sum"),
            "test_replay_status": test_status,
            "test_episode_count": test_summary.get("episode_count") if test_summary else None,
            "test_mean_net_pl_bps_sum": test_summary.get("mean_net_pl_bps_sum") if test_summary else None,
        })
    candidate_df = pd.DataFrame(candidate_rows)
    cycle_df = pd.DataFrame(cycle_rows)
    test_df = pd.DataFrame(test_rows)
    write_frame(candidate_df, run_dir / "candidate_valid_metrics.csv")
    write_frame(cycle_df, run_dir / "rolling_cycles.csv")
    write_frame(test_df, run_dir / "rolling_test_metrics.csv")
    write_json({
        **plan,
        "notes": [
            "This replay does not retrain models; it reuses existing candidate prediction and score_history files.",
            "Valid candidate selection is replayed under score_history_quantile thresholds.",
            "If the replay-selected policy was not originally tested, test replay is reported missing instead of inferred.",
        ],
    }, run_dir / "run_config.json")
    print(json.dumps({
        **plan,
        "candidate_valid_metrics": str(run_dir / "candidate_valid_metrics.csv"),
        "rolling_cycles": str(run_dir / "rolling_cycles.csv"),
        "rolling_test_metrics": str(run_dir / "rolling_test_metrics.csv"),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

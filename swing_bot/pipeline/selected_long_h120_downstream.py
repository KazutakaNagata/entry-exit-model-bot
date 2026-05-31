"""Helpers for selected long_H120 downstream validation.

The functions here deliberately stay small and path-oriented.  They do not train
models directly; scripts use them to build reviewed subprocess commands for the
existing data/entry/exit/backtest CLIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd

from swing_bot.artifacts.io import read_json, write_frame
from swing_bot.paths import btcjpy_1m_processed_dir, outputs_dir

DEFAULT_CANDIDATES = ("long_H120_v0", "long_H120_tail_v0")
DEFAULT_CANDIDATE_THRESHOLDS = {
    "long_H120_v0": 20.0,
    "long_H120_tail_v0": 25.0,
}
DEFAULT_EXIT_FEATURE_SETS = ("v2_position_aware",)
DEFAULT_EXIT_LOOKAHEADS = (30,)


@dataclass(frozen=True)
class SelectedDownstreamPlan:
    """One selected-entry downstream run combination."""

    candidate: str
    exit_lookahead_minutes: int
    exit_feature_set: str
    entry_threshold_bps: float
    hold_threshold_bps: float
    entry_oof_path: Path
    features_path: Path
    exit_dataset_dir: Path
    exit_dataset_path: Path
    exit_run_id: str
    exit_run_dir: Path
    episode_run_id: str
    episode_run_dir: Path


def parse_csv_values(text: str | None, *, default: Sequence[str] = ()) -> list[str]:
    """Parse a comma-separated option into non-empty strings."""
    if text is None or str(text).strip() == "":
        return list(default)
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        return list(default)
    return values


def parse_int_csv_values(text: str | None, *, default: Sequence[int] = ()) -> list[int]:
    """Parse a comma-separated integer option."""
    values = parse_csv_values(text, default=[str(x) for x in default])
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError as exc:
            raise ValueError(f"invalid integer list value: {value!r}") from exc
    return parsed


def parse_candidate_thresholds(text: str | None, *, default_threshold: float) -> dict[str, float]:
    """Parse ``candidate=threshold`` pairs.

    Example: ``long_H120_v0=20,long_H120_tail_v0=25``.
    """
    if text is None or str(text).strip() == "":
        return {}
    out: dict[str, float] = {}
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"candidate threshold must be candidate=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty candidate name in threshold item {item!r}")
        try:
            out[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid threshold for {key!r}: {value!r}") from exc
    # ``default_threshold`` is not inserted here; callers may need to distinguish
    # explicit candidate thresholds from defaults.
    _ = default_threshold
    return out


def threshold_for_candidate(
    candidate: str,
    *,
    explicit_thresholds: dict[str, float],
    default_threshold: float,
) -> float:
    """Return the entry threshold for a selected candidate."""
    if candidate in explicit_thresholds:
        return explicit_thresholds[candidate]
    if candidate in DEFAULT_CANDIDATE_THRESHOLDS:
        return float(DEFAULT_CANDIDATE_THRESHOLDS[candidate])
    return float(default_threshold)


def existing_table_path(path: Path) -> Path:
    """Return an existing parquet/csv path, respecting write_frame fallbacks."""
    if path.exists():
        return path
    if path.name.lower().endswith((".parquet", ".pq")):
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            return fallback
    return path


def selected_entry_oof_path(
    *,
    candidate: str,
    entry_oof_root: Path | None = None,
    entry_grid_prefix: str = "selected_long_H120",
) -> Path:
    """Path to a candidate's selected-entry OOF predictions."""
    root = entry_oof_root or outputs_dir() / "valid" / "entry_oof"
    return root / f"{entry_grid_prefix}_{candidate}" / "entry_oof_predictions.parquet"


def selected_features_path(*, candidate: str, feature_root: Path | None = None) -> Path:
    """Path to a candidate's selected feature matrix."""
    root = feature_root or btcjpy_1m_processed_dir() / "feature_selected" / "long_H120"
    return root / f"features_{candidate}.parquet"


def make_plan(
    *,
    candidate: str,
    exit_lookahead_minutes: int,
    exit_feature_set: str,
    entry_threshold_bps: float,
    hold_threshold_bps: float,
    entry_oof_root: Path | None = None,
    entry_grid_prefix: str = "selected_long_H120",
    feature_root: Path | None = None,
    exit_dataset_root: Path | None = None,
    exit_output_root: Path | None = None,
    episode_output_root: Path | None = None,
) -> SelectedDownstreamPlan:
    """Create path/run-id conventions for one downstream combination."""
    exit_dataset_root = exit_dataset_root or outputs_dir() / "valid" / "exit_dataset"
    exit_output_root = exit_output_root or outputs_dir() / "valid" / "exit_lgbm"
    episode_output_root = episode_output_root or outputs_dir() / "valid" / "episode_backtest"

    dataset_name = f"selected_long_H120_{candidate}_K{int(exit_lookahead_minutes)}"
    exit_run_id = f"selected_long_H120_{candidate}_K{int(exit_lookahead_minutes)}_{exit_feature_set}"
    entry_thr = f"{entry_threshold_bps:g}".replace(".", "p")
    hold_thr = f"{hold_threshold_bps:g}".replace(".", "p")
    episode_run_id = f"{exit_run_id}_entryThr{entry_thr}_holdThr{hold_thr}"
    exit_dataset_dir = exit_dataset_root / dataset_name
    return SelectedDownstreamPlan(
        candidate=candidate,
        exit_lookahead_minutes=int(exit_lookahead_minutes),
        exit_feature_set=exit_feature_set,
        entry_threshold_bps=float(entry_threshold_bps),
        hold_threshold_bps=float(hold_threshold_bps),
        entry_oof_path=selected_entry_oof_path(
            candidate=candidate,
            entry_oof_root=entry_oof_root,
            entry_grid_prefix=entry_grid_prefix,
        ),
        features_path=selected_features_path(candidate=candidate, feature_root=feature_root),
        exit_dataset_dir=exit_dataset_dir,
        exit_dataset_path=exit_dataset_dir / "exit_dataset.parquet",
        exit_run_id=exit_run_id,
        exit_run_dir=exit_output_root / exit_run_id,
        episode_run_id=episode_run_id,
        episode_run_dir=episode_output_root / episode_run_id,
    )


def make_plans(
    *,
    candidates: Sequence[str],
    exit_lookaheads: Sequence[int],
    exit_feature_sets: Sequence[str],
    candidate_thresholds: dict[str, float],
    default_entry_threshold_bps: float,
    hold_threshold_bps: float,
    entry_oof_root: Path | None = None,
    entry_grid_prefix: str = "selected_long_H120",
    feature_root: Path | None = None,
    exit_dataset_root: Path | None = None,
    exit_output_root: Path | None = None,
    episode_output_root: Path | None = None,
) -> list[SelectedDownstreamPlan]:
    """Create downstream plans for a small selected long_H120 candidate grid."""
    plans: list[SelectedDownstreamPlan] = []
    for candidate in candidates:
        threshold = threshold_for_candidate(
            candidate,
            explicit_thresholds=candidate_thresholds,
            default_threshold=default_entry_threshold_bps,
        )
        for lookahead in exit_lookaheads:
            for feature_set in exit_feature_sets:
                plans.append(
                    make_plan(
                        candidate=candidate,
                        exit_lookahead_minutes=int(lookahead),
                        exit_feature_set=feature_set,
                        entry_threshold_bps=threshold,
                        hold_threshold_bps=hold_threshold_bps,
                        entry_oof_root=entry_oof_root,
                        entry_grid_prefix=entry_grid_prefix,
                        feature_root=feature_root,
                        exit_dataset_root=exit_dataset_root,
                        exit_output_root=exit_output_root,
                        episode_output_root=episode_output_root,
                    )
                )
    return plans


def _infer_candidate(name: str, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in name:
            return candidate
    return None


def _infer_exit_lookahead(name: str) -> int | None:
    match = re.search(r"_K(\d+)", name)
    return int(match.group(1)) if match else None


def _infer_exit_feature_set(name: str) -> str | None:
    for feature_set in ("v0_market_only", "v1_score_decay", "v2_position_aware"):
        if feature_set in name:
            return feature_set
    return None


def summarize_episode_runs(
    run_dirs: Sequence[Path],
    *,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
) -> pd.DataFrame:
    """Read episode summary_metrics.json files into a comparison frame."""
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary_metrics.json"
        config_path = run_dir / "run_config.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing summary_metrics.json: {summary_path}")
        summary = read_json(summary_path)
        config = read_json(config_path) if config_path.exists() else {}
        run_name = run_dir.name
        policy = config.get("policy") or {}
        row: dict[str, object] = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "candidate": _infer_candidate(run_name, candidates),
            "exit_lookahead_minutes": _infer_exit_lookahead(run_name),
            "exit_feature_set": _infer_exit_feature_set(run_name),
            "entry_selection_mode": summary.get("entry_selection_mode", config.get("entry_selection_mode")),
            "entry_threshold_bps": policy.get("entry_threshold_bps", summary.get("entry_threshold_bps")),
            "entry_score_floor_bps": summary.get("entry_score_floor_bps", config.get("entry_score_floor_bps")),
            "rolling_score_window_days": summary.get("rolling_score_window_days", config.get("rolling_score_window_days")),
            "rolling_score_quantile": summary.get("rolling_score_quantile", config.get("rolling_score_quantile")),
            "rolling_score_min_periods": summary.get("rolling_score_min_periods", config.get("rolling_score_min_periods")),
            "min_entry_pred_bps": summary.get("min_entry_pred_bps", config.get("min_entry_pred_bps")),
            "min_score_margin_bps": summary.get("min_score_margin_bps", config.get("min_score_margin_bps")),
            "min_score_ratio": summary.get("min_score_ratio", config.get("min_score_ratio")),
            "score_history_rows": summary.get("score_history_rows", config.get("score_history_rows")),
            "hold_threshold_bps": policy.get("hold_threshold_bps", summary.get("hold_threshold_bps")),
        }
        for key in [
            "episode_count",
            "fold_count",
            "mean_episode_count",
            "worst_episode_count",
            "mean_gross_pl_bps_sum",
            "worst_gross_pl_bps_sum",
            "mean_net_pl_bps_sum",
            "worst_net_pl_bps_sum",
            "mean_avg_net_pl_bps",
            "worst_avg_net_pl_bps",
            "mean_median_net_pl_bps",
            "worst_median_net_pl_bps",
            "mean_win_rate",
            "worst_win_rate",
            "mean_avg_win_bps",
            "worst_avg_win_bps",
            "mean_avg_loss_bps",
            "worst_avg_loss_bps",
            "mean_profit_factor",
            "worst_profit_factor",
            "mean_avg_hold_minutes",
            "worst_avg_hold_minutes",
            "mean_median_hold_minutes",
            "worst_median_hold_minutes",
            "mean_round_trips_per_day",
            "worst_round_trips_per_day",
            "mean_fee_paid_bps",
            "worst_fee_paid_bps",
            "mean_avg_mfe_bps",
            "worst_avg_mfe_bps",
            "mean_avg_mae_bps",
            "worst_avg_mae_bps",
            "mean_avg_giveback_bps",
            "worst_avg_giveback_bps",
        ]:
            if key in summary:
                row[key] = summary[key]
        skipped = summary.get("skipped")
        if isinstance(skipped, dict):
            for k, v in skipped.items():
                row[f"skipped_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    preferred = [
        "candidate",
        "exit_lookahead_minutes",
        "exit_feature_set",
        "entry_selection_mode",
        "entry_threshold_bps",
        "entry_score_floor_bps",
        "rolling_score_quantile",
        "rolling_score_window_days",
        "min_entry_pred_bps",
        "min_score_margin_bps",
        "min_score_ratio",
        "score_history_rows",
        "hold_threshold_bps",
        "episode_count",
        "mean_net_pl_bps_sum",
        "worst_net_pl_bps_sum",
        "mean_avg_net_pl_bps",
        "worst_avg_net_pl_bps",
        "mean_profit_factor",
        "worst_profit_factor",
        "mean_round_trips_per_day",
        "mean_avg_hold_minutes",
        "mean_avg_mfe_bps",
        "mean_avg_mae_bps",
        "mean_avg_giveback_bps",
        "run_name",
        "run_dir",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[ordered]


def write_episode_comparison(df: pd.DataFrame, output: Path) -> Path:
    """Write selected downstream episode comparison table."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if "mean_net_pl_bps_sum" in df.columns:
        df = df.sort_values("mean_net_pl_bps_sum", ascending=False).reset_index(drop=True)
    return write_frame(df, output)

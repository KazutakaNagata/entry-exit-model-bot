"""Helpers for combining entry OOF predictions across runs.

Entry training writes one valid-fold prediction file per side/horizon run.
Exit dataset construction should consume only these out-of-sample valid
predictions, never in-sample fitted scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from swing_bot.artifacts.io import read_frame, read_json, write_frame, write_json
from swing_bot.models.lgbm_common import ModelTrainingError


@dataclass(frozen=True)
class EntryRunInfo:
    """Small metadata bundle for one trained entry run."""

    run_dir: Path
    run_id: str
    side: str
    horizon_minutes: int
    target_col: str
    prediction_file: Path


def _prediction_path(run_dir: Path) -> Path:
    parquet = run_dir / "predictions_valid.parquet"
    csv = run_dir / "predictions_valid.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"missing predictions_valid parquet/csv in {run_dir}")


def load_entry_run_info(run_dir: Path | str) -> EntryRunInfo:
    """Load metadata needed to identify a trained entry run."""
    rd = Path(run_dir)
    cfg_path = rd / "run_config.json"
    pred_path = _prediction_path(rd)
    if cfg_path.exists():
        cfg = read_json(cfg_path)
        side = str(cfg.get("side") or "")
        horizon = int(cfg.get("horizon_minutes"))
        target_col = str(cfg.get("target_col") or f"target_entry_net_bps_{side}_H{horizon}")
    else:
        # Fallback for manually assembled run dirs.  The prediction file must
        # contain identifying columns in this case.
        preds = read_frame(pred_path)
        if "side" not in preds.columns or "horizon_minutes" not in preds.columns:
            raise ModelTrainingError(f"{rd}: run_config.json missing and predictions do not contain side/horizon")
        side = str(preds["side"].dropna().iloc[0])
        horizon = int(preds["horizon_minutes"].dropna().iloc[0])
        target_candidates = [c for c in preds.columns if c.startswith("target_entry_net_bps_")]
        if not target_candidates:
            raise ModelTrainingError(f"{rd}: could not infer target column")
        target_col = target_candidates[0]
    if side not in {"long", "short"}:
        raise ModelTrainingError(f"{rd}: invalid side in run metadata: {side!r}")
    return EntryRunInfo(
        run_dir=rd,
        run_id=rd.name,
        side=side,
        horizon_minutes=horizon,
        target_col=target_col,
        prediction_file=pred_path,
    )


def load_entry_oof_predictions(run_dir: Path | str) -> pd.DataFrame:
    """Load one run's valid predictions and enforce OOF-only semantics."""
    info = load_entry_run_info(run_dir)
    df = read_frame(info.prediction_file)
    required = {"timestamp", "fold", "pred_entry_net_bps", "is_oof"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ModelTrainingError(f"{info.run_dir}: prediction file missing columns: {', '.join(missing)}")
    if not df["is_oof"].astype(bool).all():
        raise ModelTrainingError(f"{info.run_dir}: prediction file contains non-OOF rows")
    if info.target_col not in df.columns:
        raise ModelTrainingError(f"{info.run_dir}: missing target column {info.target_col!r}")

    out = df[["timestamp", "fold", info.target_col, "pred_entry_net_bps", "is_oof"]].copy()
    out.insert(0, "entry_run_id", info.run_id)
    out.insert(2, "side", info.side)
    out.insert(3, "horizon_minutes", int(info.horizon_minutes))
    out = out.rename(columns={info.target_col: "target_entry_net_bps"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["is_oof"] = out["is_oof"].astype(bool)
    return out.sort_values(["timestamp", "side", "horizon_minutes", "fold"]).reset_index(drop=True)


def combine_entry_oof_predictions(
    run_dirs: Sequence[Path | str],
    *,
    output_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine OOF valid predictions from multiple entry runs.

    Returns ``(combined_predictions, summary_by_run)``.  If ``output_dir`` is
    supplied, writes:

    - ``entry_oof_predictions.parquet``
    - ``entry_oof_summary.csv``
    - ``entry_oof_manifest.json``
    """
    if not run_dirs:
        raise ValueError("at least one run_dir is required")
    frames = [load_entry_oof_predictions(rd) for rd in run_dirs]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp", "side", "horizon_minutes", "entry_run_id"]).reset_index(drop=True)

    # One row per timestamp/side/horizon is expected.  Duplicate run ids for the
    # same side/horizon are ambiguous for downstream exit dataset creation.
    dup_cols = ["timestamp", "side", "horizon_minutes"]
    duplicate_count = int(combined.duplicated(dup_cols).sum())
    if duplicate_count:
        raise ModelTrainingError(
            "combined OOF predictions contain duplicate timestamp/side/horizon rows; "
            f"duplicates={duplicate_count}. Pass only one run per side/horizon."
        )

    summary = (
        combined.groupby(["entry_run_id", "side", "horizon_minutes"], as_index=False)
        .agg(
            rows=("timestamp", "size"),
            fold_count=("fold", "nunique"),
            pred_mean_bps=("pred_entry_net_bps", "mean"),
            pred_p95_bps=("pred_entry_net_bps", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
            target_mean_bps=("target_entry_net_bps", "mean"),
            target_gt0_rate=("target_entry_net_bps", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
        )
        .sort_values(["side", "horizon_minutes"])
        .reset_index(drop=True)
    )

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = write_frame(combined, out_dir / "entry_oof_predictions.parquet")
        summary_path = write_frame(summary, out_dir / "entry_oof_summary.csv")
        write_json(
            {
                "role": "entry_oof_predictions",
                "source_run_dirs": [str(Path(rd)) for rd in run_dirs],
                "prediction_file": str(pred_path),
                "summary_file": str(summary_path),
                "row_count": int(len(combined)),
                "run_count": int(len(run_dirs)),
                "notes": [
                    "All rows must have is_oof=true.",
                    "Downstream exit dataset builders must reject non-OOF entry scores.",
                    "This artifact contains valid-fold predictions only; it is not a test evaluation.",
                ],
            },
            out_dir / "entry_oof_manifest.json",
        )
    return combined, summary


def summarize_entry_run_dirs(run_dirs: Iterable[Path | str]) -> pd.DataFrame:
    """Build a compact comparison table from entry run artifacts."""
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        info = load_entry_run_info(run_dir)
        summary_path = info.run_dir / "summary_metrics.json"
        row: dict[str, object] = {
            "entry_run_id": info.run_id,
            "run_dir": str(info.run_dir),
            "side": info.side,
            "horizon_minutes": int(info.horizon_minutes),
            "target_col": info.target_col,
        }
        if summary_path.exists():
            summary = read_json(summary_path)
            for key in (
                "mean_top_q90_avg_target_bps",
                "worst_top_q90_avg_target_bps",
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
            ):
                if key in summary:
                    row[key] = summary[key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["side", "horizon_minutes", "entry_run_id"]).reset_index(drop=True)

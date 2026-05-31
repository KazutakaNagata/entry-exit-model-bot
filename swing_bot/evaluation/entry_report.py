"""Entry model report helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json
from swing_bot.evaluation.fold_metrics import summarize_fold_metrics
from swing_bot.evaluation.topk import score_decile_summary


def write_entry_report(
    *,
    predictions: pd.DataFrame,
    fold_metrics: Sequence[dict[str, object]],
    output_dir: Path,
    target_col: str,
    pred_col: str = "pred_entry_net_bps",
) -> dict[str, object]:
    """Write fold metrics, decile metrics, predictions, and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(fold_metrics)
    write_frame(metrics_df, output_dir / "fold_metrics.csv")

    decile_rows = []
    for fold_name, fold_df in predictions.groupby("fold", sort=True):
        deciles = score_decile_summary(fold_df[target_col], fold_df[pred_col])
        if not deciles.empty:
            deciles.insert(0, "fold", fold_name)
            decile_rows.append(deciles)
    decile_df = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()
    write_frame(decile_df, output_dir / "score_deciles.csv")
    write_frame(predictions, output_dir / "predictions_valid.parquet")

    summary = summarize_fold_metrics(fold_metrics)
    write_json(summary, output_dir / "summary_metrics.json")
    return summary

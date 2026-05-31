"""Writers for episode-backtest valid reports."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from swing_bot.artifacts.io import write_frame, write_json


def write_episode_report(
    *,
    episodes: pd.DataFrame,
    fold_metrics: Sequence[dict[str, object]],
    summary: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """Write episode rows, fold metrics, and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(episodes, output_dir / "episodes.parquet")
    write_frame(pd.DataFrame(fold_metrics), output_dir / "fold_metrics.csv")
    write_json(summary, output_dir / "summary_metrics.json")
    return summary

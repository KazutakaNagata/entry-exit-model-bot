"""Episode entry-overlap diagnostics.

The diagnostics in this module compare two already-generated valid-fold episode
backtests, for example:

* fixed-train entry model + fixed-hold exit
* walk-forward/rolling entry model + fixed-hold exit

It does not train models, create predictions, tune thresholds, or read test data.
The main question answered is:

    Are two entry policies selecting the same timestamps, and where does the P/L
    difference come from?

The primary matching key is exact ``(fold, side, entry_time)``.  This is
intentionally strict and live-safe.  If near-time clustering diagnostics are
needed later, add them as a separate diagnostic rather than silently relaxing the
main overlap definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class EntryOverlapError(ValueError):
    """Raised when episode overlap inputs are invalid."""


@dataclass(frozen=True)
class EpisodeOverlapResult:
    """Container for overlap diagnostics."""

    summary: dict[str, object]
    by_group: pd.DataFrame
    by_fold: pd.DataFrame
    episodes: pd.DataFrame


DEFAULT_PNL_COLUMNS = [
    "gross_pl_bps",
    "net_pl_bps",
    "mfe_bps",
    "mae_bps",
    "giveback_bps",
]

KEY_COLUMNS = ["fold", "side", "entry_time"]
TIME_COLUMNS = ["entry_time", "entry_exec_time", "exit_decision_time", "exit_exec_time"]


def read_episode_frame(path: Path) -> pd.DataFrame:
    """Read an episode parquet/csv file.

    ``path`` may be a file path or a run directory.  If it is a directory, this
    function looks for ``episodes.parquet`` first and then ``episodes.csv``.
    """
    p = Path(path)
    if p.is_dir():
        pq = p / "episodes.parquet"
        csv = p / "episodes.csv"
        if pq.exists():
            p = pq
        elif csv.exists():
            p = csv
        else:
            raise FileNotFoundError(f"episode run dir does not contain episodes.parquet/csv: {path}")

    if not p.exists():
        alt = p.with_suffix(".csv") if p.suffix == ".parquet" else p.with_suffix(".parquet")
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"episode file does not exist: {path}")

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise EntryOverlapError(f"unsupported episode file extension: {p.suffix}")


def write_frame(df: pd.DataFrame, path: Path) -> Path:
    """Write a dataframe, falling back to csv when parquet support is missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    df.to_csv(path, index=False)
    return path


def _normalise_episodes(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"entry_time", "side", "net_pl_bps"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise EntryOverlapError(f"{label} episodes missing required columns: {missing}")

    out = df.copy()
    if "fold" not in out.columns:
        if "fold_id" in out.columns:
            out = out.rename(columns={"fold_id": "fold"})
        else:
            raise EntryOverlapError(f"{label} episodes missing required column: fold or fold_id")

    out["fold"] = out["fold"].astype(str)
    out["side"] = out["side"].astype(str)
    for col in TIME_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    if out["entry_time"].isna().any():
        bad = int(out["entry_time"].isna().sum())
        raise EntryOverlapError(f"{label} episodes have {bad} unparseable entry_time rows")

    for col in DEFAULT_PNL_COLUMNS + ["entry_pred_net_bps", "effective_entry_threshold_bps", "rolling_score_threshold_bps"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    dup = int(out.duplicated(KEY_COLUMNS).sum())
    if dup:
        raise EntryOverlapError(
            f"{label} episodes contain {dup} duplicate exact entry keys {KEY_COLUMNS}. "
            "Deduplicate or compare run outputs before using overlap diagnostics."
        )
    return out.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def _prefix_non_key_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {c: f"{prefix}_{c}" for c in df.columns if c not in KEY_COLUMNS}
    return df.rename(columns=rename)


def make_overlap_episode_table(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return exact-overlap episode table for two episode sets."""
    l = _prefix_non_key_columns(left, "left")
    r = _prefix_non_key_columns(right, "right")
    merged = l.merge(r, on=KEY_COLUMNS, how="outer", indicator=True)
    group_map = {"both": "both", "left_only": "left_only", "right_only": "right_only"}
    merged["overlap_group"] = merged["_merge"].map(group_map).astype(str)
    merged = merged.drop(columns=["_merge"])

    # A single P/L column for group-level summaries.  For both entries, the left
    # and right P/L should usually be identical if the exit policy is identical;
    # keep both raw columns and use the left value as canonical.
    merged["canonical_net_pl_bps"] = np.where(
        merged["overlap_group"].eq("right_only"),
        merged.get("right_net_pl_bps"),
        merged.get("left_net_pl_bps"),
    )
    merged["canonical_gross_pl_bps"] = np.where(
        merged["overlap_group"].eq("right_only"),
        merged.get("right_gross_pl_bps"),
        merged.get("left_gross_pl_bps"),
    )
    for stat in ["mfe_bps", "mae_bps", "giveback_bps"]:
        left_col = f"left_{stat}"
        right_col = f"right_{stat}"
        if left_col in merged.columns or right_col in merged.columns:
            merged[f"canonical_{stat}"] = np.where(
                merged["overlap_group"].eq("right_only"),
                merged.get(right_col, np.nan),
                merged.get(left_col, np.nan),
            )

    if "left_net_pl_bps" in merged.columns and "right_net_pl_bps" in merged.columns:
        merged["net_pl_bps_diff_right_minus_left"] = merged["right_net_pl_bps"] - merged["left_net_pl_bps"]
    else:
        merged["net_pl_bps_diff_right_minus_left"] = np.nan
    return merged.sort_values(["fold", "entry_time", "side"], kind="mergesort").reset_index(drop=True)


def _profit_factor(net: pd.Series) -> float:
    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def _summarize_subset(df: pd.DataFrame, *, group: str | None = None, fold: str | None = None) -> dict[str, object]:
    net = pd.to_numeric(df.get("canonical_net_pl_bps", pd.Series(dtype="float64")), errors="coerce").dropna()
    gross = pd.to_numeric(df.get("canonical_gross_pl_bps", pd.Series(dtype="float64")), errors="coerce").dropna()
    row: dict[str, object] = {}
    if group is not None:
        row["overlap_group"] = group
    if fold is not None:
        row["fold"] = fold
    row.update({
        "episode_count": int(len(net)),
        "gross_pl_bps_sum": float(gross.sum()) if len(gross) else 0.0,
        "net_pl_bps_sum": float(net.sum()) if len(net) else 0.0,
        "avg_net_pl_bps": float(net.mean()) if len(net) else float("nan"),
        "median_net_pl_bps": float(net.median()) if len(net) else float("nan"),
        "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "avg_win_bps": float(net[net > 0].mean()) if (net > 0).any() else float("nan"),
        "avg_loss_bps": float(net[net < 0].mean()) if (net < 0).any() else float("nan"),
        "profit_factor": _profit_factor(net),
    })
    for stat in ["mfe_bps", "mae_bps", "giveback_bps"]:
        col = f"canonical_{stat}"
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"avg_{stat}"] = float(s.mean()) if len(s) else float("nan")
    return row


def summarize_overlap_groups(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["both", "left_only", "right_only"]:
        part = episodes[episodes["overlap_group"] == group]
        rows.append(_summarize_subset(part, group=group))
    return pd.DataFrame(rows)


def summarize_overlap_by_fold(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    folds = sorted(episodes["fold"].astype(str).unique().tolist())
    for fold in folds:
        fold_df = episodes[episodes["fold"].astype(str) == fold]
        for group in ["both", "left_only", "right_only"]:
            part = fold_df[fold_df["overlap_group"] == group]
            rows.append(_summarize_subset(part, group=group, fold=fold))
        rows.append(_summarize_subset(fold_df, group="all_union", fold=fold))
    return pd.DataFrame(rows)


def _sum_net(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").sum())


def make_overlap_summary(
    *,
    episodes: pd.DataFrame,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> dict[str, object]:
    both = episodes[episodes["overlap_group"] == "both"]
    left_only = episodes[episodes["overlap_group"] == "left_only"]
    right_only = episodes[episodes["overlap_group"] == "right_only"]
    union_count = int(len(episodes))
    left_count = int(len(left))
    right_count = int(len(right))
    both_count = int(len(both))

    left_net_sum = float(pd.to_numeric(left["net_pl_bps"], errors="coerce").sum()) if "net_pl_bps" in left.columns else float("nan")
    right_net_sum = float(pd.to_numeric(right["net_pl_bps"], errors="coerce").sum()) if "net_pl_bps" in right.columns else float("nan")

    return {
        "left_label": left_label,
        "right_label": right_label,
        "match_key": KEY_COLUMNS,
        "left_episode_count": left_count,
        "right_episode_count": right_count,
        "union_episode_count": union_count,
        "both_episode_count": both_count,
        "left_only_episode_count": int(len(left_only)),
        "right_only_episode_count": int(len(right_only)),
        "overlap_rate_left": float(both_count / left_count) if left_count else float("nan"),
        "overlap_rate_right": float(both_count / right_count) if right_count else float("nan"),
        "jaccard_overlap": float(both_count / union_count) if union_count else float("nan"),
        "left_net_pl_bps_sum": left_net_sum,
        "right_net_pl_bps_sum": right_net_sum,
        "right_minus_left_net_pl_bps_sum": right_net_sum - left_net_sum,
        "both_left_net_pl_bps_sum": _sum_net(both, "left_net_pl_bps"),
        "both_right_net_pl_bps_sum": _sum_net(both, "right_net_pl_bps"),
        "left_only_net_pl_bps_sum": _sum_net(left_only, "left_net_pl_bps"),
        "right_only_net_pl_bps_sum": _sum_net(right_only, "right_net_pl_bps"),
        "notes": [
            "Exact overlap uses fold + side + entry_time.",
            "left_only P/L answers what the left policy found that the right policy missed.",
            "right_only P/L answers what the right policy added relative to the left policy.",
            "This diagnostic reads valid-fold episode outputs only; it does not touch test data.",
        ],
    }


def compare_episode_entries(
    *,
    left_episodes: pd.DataFrame,
    right_episodes: pd.DataFrame,
    left_label: str = "left",
    right_label: str = "right",
) -> EpisodeOverlapResult:
    """Compare two episode backtest outputs by exact entry overlap."""
    left = _normalise_episodes(left_episodes, label=left_label)
    right = _normalise_episodes(right_episodes, label=right_label)
    episodes = make_overlap_episode_table(left, right)
    by_group = summarize_overlap_groups(episodes)
    by_fold = summarize_overlap_by_fold(episodes)
    summary = make_overlap_summary(
        episodes=episodes,
        left=left,
        right=right,
        left_label=left_label,
        right_label=right_label,
    )
    return EpisodeOverlapResult(summary=summary, by_group=by_group, by_fold=by_fold, episodes=episodes)

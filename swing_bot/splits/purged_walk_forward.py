"""Purged walk-forward helpers for timestamp-indexed minute data.

These helpers do not create labels.  They only remove train rows whose label
lookahead could overlap an evaluation range, plus an optional embargo after the
evaluation range.  For this project, tuning must happen on valid folds; test is
reserved for locked final audit.
"""
from __future__ import annotations

import pandas as pd

from swing_bot.splits.split_manifest import FoldSpec, SplitManifest, TimeRange


def _as_utc_series(timestamps: pd.Series | pd.DatetimeIndex) -> pd.Series:
    parsed = pd.to_datetime(timestamps, utc=True)
    if isinstance(parsed, pd.DatetimeIndex):
        return pd.Series(parsed)
    return parsed.reset_index(drop=True)


def label_interval_end(timestamps: pd.Series | pd.DatetimeIndex, max_label_minutes: int) -> pd.Series:
    """Return the latest future timestamp touched by a row's label.

    Entry labels use `t+1 open` through `t+1+H open`, so a conservative interval
    end for decision row `t` is `t + (1 + max_label_minutes)`.
    """
    ts = _as_utc_series(timestamps)
    return ts + pd.Timedelta(minutes=1 + int(max_label_minutes))


def purged_train_mask_for_eval_range(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    base_train_range: TimeRange,
    eval_range: TimeRange,
    max_label_minutes: int,
    embargo_minutes: int = 0,
) -> pd.Series:
    """Return mask for rows allowed in train when evaluating a time range.

    A train row is removed if its future label interval can overlap the eval
    range.  Rows inside the optional embargo window after eval are also removed.
    """
    ts = _as_utc_series(timestamps)
    train_mask = base_train_range.contains(ts)
    interval_end = label_interval_end(ts, max_label_minutes)

    overlaps_eval = (ts <= eval_range.end) & (interval_end >= eval_range.start)
    if embargo_minutes > 0:
        embargo_end = eval_range.end + pd.Timedelta(minutes=int(embargo_minutes))
        in_embargo = (ts > eval_range.end) & (ts <= embargo_end)
    else:
        in_embargo = pd.Series(False, index=ts.index)
    return (train_mask & ~overlaps_eval & ~in_embargo).reset_index(drop=True)


def valid_fold_masks(
    timestamps: pd.Series | pd.DatetimeIndex,
    manifest: SplitManifest,
    *,
    max_label_minutes: int | None = None,
) -> dict[str, dict[str, pd.Series]]:
    """Build train/eval masks for each valid fold.

    The training base is the manifest train range.  The eval mask is the fold
    range.  `max_label_minutes` defaults to manifest.purge_minutes when omitted.
    """
    effective_label_minutes = manifest.purge_minutes if max_label_minutes is None else int(max_label_minutes)
    masks: dict[str, dict[str, pd.Series]] = {}
    ts = _as_utc_series(timestamps)
    for fold in manifest.folds:
        train_mask = purged_train_mask_for_eval_range(
            ts,
            base_train_range=manifest.train,
            eval_range=fold.range,
            max_label_minutes=effective_label_minutes,
            embargo_minutes=manifest.embargo_minutes,
        )
        eval_mask = fold.range.contains(ts).reset_index(drop=True)
        masks[fold.name] = {"train": train_mask, "eval": eval_mask}
    return masks


def drop_fold_boundary_rows(
    timestamps: pd.Series | pd.DatetimeIndex,
    fold: FoldSpec,
    *,
    required_future_minutes: int,
) -> pd.Series:
    """Return rows whose future target stays inside the fold range.

    This is used for exit/episode datasets.  If a row's target would run past the
    fold end, drop it instead of allowing fold-boundary leakage.
    """
    ts = _as_utc_series(timestamps)
    target_end = ts + pd.Timedelta(minutes=1 + int(required_future_minutes))
    return ((ts >= fold.range.start) & (target_end <= fold.range.end)).reset_index(drop=True)

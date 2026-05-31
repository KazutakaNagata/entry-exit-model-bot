"""Fixed-hold valid-fold entry backtest.

This module evaluates an entry model without an exit model.  It is meant to
answer a narrow question before exit research:

    If the entry OOF signal fires, what happens if we simply hold for a fixed
    number of minutes and charge one roundtrip cost?

It does not read test data, it requires OOF entry predictions, and it supports
only live-feasible entry rules:

* fixed absolute threshold
* rolling quantile threshold computed from past scores only

Do not use fold-wide top quantiles as a backtest rule; that would use future
score distribution information from the validation fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from swing_bot.backtest.metrics import episode_fold_metrics, summarize_episode_metrics
from swing_bot.backtest.policy import EpisodePolicyConfig, should_enter
from swing_bot.labels.costs import side_normalized_return_bps, side_to_sign


class FixedHoldBacktestError(ValueError):
    """Raised when fixed-hold entry backtest inputs are unsafe."""


EntrySelectionMode = Literal["threshold", "rolling_quantile"]
RollingHistoryMode = Literal["fold_local", "continuous_valid", "score_history"]


@dataclass(frozen=True)
class FixedHoldBacktestConfig:
    """Configuration for fixed-hold entry backtests.

    ``fixed_hold_minutes`` is measured from execution open to exit open.
    A decision row at timestamp ``t`` enters at ``t+1min open`` and exits at
    ``t+1min+fixed_hold_minutes open``.

    ``entry_selection_mode='rolling_quantile'`` computes the score threshold
    from scores strictly before the current timestamp.  ``rolling_history_mode``
    controls whether the past-score history resets at each fold, carries over
    across prior valid folds, or is warmed up from an explicit score-history
    file.  In all modes, the current row and future rows are excluded.
    """

    side: str = "long"
    entry_horizon_minutes: int = 120
    fixed_hold_minutes: int = 120
    policy: EpisodePolicyConfig = EpisodePolicyConfig()
    entry_pred_col: str = "pred_entry_net_bps"
    entry_selection_mode: EntrySelectionMode = "threshold"
    rolling_score_window_days: int = 60
    rolling_score_quantile: float = 0.99
    rolling_score_min_periods: int = 1000
    rolling_history_mode: RollingHistoryMode = "fold_local"
    entry_score_floor_bps: float | None = None
    min_entry_pred_bps: float | None = None
    min_score_margin_bps: float | None = None
    min_score_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.fixed_hold_minutes <= 0:
            raise FixedHoldBacktestError("fixed_hold_minutes must be positive")
        if self.entry_horizon_minutes <= 0:
            raise FixedHoldBacktestError("entry_horizon_minutes must be positive")
        if self.entry_selection_mode not in {"threshold", "rolling_quantile"}:
            raise FixedHoldBacktestError("entry_selection_mode must be 'threshold' or 'rolling_quantile'")
        if self.rolling_score_window_days <= 0:
            raise FixedHoldBacktestError("rolling_score_window_days must be positive")
        if not 0.0 < float(self.rolling_score_quantile) < 1.0:
            raise FixedHoldBacktestError("rolling_score_quantile must be between 0 and 1")
        if self.rolling_score_min_periods < 1:
            raise FixedHoldBacktestError("rolling_score_min_periods must be positive")
        if self.rolling_history_mode not in {"fold_local", "continuous_valid", "score_history"}:
            raise FixedHoldBacktestError("rolling_history_mode must be 'fold_local', 'continuous_valid', or 'score_history'")
        if self.min_score_ratio is not None and float(self.min_score_ratio) <= 0.0:
            raise FixedHoldBacktestError("min_score_ratio must be positive when provided")
        # Validate side early.
        side_to_sign(self.side)


def fixed_hold_episode_id(side: str, entry_horizon_minutes: int, entry_time: pd.Timestamp) -> str:
    """Return a stable episode_id for fixed-hold entry rows."""
    ts = pd.Timestamp(entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return f"fixed_{side}_H{int(entry_horizon_minutes)}_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def _validate_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(ohlcv.columns))
    if missing:
        raise FixedHoldBacktestError("OHLCV missing required columns: " + ", ".join(missing))
    out = ohlcv.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise FixedHoldBacktestError("OHLCV contains unparseable timestamps")
    if out["timestamp"].duplicated().any():
        raise FixedHoldBacktestError("OHLCV timestamps must be unique")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _validate_entry_oof(entry_oof: pd.DataFrame, config: FixedHoldBacktestConfig) -> pd.DataFrame:
    required = {"timestamp", "fold", "side", "horizon_minutes", config.entry_pred_col, "is_oof"}
    missing = sorted(required - set(entry_oof.columns))
    if missing:
        raise FixedHoldBacktestError("entry predictions missing columns: " + ", ".join(missing))
    out = entry_oof.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise FixedHoldBacktestError("entry predictions contain unparseable timestamps")
    if not out["is_oof"].astype(bool).all():
        raise FixedHoldBacktestError("entry predictions contain non-OOF rows; refusing to backtest")
    out["side"] = out["side"].astype(str)
    out["horizon_minutes"] = pd.to_numeric(out["horizon_minutes"], errors="raise").astype(int)
    out[config.entry_pred_col] = pd.to_numeric(out[config.entry_pred_col], errors="coerce")
    out = out[(out["side"] == config.side) & (out["horizon_minutes"] == int(config.entry_horizon_minutes))].copy()
    if out.empty:
        raise FixedHoldBacktestError(f"no entry rows for side={config.side} H={config.entry_horizon_minutes}")
    dup = int(out.duplicated(["timestamp", "side", "horizon_minutes"]).sum())
    if dup:
        raise FixedHoldBacktestError(f"entry predictions contain duplicate timestamp/side/horizon rows: {dup}")
    return out.sort_values(["fold", "timestamp"], kind="mergesort").reset_index(drop=True)


def _validate_score_history(score_history: pd.DataFrame | None, config: FixedHoldBacktestConfig) -> pd.DataFrame | None:
    """Validate optional past score history used only for rolling thresholds.

    The history is never used as tradable rows.  It is only a calibration buffer
    for rolling score quantiles, e.g. train-period scores immediately before the
    first valid fold.  If a ``fold`` column is present, history rows are matched
    to the same evaluation fold; otherwise every history row is available as
    past history for every fold.
    """
    if score_history is None:
        return None
    required = {"timestamp", "side", "horizon_minutes", config.entry_pred_col}
    missing = sorted(required - set(score_history.columns))
    if missing:
        raise FixedHoldBacktestError("score history missing columns: " + ", ".join(missing))
    out = score_history.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise FixedHoldBacktestError("score history contains unparseable timestamps")
    out["side"] = out["side"].astype(str)
    out["horizon_minutes"] = pd.to_numeric(out["horizon_minutes"], errors="raise").astype(int)
    out[config.entry_pred_col] = pd.to_numeric(out[config.entry_pred_col], errors="coerce")
    out = out[(out["side"] == config.side) & (out["horizon_minutes"] == int(config.entry_horizon_minutes))].copy()
    if out.empty:
        return out
    if "fold" in out.columns:
        out["fold"] = out["fold"].astype(str)
        dup_cols = ["timestamp", "side", "horizon_minutes", "fold"]
    else:
        dup_cols = ["timestamp", "side", "horizon_minutes"]
    dup = int(out.duplicated(dup_cols).sum())
    if dup:
        raise FixedHoldBacktestError(f"score history contains duplicate rows for {dup_cols}: {dup}")
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _history_for_fold(score_history: pd.DataFrame | None, fold_name: str, eval_timestamps: pd.Series) -> pd.DataFrame:
    if score_history is None or score_history.empty:
        return pd.DataFrame()
    hist = score_history
    if "fold" in hist.columns:
        # Fold-specific history is preferred.  Rows tagged '*' or 'all' can be
        # used as shared global warmup history.
        fold_values = {str(fold_name), "*", "all", "global", "train"}
        hist = hist[hist["fold"].astype(str).isin(fold_values)].copy()
    else:
        hist = hist.copy()
    if hist.empty:
        return hist
    # Avoid duplicate current-timestamp calibration rows when history and eval
    # predictions overlap.  Future history rows are harmless for rolling windows
    # because they are after timestamp t, but same-timestamp duplicates can make
    # closed='left' semantics ambiguous.
    eval_ts = set(pd.to_datetime(eval_timestamps, utc=True))
    hist = hist[~hist["timestamp"].isin(eval_ts)].copy()
    return hist


def _compute_time_rolling_thresholds(
    combined: pd.DataFrame,
    *,
    pred_col: str,
    window: str,
    quantile: float,
    min_periods: int,
) -> pd.DataFrame:
    """Compute past-only rolling thresholds for rows marked ``_is_eval``.

    ``combined`` must contain timestamp, pred_col, _is_eval, and _eval_order.
    It may include non-tradable history rows.  The current row is excluded by
    ``closed='left'``.
    """
    if combined.empty:
        return combined
    combined = combined.sort_values(["timestamp", "_is_eval", "_eval_order"], kind="mergesort").reset_index(drop=True)
    s = combined.set_index("timestamp")[pred_col].astype(float)
    rolling = s.rolling(window=window, closed="left", min_periods=int(min_periods))
    combined["rolling_score_threshold_bps"] = rolling.quantile(float(quantile)).to_numpy(dtype="float64")
    combined["rolling_score_history_count"] = rolling.count().to_numpy(dtype="float64")
    return combined


def _finalize_rolling_threshold_frame(df: pd.DataFrame, *, floor: float) -> pd.DataFrame:
    out = df.copy()
    out["entry_score_floor_bps"] = float(floor)
    out["effective_entry_threshold_bps"] = np.maximum(
        out["rolling_score_threshold_bps"].to_numpy(dtype="float64"),
        float(floor),
    )
    return out


def _add_rolling_score_thresholds(entry: pd.DataFrame, config: FixedHoldBacktestConfig, score_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add live-safe rolling quantile thresholds to entry rows.

    Modes:
    * fold_local: each validation fold uses only its own past scores.
    * continuous_valid: validation folds are treated as one time-ordered stream;
      prior valid folds can warm later folds, but future rows are never used.
    * score_history: each fold can be warmed by an explicit score history file,
      e.g. train-tail scores produced by the same fold model.
    """
    if config.entry_selection_mode != "rolling_quantile":
        out = entry.copy()
        out["rolling_score_threshold_bps"] = np.nan
        out["entry_score_floor_bps"] = float(config.policy.entry_threshold_bps if config.entry_score_floor_bps is None else config.entry_score_floor_bps)
        out["effective_entry_threshold_bps"] = float(config.policy.entry_threshold_bps)
        out["rolling_score_history_count"] = 0.0
        return out

    window = f"{int(config.rolling_score_window_days)}D"
    floor = float(config.policy.entry_threshold_bps if config.entry_score_floor_bps is None else config.entry_score_floor_bps)

    if config.rolling_history_mode == "continuous_valid":
        eval_for_roll = entry.sort_values("timestamp", kind="mergesort").copy()
        eval_for_roll["_is_eval"] = True
        eval_for_roll["_eval_order"] = np.arange(len(eval_for_roll), dtype=int)
        combined = _compute_time_rolling_thresholds(
            eval_for_roll[["timestamp", config.entry_pred_col, "_is_eval", "_eval_order"]].copy(),
            pred_col=config.entry_pred_col,
            window=window,
            quantile=float(config.rolling_score_quantile),
            min_periods=int(config.rolling_score_min_periods),
        )
        thresholds = combined.loc[combined["_is_eval"].astype(bool), [
            "_eval_order",
            "rolling_score_threshold_bps",
            "rolling_score_history_count",
        ]].sort_values("_eval_order")
        if len(thresholds) != len(eval_for_roll):
            raise FixedHoldBacktestError("internal continuous rolling threshold alignment error")
        eval_for_roll["rolling_score_threshold_bps"] = thresholds["rolling_score_threshold_bps"].to_numpy(dtype="float64")
        eval_for_roll["rolling_score_history_count"] = thresholds["rolling_score_history_count"].to_numpy(dtype="float64")
        return _finalize_rolling_threshold_frame(eval_for_roll.drop(columns=["_is_eval", "_eval_order"]), floor=floor).sort_values(["fold", "timestamp"], kind="mergesort").reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for fold_name, group in entry.groupby("fold", sort=True):
        g = group.sort_values("timestamp", kind="mergesort").copy()
        if config.rolling_history_mode == "score_history":
            hist = _history_for_fold(score_history, str(fold_name), g["timestamp"])
        else:
            hist = pd.DataFrame()

        hist_cols = ["timestamp", config.entry_pred_col]
        hist_for_roll = hist[hist_cols].copy() if not hist.empty else pd.DataFrame(columns=hist_cols)
        hist_for_roll["_is_eval"] = False
        hist_for_roll["_eval_order"] = -1
        eval_for_roll = g[["timestamp", config.entry_pred_col]].copy()
        eval_for_roll["_is_eval"] = True
        eval_for_roll["_eval_order"] = np.arange(len(eval_for_roll), dtype=int)
        combined = pd.concat([hist_for_roll, eval_for_roll], axis=0, ignore_index=True)
        combined = _compute_time_rolling_thresholds(
            combined,
            pred_col=config.entry_pred_col,
            window=window,
            quantile=float(config.rolling_score_quantile),
            min_periods=int(config.rolling_score_min_periods),
        )
        eval_thresholds = combined.loc[combined["_is_eval"].astype(bool), [
            "_eval_order",
            "rolling_score_threshold_bps",
            "rolling_score_history_count",
        ]].sort_values("_eval_order")
        if len(eval_thresholds) != len(g):
            raise FixedHoldBacktestError("internal rolling threshold alignment error")
        g["rolling_score_threshold_bps"] = eval_thresholds["rolling_score_threshold_bps"].to_numpy(dtype="float64")
        g["rolling_score_history_count"] = eval_thresholds["rolling_score_history_count"].to_numpy(dtype="float64")
        parts.append(_finalize_rolling_threshold_frame(g, floor=floor))
    return pd.concat(parts, axis=0, ignore_index=True).sort_values(["fold", "timestamp"], kind="mergesort").reset_index(drop=True)


def _entry_threshold_for_row(row: object, config: FixedHoldBacktestConfig) -> tuple[float, float | None, str | None]:
    """Return (effective_threshold, rolling_threshold, skip_key_if_not_ready)."""
    if config.entry_selection_mode == "threshold":
        return float(config.policy.entry_threshold_bps), None, None
    threshold = float(getattr(row, "effective_entry_threshold_bps"))
    rolling_threshold = float(getattr(row, "rolling_score_threshold_bps"))
    if not np.isfinite(rolling_threshold) or not np.isfinite(threshold):
        return float("nan"), rolling_threshold, "rolling_quantile_not_ready"
    return threshold, rolling_threshold, None


def _score_ratio(entry_pred: float, threshold: float) -> float:
    """Return score / threshold for positive thresholds, otherwise NaN.

    The ratio filter is meant for positive predicted-net-return thresholds.
    If the threshold is zero or negative, using a ratio is ill-defined, so the
    caller should treat the ratio filter as not satisfied.
    """
    if not np.isfinite(entry_pred) or not np.isfinite(threshold) or threshold <= 0.0:
        return float("nan")
    return float(entry_pred / threshold)


def _should_enter_row(entry_pred: float, row: object, config: FixedHoldBacktestConfig) -> tuple[bool, str]:
    threshold, _, not_ready_key = _entry_threshold_for_row(row, config)
    if not_ready_key is not None:
        return False, not_ready_key

    if config.entry_selection_mode == "threshold":
        if not should_enter(entry_pred, config.policy):
            return False, "entry_below_threshold"
    else:
        if not entry_pred > threshold:
            return False, "entry_below_rolling_quantile"

    if config.min_entry_pred_bps is not None and entry_pred < float(config.min_entry_pred_bps):
        return False, "entry_below_min_pred"

    margin = float(entry_pred - threshold)
    if config.min_score_margin_bps is not None and margin < float(config.min_score_margin_bps):
        return False, "entry_below_min_score_margin"

    ratio = _score_ratio(entry_pred, threshold)
    if config.min_score_ratio is not None and (not np.isfinite(ratio) or ratio < float(config.min_score_ratio)):
        return False, "entry_below_min_score_ratio"

    return True, ""


def _price_lookup(ohlcv: pd.DataFrame) -> tuple[dict[pd.Timestamp, int], dict[str, np.ndarray], np.ndarray]:
    ts = pd.to_datetime(ohlcv["timestamp"], utc=True)
    pos_by_ts = {pd.Timestamp(t): int(i) for i, t in enumerate(ts)}
    arrays = {c: pd.to_numeric(ohlcv[c], errors="coerce").to_numpy(dtype="float64") for c in ["open", "high", "low", "close"]}
    return pos_by_ts, arrays, ts.to_numpy()


def _path_is_contiguous(ts_values: np.ndarray, start_pos: int, end_pos: int) -> bool:
    if end_pos < start_pos:
        return False
    start = pd.Timestamp(ts_values[start_pos])
    end = pd.Timestamp(ts_values[end_pos])
    expected_minutes = int((end - start) / pd.Timedelta(minutes=1)) + 1
    return expected_minutes == (end_pos - start_pos + 1)


def _episode_path_stats(
    *,
    arrays: dict[str, np.ndarray],
    ts_values: np.ndarray,
    pos_by_ts: dict[pd.Timestamp, int],
    side: str,
    entry_exec_time: pd.Timestamp,
    exit_exec_time: pd.Timestamp,
    roundtrip_cost_bps: float,
) -> dict[str, float] | None:
    entry_pos = pos_by_ts.get(entry_exec_time)
    exit_pos = pos_by_ts.get(exit_exec_time)
    if entry_pos is None or exit_pos is None or exit_pos < entry_pos:
        return None
    if not _path_is_contiguous(ts_values, entry_pos, exit_pos):
        return None
    entry_price = float(arrays["open"][entry_pos])
    exit_price = float(arrays["open"][exit_pos])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        return None

    path_high = arrays["high"][entry_pos : exit_pos + 1]
    path_low = arrays["low"][entry_pos : exit_pos + 1]
    if np.isnan(path_high).any() or np.isnan(path_low).any():
        return None

    sign = side_to_sign(side)
    if sign == 1:
        mfe = float(np.log(np.nanmax(path_high) / entry_price) * 10000.0)
        mae = float(np.log(np.nanmin(path_low) / entry_price) * 10000.0)
    else:
        mfe = float(-np.log(np.nanmin(path_low) / entry_price) * 10000.0)
        mae = float(-np.log(np.nanmax(path_high) / entry_price) * 10000.0)
    gross = float(side_normalized_return_bps(entry_price, exit_price, side))
    net = gross - float(roundtrip_cost_bps)
    giveback = float(max(0.0, mfe - gross)) if np.isfinite(mfe) and np.isfinite(gross) else float("nan")
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_pl_bps": gross,
        "net_pl_bps": net,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "giveback_bps": giveback,
    }


def run_fixed_hold_backtest(
    *,
    ohlcv: pd.DataFrame,
    entry_oof: pd.DataFrame,
    config: FixedHoldBacktestConfig,
    score_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """Run a fixed-hold valid-fold entry backtest.

    Returns ``(episodes, fold_metrics, summary)``.  No exit model predictions are
    used.  This is the baseline that should be compared with learned exits.
    """
    ohlcv_clean = _validate_ohlcv(ohlcv)
    history_clean = _validate_score_history(score_history, config)
    entry_clean = _add_rolling_score_thresholds(_validate_entry_oof(entry_oof, config), config, history_clean)
    pos_by_ts, arrays, ts_values = _price_lookup(ohlcv_clean)

    episodes: list[dict[str, object]] = []
    skipped = {
        "entry_below_threshold": 0,
        "entry_below_rolling_quantile": 0,
        "rolling_quantile_not_ready": 0,
        "entry_below_min_pred": 0,
        "entry_below_min_score_margin": 0,
        "entry_below_min_score_ratio": 0,
        "cooldown_or_reentry_block": 0,
        "max_round_trips_per_day": 0,
        "missing_price_path": 0,
    }

    for fold_name, fold_entries in entry_clean.groupby("fold", sort=True):
        next_allowed_entry_time = pd.Timestamp.min.tz_localize("UTC")
        daily_trips: dict[object, int] = {}
        fold_entries = fold_entries.sort_values("timestamp", kind="mergesort")
        for entry in fold_entries.itertuples(index=False):
            entry_time = pd.Timestamp(entry.timestamp)
            if entry_time < next_allowed_entry_time:
                skipped["cooldown_or_reentry_block"] += 1
                continue
            entry_pred = float(getattr(entry, config.entry_pred_col))
            allowed, skip_key = _should_enter_row(entry_pred, entry, config)
            if not allowed:
                skipped[skip_key] += 1
                continue
            entry_date = entry_time.date()
            if config.policy.max_round_trips_per_day is not None:
                if daily_trips.get(entry_date, 0) >= int(config.policy.max_round_trips_per_day):
                    skipped["max_round_trips_per_day"] += 1
                    continue

            entry_exec_time = entry_time + pd.Timedelta(minutes=1)
            exit_exec_time = entry_exec_time + pd.Timedelta(minutes=int(config.fixed_hold_minutes))
            stats = _episode_path_stats(
                arrays=arrays,
                ts_values=ts_values,
                pos_by_ts=pos_by_ts,
                side=config.side,
                entry_exec_time=entry_exec_time,
                exit_exec_time=exit_exec_time,
                roundtrip_cost_bps=config.policy.roundtrip_cost_bps,
            )
            if stats is None:
                skipped["missing_price_path"] += 1
                continue

            episode = {
                "fold": str(getattr(entry, "fold")),
                "episode_id": fixed_hold_episode_id(config.side, config.entry_horizon_minutes, entry_time),
                "side": config.side,
                "entry_horizon_minutes": int(config.entry_horizon_minutes),
                "fixed_hold_minutes": int(config.fixed_hold_minutes),
                "entry_time": entry_time,
                "entry_exec_time": entry_exec_time,
                "exit_decision_time": exit_exec_time - pd.Timedelta(minutes=1),
                "exit_exec_time": exit_exec_time,
                "exit_reason": "fixed_hold",
                "entry_selection_mode": config.entry_selection_mode,
                "rolling_history_mode": config.rolling_history_mode,
                "entry_pred_net_bps": entry_pred,
                "entry_threshold_bps": float(config.policy.entry_threshold_bps),
                "rolling_score_threshold_bps": float(getattr(entry, "rolling_score_threshold_bps", float("nan"))),
                "effective_entry_threshold_bps": float(getattr(entry, "effective_entry_threshold_bps", config.policy.entry_threshold_bps)),
                "entry_score_floor_bps": float(getattr(entry, "entry_score_floor_bps", config.policy.entry_threshold_bps)),
                "rolling_score_history_count": float(getattr(entry, "rolling_score_history_count", float("nan"))),
                "score_margin_bps": float(entry_pred - float(getattr(entry, "effective_entry_threshold_bps", config.policy.entry_threshold_bps))),
                "entry_score_margin_bps": float(entry_pred - float(getattr(entry, "effective_entry_threshold_bps", config.policy.entry_threshold_bps))),
                "score_ratio": _score_ratio(entry_pred, float(getattr(entry, "effective_entry_threshold_bps", config.policy.entry_threshold_bps))),
                "entry_score_ratio": _score_ratio(entry_pred, float(getattr(entry, "effective_entry_threshold_bps", config.policy.entry_threshold_bps))),
                "min_entry_pred_bps": float("nan") if config.min_entry_pred_bps is None else float(config.min_entry_pred_bps),
                "min_score_margin_bps": float("nan") if config.min_score_margin_bps is None else float(config.min_score_margin_bps),
                "min_score_ratio": float("nan") if config.min_score_ratio is None else float(config.min_score_ratio),
                "last_exit_pred_hold_delta_bps": float("nan"),
                "decision_count": 0,
                "hold_minutes": int(config.fixed_hold_minutes),
                "roundtrip_cost_bps": float(config.policy.roundtrip_cost_bps),
                **stats,
            }
            episodes.append(episode)
            daily_trips[entry_date] = daily_trips.get(entry_date, 0) + 1
            block_minutes = max(
                int(config.policy.cooldown_after_exit_minutes),
                int(config.policy.same_side_reentry_block_minutes),
            )
            next_allowed_entry_time = exit_exec_time + pd.Timedelta(minutes=block_minutes)

    episodes_df = pd.DataFrame(episodes)
    if not episodes_df.empty:
        for col in ["entry_time", "entry_exec_time", "exit_decision_time", "exit_exec_time"]:
            episodes_df[col] = pd.to_datetime(episodes_df[col], utc=True, errors="coerce")
        episodes_df = episodes_df.sort_values(["fold", "entry_time"], kind="mergesort").reset_index(drop=True)

    metrics = []
    for fold_name in sorted(entry_clean["fold"].astype(str).unique().tolist()):
        fold_eps = episodes_df[episodes_df["fold"] == fold_name] if not episodes_df.empty else pd.DataFrame()
        metrics.append(episode_fold_metrics(fold_eps, fold_name=fold_name))
    summary = summarize_episode_metrics(metrics)
    summary.update({
        "episode_count": int(len(episodes_df)),
        "side": config.side,
        "entry_horizon_minutes": int(config.entry_horizon_minutes),
        "fixed_hold_minutes": int(config.fixed_hold_minutes),
        "entry_selection_mode": config.entry_selection_mode,
        "rolling_history_mode": config.rolling_history_mode,
        "entry_threshold_bps": float(config.policy.entry_threshold_bps),
        "entry_score_floor_bps": float(config.policy.entry_threshold_bps if config.entry_score_floor_bps is None else config.entry_score_floor_bps),
        "score_history_rows": int(0 if history_clean is None else len(history_clean)),
        "rolling_score_window_days": int(config.rolling_score_window_days),
        "rolling_score_quantile": float(config.rolling_score_quantile),
        "rolling_score_min_periods": int(config.rolling_score_min_periods),
        "min_entry_pred_bps": None if config.min_entry_pred_bps is None else float(config.min_entry_pred_bps),
        "min_score_margin_bps": None if config.min_score_margin_bps is None else float(config.min_score_margin_bps),
        "min_score_ratio": None if config.min_score_ratio is None else float(config.min_score_ratio),
        "hold_threshold_bps": float("nan"),
        "roundtrip_cost_bps": float(config.policy.roundtrip_cost_bps),
        "skipped": skipped,
        "notes": [
            "Fixed-hold valid-fold entry baseline; no exit model predictions are used.",
            "Entry predictions must be OOF; non-OOF inputs are rejected.",
            "Roundtrip cost is charged once per completed episode.",
            "rolling_quantile mode uses past scores only; fold-wide future score quantiles are not used.",
            "rolling_history_mode controls whether history resets per fold, carries prior valid folds, or uses explicit score_history.",
            "Optional score-strength filters use only current score and the live-safe effective threshold.",
        ],
    })
    return episodes_df, metrics, summary

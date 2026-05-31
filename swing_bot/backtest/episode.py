"""Valid-fold episode backtest from OOF entry and exit predictions.

This MVP backtester is intentionally narrow:

* one side / one entry horizon at a time;
* OOF entry predictions only;
* OOF exit predictions only;
* roundtrip cost charged once per completed episode;
* no test-period logic.

The goal is plumbing validation before feature expansion, not strategy selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from swing_bot.backtest.metrics import episode_fold_metrics, summarize_episode_metrics
from swing_bot.backtest.policy import EpisodePolicyConfig, should_enter, should_hold
from swing_bot.labels.costs import side_normalized_return_bps, side_to_sign


class EpisodeBacktestError(ValueError):
    """Raised when an episode backtest cannot run safely."""


@dataclass(frozen=True)
class EpisodeBacktestConfig:
    side: str = "long"
    entry_horizon_minutes: int = 60
    policy: EpisodePolicyConfig = EpisodePolicyConfig()
    entry_pred_col: str = "pred_entry_net_bps"
    exit_pred_col: str = "pred_exit_hold_delta_bps"

    def __post_init__(self) -> None:
        side_to_sign(self.side)
        if self.entry_horizon_minutes <= 0:
            raise EpisodeBacktestError("entry_horizon_minutes must be positive")


def episode_id_for_entry(side: str, entry_horizon_minutes: int, entry_time: pd.Timestamp) -> str:
    """Return the same episode_id format used by make_exit_dataset.py."""
    ts = pd.Timestamp(entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return f"{side}_H{int(entry_horizon_minutes)}_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def _validate_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(ohlcv.columns))
    if missing:
        raise EpisodeBacktestError("OHLCV missing required columns: " + ", ".join(missing))
    out = ohlcv.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise EpisodeBacktestError("OHLCV contains unparseable timestamps")
    if out["timestamp"].duplicated().any():
        raise EpisodeBacktestError("OHLCV timestamps must be unique")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _validate_entry_oof(entry_oof: pd.DataFrame, config: EpisodeBacktestConfig) -> pd.DataFrame:
    required = {"timestamp", "fold", "side", "horizon_minutes", config.entry_pred_col, "is_oof"}
    missing = sorted(required - set(entry_oof.columns))
    if missing:
        raise EpisodeBacktestError("entry predictions missing columns: " + ", ".join(missing))
    out = entry_oof.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise EpisodeBacktestError("entry predictions contain unparseable timestamps")
    if not out["is_oof"].astype(bool).all():
        raise EpisodeBacktestError("entry predictions contain non-OOF rows; refusing to backtest")
    out["side"] = out["side"].astype(str)
    out["horizon_minutes"] = pd.to_numeric(out["horizon_minutes"], errors="raise").astype(int)
    out[config.entry_pred_col] = pd.to_numeric(out[config.entry_pred_col], errors="coerce")
    out = out[(out["side"] == config.side) & (out["horizon_minutes"] == int(config.entry_horizon_minutes))].copy()
    if out.empty:
        raise EpisodeBacktestError(f"no entry rows for side={config.side} H={config.entry_horizon_minutes}")
    dup = int(out.duplicated(["timestamp", "side", "horizon_minutes"]).sum())
    if dup:
        raise EpisodeBacktestError(f"entry predictions contain duplicate timestamp/side/horizon rows: {dup}")
    return out.sort_values(["fold", "timestamp"], kind="mergesort").reset_index(drop=True)


def _validate_exit_predictions(exit_predictions: pd.DataFrame, config: EpisodeBacktestConfig) -> pd.DataFrame:
    required = {"timestamp", "fold", "episode_id", config.exit_pred_col, "is_oof"}
    missing = sorted(required - set(exit_predictions.columns))
    if missing:
        raise EpisodeBacktestError("exit predictions missing columns: " + ", ".join(missing))
    out = exit_predictions.copy()
    for col in ["timestamp", "entry_time", "current_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            if out[col].isna().any():
                raise EpisodeBacktestError(f"exit predictions contain unparseable timestamps in {col}")
    if not out["is_oof"].astype(bool).all():
        raise EpisodeBacktestError("exit predictions contain non-OOF rows; refusing to backtest")
    out["fold"] = out["fold"].astype(str)
    out["episode_id"] = out["episode_id"].astype(str)
    out[config.exit_pred_col] = pd.to_numeric(out[config.exit_pred_col], errors="coerce")
    if "side" in out.columns:
        out = out[out["side"].astype(str) == config.side].copy()
    if "entry_horizon_minutes" in out.columns:
        out = out[pd.to_numeric(out["entry_horizon_minutes"], errors="coerce") == int(config.entry_horizon_minutes)].copy()
    if out.empty:
        raise EpisodeBacktestError("exit predictions are empty after side/horizon filtering")
    return out.sort_values(["fold", "episode_id", "timestamp"], kind="mergesort").reset_index(drop=True)


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


def _hard_stop_hit(mae_bps: float, hard_stop_bps: float | None) -> bool:
    if hard_stop_bps is None:
        return False
    if not np.isfinite(float(mae_bps)):
        return False
    return float(mae_bps) <= -abs(float(hard_stop_bps))


def _infer_exit_exec_time(exit_rows: pd.DataFrame, current_time: pd.Timestamp) -> pd.Timestamp:
    if "current_exit_time" in exit_rows.columns:
        vals = exit_rows.loc[exit_rows["timestamp"] == current_time, "current_exit_time"]
        if len(vals):
            value = pd.Timestamp(vals.iloc[0])
            return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return pd.Timestamp(current_time) + pd.Timedelta(minutes=1)


def _simulate_one_episode(
    *,
    entry_row: pd.Series,
    exit_rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    ts_values: np.ndarray,
    pos_by_ts: dict[pd.Timestamp, int],
    config: EpisodeBacktestConfig,
) -> dict[str, object] | None:
    policy = config.policy
    entry_time = pd.Timestamp(entry_row["timestamp"])
    entry_exec_time = entry_time + pd.Timedelta(minutes=1)
    episode_id = episode_id_for_entry(config.side, config.entry_horizon_minutes, entry_time)
    rows = exit_rows[exit_rows["episode_id"] == episode_id].copy().sort_values("timestamp")
    if rows.empty:
        return None

    exit_confirm_count = 0
    last_exit_pred = float("nan")
    last_decision_time = None
    exit_reason = "data_boundary"
    exit_exec_time = None
    decision_count = 0

    for row in rows.itertuples(index=False):
        current_time = pd.Timestamp(getattr(row, "timestamp"))
        last_decision_time = current_time
        decision_count += 1
        pred = float(getattr(row, config.exit_pred_col))
        last_exit_pred = pred
        current_exit_time = _infer_exit_exec_time(rows, current_time)

        stats_now = _episode_path_stats(
            arrays=arrays,
            ts_values=ts_values,
            pos_by_ts=pos_by_ts,
            side=config.side,
            entry_exec_time=entry_exec_time,
            exit_exec_time=current_exit_time,
            roundtrip_cost_bps=policy.roundtrip_cost_bps,
        )
        if stats_now is None:
            continue

        age_minutes = int((current_time - entry_time) / pd.Timedelta(minutes=1))
        if _hard_stop_hit(float(stats_now["mae_bps"]), policy.hard_stop_bps):
            exit_reason = "hard_stop"
            exit_exec_time = current_exit_time
            break
        if age_minutes >= int(policy.max_hold_minutes):
            exit_reason = "max_hold"
            exit_exec_time = current_exit_time
            break

        if should_hold(pred, policy):
            exit_confirm_count = 0
            continue
        exit_confirm_count += 1
        if exit_confirm_count >= int(policy.exit_confirm_bars):
            exit_reason = "exit_signal"
            exit_exec_time = current_exit_time
            break

    if exit_exec_time is None:
        if last_decision_time is None:
            return None
        exit_exec_time = pd.Timestamp(last_decision_time) + pd.Timedelta(minutes=1)

    final_stats = _episode_path_stats(
        arrays=arrays,
        ts_values=ts_values,
        pos_by_ts=pos_by_ts,
        side=config.side,
        entry_exec_time=entry_exec_time,
        exit_exec_time=exit_exec_time,
        roundtrip_cost_bps=policy.roundtrip_cost_bps,
    )
    if final_stats is None:
        return None

    hold_minutes = int((pd.Timestamp(exit_exec_time) - entry_exec_time) / pd.Timedelta(minutes=1))
    return {
        "fold": str(entry_row["fold"]),
        "episode_id": episode_id,
        "side": config.side,
        "entry_horizon_minutes": int(config.entry_horizon_minutes),
        "entry_time": entry_time,
        "entry_exec_time": entry_exec_time,
        "exit_decision_time": last_decision_time,
        "exit_exec_time": exit_exec_time,
        "exit_reason": exit_reason,
        "entry_pred_net_bps": float(entry_row[config.entry_pred_col]),
        "last_exit_pred_hold_delta_bps": last_exit_pred,
        "decision_count": int(decision_count),
        "hold_minutes": int(hold_minutes),
        "roundtrip_cost_bps": float(policy.roundtrip_cost_bps),
        **final_stats,
    }


def run_episode_backtest(
    *,
    ohlcv: pd.DataFrame,
    entry_oof: pd.DataFrame,
    exit_predictions: pd.DataFrame,
    config: EpisodeBacktestConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """Run a deterministic valid-fold episode backtest.

    The function returns ``(episodes, fold_metrics, summary)``.
    """
    ohlcv_clean = _validate_ohlcv(ohlcv)
    entry_clean = _validate_entry_oof(entry_oof, config)
    exit_clean = _validate_exit_predictions(exit_predictions, config)

    pos_by_ts, arrays, ts_values = _price_lookup(ohlcv_clean)
    exit_episode_ids = set(exit_clean["episode_id"].astype(str).unique())

    episodes: list[dict[str, object]] = []
    skipped = {
        "entry_below_threshold": 0,
        "cooldown_or_reentry_block": 0,
        "max_round_trips_per_day": 0,
        "missing_exit_path": 0,
        "missing_price_path": 0,
    }

    for fold_name, fold_entries in entry_clean.groupby("fold", sort=True):
        next_allowed_entry_time = pd.Timestamp.min.tz_localize("UTC")
        daily_trips: dict[object, int] = {}
        fold_entries = fold_entries.sort_values("timestamp", kind="mergesort")
        fold_exits = exit_clean[exit_clean["fold"] == str(fold_name)].copy()
        for entry in fold_entries.itertuples(index=False):
            entry_time = pd.Timestamp(entry.timestamp)
            if entry_time < next_allowed_entry_time:
                skipped["cooldown_or_reentry_block"] += 1
                continue
            entry_pred = float(getattr(entry, config.entry_pred_col))
            if not should_enter(entry_pred, config.policy):
                skipped["entry_below_threshold"] += 1
                continue
            entry_date = entry_time.date()
            if config.policy.max_round_trips_per_day is not None:
                if daily_trips.get(entry_date, 0) >= int(config.policy.max_round_trips_per_day):
                    skipped["max_round_trips_per_day"] += 1
                    continue
            ep_id = episode_id_for_entry(config.side, config.entry_horizon_minutes, entry_time)
            if ep_id not in exit_episode_ids:
                skipped["missing_exit_path"] += 1
                continue

            entry_series = pd.Series(entry._asdict())
            episode = _simulate_one_episode(
                entry_row=entry_series,
                exit_rows=fold_exits,
                arrays=arrays,
                ts_values=ts_values,
                pos_by_ts=pos_by_ts,
                config=config,
            )
            if episode is None:
                skipped["missing_price_path"] += 1
                continue
            episodes.append(episode)
            daily_trips[entry_date] = daily_trips.get(entry_date, 0) + 1
            exit_exec_time = pd.Timestamp(episode["exit_exec_time"])
            block_minutes = max(int(config.policy.cooldown_after_exit_minutes), int(config.policy.same_side_reentry_block_minutes))
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
        "entry_threshold_bps": float(config.policy.entry_threshold_bps),
        "hold_threshold_bps": float(config.policy.hold_threshold_bps),
        "roundtrip_cost_bps": float(config.policy.roundtrip_cost_bps),
        "skipped": skipped,
        "notes": [
            "This is a plumbing MVP. Performance should not be interpreted before feature expansion.",
            "Entry and exit predictions must be OOF; non-OOF inputs are rejected.",
            "Roundtrip cost is charged once per completed episode.",
        ],
    })
    return episodes_df, metrics, summary

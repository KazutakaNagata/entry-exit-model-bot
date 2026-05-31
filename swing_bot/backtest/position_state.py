"""Build position-state datasets for supervised exit models.

Exit rows are conditioned on an entry policy.  In train/valid research this means
entry candidates must come from OOF entry predictions, not in-sample scores.
Each output row represents one position state:

    (entry_time, current_time, side)

The target is a simple hold-delta: exit at the next open after current_time versus
exit at the next open after ``current_time + lookahead``.  Position-state
features are computed only from bars available by ``current_time``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from swing_bot.labels.costs import net_pl_bps, side_normalized_return_bps, side_to_sign
from swing_bot.splits.split_manifest import FoldSpec, SplitManifest


class ExitDatasetError(ValueError):
    """Raised when an exit dataset cannot be built safely."""


@dataclass(frozen=True)
class ExitDatasetConfig:
    """Configuration for one exit training dataset."""

    side: str
    entry_horizon_minutes: int
    exit_lookahead_minutes: int
    max_hold_minutes: int
    decision_interval_minutes: int = 5
    candidate_interval_minutes: int = 5
    min_entry_pred_bps: float | None = 0.0
    top_entry_quantile: float | None = None
    roundtrip_cost_bps: float = 15.0

    def __post_init__(self) -> None:
        side_to_sign(self.side)
        if self.entry_horizon_minutes <= 0:
            raise ExitDatasetError("entry_horizon_minutes must be positive")
        if self.exit_lookahead_minutes <= 0:
            raise ExitDatasetError("exit_lookahead_minutes must be positive")
        if self.max_hold_minutes <= 0:
            raise ExitDatasetError("max_hold_minutes must be positive")
        if self.decision_interval_minutes <= 0:
            raise ExitDatasetError("decision_interval_minutes must be positive")
        if self.candidate_interval_minutes <= 0:
            raise ExitDatasetError("candidate_interval_minutes must be positive")
        if self.top_entry_quantile is not None and not (0.0 < float(self.top_entry_quantile) < 1.0):
            raise ExitDatasetError("top_entry_quantile must be between 0 and 1 when provided")

    @property
    def target_col(self) -> str:
        return f"target_exit_hold_delta_bps_{self.side}_K{self.exit_lookahead_minutes}"

    @property
    def dataset_name(self) -> str:
        return f"exit_{self.side}_entryH{self.entry_horizon_minutes}_K{self.exit_lookahead_minutes}"


def _utc_series(values: pd.Series | Sequence[object]) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True, errors="coerce")


def _validate_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(ohlcv.columns))
    if missing:
        raise ExitDatasetError("OHLCV missing required columns: " + ", ".join(missing))
    out = ohlcv.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ExitDatasetError("OHLCV contains unparseable timestamps")
    if out["timestamp"].duplicated().any():
        raise ExitDatasetError("OHLCV timestamps must be unique")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _validate_features(features: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in features.columns:
        raise ExitDatasetError("features must contain timestamp")
    forbidden = ("target", "future", "hold_delta", "mfe", "mae", "diag_")
    bad = [c for c in features.columns if c != "timestamp" and any(token in c.lower() for token in forbidden)]
    if bad:
        raise ExitDatasetError("features contain forbidden label/diagnostic columns: " + ", ".join(bad[:20]))
    out = features.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ExitDatasetError("features contain unparseable timestamps")
    if out["timestamp"].duplicated().any():
        raise ExitDatasetError("feature timestamps must be unique")
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _validate_entry_oof(entry_oof: pd.DataFrame, config: ExitDatasetConfig) -> pd.DataFrame:
    required = {"timestamp", "fold", "side", "horizon_minutes", "pred_entry_net_bps", "is_oof"}
    missing = sorted(required - set(entry_oof.columns))
    if missing:
        raise ExitDatasetError("entry OOF predictions missing columns: " + ", ".join(missing))
    out = entry_oof.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ExitDatasetError("entry OOF predictions contain unparseable timestamps")
    if not out["is_oof"].astype(bool).all():
        raise ExitDatasetError("entry OOF predictions contain non-OOF rows; refusing to build exit dataset")
    out["side"] = out["side"].astype(str)
    out["horizon_minutes"] = pd.to_numeric(out["horizon_minutes"], errors="raise").astype(int)
    out["pred_entry_net_bps"] = pd.to_numeric(out["pred_entry_net_bps"], errors="coerce")
    out = out[(out["side"] == config.side) & (out["horizon_minutes"] == int(config.entry_horizon_minutes))].copy()
    if out.empty:
        raise ExitDatasetError(
            f"no OOF rows found for side={config.side} horizon={config.entry_horizon_minutes}"
        )
    dup_count = int(out.duplicated(["timestamp", "side", "horizon_minutes"]).sum())
    if dup_count:
        raise ExitDatasetError(f"entry OOF predictions contain duplicate timestamp/side/horizon rows: {dup_count}")
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _candidate_time_mask(timestamps: pd.Series, interval_minutes: int) -> pd.Series:
    if interval_minutes <= 1:
        return pd.Series(True, index=timestamps.index)
    ts = pd.to_datetime(timestamps, utc=True)
    minute_number = (ts.astype("int64") // (60 * 1_000_000_000)).astype("int64")
    return pd.Series((minute_number % int(interval_minutes)) == 0, index=timestamps.index)


def _filter_entry_candidates(entry_oof: pd.DataFrame, config: ExitDatasetConfig) -> pd.DataFrame:
    candidates = entry_oof[entry_oof["pred_entry_net_bps"].notna()].copy()
    candidates = candidates[_candidate_time_mask(candidates["timestamp"], config.candidate_interval_minutes)].copy()
    if config.min_entry_pred_bps is not None:
        candidates = candidates[candidates["pred_entry_net_bps"] > float(config.min_entry_pred_bps)].copy()
    if config.top_entry_quantile is not None:
        q = float(config.top_entry_quantile)
        # Select within each fold so one high-volatility fold does not dominate the dataset.
        thresholds = candidates.groupby("fold")["pred_entry_net_bps"].transform(lambda s: s.quantile(q))
        candidates = candidates[candidates["pred_entry_net_bps"] >= thresholds].copy()
    return candidates.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _fold_by_name(manifest: SplitManifest) -> dict[str, FoldSpec]:
    return {fold.name: fold for fold in manifest.folds}


def _make_score_lookup(entry_oof: pd.DataFrame) -> dict[pd.Timestamp, float]:
    return {
        pd.Timestamp(row.timestamp): float(row.pred_entry_net_bps)
        for row in entry_oof[["timestamp", "pred_entry_net_bps"]].itertuples(index=False)
        if pd.notna(row.pred_entry_net_bps)
    }


def _market_features_at_current_time(features: pd.DataFrame) -> tuple[list[str], dict[pd.Timestamp, dict[str, object]]]:
    feature_cols = [c for c in features.columns if c != "timestamp"]
    records = features.set_index("timestamp")[feature_cols].to_dict(orient="index")
    return feature_cols, records


def _price_maps(ohlcv: pd.DataFrame) -> tuple[dict[pd.Timestamp, int], dict[str, np.ndarray], np.ndarray]:
    ts = pd.to_datetime(ohlcv["timestamp"], utc=True)
    pos_by_ts = {pd.Timestamp(t): int(i) for i, t in enumerate(ts)}
    arrays = {col: pd.to_numeric(ohlcv[col], errors="coerce").to_numpy(dtype="float64") for col in ["open", "high", "low", "close"]}
    return pos_by_ts, arrays, ts.to_numpy()


def _path_is_contiguous(ts_values: np.ndarray, start_pos: int, end_pos: int) -> bool:
    expected_rows = end_pos - start_pos + 1
    if expected_rows <= 0:
        return False
    start = pd.Timestamp(ts_values[start_pos])
    end = pd.Timestamp(ts_values[end_pos])
    expected_minutes = int((end - start) / pd.Timedelta(minutes=1)) + 1
    return expected_minutes == expected_rows


def _position_path_features(
    *,
    arrays: dict[str, np.ndarray],
    ts_values: np.ndarray,
    pos_by_ts: dict[pd.Timestamp, int],
    side: str,
    entry_exec_time: pd.Timestamp,
    current_time: pd.Timestamp,
    roundtrip_cost_bps: float,
) -> dict[str, float] | None:
    entry_pos = pos_by_ts.get(entry_exec_time)
    current_pos = pos_by_ts.get(current_time)
    if entry_pos is None or current_pos is None or current_pos < entry_pos:
        return None
    if not _path_is_contiguous(ts_values, entry_pos, current_pos):
        return None

    entry_price = float(arrays["open"][entry_pos])
    current_close = float(arrays["close"][current_pos])
    if not np.isfinite(entry_price) or not np.isfinite(current_close) or entry_price <= 0 or current_close <= 0:
        return None

    path_high = arrays["high"][entry_pos : current_pos + 1]
    path_low = arrays["low"][entry_pos : current_pos + 1]
    if np.isnan(path_high).any() or np.isnan(path_low).any():
        return None

    sign = side_to_sign(side)
    if sign == 1:
        mfe = float(np.log(np.nanmax(path_high) / entry_price) * 10000.0)
        mae = float(np.log(np.nanmin(path_low) / entry_price) * 10000.0)
    else:
        mfe = float(-np.log(np.nanmin(path_low) / entry_price) * 10000.0)
        mae = float(-np.log(np.nanmax(path_high) / entry_price) * 10000.0)

    unrealized = float(side_normalized_return_bps(entry_price, current_close, side))
    unrealized_net = float(net_pl_bps(entry_price, current_close, side, roundtrip_cost_bps=roundtrip_cost_bps))
    giveback = float(max(0.0, mfe - unrealized)) if np.isfinite(mfe) and np.isfinite(unrealized) else np.nan
    return {
        "entry_price": entry_price,
        "current_close": current_close,
        "unrealized_pl_bps": unrealized,
        "unrealized_pl_after_cost_bps": unrealized_net,
        "mfe_since_entry_bps": mfe,
        "mae_since_entry_bps": mae,
        "giveback_from_mfe_bps": giveback,
    }


def _hold_delta_target(
    *,
    arrays: dict[str, np.ndarray],
    pos_by_ts: dict[pd.Timestamp, int],
    side: str,
    current_exit_time: pd.Timestamp,
    future_exit_time: pd.Timestamp,
) -> float | None:
    current_pos = pos_by_ts.get(current_exit_time)
    future_pos = pos_by_ts.get(future_exit_time)
    if current_pos is None or future_pos is None:
        return None
    current_exit_price = float(arrays["open"][current_pos])
    future_exit_price = float(arrays["open"][future_pos])
    if not np.isfinite(current_exit_price) or not np.isfinite(future_exit_price):
        return None
    if current_exit_price <= 0 or future_exit_price <= 0:
        return None
    return float(side_normalized_return_bps(current_exit_price, future_exit_price, side))


def build_exit_position_dataset(
    *,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame | None,
    entry_oof: pd.DataFrame,
    split_manifest: SplitManifest,
    config: ExitDatasetConfig,
    include_market_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a policy-conditioned exit dataset from OOF entry candidates.

    The function returns ``(dataset, summary)``.  It refuses non-OOF entry scores
    and drops any row whose target would cross the valid fold boundary.
    """
    ohlcv_clean = _validate_ohlcv(ohlcv)
    features_clean = _validate_features(features) if include_market_features else None
    entry_clean = _validate_entry_oof(entry_oof, config)
    candidates = _filter_entry_candidates(entry_clean, config)

    pos_by_ts, arrays, ts_values = _price_maps(ohlcv_clean)
    if include_market_features:
        if features_clean is None:
            raise ExitDatasetError("features must be provided when include_market_features=True")
        feature_cols, feature_lookup = _market_features_at_current_time(features_clean)
    else:
        feature_cols, feature_lookup = [], {}
    score_lookup = _make_score_lookup(entry_clean)
    folds = _fold_by_name(split_manifest)
    side_sign = side_to_sign(config.side)
    del side_sign  # validation only; sign is used by helper functions.

    rows: list[dict[str, object]] = []
    skipped = {
        "unknown_fold": 0,
        "entry_outside_fold": 0,
        "no_market_features": 0,
        "bad_position_path": 0,
        "missing_target_price": 0,
        "fold_boundary": 0,
    }

    for candidate in candidates.itertuples(index=False):
        fold_name = str(candidate.fold)
        fold = folds.get(fold_name)
        if fold is None:
            skipped["unknown_fold"] += 1
            continue
        entry_time = pd.Timestamp(candidate.timestamp)
        if not bool(fold.range.contains(pd.DatetimeIndex([entry_time]))[0]):
            skipped["entry_outside_fold"] += 1
            continue
        entry_exec_time = entry_time + pd.Timedelta(minutes=1)
        entry_score = float(candidate.pred_entry_net_bps)
        episode_id = f"{config.side}_H{config.entry_horizon_minutes}_{entry_time.strftime('%Y%m%dT%H%M%SZ')}"

        for age_minutes in range(
            int(config.decision_interval_minutes),
            int(config.max_hold_minutes) + 1,
            int(config.decision_interval_minutes),
        ):
            current_time = entry_time + pd.Timedelta(minutes=age_minutes)
            current_exit_time = current_time + pd.Timedelta(minutes=1)
            future_exit_time = current_time + pd.Timedelta(minutes=1 + int(config.exit_lookahead_minutes))
            if current_time < fold.range.start or future_exit_time > fold.range.end:
                skipped["fold_boundary"] += 1
                continue
            if include_market_features and current_time not in feature_lookup:
                skipped["no_market_features"] += 1
                continue

            path_features = _position_path_features(
                arrays=arrays,
                ts_values=ts_values,
                pos_by_ts=pos_by_ts,
                side=config.side,
                entry_exec_time=entry_exec_time,
                current_time=current_time,
                roundtrip_cost_bps=float(config.roundtrip_cost_bps),
            )
            if path_features is None:
                skipped["bad_position_path"] += 1
                continue
            target = _hold_delta_target(
                arrays=arrays,
                pos_by_ts=pos_by_ts,
                side=config.side,
                current_exit_time=current_exit_time,
                future_exit_time=future_exit_time,
            )
            if target is None or not np.isfinite(target):
                skipped["missing_target_price"] += 1
                continue

            current_score = score_lookup.get(current_time, np.nan)
            row: dict[str, object] = {
                "timestamp": current_time,
                "entry_time": entry_time,
                "entry_exec_time": entry_exec_time,
                "current_time": current_time,
                "current_exit_time": current_exit_time,
                "future_exit_time": future_exit_time,
                "fold": fold_name,
                "episode_id": episode_id,
                "side": config.side,
                "entry_horizon_minutes": int(config.entry_horizon_minutes),
                "exit_lookahead_minutes": int(config.exit_lookahead_minutes),
                "age_minutes": int(age_minutes),
                "entry_pred_net_bps": entry_score,
                "current_pred_net_bps": current_score,
                "score_decay_bps": float(current_score - entry_score) if np.isfinite(current_score) else np.nan,
                "is_entry_score_oof": True,
                "is_current_score_oof": bool(np.isfinite(current_score)),
                config.target_col: target,
                "target_exit_hold_delta_bps": target,
            }
            row.update(path_features)
            if include_market_features:
                row.update(feature_lookup[current_time])
            rows.append(row)

    dataset = pd.DataFrame(rows)
    if not dataset.empty:
        dataset = dataset.sort_values(["timestamp", "episode_id", "age_minutes"]).reset_index(drop=True)

    summary: dict[str, object] = {
        "dataset_name": config.dataset_name,
        "side": config.side,
        "entry_horizon_minutes": int(config.entry_horizon_minutes),
        "exit_lookahead_minutes": int(config.exit_lookahead_minutes),
        "max_hold_minutes": int(config.max_hold_minutes),
        "decision_interval_minutes": int(config.decision_interval_minutes),
        "candidate_interval_minutes": int(config.candidate_interval_minutes),
        "min_entry_pred_bps": config.min_entry_pred_bps,
        "top_entry_quantile": config.top_entry_quantile,
        "roundtrip_cost_bps": float(config.roundtrip_cost_bps),
        "entry_oof_rows_for_side_horizon": int(len(entry_clean)),
        "candidate_entries": int(len(candidates)),
        "exit_rows": int(len(dataset)),
        "episode_count": int(dataset["episode_id"].nunique()) if not dataset.empty else 0,
        "feature_count_market": int(len(feature_cols)) if include_market_features else 0,
        "skipped": skipped,
        "notes": [
            "Entry candidates are selected from OOF entry predictions only.",
            "Rows whose future exit target crosses a valid fold boundary are dropped.",
            "Position features use only bars from entry execution through current_time.",
            "target_exit_hold_delta_bps does not subtract a fresh roundtrip cost.",
        ],
    }
    return dataset, summary


def exit_dataset_feature_columns(df: pd.DataFrame, *, target_col: str = "target_exit_hold_delta_bps") -> list[str]:
    """Return candidate feature columns for exit model training."""
    meta_cols = {
        "timestamp",
        "entry_time",
        "entry_exec_time",
        "current_time",
        "current_exit_time",
        "future_exit_time",
        "fold",
        "episode_id",
        "side",
        "entry_horizon_minutes",
        "exit_lookahead_minutes",
        target_col,
    }
    forbidden_tokens = ("target_exit_hold_delta_bps_",)
    cols = []
    for col in df.columns:
        if col in meta_cols:
            continue
        if any(token in col for token in forbidden_tokens):
            continue
        cols.append(col)
    return cols

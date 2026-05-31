"""Monthly rolling research protocol utilities.

This module evaluates a *research process* rather than one hand-picked feature
set.  For each rolling test month, candidate feature policies are selected from
three prior monthly validation folds, then the selected policy is trained on the
nine months immediately before the test month and evaluated on the next month.

The module can evaluate either a small explicit candidate list or a mechanically
generated family-ablation policy universe; code applies a fixed selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import gc

import numpy as np
import pandas as pd

from swing_bot.artifacts.io import read_yaml, write_frame, write_json
from swing_bot.backtest.fixed_hold import FixedHoldBacktestConfig, run_fixed_hold_backtest
from swing_bot.backtest.metrics import summarize_episode_metrics
from swing_bot.backtest.policy import EpisodePolicyConfig
from swing_bot.evaluation.episode_report import write_episode_report
from swing_bot.labels.entry_net_return import entry_target_column
from swing_bot.models.entry_lgbm import merge_features_and_labels
from swing_bot.models.lgbm_common import (
    ModelTrainingError,
    assert_no_forbidden_feature_columns,
    load_lgbm_params,
    make_lgbm_regressor,
    numeric_feature_columns,
    predict_regressor,
    save_lgbm_model,
)
from swing_bot.splits.purged_walk_forward import label_interval_end
from swing_bot.research.family_policy_selection import selector_metrics_from_fold_metrics


@dataclass(frozen=True)
class FeaturePolicy:
    name: str
    features_path: Path
    description: str = ""


@dataclass(frozen=True)
class RollingMonthlyConfig:
    name: str
    side: str
    horizon_minutes: int
    fixed_hold_minutes: int
    train_months: int
    valid_months: int
    test_months: int
    step_months: int
    score_history_days: int
    score_window_days: int
    score_quantile: float
    score_floor_bps: float
    rolling_score_min_periods: int
    entry_selection_mode: str
    score_quantile_min_periods: int
    roundtrip_cost_bps: float
    cooldown_after_exit_minutes: int
    same_side_reentry_block_minutes: int
    label_lag_minutes: int
    model_config_path: Path
    labels_path: Path
    ohlcv_path: Path
    feature_policies: tuple[FeaturePolicy, ...]
    min_valid_episode_count: int = 10
    min_delta_to_switch_bps: float = 0.0
    robust_worst_weight: float = 0.5
    robust_median_weight: float = 0.25
    min_active_valid_folds: int = 2
    max_worst_valid_loss_bps: float | None = None
    selector_fail_penalty: float = 1_000_000.0
    max_train_rows: int | None = None
    force_col_wise: bool = True
    downcast_float32: bool = True
    save_models: bool = False

    @property
    def target_col(self) -> str:
        return entry_target_column(self.side, self.horizon_minutes)


@dataclass(frozen=True)
class MonthlyCycle:
    cycle_id: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    valid_months: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]


def month_start(ts: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(pd.to_datetime(ts, utc=True))
    return pd.Timestamp(year=parsed.year, month=parsed.month, day=1, tz="UTC")


def _ensure_utc_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    parsed = pd.Timestamp(pd.to_datetime(ts, utc=True))
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    base = _ensure_utc_timestamp(ts)
    p = pd.Period(base.strftime("%Y-%m"), freq="M") + int(months)
    return pd.Timestamp(p.start_time, tz="UTC")


def month_end_exclusive(start: pd.Timestamp, months: int = 1) -> pd.Timestamp:
    return add_months(start, months)


def inclusive_end(exclusive_end: pd.Timestamp) -> pd.Timestamp:
    return _ensure_utc_timestamp(exclusive_end) - pd.Timedelta(minutes=1)


def build_monthly_cycles(
    *,
    first_test_month: str | pd.Timestamp,
    last_test_month: str | pd.Timestamp,
    valid_months: int,
    test_months: int = 1,
    step_months: int = 1,
) -> list[MonthlyCycle]:
    """Build rolling monthly cycles from first through last test month inclusive."""
    first = month_start(first_test_month)
    last = month_start(last_test_month)
    cycles: list[MonthlyCycle] = []
    test_start = first
    while test_start <= last:
        test_excl = month_end_exclusive(test_start, test_months)
        valid_ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        for i in range(valid_months, 0, -1):
            v_start = add_months(test_start, -i)
            v_excl = add_months(v_start, 1)
            valid_ranges.append((f"valid_m{valid_months - i + 1:02d}", v_start, inclusive_end(v_excl)))
        cycles.append(
            MonthlyCycle(
                cycle_id=f"test_{test_start.strftime('%Y%m')}",
                test_start=test_start,
                test_end=inclusive_end(test_excl),
                valid_months=tuple(valid_ranges),
            )
        )
        test_start = add_months(test_start, step_months)
    return cycles


def load_rolling_monthly_config(path: Path | str, *, allow_empty_feature_policies: bool = False) -> RollingMonthlyConfig:
    raw = read_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"rolling monthly config must be a mapping: {path}")
    policies = []
    for item in raw.get("feature_policies") or []:
        policies.append(
            FeaturePolicy(
                name=str(item["name"]),
                features_path=Path(item["features"]),
                description=str(item.get("description") or ""),
            )
        )
    if not policies and not allow_empty_feature_policies:
        raise ValueError("rolling monthly config must define feature_policies")
    entry = raw.get("entry_selection") or {}
    train = raw.get("training") or {}
    paths = raw.get("paths") or {}
    selector = raw.get("selector") or {}
    return RollingMonthlyConfig(
        name=str(raw.get("name") or Path(path).stem),
        side=str(raw.get("side", "long")),
        horizon_minutes=int(raw.get("horizon_minutes", 120)),
        fixed_hold_minutes=int(raw.get("fixed_hold_minutes", raw.get("horizon_minutes", 120))),
        train_months=int(train.get("train_months", 9)),
        valid_months=int(train.get("valid_months", 3)),
        test_months=int(train.get("test_months", 1)),
        step_months=int(train.get("step_months", 1)),
        score_history_days=int(train.get("score_history_days", 120)),
        label_lag_minutes=int(train.get("label_lag_minutes", raw.get("horizon_minutes", 120))),
        model_config_path=Path(paths.get("model_config", "configs/models/lgbm_entry_v0.yaml")),
        labels_path=Path(paths.get("labels", "outputs/valid/labels/btcjpy_1m_labels.parquet")),
        ohlcv_path=Path(paths.get("ohlcv", "data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet")),
        score_window_days=int(entry.get("score_window_days", 60)),
        score_quantile=float(entry.get("score_quantile", entry.get("train_score_quantile", 0.995))),
        score_floor_bps=float(entry.get("score_floor_bps", 40.0)),
        rolling_score_min_periods=int(entry.get("rolling_score_min_periods", 30000)),
        entry_selection_mode=str(entry.get("mode", "rolling_quantile")),
        score_quantile_min_periods=int(entry.get("score_quantile_min_periods", entry.get("rolling_score_min_periods", 30000))),
        roundtrip_cost_bps=float(raw.get("roundtrip_cost_bps", 15.0)),
        cooldown_after_exit_minutes=int(raw.get("cooldown_after_exit_minutes", 15)),
        same_side_reentry_block_minutes=int(raw.get("same_side_reentry_block_minutes", 30)),
        min_valid_episode_count=int(selector.get("min_valid_episode_count", selector.get("min_total_episode_count", 10))),
        min_delta_to_switch_bps=float(selector.get("min_delta_to_switch_bps", 0.0)),
        robust_worst_weight=float(selector.get("robust_worst_weight", 0.5)),
        robust_median_weight=float(selector.get("robust_median_weight", 0.25)),
        min_active_valid_folds=int(selector.get("min_active_valid_folds", 2)),
        max_worst_valid_loss_bps=(None if selector.get("max_worst_valid_loss_bps") in (None, "") else float(selector.get("max_worst_valid_loss_bps"))),
        selector_fail_penalty=float(selector.get("selector_fail_penalty", 1_000_000.0)),
        max_train_rows=(None if train.get("max_train_rows") in (None, "", 0) else int(train.get("max_train_rows"))),
        force_col_wise=bool(train.get("force_col_wise", True)),
        downcast_float32=bool(train.get("downcast_float32", True)),
        save_models=bool(train.get("save_models", False)),
        feature_policies=tuple(policies),
    )


def _as_utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True)
    if isinstance(parsed, pd.DatetimeIndex):
        return pd.Series(parsed)
    return parsed.reset_index(drop=True)


def _downcast(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float32", copy=False)
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("float32")
    return df


def _fit_predict_segment(
    *,
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
    fold_name: str,
    side: str,
    horizon_minutes: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    score_history_days: int,
    label_lag_minutes: int,
    max_train_rows: int | None,
    save_model_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ts = data["timestamp"]
    safe_label_end = label_interval_end(ts, int(horizon_minutes)) < eval_start
    train_mask = (ts >= train_start) & (ts <= train_end) & safe_label_end & data[target_col].notna()
    train_idx = train_mask[train_mask].index
    if max_train_rows is not None and int(max_train_rows) > 0 and len(train_idx) > int(max_train_rows):
        train_idx = ts.loc[train_idx].sort_values().index[-int(max_train_rows):]
    eval_mask = (ts >= eval_start) & (ts <= eval_end) & data[target_col].notna()
    eval_idx = eval_mask[eval_mask].index
    hist_start = eval_start - pd.Timedelta(days=int(score_history_days))
    history_mask = (ts >= hist_start) & (ts < eval_start)
    history_idx = history_mask[history_mask].index
    if len(train_idx) == 0:
        raise ModelTrainingError(f"{fold_name}: no train rows")
    if len(eval_idx) == 0:
        raise ModelTrainingError(f"{fold_name}: no eval rows")
    y = pd.to_numeric(data.loc[train_idx, target_col], errors="coerce")
    ok = y.notna()
    fit_idx = train_idx[ok.to_numpy()]
    model = make_lgbm_regressor(params)
    model.fit(data.loc[fit_idx, feature_cols], y.loc[fit_idx])
    eval_view = data.loc[eval_idx, ["timestamp", target_col] + feature_cols]
    pred = predict_regressor(model, eval_view, feature_cols)
    pred_df = pd.DataFrame({
        "timestamp": eval_view["timestamp"].to_numpy(),
        "fold": fold_name,
        "side": side,
        "horizon_minutes": int(horizon_minutes),
        target_col: eval_view[target_col].to_numpy(),
        "pred_entry_net_bps": pred,
        "is_oof": True,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "train_count": int(len(fit_idx)),
    })
    if len(history_idx) > 0:
        hist_view = data.loc[history_idx, ["timestamp"] + feature_cols]
        hist_pred = predict_regressor(model, hist_view, feature_cols)
        hist_df = pd.DataFrame({
            "timestamp": hist_view["timestamp"].to_numpy(),
            "fold": fold_name,
            "side": side,
            "horizon_minutes": int(horizon_minutes),
            "pred_entry_net_bps": hist_pred,
            "source": "monthly_pre_fold_score_history",
        })
    else:
        hist_df = pd.DataFrame(columns=["timestamp", "fold", "side", "horizon_minutes", "pred_entry_net_bps", "source"])
    if save_model_path is not None:
        save_lgbm_model(model, save_model_path)
    meta = {
        "fold": fold_name,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "eval_start": eval_start.isoformat(),
        "eval_end": eval_end.isoformat(),
        "train_count": int(len(fit_idx)),
        "eval_count": int(len(eval_idx)),
        "score_history_count": int(len(history_idx)),
    }
    del model, eval_view
    if "hist_view" in locals():
        del hist_view
    gc.collect()
    return pred_df, hist_df, meta


def _sum_skipped(summaries: list[dict[str, Any]]) -> dict[str, int]:
    keys: set[str] = set()
    for summary in summaries:
        skipped = summary.get("skipped") or {}
        if isinstance(skipped, dict):
            keys.update(str(k) for k in skipped.keys())
    return {key: int(sum(int((summary.get("skipped") or {}).get(key, 0) or 0) for summary in summaries)) for key in sorted(keys)}


def _threshold_from_history(
    *,
    score_history: pd.DataFrame,
    fold_name: str,
    pred_col: str,
    quantile: float,
    floor_bps: float,
    min_periods: int,
) -> tuple[float, float, int]:
    if score_history is None or score_history.empty:
        return float("inf"), float("nan"), 0
    hist = score_history.copy()
    if "fold" in hist.columns:
        allowed = {str(fold_name), "*", "all", "global", "train"}
        hist = hist[hist["fold"].astype(str).isin(allowed)].copy()
    if hist.empty or pred_col not in hist.columns:
        return float("inf"), float("nan"), 0
    scores = pd.to_numeric(hist[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    count = int(len(scores))
    if count < int(min_periods):
        return float("inf"), float("nan"), count
    q = float(np.quantile(scores.to_numpy(dtype="float64"), float(quantile)))
    threshold = float(max(q, float(floor_bps)))
    return threshold, q, count


def _backtest_score_history_quantile(
    *,
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
    score_history: pd.DataFrame,
    cfg: RollingMonthlyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Backtest using one fixed threshold per evaluation fold/month.

    The threshold is computed from strictly pre-period score_history rows:
        max(quantile(score_history.pred, cfg.score_quantile), cfg.score_floor_bps)

    This is intended for monthly rolling research where score_history contains
    the 120d (or configured) pre-period scores produced by that month/fold's
    model.  It avoids rolling-in-period warmup and does not use evaluation-month
    score distribution.
    """
    all_episodes: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, Any]] = []
    fold_threshold_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    pred_col = "pred_entry_net_bps"
    for fold_name, group in predictions.groupby("fold", sort=True):
        threshold, raw_q, hist_count = _threshold_from_history(
            score_history=score_history,
            fold_name=str(fold_name),
            pred_col=pred_col,
            quantile=cfg.score_quantile,
            floor_bps=cfg.score_floor_bps,
            min_periods=cfg.score_quantile_min_periods,
        )
        policy = EpisodePolicyConfig(
            entry_threshold_bps=threshold,
            hold_threshold_bps=0.0,
            roundtrip_cost_bps=cfg.roundtrip_cost_bps,
            max_hold_minutes=cfg.fixed_hold_minutes,
            decision_interval_minutes=5,
            cooldown_after_exit_minutes=cfg.cooldown_after_exit_minutes,
            same_side_reentry_block_minutes=cfg.same_side_reentry_block_minutes,
        )
        backtest_cfg = FixedHoldBacktestConfig(
            side=cfg.side,
            entry_horizon_minutes=cfg.horizon_minutes,
            fixed_hold_minutes=cfg.fixed_hold_minutes,
            policy=policy,
            entry_selection_mode="threshold",
            entry_score_floor_bps=threshold,
        )
        episodes, fold_metrics, summary = run_fixed_hold_backtest(ohlcv=ohlcv, entry_oof=group.copy(), config=backtest_cfg)
        if not episodes.empty:
            episodes = episodes.copy()
            episodes["entry_selection_mode"] = "score_history_quantile"
            episodes["score_history_quantile"] = float(cfg.score_quantile)
            episodes["score_history_quantile_threshold_bps"] = float(raw_q)
            episodes["effective_entry_threshold_bps"] = float(threshold)
            episodes["entry_threshold_bps"] = float(threshold)
            episodes["entry_score_floor_bps"] = float(cfg.score_floor_bps)
            episodes["score_history_rows_for_threshold"] = int(hist_count)
        all_episodes.append(episodes)
        all_fold_metrics.extend(fold_metrics)
        summaries.append(summary)
        fold_threshold_rows.append({
            "fold": str(fold_name),
            "score_history_rows_for_threshold": int(hist_count),
            "score_history_quantile": float(cfg.score_quantile),
            "score_history_quantile_threshold_bps": raw_q,
            "score_floor_bps": float(cfg.score_floor_bps),
            "effective_entry_threshold_bps": threshold,
        })
    episodes_df = pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()
    summary = summarize_episode_metrics(all_fold_metrics)
    summary.update({
        "episode_count": int(len(episodes_df)),
        "side": cfg.side,
        "entry_horizon_minutes": int(cfg.horizon_minutes),
        "fixed_hold_minutes": int(cfg.fixed_hold_minutes),
        "entry_selection_mode": "score_history_quantile",
        "rolling_history_mode": "score_history",
        "entry_threshold_bps": float("nan"),
        "entry_score_floor_bps": float(cfg.score_floor_bps),
        "score_history_rows": int(0 if score_history is None else len(score_history)),
        "score_history_quantile": float(cfg.score_quantile),
        "score_quantile_min_periods": int(cfg.score_quantile_min_periods),
        "fold_thresholds": fold_threshold_rows,
        "hold_threshold_bps": float("nan"),
        "roundtrip_cost_bps": float(cfg.roundtrip_cost_bps),
        "skipped": _sum_skipped(summaries),
        "notes": [
            "Fixed-hold monthly research backtest using pre-period score-history quantile thresholds.",
            "The score quantile is computed from score_history rows only; evaluation-month score distribution is not used.",
            "Roundtrip cost is charged once per completed episode.",
        ],
    })
    return episodes_df, pd.DataFrame(all_fold_metrics), summary


def _backtest(
    *,
    ohlcv: pd.DataFrame,
    predictions: pd.DataFrame,
    score_history: pd.DataFrame,
    cfg: RollingMonthlyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mode = str(cfg.entry_selection_mode)
    if mode in {"score_history_quantile", "preperiod_quantile", "train_tail_quantile"}:
        return _backtest_score_history_quantile(ohlcv=ohlcv, predictions=predictions, score_history=score_history, cfg=cfg)
    if mode != "rolling_quantile":
        raise ValueError(f"unsupported monthly entry_selection mode: {mode}")
    policy = EpisodePolicyConfig(
        entry_threshold_bps=cfg.score_floor_bps,
        hold_threshold_bps=0.0,
        roundtrip_cost_bps=cfg.roundtrip_cost_bps,
        max_hold_minutes=cfg.fixed_hold_minutes,
        decision_interval_minutes=5,
        cooldown_after_exit_minutes=cfg.cooldown_after_exit_minutes,
        same_side_reentry_block_minutes=cfg.same_side_reentry_block_minutes,
    )
    backtest_cfg = FixedHoldBacktestConfig(
        side=cfg.side,
        entry_horizon_minutes=cfg.horizon_minutes,
        fixed_hold_minutes=cfg.fixed_hold_minutes,
        policy=policy,
        entry_selection_mode="rolling_quantile",
        rolling_history_mode="score_history",
        rolling_score_window_days=cfg.score_window_days,
        rolling_score_quantile=cfg.score_quantile,
        rolling_score_min_periods=cfg.rolling_score_min_periods,
        entry_score_floor_bps=cfg.score_floor_bps,
    )
    return run_fixed_hold_backtest(ohlcv=ohlcv, entry_oof=predictions, config=backtest_cfg, score_history=score_history)


def robust_selector_score(
    summary: dict[str, Any],
    cfg: RollingMonthlyConfig,
    *,
    fold_metrics: Iterable[dict[str, Any]] | pd.DataFrame | None = None,
) -> float:
    """Return the fixed 3-fold robust selector score.

    Preferred formula, using validation fold/month metrics:

        total_net + worst_weight * worst_month_net + median_weight * median_month_net

    Candidate policies that do not satisfy minimum activity / loss constraints are
    penalized rather than silently removed, so reports still show them.
    """
    if fold_metrics is None:
        # Backward-compatible fallback for older call sites.  This is less exact
        # because summary stores mean/worst rather than total/median.
        mean_net = float(summary.get("mean_net_pl_bps_sum", 0.0) or 0.0)
        worst_net = float(summary.get("worst_net_pl_bps_sum", 0.0) or 0.0)
        episode_count = int(summary.get("episode_count", 0) or 0)
        penalty = 0.0 if episode_count >= int(cfg.min_valid_episode_count) else cfg.selector_fail_penalty
        return mean_net + cfg.robust_worst_weight * worst_net - penalty
    fm = pd.DataFrame(fold_metrics)
    metrics = selector_metrics_from_fold_metrics(
        fm,
        min_total_episode_count=int(cfg.min_valid_episode_count),
        min_active_fold_count=int(cfg.min_active_valid_folds),
        max_worst_fold_loss_bps=cfg.max_worst_valid_loss_bps,
        worst_weight=float(cfg.robust_worst_weight),
        median_weight=float(cfg.robust_median_weight),
        fail_penalty=float(cfg.selector_fail_penalty),
    )
    return float(metrics["robust_score"])


@dataclass(frozen=True)
class MonthlyResearchResult:
    run_dir: Path
    cycles_csv: Path
    candidate_csv: Path
    test_metrics_csv: Path


def run_rolling_monthly_research(
    *,
    config: RollingMonthlyConfig,
    first_test_month: str,
    last_test_month: str,
    output_root: Path,
    run_id: str,
    max_cycles: int | None = None,
) -> MonthlyResearchResult:
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cycles = build_monthly_cycles(
        first_test_month=first_test_month,
        last_test_month=last_test_month,
        valid_months=config.valid_months,
        test_months=config.test_months,
        step_months=config.step_months,
    )
    if max_cycles is not None:
        cycles = cycles[: int(max_cycles)]
    ohlcv = pd.read_parquet(config.ohlcv_path) if str(config.ohlcv_path).endswith(".parquet") else pd.read_csv(config.ohlcv_path)
    labels = pd.read_parquet(config.labels_path) if str(config.labels_path).endswith(".parquet") else pd.read_csv(config.labels_path)
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    params = load_lgbm_params(config.model_config_path)
    if config.force_col_wise:
        params = {**params, "force_col_wise": True}

    candidate_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    cached_data: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    def data_for_policy(policy: FeaturePolicy) -> tuple[pd.DataFrame, list[str]]:
        if policy.name in cached_data:
            return cached_data[policy.name]
        features = pd.read_parquet(policy.features_path) if str(policy.features_path).endswith(".parquet") else pd.read_csv(policy.features_path)
        data = merge_features_and_labels(features, labels, target_col=config.target_col)
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        feature_cols = numeric_feature_columns(data, exclude={config.target_col})
        assert_no_forbidden_feature_columns(feature_cols)
        if config.downcast_float32:
            data = _downcast(data, feature_cols, config.target_col)
        cached_data[policy.name] = (data, feature_cols)
        return data, feature_cols

    for cycle in cycles:
        cycle_dir = run_dir / cycle.cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)
        best_policy: FeaturePolicy | None = None
        best_score = -np.inf
        best_valid_summary: dict[str, Any] | None = None
        for policy in config.feature_policies:
            data, feature_cols = data_for_policy(policy)
            preds: list[pd.DataFrame] = []
            hists: list[pd.DataFrame] = []
            segment_meta: list[dict[str, Any]] = []
            for valid_name, v_start, v_end in cycle.valid_months:
                train_end = v_start - pd.Timedelta(minutes=int(config.label_lag_minutes))
                train_start = v_start - pd.DateOffset(months=int(config.train_months))
                fold_name = f"{cycle.cycle_id}_{valid_name}_{policy.name}"
                pred_df, hist_df, meta = _fit_predict_segment(
                    data=data,
                    feature_cols=feature_cols,
                    target_col=config.target_col,
                    params=params,
                    fold_name=fold_name,
                    side=config.side,
                    horizon_minutes=config.horizon_minutes,
                    train_start=_ensure_utc_timestamp(train_start),
                    train_end=_ensure_utc_timestamp(train_end),
                    eval_start=v_start,
                    eval_end=v_end,
                    score_history_days=config.score_history_days,
                    label_lag_minutes=config.label_lag_minutes,
                    max_train_rows=config.max_train_rows,
                )
                preds.append(pred_df)
                hists.append(hist_df)
                segment_meta.append(meta)
            valid_preds = pd.concat(preds, ignore_index=True)
            valid_hist = pd.concat(hists, ignore_index=True) if hists else pd.DataFrame()
            episodes, fold_metrics, summary = _backtest(ohlcv=ohlcv, predictions=valid_preds, score_history=valid_hist, cfg=config)
            selector_metrics = selector_metrics_from_fold_metrics(
                pd.DataFrame(fold_metrics),
                min_total_episode_count=int(config.min_valid_episode_count),
                min_active_fold_count=int(config.min_active_valid_folds),
                max_worst_fold_loss_bps=config.max_worst_valid_loss_bps,
                worst_weight=float(config.robust_worst_weight),
                median_weight=float(config.robust_median_weight),
                fail_penalty=float(config.selector_fail_penalty),
            )
            score = float(selector_metrics["robust_score"])
            candidate_row = {
                "cycle_id": cycle.cycle_id,
                "policy": policy.name,
                "robust_score": score,
                "selected": False,
                **selector_metrics,
                **{k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool)) or v is None},
            }
            candidate_rows.append(candidate_row)
            policy_dir = cycle_dir / "valid_candidates" / policy.name
            write_episode_report(episodes=episodes, fold_metrics=fold_metrics, summary=summary, output_dir=policy_dir)
            write_frame(valid_preds, policy_dir / "entry_oof_predictions.parquet")
            if not valid_hist.empty:
                write_frame(valid_hist, policy_dir / "score_history.parquet")
            write_json({"segments": segment_meta, "policy": policy.__dict__, "summary": summary, "robust_score": score}, policy_dir / "candidate_run.json")
            if score > best_score:
                best_score = score
                best_policy = policy
                best_valid_summary = summary
        assert best_policy is not None
        for row in candidate_rows:
            if row["cycle_id"] == cycle.cycle_id and row["policy"] == best_policy.name:
                row["selected"] = True
        # Train selected policy for the rolling test month.
        data, feature_cols = data_for_policy(best_policy)
        test_train_end = cycle.test_start - pd.Timedelta(minutes=int(config.label_lag_minutes))
        test_train_start = cycle.test_start - pd.DateOffset(months=int(config.train_months))
        test_fold = f"{cycle.cycle_id}_test_{best_policy.name}"
        pred_df, hist_df, test_meta = _fit_predict_segment(
            data=data,
            feature_cols=feature_cols,
            target_col=config.target_col,
            params=params,
            fold_name=test_fold,
            side=config.side,
            horizon_minutes=config.horizon_minutes,
            train_start=_ensure_utc_timestamp(test_train_start),
            train_end=_ensure_utc_timestamp(test_train_end),
            eval_start=cycle.test_start,
            eval_end=cycle.test_end,
            score_history_days=config.score_history_days,
            label_lag_minutes=config.label_lag_minutes,
            max_train_rows=config.max_train_rows,
        )
        test_episodes, test_fold_metrics, test_summary = _backtest(ohlcv=ohlcv, predictions=pred_df, score_history=hist_df, cfg=config)
        test_dir = cycle_dir / "test" / best_policy.name
        write_episode_report(episodes=test_episodes, fold_metrics=test_fold_metrics, summary=test_summary, output_dir=test_dir)
        write_frame(pred_df, test_dir / "entry_oof_predictions.parquet")
        if not hist_df.empty:
            write_frame(hist_df, test_dir / "score_history.parquet")
        test_row = {
            "cycle_id": cycle.cycle_id,
            "test_start": cycle.test_start.isoformat(),
            "test_end": cycle.test_end.isoformat(),
            "selected_policy": best_policy.name,
            "valid_robust_score": best_score,
            **{f"test_{k}": v for k, v in test_summary.items() if isinstance(v, (int, float, str, bool)) or v is None},
        }
        test_rows.append(test_row)
        cycle_rows.append({
            "cycle_id": cycle.cycle_id,
            "test_start": cycle.test_start.isoformat(),
            "test_end": cycle.test_end.isoformat(),
            "selected_policy": best_policy.name,
            "valid_robust_score": best_score,
            "valid_mean_net_pl_bps_sum": (best_valid_summary or {}).get("mean_net_pl_bps_sum"),
            "test_mean_net_pl_bps_sum": test_summary.get("mean_net_pl_bps_sum"),
            "test_worst_net_pl_bps_sum": test_summary.get("worst_net_pl_bps_sum"),
            "test_episode_count": test_summary.get("episode_count"),
            "train_start": test_meta["train_start"],
            "train_end": test_meta["train_end"],
        })
        write_json({"cycle": cycle_rows[-1], "selected_policy": best_policy.__dict__, "test_segment": test_meta}, cycle_dir / "cycle_summary.json")
    candidates_df = pd.DataFrame(candidate_rows)
    cycles_df = pd.DataFrame(cycle_rows)
    tests_df = pd.DataFrame(test_rows)
    candidate_path = run_dir / "candidate_valid_metrics.csv"
    cycles_path = run_dir / "rolling_cycles.csv"
    test_path = run_dir / "rolling_test_metrics.csv"
    write_frame(candidates_df, candidate_path)
    write_frame(cycles_df, cycles_path)
    write_frame(tests_df, test_path)
    write_json(
        {
            "run_id": run_id,
            "config_name": config.name,
            "cycle_count": len(cycles),
            "feature_policies": [p.__dict__ for p in config.feature_policies],
            "entry_selection_mode": config.entry_selection_mode,
            "score_quantile": config.score_quantile,
            "score_floor_bps": config.score_floor_bps,
            "score_quantile_min_periods": config.score_quantile_min_periods,
            "notes": [
                "Each rolling test month selects a feature policy from the prior valid months only.",
                "Test month metrics are not used to change selector rules inside this run.",
                "This is a research-process backtest, not a final locked global test audit.",
            ],
        },
        run_dir / "run_config.json",
    )
    return MonthlyResearchResult(run_dir=run_dir, cycles_csv=cycles_path, candidate_csv=candidate_path, test_metrics_csv=test_path)

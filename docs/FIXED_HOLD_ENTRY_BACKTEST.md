# Fixed-Hold Entry Backtest

This is the baseline for checking whether an entry model's fixed-horizon target
actually survives episode construction.

It intentionally does **not** use the learned exit model.

## Purpose

For an entry row at decision time `t`:

```text
entry execution = open[t + 1]
fixed exit      = open[t + 1 + fixed_hold_minutes]
net P/L         = side-normalized gross P/L - roundtrip_cost_bps
```

For `long_H120`, the default fixed hold is 120 minutes.

## Important: no fold-wide top quantile entry rules

Do **not** use the whole valid fold's score distribution to decide whether the
current row is top q99/q99.5.  That uses future validation scores and is not a
live-feasible entry rule.

Allowed fixed-hold entry rules are:

```text
threshold:
  current_score > fixed_bps_threshold

rolling_quantile:
  current_score > quantile(past_scores_only)
  and current_score > absolute_floor_bps
```

The rolling quantile threshold uses only scores strictly before the current
row inside the same fold.  The current row and future rows are excluded.

Offline top-q metrics can remain as ranking diagnostics, but they are not a
strategy backtest rule.

## Run one candidate with fixed threshold

```bash
python3 scripts/backtest_fixed_hold_entry_valid.py \
  --ohlcv data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --entry-oof outputs/valid/entry_oof/selected_long_H120_long_H120_tail_v0/entry_oof_predictions.parquet \
  --side long \
  --entry-horizon 120 \
  --fixed-hold-minutes 120 \
  --entry-selection-mode threshold \
  --entry-threshold-bps 25 \
  --roundtrip-cost-bps 15
```

## Run one candidate with rolling quantile

```bash
python3 scripts/backtest_fixed_hold_entry_valid.py \
  --ohlcv data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --entry-oof outputs/valid/entry_oof/selected_long_H120_long_H120_tail_v0/entry_oof_predictions.parquet \
  --side long \
  --entry-horizon 120 \
  --fixed-hold-minutes 120 \
  --entry-selection-mode rolling_quantile \
  --rolling-score-window-days 60 \
  --rolling-score-quantile 0.99 \
  --rolling-score-min-periods 1000 \
  --entry-score-floor-bps 25 \
  --entry-threshold-bps 25 \
  --roundtrip-cost-bps 15
```

## Run selected long_H120 candidates

```bash
python3 scripts/run_selected_long_h120_fixed_hold.py
```

Defaults:

```text
long_H120_v0      threshold = 20bps
long_H120_tail_v0 threshold = 25bps
fixed_hold        = 120m
roundtrip cost    = 15bps
```

Rolling-quantile version:

```bash
python3 scripts/run_selected_long_h120_fixed_hold.py \
  --entry-selection-mode rolling_quantile \
  --rolling-score-window-days 60 \
  --rolling-score-quantile 0.99 \
  --rolling-score-min-periods 1000
```

## Sweep small valid-only grids

```bash
python3 scripts/sweep_selected_long_h120_fixed_hold.py --dry-run
python3 scripts/sweep_selected_long_h120_fixed_hold.py
```

The sweep runs absolute thresholds and rolling quantiles.  It does not use
fold-wide future quantiles.

Compare results:

```bash
python3 scripts/compare_selected_long_h120_episode.py \
  --run-prefix selected_long_H120_fixed_hold
```

or for the sweep:

```bash
python3 scripts/compare_selected_long_h120_episode.py \
  --run-prefix selected_long_H120_fixed_hold_sweep
```

Output:

```text
outputs/valid/episode_backtest/selected_long_H120_episode_comparison.csv
```

## Rules

- Entry predictions must be OOF.
- Test data is not read.
- No exit model predictions are used.
- Roundtrip cost is charged once per completed episode.
- Fold-wide top quantile entry rules are prohibited because they use future
  score distribution information.
- This is a valid-fold research baseline, not a locked test audit.

## Additional score-strength filters

After the base entry rule passes, the backtester can apply optional live-safe
score filters:

```text
min_entry_pred_bps:
  current_score >= value

min_score_margin_bps:
  current_score - effective_threshold >= value

min_score_ratio:
  current_score / effective_threshold >= value
```

For rolling quantile mode, `effective_threshold` is:

```text
max(past_rolling_quantile_threshold, entry_score_floor_bps)
```

These filters use only the current model score and the already-computed
past-only threshold.  They do not use future prices or future score
distributions.

Example: rolling q99.5, floor 40bps, and at least 20bps above the effective
threshold:

```bash
python3 scripts/backtest_fixed_hold_entry_valid.py \
  --ohlcv data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --entry-oof outputs/valid/entry_oof/selected_long_H120_long_H120_tail_v0/entry_oof_predictions.parquet \
  --side long \
  --entry-horizon 120 \
  --fixed-hold-minutes 120 \
  --entry-selection-mode rolling_quantile \
  --rolling-score-window-days 60 \
  --rolling-score-quantile 0.995 \
  --rolling-score-min-periods 1000 \
  --entry-score-floor-bps 40 \
  --entry-threshold-bps 40 \
  --min-score-margin-bps 20 \
  --roundtrip-cost-bps 15
```

Example: sweep only the current best region with one filter at a time:

```bash
python3 scripts/sweep_selected_long_h120_fixed_hold.py \
  --candidates long_H120_tail_v0 \
  --modes rolling_quantile \
  --threshold-grids 'long_H120_tail_v0=40' \
  --rolling-score-quantiles 0.995 \
  --min-entry-pred-values 60 \
  --min-score-margin-values 20,30 \
  --min-score-ratio-values 1.25,1.5
```

## Rolling history note

The current implementation computes rolling quantile thresholds inside each
valid fold using only earlier scores from that same fold.  It does **not** seed
fold_01 with the final 60 days of train predictions.  This is conservative and
prevents accidentally mixing in in-sample train predictions, but it also creates
a warm-up period at the start of each fold.  If train-period score history is
used later, it must be generated from a strictly out-of-sample/calibration-safe
prediction stream and passed explicitly; do not silently use in-sample train
scores as rolling history.

## Rolling history modes

`rolling_quantile` supports three history modes:

```text
fold_local       # fold starts cold; previous folds are not used
continuous_valid # previous valid-fold scores can warm later folds
score_history    # explicit train-tail / pre-fold score history
```

All modes are past-only. `continuous_valid` never uses future valid scores; it only carries scores with timestamps earlier than the current row.

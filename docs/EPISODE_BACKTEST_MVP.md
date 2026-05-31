# Episode Backtest MVP

This step only verifies plumbing from entry OOF predictions to exit OOF predictions to episode-level P/L.
It is not a strategy-selection step.  The current feature set is intentionally minimal, so poor P/L is expected.

## Inputs

The script consumes:

```text
canonical 1m OHLCV
entry_oof_predictions.parquet/csv
exit model predictions_valid.parquet/csv
```

Both entry and exit predictions must have `is_oof = true`.  Non-OOF inputs are rejected.

## State machine

The MVP policy has two states:

```text
flat
position
```

When flat:

```text
enter if pred_entry_net_bps > entry_threshold_bps
```

When in position:

```text
hold if pred_exit_hold_delta_bps > hold_threshold_bps
otherwise exit
```

Forced exits:

```text
max_hold_minutes
hard_stop_bps, if configured
data boundary / no later exit prediction
```

Churn controls:

```text
exit_confirm_bars
cooldown_after_exit_minutes
same_side_reentry_block_minutes
max_round_trips_per_day
```

## Timing

For an entry decision at timestamp `t`:

```text
entry_exec_time = t + 1 minute
entry_price = open[entry_exec_time]
```

For an exit decision at `current_time`:

```text
exit_exec_time = current_time + 1 minute
exit_price = open[exit_exec_time]
```

This preserves the project rule: decisions are made after a bar closes and execution happens on the next open.

## Cost handling

The backtest charges roundtrip cost exactly once per completed episode:

```text
gross_pl_bps = side_normalized_log_return(entry_price, exit_price)
net_pl_bps = gross_pl_bps - roundtrip_cost_bps
```

The exit model target does not subtract a fresh roundtrip cost on every row.  Episode P/L is where the actual one-roundtrip cost is applied.

## Example

```bash
python scripts/backtest_episode_valid.py \
  --ohlcv data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --entry-oof outputs/valid/entry_oof/<grid_id>/entry_oof_predictions.parquet \
  --exit-predictions outputs/valid/exit_lgbm/<exit_run_id>/predictions_valid.parquet \
  --side long \
  --entry-horizon 60 \
  --entry-threshold-bps 20 \
  --hold-threshold-bps 0 \
  --roundtrip-cost-bps 15 \
  --max-hold-minutes 240
```

Output:

```text
outputs/valid/episode_backtest/<run_id>/
  run_config.json
  episodes.parquet or episodes.csv
  fold_metrics.csv
  summary_metrics.json
```

## Interpretation

At this stage, only check:

```text
OOF inputs are accepted and non-OOF inputs are rejected
positions do not overlap
roundtrip cost is charged once
fold-level episode metrics are written
state transitions are plausible
```

Do not tune thresholds aggressively or interpret P/L seriously until stronger feature families are migrated and reviewed.

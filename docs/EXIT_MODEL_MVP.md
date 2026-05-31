# Exit model MVP

This phase trains supervised LightGBM regressors on an exit position-state dataset
created by `scripts/make_exit_dataset.py`.

## Target

The target is continuous regression, not classification:

```text
target_exit_hold_delta_bps
```

It means:

```text
P/L if holding until current_time + K
minus
P/L if exiting at current_time
```

A positive value means holding was better than exiting immediately. A negative
value means exiting immediately was better. The target does not subtract a fresh
roundtrip cost; fixed costs mostly cancel in the hold-vs-exit comparison.
Roundtrip cost is charged later in episode backtests.

## Feature sets

Train the three feature sets separately:

```bash
python scripts/train_exit_lgbm.py \
  --exit-dataset outputs/valid/exit_dataset/exit_long_entryH60_K30/exit_dataset.parquet \
  --feature-set v0_market_only

python scripts/train_exit_lgbm.py \
  --exit-dataset outputs/valid/exit_dataset/exit_long_entryH60_K30/exit_dataset.parquet \
  --feature-set v1_score_decay

python scripts/train_exit_lgbm.py \
  --exit-dataset outputs/valid/exit_dataset/exit_long_entryH60_K30/exit_dataset.parquet \
  --feature-set v2_position_aware
```

### v0_market_only

Uses only current market features.

### v1_score_decay

Uses market features plus:

```text
age_minutes
entry_pred_net_bps
current_pred_net_bps
score_decay_bps
```

This tests whether the entry hypothesis still appears alive.

### v2_position_aware

Uses v1 plus:

```text
unrealized_pl_bps
unrealized_pl_after_cost_bps
mfe_since_entry_bps
mae_since_entry_bps
giveback_from_mfe_bps
```

This tests whether the post-entry path helps exit decisions.

## Validation

Training is fold-crossed inside the valid period:

```text
fold_01 eval: train on fold_02..fold_05
fold_02 eval: train on fold_01, fold_03..fold_05
...
```

Rows from the evaluation fold are not used for training. Evaluation episode IDs
are explicitly removed from the training rows as an additional guard.

## Outputs

Each run writes:

```text
outputs/valid/exit_lgbm/<run_id>/
  run_config.json
  input_snapshot.json
  fold_metrics.csv
  score_deciles.csv
  predictions_valid.parquet or predictions_valid.csv
  summary_metrics.json
  feature_importance_by_fold.csv
  feature_importance_mean.csv
```

## What to inspect

For a useful exit model:

```text
Top predicted rows should have positive realized hold_delta.
Bottom predicted rows should have negative realized hold_delta.
Top-bottom spread should be positive across folds.
Score deciles should be roughly monotonic.
Worst fold should not collapse.
```

Do not move to episode backtest solely because RMSE improves. The point of this
model is ranking hold-vs-exit states, not minimizing generic error.

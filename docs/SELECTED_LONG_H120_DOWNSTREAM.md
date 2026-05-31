# Selected long_H120 Downstream Validation

This step sends the reviewed long_H120 entry candidates through the already-built
valid-fold downstream plumbing:

1. `make_exit_dataset.py`
2. `train_exit_lgbm.py`
3. `backtest_episode_valid.py`

It is **not** a test evaluation and it does not tune on test.  It assumes the
selected entry candidates have already been trained by:

```bash
python3 scripts/run_selected_long_h120_entry.py
```

That command should create candidate-specific OOF entry predictions under:

```text
outputs/valid/entry_oof/selected_long_H120_long_H120_v0/
outputs/valid/entry_oof/selected_long_H120_long_H120_tail_v0/
```

## Candidates

The default candidates are:

```text
long_H120_v0
  primary candidate: v3a - compressed_activity
  default entry threshold: 20bps

long_H120_tail_v0
  tail candidate: long_H120_v0 + compressed_price_geometry
  default entry threshold: 25bps
```

The defaults reflect the entry-only analysis: `long_H120_v0` is more robust
through q97, while `long_H120_tail_v0` is a stricter q99 candidate.

## Run both candidates through downstream plumbing

```bash
python3 scripts/run_selected_long_h120_downstream.py
```

By default this runs:

```text
side = long
entry_horizon = 120
exit_lookahead = 30
exit_feature_set = v2_position_aware
roundtrip_cost_bps = 15
max_hold_minutes = 240
hold_threshold_bps = 0
entry_threshold_bps:
  long_H120_v0 = 20
  long_H120_tail_v0 = 25
```

## Dry-run

```bash
python3 scripts/run_selected_long_h120_downstream.py --dry-run
```

## Run only one candidate

```bash
python3 scripts/run_selected_long_h120_downstream.py \
  --candidates long_H120_v0
```

## Try K30 and K60 exit lookaheads

```bash
python3 scripts/run_selected_long_h120_downstream.py \
  --exit-lookaheads 30,60
```

## Compare episode summaries

```bash
python3 scripts/compare_selected_long_h120_episode.py
```

Default output:

```text
outputs/valid/episode_backtest/selected_long_H120_episode_comparison.csv
```

## Important constraints

- Entry predictions must be OOF valid-fold predictions.
- Exit datasets must be rebuilt per candidate, because the entry policy changes.
- For `entry_horizon=120`, do not reuse old `long_H60` exit datasets.
- Roundtrip cost is charged once per completed episode in the episode backtest.
- This step does not select a production policy and does not touch test.

## Memory note for selected long_H120 exit datasets

`features_long_H120_v0.parquet` and `features_long_H120_tail_v0.parquet` can contain hundreds of entry features.  Expanding all of those market features onto every exit position-state row can use a large amount of memory because one entry can create many exit rows.

For the selected long_H120 downstream runner, the default is now:

```bash
--exit-market-features none
```

This builds the exit dataset from OOF entry score columns and position-state columns only:

- `age_minutes`
- `entry_pred_net_bps`
- `current_pred_net_bps`
- `score_decay_bps`
- `unrealized_pl_bps`
- `mfe_since_entry_bps`
- `mae_since_entry_bps`
- `giveback_from_mfe_bps`

This is enough for `v2_position_aware` plumbing and avoids materializing the full entry feature matrix into the exit dataset.  To explicitly include all market features, pass:

```bash
--exit-market-features all
```

Use `all` only when the machine has enough memory or when a compact exit-specific market feature set has been prepared.

# Entry Grid and OOF Prediction Artifacts

This step expands the entry model MVP from one run (`long H60`) to a small,
reviewable grid:

- `long H60`, `long H120`, `long H240`
- `short H60`, `short H120`, `short H240`

The model is still a **LightGBM regressor**.  It predicts the continuous
cost-adjusted entry target:

```text
target_entry_net_bps_<side>_H<horizon>
```

No binary entry classifier is implemented in this step.

## Why OOF predictions matter

The future exit dataset will be built from positions that the entry model would
have opened.  To avoid optimistic leakage, the entry score used to create those
positions must be out-of-sample for the timestamp being scored.

The current convention is:

```text
entry model fit:   manifest.train only
evaluation score: valid fold only
is_oof:           true
```

These valid-fold predictions are collected into a single artifact for later exit
dataset construction.

## Train the default grid

```bash
python scripts/train_entry_grid.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml
```

Default inputs:

```text
data/processed/binance_japan/BTCJPY/1m/features_entry_v0.parquet
outputs/valid/labels/btcjpy_1m_labels.parquet
configs/models/lgbm_entry_v0.yaml
```

Default grid:

```text
--sides long,short
--horizons 60,120,240
```

Outputs:

```text
outputs/valid/entry_lgbm/<grid_id>_<side>_H<horizon>/
outputs/valid/entry_grid/<grid_id>/
outputs/valid/entry_oof/<grid_id>/entry_oof_predictions.parquet
```

## Dry run

Use this before training to confirm that all target columns exist:

```bash
python scripts/train_entry_grid.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --dry-run
```

## Collect OOF predictions from existing runs

If individual runs were already trained with `scripts/train_entry_lgbm.py`, combine
them manually:

```bash
python scripts/collect_entry_oof_predictions.py \
  --run-dir outputs/valid/entry_lgbm/entry_long_H60_... \
  --run-dir outputs/valid/entry_lgbm/entry_long_H120_... \
  --run-dir outputs/valid/entry_lgbm/entry_short_H60_...
```

The collector refuses prediction files where `is_oof` is not true for every row.
It also refuses duplicate `(timestamp, side, horizon)` rows, because downstream
exit dataset generation needs one unambiguous entry score per side/horizon.

## Compare runs

```bash
python scripts/compare_entry_runs.py \
  --run-dir outputs/valid/entry_lgbm/<run_a> \
  --run-dir outputs/valid/entry_lgbm/<run_b> \
  --output outputs/valid/entry_grid/manual_comparison.csv
```

## Review checklist

Before moving to exit dataset construction, review:

- `entry_model_comparison.csv`
- `entry_oof_predictions.parquet` or CSV fallback
- `entry_oof_manifest.json`
- each member run's `fold_metrics.csv`
- worst-fold metrics, not just mean metrics

Do not use test results to choose side, horizon, threshold, or feature set.

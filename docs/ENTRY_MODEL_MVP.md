# Entry Model MVP

This phase adds the first LightGBM entry model.  It is intentionally narrow:

- model role: `entry`
- estimator: `LGBMRegressor`
- default side: `long`
- default horizon: `H60`
- target: `target_entry_net_bps_long_H60`
- train data: split manifest `train` range only, purged per valid fold
- evaluation data: split manifest `valid` folds only
- test data: not used

The target is a continuous net-return target, not a binary classifier:

```text
long H60 target = log(open[t+1+60] / open[t+1]) * 10000 - roundtrip_cost_bps
```

The default cost is 15 bps, inherited from `configs/labels/entry_net_return_v0.yaml`.

## Expected inputs

Build data in this order:

```bash
python scripts/audit_data.py \
  --input data/raw/binance_japan/BTCJPY/1m \
  --write-canonical

python scripts/build_labels.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet

python scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet
```

Then create a concrete split file:

```text
configs/splits/btcjpy_1m_split_v1.yaml
```

Do not use the `.template.yaml` file directly.  The concrete split must contain
real UTC dates and `test.usage: locked_audit_only`.

## Train the first model

```bash
python scripts/train_entry_lgbm.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --side long \
  --horizon 60
```

The script trains one model per valid fold.  For each fold, train rows come only
from the manifest train range; evaluation rows come only from that valid fold.
The script does not predict or report test rows.

## Outputs

A run is written under:

```text
outputs/valid/entry_lgbm/<run_id>/
```

Important files:

```text
run_config.json
input_snapshot.json
fold_metrics.csv
score_deciles.csv
predictions_valid.parquet   # or .csv fallback if pyarrow is unavailable
summary_metrics.json
feature_importance_by_fold.csv
feature_importance_mean.csv
```

Fold models are saved under:

```text
models/entry/<run_id>/fold_01.txt
models/entry/<run_id>/fold_02.txt
...
```

## Main review points

Check these before expanding to more horizons/sides:

- `run_config.json` says `test_usage: locked_audit_only`.
- predictions are valid-fold only.
- `is_oof` is true for all saved valid predictions.
- feature count matches the reviewed feature matrix.
- target column is the expected continuous target.
- `top_q95` / `top_q97` / `top_q99` realized target bps improve consistently across folds.
- worst fold is not ignored.

Generic RMSE is reported, but it is not the main decision metric.  For this
strategy, the key metric is realized net bps among the highest predicted rows.

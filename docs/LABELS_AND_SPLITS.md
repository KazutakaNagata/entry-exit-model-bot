# Splits and Labels

This project separates data ingestion, label generation, feature generation, and model training.
This stage implements only split/label helpers. It does not train LightGBM and does not evaluate test.

## Split rule

Use a concrete split file copied from:

```text
configs/splits/btcjpy_1m_split_v1.template.yaml
```

The concrete file must define:

- `train`
- `valid`
- exactly five valid folds
- `test` with `usage: locked_audit_only`
- `purge.purge_minutes`
- `purge.embargo_minutes`

Normal research scripts must tune only on train/valid. Test is read only by the final locked audit script.

Inspect a concrete split:

```bash
python scripts/inspect_splits.py --split configs/splits/btcjpy_1m_split_v1.yaml
```

Add row counts if canonical OHLCV is available:

```bash
python scripts/inspect_splits.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --data data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet
```

## Entry label

For decision row `t`:

```text
decision: after bar t close
entry:    open[t + 1]
exit:     open[t + 1 + H]
```

Long:

```text
log(open[t+1+H] / open[t+1]) * 10000 - roundtrip_cost_bps
```

Short:

```text
-log(open[t+1+H] / open[t+1]) * 10000 - roundtrip_cost_bps
```

Default horizons:

```text
H = 60, 120, 240 minutes
```

Default cost:

```text
roundtrip_cost_bps = 15.0
```

## Exit label

For decision row `t`:

```text
current exit: open[t + 1]
future exit:  open[t + 1 + K]
```

Generic side-normalized hold delta:

```text
side * log(open[t+1+K] / open[t+1]) * 10000
```

Default lookaheads:

```text
long:  K = 30, 60 minutes
short: K = 15, 30 minutes
```

The exit target does not subtract a new 15 bps roundtrip cost on every hold decision.
Episode backtests will subtract one roundtrip cost per position.

## Build labels

Dry run:

```bash
python scripts/build_labels.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --dry-run
```

Write labels:

```bash
python scripts/build_labels.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --output outputs/valid/labels/btcjpy_1m_labels.parquet
```

MFE/MAE diagnostics are optional and should not be used as entry features:

```bash
python scripts/build_labels.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --include-mfe-mae \
  --dry-run
```

## Leakage guardrails

- Feature row `t` may use only data available by bar `t` close.
- Entry/exit labels use future data but must never enter feature matrices.
- Entry execution is next bar open, not current close.
- Exit execution is next bar open, not current close.
- Exact timestamp lookup is used for entry/exit labels; missing target minutes become `NaN`.
- MFE/MAE are diagnostics/position-state values, not ordinary entry features.

# Feature Factory v3b

`v3b` adds structure-oriented families before moving the long_H120 candidate into exit modeling.

The current long_H120 best candidate from v3a was:

```text
v3a - compressed_activity
```

The new model-specific config keeps that decision and adds structure families:

```text
configs/features/model_specific/entry_feature_set_long_H120_v3b_candidate.yaml
```

## Added families

```text
enhanced_support_resistance
enhanced_market_structure_void
enhanced_pullback_geometry
fractal_structure
pivot_line
compressed_price_geometry
compressed_breakout
```

These families target:

```text
support / resistance distance
upside room
pullback quality
price range position
higher-high / higher-low structure
pivot-line geometry
compressed price geometry
compressed breakout context
```

## Leakage rules

Structural levels are previous levels wherever current price is compared to resistance/support:

```python
high.shift(1).rolling(...)
low.shift(1).rolling(...)
```

The current finalized bar may be used as a feature at decision time, but it is not used to define the prior level being compared against.

## Build commands

Full v3b exploration universe:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/entry_feature_set_v3b.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_v3b.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_v3b.json \
  --report-output outputs/valid/features/entry_feature_audit_v3b.json
```

Current long_H120 candidate:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/model_specific/entry_feature_set_long_H120_v3b_candidate.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_v3b_candidate.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_long_H120_v3b_candidate.json \
  --report-output outputs/valid/features/entry_feature_audit_long_H120_v3b_candidate.json
```

Then train only long_H120 first:

```bash
python3 scripts/train_entry_grid.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_v3b_candidate.parquet \
  --grid-run-id entry_grid_long_H120_v3b_candidate \
  --sides long \
  --horizons 120
```

Do not evaluate test while tuning this candidate.

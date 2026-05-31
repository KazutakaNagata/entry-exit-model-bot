# Feature Migration v2

v2 adds three deterministic, past-only feature families on top of v1:

- `breakout`
- `volume_price_pressure`
- `rejection_reversal`

The goal is to bring the new repository closer to the old long-tail research
without adding supervised compression, target-derived values, or complicated
stateful logic.

## Families

### breakout

`breakout` compares the finalized current bar with previous rolling highs/lows.
Previous levels are always built with `shift(1).rolling(...)`, so the current
bar is not used as its own resistance/support level.

Examples:

- close above previous rolling high
- close below previous rolling low
- positive up/down breakout strength
- high/low extension relative to previous range
- current range expansion
- current volume expansion
- breakout strength with positive volume z-score

### volume_price_pressure

`volume_price_pressure` summarizes whether volume is aligned with short-term
returns over rolling windows ending at the current finalized bar.

Examples:

- signed volume imbalance
- up/down volume share
- volume-weighted return
- up/down return pressure
- return/log-volume rolling correlation

### rejection_reversal

`rejection_reversal` captures current finalized candle shape and failed breaks
of previous rolling levels.

Examples:

- upper/lower wick size and ratio
- close position inside candle
- failed breakout against previous rolling high
- failed breakdown against previous rolling low
- wick surprise z-score versus previous candles
- upper/lower rejection scores

## Leakage rules

All v2 features must satisfy:

- no `shift(-n)`
- no centered rolling windows
- no target, hold-delta, MFE/MAE, or future-return columns
- no test data usage for selection or tuning
- every feature appears in the manifest with `uses_future=false`

`rejection_reversal` uses the current finalized candle. This is allowed because
features are assumed to be computed after bar close, and execution happens at the
next bar open.

## Expected usage

Build v2 features:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/entry_feature_set_v2.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_v2.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_v2.json \
  --report-output outputs/valid/features/entry_feature_audit_v2.json
```

Audit:

```bash
python3 scripts/audit_features.py \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_v2.parquet \
  --manifest outputs/valid/features/entry_feature_manifest_v2.json \
  --config configs/features/entry_feature_set_v2.yaml \
  --strict
```

Then rerun the downstream pipeline:

1. entry grid
2. entry OOF collection
3. exit dataset
4. exit LGBM
5. episode backtest

Do not interpret v2 as final. It is a deterministic family expansion checkpoint.

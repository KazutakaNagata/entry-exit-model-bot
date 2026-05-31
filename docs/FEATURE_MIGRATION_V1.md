# Feature migration v1

This delta adds the first migrated structure-style feature families for the
entry/exit pipeline.  The goal is still safe feature expansion and plumbing, not
final performance tuning.

## Added families

- `support_resistance`
  - distance from close to rolling support/resistance levels
  - close position inside the rolling high/low range
  - breakout/breakdown versus *prior* rolling levels using `shift(1)`
- `market_structure_void`
  - upside/downside room to prior high/low levels
  - close position relative to the previous high/low range
  - positive breakout/breakdown extension beyond previous range
  - bars since high/low inside a rolling window
- `pullback_geometry`
  - drawdown from rolling high and bounce from rolling low
  - drawdown/bounce ratios inside recent ranges
  - simple differences between short and long pullback windows

## Leakage rules

All v1 features are computed from finalized 1-minute OHLCV at or before the
feature row timestamp.  Features that compare against a previous structure level
use `shift(1)` before rolling, so the reference level existed before the current
candle.

Still forbidden in generic market features:

- target columns
- future returns
- labels
- hold-delta targets
- MFE/MAE/giveback from future episode paths
- entry/exit prices

MFE/MAE/giveback remain allowed only inside the explicitly separate exit
position-state dataset.

## How to build v1 features

```bash
python scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/entry_feature_set_v1.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_v1.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_v1.json \
  --report-output outputs/valid/features/entry_feature_audit_v1.json
```

Then run the feature audit:

```bash
python scripts/audit_features.py \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_v1.parquet \
  --manifest outputs/valid/features/entry_feature_manifest_v1.json \
  --strict
```

After this, regenerate the full downstream chain from v1 features:

1. entry grid training
2. entry OOF collection
3. exit dataset
4. exit LGBM
5. episode backtest

Do not compare v1 episode performance against v0 unless every downstream artifact
has been regenerated from the same feature set.

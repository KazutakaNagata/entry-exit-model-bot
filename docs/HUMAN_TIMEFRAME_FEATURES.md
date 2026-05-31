# Human-watched timeframe features

This delta adds three reviewed, past-only feature families intended to test a different hypothesis from the existing rolling-window families.

The existing feature universe is mostly rolling-window based: returns, volatility, rolling high/low geometry, range position, compressed trend/volatility, and related summaries measured from the current timestamp over the last `N` minutes. Those features are useful, but they do not explicitly encode the chart objects many traders and bots watch: closed 5m/15m/1h/4h candles, their high/low/close, common indicators, and the phase of the current minute inside those bars.

## Added families

### `anchored_timeframe_candle`

Uses closed anchored candles for:

```text
1m, 5m, 15m, 60m, 240m
```

Features include:

```text
body_bps
range_bps
body_to_range
upper_wick_ratio
lower_wick_ratio
close_position
current close vs last closed timeframe high/low/close
previous 4/12 anchored-bar high/low distances
range position within previous 4/12 anchored bars
break above/below previous anchored levels
```

Higher timeframe bars are aligned by close time. For example, at `10:37`, the 60m features use the latest 60m candle whose close time is `<= 10:37`, not the currently forming `10:00-11:00` candle.

### `anchored_timeframe_indicators`

Computes common indicators on closed anchored candles:

```text
EMA 10/20/50 distance and slope
EMA10 vs EMA20
EMA20 vs EMA50
ATR14
RSI14
Bollinger 20 z-score / width / band position
MACD 12/26/9
```

Supported timeframes:

```text
1m, 5m, 15m, 60m, 240m
```

### `timeframe_phase`

Calendar-only phase features:

```text
minute within 5m / 15m / 60m / 240m bar
fractional phase
sin/cos phase
minutes to anchored bar close
first-minute and last-minute flags
UTC minute-of-day sin/cos
```

These features intentionally do not use price data.

## Configs

Full exploration universe:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/entry_feature_set_v4_timeframe.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_v4_timeframe.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_v4_timeframe.json \
  --report-output outputs/valid/features/entry_feature_audit_v4_timeframe.json
```

Long H120 tail candidate plus timeframe families:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/model_specific/entry_feature_set_long_H120_tail_timeframe_v0.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_tail_timeframe_v0.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_long_H120_tail_timeframe_v0.json \
  --report-output outputs/valid/features/entry_feature_audit_long_H120_tail_timeframe_v0.json
```

Audit:

```bash
python3 scripts/audit_features.py \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_tail_timeframe_v0.parquet \
  --manifest outputs/valid/features/entry_feature_manifest_long_H120_tail_timeframe_v0.json \
  --config configs/features/model_specific/entry_feature_set_long_H120_tail_timeframe_v0.yaml \
  --strict
```

## Suggested fixed-train valid check

After building the model-specific file above, compare it against the existing `long_H120_tail_v0` fixed-train baseline.

```bash
python3 scripts/train_entry_grid.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_tail_timeframe_v0.parquet \
  --grid-run-id entry_grid_long_H120_tail_timeframe_v0 \
  --sides long \
  --horizons 120
```

Then inspect:

```text
outputs/valid/entry_grid/entry_grid_long_H120_tail_timeframe_v0/entry_model_comparison.csv
outputs/valid/entry_lgbm/entry_grid_long_H120_tail_timeframe_v0_long_H120/fold_metrics.csv
outputs/valid/entry_lgbm/entry_grid_long_H120_tail_timeframe_v0_long_H120/score_deciles.csv
```

## Leakage notes

The most important leakage risk is higher-timeframe candles. This implementation uses only closed higher-timeframe candles. It does not use a still-forming 1h/4h candle's final high/low/close.

For current 1m rows, the project convention is that row-level features are available after the 1m bar closes and entry happens at the next open. This is the same convention used by the existing rolling feature families.

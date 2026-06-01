# Long_H120 curated existing families + full human-timeframe features

This patch adds Mac-friendlier feature configs for testing whether human-watched timeframe features help `long_H120` without carrying the whole v3/v4 feature universe.

The intent is:

```text
Keep selected existing rolling families that look structurally useful for long_H120.
Add all new full human-timeframe families.
Do not use the lite timeframe families in these configs.
```

## Configs

### `entry_feature_set_long_H120_curated_core_plus_timeframe_v0.yaml`

Recommended first run.

Existing families kept:

```text
price
return_path
volatility
trend_persistence
volume
regime_switch
compressed_trend
compressed_volatility
volume_spike
compressed_price_geometry
```

New full timeframe families added:

```text
anchored_timeframe_candle
anchored_timeframe_indicators
timeframe_phase
```

This excludes heavier/noisier existing families such as `pivot_line`, `compressed_activity`, `rejection_reversal`, enhanced structure families, raw support/void/pullback structure, and full breakout/volume-pressure blocks.

### `entry_feature_set_long_H120_curated_broad_plus_timeframe_v0.yaml`

Use if the core candidate is too sparse or if memory permits.

It keeps the core plus:

```text
trend_regime
micro_range
market_structure_void
pullback_geometry
support_resistance
```

It still excludes:

```text
compressed_activity
pivot_line
rejection_reversal
compressed_breakout
fractal_structure
enhanced_support_resistance
enhanced_market_structure_void
enhanced_pullback_geometry
breakout
volume_price_pressure
acceleration
```

### `entry_feature_set_long_H120_curated_timeframe_only_v0.yaml`

Diagnostic only. It includes only:

```text
anchored_timeframe_candle
anchored_timeframe_indicators
timeframe_phase
```

Use it to check whether human-watched timeframe features have standalone signal.

## Why these existing families?

For this quick fixed-train check, the hypothesis is:

```text
long_H120 edge = regime transition + trend continuation + volatility state + human-watched timeframe context.
```

So the core keeps basic normalization, trend persistence, regime switch, compressed trend/volatility, volume spike, and compressed price geometry.

The excluded families are not permanent drops. They are excluded only to make the human-timeframe experiment small enough to run locally while avoiding known noisy or short-oriented blocks.

## One-command check

Build, audit, and train `core` and `broad`:

```bash
python3 scripts/run_long_h120_curated_timeframe_check.py \
  --candidates core,broad
```

Dry run:

```bash
python3 scripts/run_long_h120_curated_timeframe_check.py \
  --candidates core,broad \
  --dry-run
```

Only train after features already exist:

```bash
python3 scripts/run_long_h120_curated_timeframe_check.py \
  --candidates core,broad \
  --skip-build \
  --skip-audit
```

## Manual commands

Core:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/model_specific/entry_feature_set_long_H120_curated_core_plus_timeframe_v0.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_core_plus_timeframe_v0.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_long_H120_curated_core_plus_timeframe_v0.json \
  --report-output outputs/valid/features/entry_feature_audit_long_H120_curated_core_plus_timeframe_v0.json
```

```bash
python3 scripts/train_entry_grid.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_core_plus_timeframe_v0.parquet \
  --grid-run-id entry_grid_long_H120_curated_core_plus_timeframe_v0 \
  --sides long \
  --horizons 120
```

Broad:

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/model_specific/entry_feature_set_long_H120_curated_broad_plus_timeframe_v0.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_broad_plus_timeframe_v0.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_long_H120_curated_broad_plus_timeframe_v0.json \
  --report-output outputs/valid/features/entry_feature_audit_long_H120_curated_broad_plus_timeframe_v0.json
```

```bash
python3 scripts/train_entry_grid.py \
  --split configs/splits/btcjpy_1m_split_v1.yaml \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_long_H120_curated_broad_plus_timeframe_v0.parquet \
  --grid-run-id entry_grid_long_H120_curated_broad_plus_timeframe_v0 \
  --sides long \
  --horizons 120
```

## Compare

Check:

```text
outputs/valid/entry_grid/entry_grid_long_H120_curated_core_plus_timeframe_v0/entry_model_comparison.csv
outputs/valid/entry_grid/entry_grid_long_H120_curated_broad_plus_timeframe_v0/entry_model_comparison.csv
```

Compare against the previous selected baseline:

```text
long_H120_tail_v0
```

Look mainly at:

```text
mean_top_q95_avg_target_bps
worst_top_q95_avg_target_bps
mean_top_q97_avg_target_bps
worst_top_q97_avg_target_bps
mean_top_q99_avg_target_bps
worst_top_q99_avg_target_bps
mean_top_q99_precision_gt0
```

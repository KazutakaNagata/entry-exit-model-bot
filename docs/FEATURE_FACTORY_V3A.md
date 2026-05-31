# Feature Factory v3a

`v3a` expands the feature universe before serious feature selection. It is not a selected production feature set.

The goal is to restore more of the old research search space while keeping feature generation deterministic, reviewable, and past-only. This phase focuses on trend / return / volatility / activity families:

- `acceleration`
- `trend_regime`
- `regime_switch`
- `micro_range`
- `volume_spike`
- `compressed_trend`
- `compressed_activity`
- `compressed_volatility`

The existing v2 families remain available:

- `price`
- `return_path`
- `volatility`
- `trend_persistence`
- `volume`
- `support_resistance`
- `market_structure_void`
- `pullback_geometry`
- `breakout`
- `volume_price_pressure`
- `rejection_reversal`

## Usage

```bash
python3 scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet \
  --config configs/features/entry_feature_set_v3a.yaml \
  --output data/processed/binance_japan/BTCJPY/1m/features_entry_v3a.parquet \
  --manifest-output outputs/valid/features/entry_feature_manifest_v3a.json \
  --report-output outputs/valid/features/entry_feature_audit_v3a.json
```

Then audit:

```bash
python3 scripts/audit_features.py \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_v3a.parquet \
  --manifest outputs/valid/features/entry_feature_manifest_v3a.json \
  --config configs/features/entry_feature_set_v3a.yaml \
  --strict
```

## Rules

- Feature row `t` may use only finalized bars up to `t`.
- No `target_*`, `future_*`, `hold_delta_*`, `mfe_*`, or `mae_*` columns may enter entry features.
- `compressed_*` families in v3a are deterministic summaries. They are not trained supervised compressors.
- Feature selection comes later and should use domain hypothesis blocks, not blind unit-level full search.

## Next recommended research steps

1. Build v3a and run strict audit.
2. Run the legacy long 60m tail benchmark once that benchmark is implemented.
3. Run entry regression on focused models, especially `long_H60`, `long_H120`, and `short_H240`.
4. Use hypothesis-block search to select from the larger feature universe.

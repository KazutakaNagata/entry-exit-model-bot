# Feature Pipeline v0

This phase adds only a small, reviewed feature pipeline. It is intentionally not
the full legacy feature set.

## Decision-time rule

A feature row at timestamp `t` may use only finalized 1-minute bars up to and
including row `t`. The trading decision is assumed to happen after bar `t` closes;
entry/exit execution happens at `t+1 open` in the label/backtest code.

## Implemented families

The initial families are:

- `price`: current finalized 1-minute candle body/wicks/range.
- `return_path`: past close-to-close returns over 1, 3, 5, 15, 30, 60, 120, 240 minutes.
- `volatility`: rolling realized volatility and high/low range.
- `trend_persistence`: close-vs-MA, rolling slope, positive-return share.
- `volume`: log-volume z-score and relative volume.

Legacy families such as support/resistance, market-structure void, rejection, and
pullback geometry should be migrated one family at a time after leakage review.

## Manifest requirement

Every generated feature has a manifest row with:

- feature name
- family
- lookback minutes
- `uses_future=false`
- source
- availability timing

The feature matrix must not include raw OHLCV columns other than `timestamp`, nor
any target/diagnostic columns such as `target_*`, `hold_delta_*`, `mfe_*`, or
`mae_*`.

## Commands

```bash
python scripts/build_features.py \
  --input data/processed/binance_japan/BTCJPY/1m/ohlcv.parquet

python scripts/audit_features.py --strict
```

The default output paths are:

```text
data/processed/binance_japan/BTCJPY/1m/features_entry_v0.parquet
outputs/valid/features/entry_feature_manifest_v0.json
outputs/valid/features/entry_feature_audit_v0.json
```

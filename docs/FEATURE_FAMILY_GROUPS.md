# Feature Family Groups

This document defines the cleaned Long_H120 feature-family organization used
after the v3b all-family experiments.

The previous `features_entry_v3b` universe mixed base, enhanced, compressed, and
timeframe variants in one "full" matrix. That made family ablation hard to
interpret: dropping `breakout` still left `compressed_breakout`, and dropping
`support_resistance` still left `enhanced_support_resistance` and `pivot_line`.

The new rule is:

- `canonical_full_v1` uses one canonical implementation per group.
- Archived variants remain in the repo, but are excluded from the canonical baseline.
- Group ablation should be run against `canonical_full_v1`.
- Variant swaps are a later, narrower experiment after group usefulness is clear.

The machine-readable registry is
`configs/features/family_groups_v1.yaml`.

## Canonical Full V1

```text
price
micro_range
return_path
compressed_trend
regime_switch
compressed_volatility
compressed_activity
volume_spike
volume_price_pressure
breakout
support_resistance
market_structure_void
pullback_geometry
compressed_price_geometry
anchored_timeframe_candle
anchored_timeframe_indicators
timeframe_phase
```

## Groups

| Group | Canonical | Archived variants |
| --- | --- | --- |
| `price_base` | `price` | none |
| `bar_microstructure` | `micro_range` | `rejection_reversal` |
| `return_momentum` | `return_path` | `acceleration` |
| `trend` | `compressed_trend` | `trend_persistence`, `trend_regime` |
| `regime_shift` | `regime_switch` | none |
| `volatility` | `compressed_volatility` | `volatility` |
| `activity` | `compressed_activity` | none |
| `volume_level` | `volume_spike` | `volume` |
| `volume_pressure` | `volume_price_pressure` | none |
| `breakout` | `breakout` | `compressed_breakout` |
| `support_resistance` | `support_resistance` | `enhanced_support_resistance`, `pivot_line` |
| `market_structure` | `market_structure_void` | `enhanced_market_structure_void`, `fractal_structure` |
| `pullback` | `pullback_geometry` | `enhanced_pullback_geometry` |
| `price_geometry` | `compressed_price_geometry` | none |
| `timeframe_candle` | `anchored_timeframe_candle` | `anchored_timeframe_candle_lite` |
| `timeframe_indicator` | `anchored_timeframe_indicators` | none |
| `timeframe_phase` | `timeframe_phase` | none |

## Notes

`trend_regime` is archived in v1 because fixed-threshold Long_H120 experiments
showed that dropping it improved both validation and rolling test results.

`compressed_activity` remains canonical in v1, but it overlaps with volatility
and range information. It should be treated as an early group-ablation candidate,
not as a permanent feature.

`enhanced_*` families are archived for the first cleanup pass because they add
large, overlapping structure blocks. They are better evaluated later as variant
swaps against the simpler canonical structure families.


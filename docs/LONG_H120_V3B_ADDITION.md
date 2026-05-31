# long_H120 v3b family addition

Purpose: isolate which v3b structure families help the current long_H120 base before moving to exit datasets.

Current base:

```text
long_H120 base_v0 = v3a - compressed_activity
```

This base came from valid-fold ablation. It keeps breakout, rejection_reversal, volume_price_pressure, volume_spike, and the trend/regime/volatility families because combined ablation showed these interactions should not be removed blindly.

This experiment adds v3b families one at a time and in a few domain-motivated blocks:

```text
enhanced_support_resistance
enhanced_market_structure_void
enhanced_pullback_geometry
fractal_structure
pivot_line
compressed_price_geometry
compressed_breakout
room_pullback
support_room_pullback
structure_core
all_v3b
```

Run:

```bash
python3 scripts/run_long_h120_v3b_addition.py --dry-run
python3 scripts/run_long_h120_v3b_addition.py
python3 scripts/compare_long_h120_v3b_addition.py
```

Focused run:

```bash
python3 scripts/run_long_h120_v3b_addition.py \
  --additions base_v0,base_plus_enhanced_market_structure_void,base_plus_enhanced_pullback_geometry,base_plus_structure_core
```

Comparison output:

```text
outputs/valid/entry_grid/long_H120_v3b_addition_comparison.csv
```

Selection rule:

- Do not use test data.
- Compare against `base_v0`.
- Prefer variants that improve q95/q97 and worst-fold metrics while preserving q99 strength.
- Treat q99-only improvements as unstable until confirmed by q95/q97 and fold stability.
- If all-v3b is worse, do not reject all structure features; inspect individual additions.

The next step, if a candidate improves over base_v0, is to rebuild entry OOF for long_H120, then rebuild the long_H120 exit dataset. Existing exit dataset rules still apply: entry candidates must come from OOF predictions, fold-boundary-crossing targets must be dropped, and roundtrip cost is charged once per episode in episode backtests.

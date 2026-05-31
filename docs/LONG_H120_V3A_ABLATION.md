# long_H120 v3a family ablation

This experiment tests which v3a feature families are unnecessary or harmful for
`entry_long_H120`.

It is **not** a production feature-selection result. It is a valid-fold research
experiment that keeps the model, labels, split, and target fixed while changing
only the feature-family set.

## Why this exists

`entry_feature_set_v3a` improved long models, especially `long_H120`. However,
v3a contains both long-oriented trend/regime families and possible noise families
such as pressure/rejection/spike features. The first long_H120 feature-selection
step is therefore removal, not more feature addition.

## Generated configs

Configs live under:

```text
configs/features/ablations/long_H120_v3a/
```

Important variants:

```text
v3a_full
v3a_minus_breakout
v3a_minus_volume_price_pressure
v3a_minus_rejection_reversal
v3a_minus_acceleration
v3a_minus_trend_regime
v3a_minus_regime_switch
v3a_minus_micro_range
v3a_minus_volume_spike
v3a_minus_compressed_trend
v3a_minus_compressed_activity
v3a_minus_compressed_volatility
v3a_minus_v2_noise
v3a_minus_short_noise
hypothesis_core
```

## Run

Dry-run first:

```bash
python3 scripts/run_long_h120_v3a_ablation.py --dry-run
```

Then run all variants:

```bash
python3 scripts/run_long_h120_v3a_ablation.py
```

To run only a subset:

```bash
python3 scripts/run_long_h120_v3a_ablation.py \
  --ablations v3a_full,v3a_minus_volume_price_pressure,v3a_minus_rejection_reversal,hypothesis_core
```

If features are already built and only training is needed:

```bash
python3 scripts/run_long_h120_v3a_ablation.py --only-train --skip-existing-features
```

## Compare

```bash
python3 scripts/compare_long_h120_v3a_ablation.py
```

Default output:

```text
outputs/valid/entry_grid/long_H120_v3a_ablation_comparison.csv
```

## Interpretation

A family is a removal candidate if dropping it improves q95/q97/worst-fold
metrics or preserves q99 while reducing feature count. A family is likely needed
if removing it hurts q95/q97/q99 and worst-fold metrics.

Avoid adopting a variant based on q99 alone. For long_H120, the desired movement
is:

```text
q99 remains positive or near positive
q95/q97 move up
worst fold improves
feature count drops or remains justified
```

The test set must not be used in this experiment.

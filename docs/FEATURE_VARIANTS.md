# Feature Variant Comparison

This stage decomposes `entry_feature_set_v2` into smaller family additions so we
can see which family helped or hurt the v1 baseline.

The current comparison is intentionally simple:

```text
v1 baseline
v1 + breakout
v1 + volume_price_pressure
v1 + rejection_reversal
v1 + breakout + volume_price_pressure
v1 + breakout + rejection_reversal
v1 + volume_price_pressure + rejection_reversal
v2 all
```

The first focus models are:

```text
long_H120
short_H240
```

Rationale:

- `long_H120` was the best-looking v1 long candidate, but worsened in v2.
- `short_H240` improved the most in v2, but is still not net-profitable.

This is still valid-only research.  Do not use test results to pick variants.

## Generate variant configs

The delta already includes generated configs under `configs/features/variants/`.
If they need to be regenerated:

```bash
python3 scripts/build_feature_set_variants.py --force
```

## Build and train all variants

A full run can be expensive because it builds features and trains entry grids for
each variant.  Start with a dry run:

```bash
python3 scripts/run_entry_feature_variants.py \
  --grid-prefix entry_variant \
  --sides long,short \
  --horizons 60,120,240 \
  --dry-run
```

Then run:

```bash
python3 scripts/run_entry_feature_variants.py \
  --grid-prefix entry_variant \
  --sides long,short \
  --horizons 60,120,240
```

For a quicker first pass, focus only on long H120 and short H240 by running two
separate calls:

```bash
python3 scripts/run_entry_feature_variants.py \
  --grid-prefix entry_variant_long120 \
  --sides long \
  --horizons 120

python3 scripts/run_entry_feature_variants.py \
  --grid-prefix entry_variant_short240 \
  --sides short \
  --horizons 240
```

## Compare results

For full-grid runs:

```bash
python3 scripts/compare_entry_feature_variants.py \
  --grid-prefix entry_variant \
  --baseline-variant v1_baseline
```

For focused long H120:

```bash
python3 scripts/compare_entry_feature_variants.py \
  --grid-prefix entry_variant_long120 \
  --baseline-variant v1_baseline \
  --focus long:120
```

For focused short H240:

```bash
python3 scripts/compare_entry_feature_variants.py \
  --grid-prefix entry_variant_short240 \
  --baseline-variant v1_baseline \
  --focus short:240
```

The main outputs are:

```text
outputs/valid/entry_grid/feature_variant_comparison.csv
outputs/valid/entry_grid/feature_variant_focus_comparison.csv
```

## How to judge

Prefer variants that improve more than one top-quantile bucket and do not damage
worst-fold behavior.

Useful columns:

```text
mean_top_q95_avg_target_bps
worst_top_q95_avg_target_bps
mean_top_q97_avg_target_bps
worst_top_q97_avg_target_bps
mean_top_q99_avg_target_bps
worst_top_q99_avg_target_bps
mean_top_q99_precision_gt0
worst_top_q99_precision_gt0
```

A family is not adopted merely because one mean metric improves.  It needs to
hold up across valid folds.

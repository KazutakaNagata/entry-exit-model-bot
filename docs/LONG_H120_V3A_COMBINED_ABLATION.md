# long_H120 v3a combined ablation

This experiment is the second pass after the `long_H120_v3a` leave-one-family-out ablation.

The first pass suggested that the following families are suspicious for the `long_H120` entry model:

- `compressed_activity`
- `rejection_reversal`
- `breakout`

It also suggested that these families should be treated carefully rather than dropped blindly:

- `volume_price_pressure`: mixed. Wider q95/q97 improved when removed, but q99 worst-fold behavior worsened.
- `volume_spike`: removing it hurt long_H120, so it is included only as a stress-test removal.

## Goal

Test whether the suspicious families hurt additively or only independently.

This is still a valid-fold research experiment. It is not a production selection step and must not read test data.

## Generated configs

Configs live under:

```text
configs/features/ablations/long_H120_v3a_combined/
```

The variants are:

```text
v3a_full
v3a_minus_compressed_activity
v3a_minus_compressed_activity_rejection_reversal
v3a_minus_compressed_activity_breakout
v3a_minus_rejection_reversal_breakout
v3a_minus_compressed_activity_rejection_reversal_breakout
v3a_minus_compressed_activity_volume_price_pressure
v3a_minus_compressed_activity_rejection_reversal_breakout_volume_price_pressure
v3a_minus_compressed_activity_rejection_reversal_breakout_volume_spike
```

## Run

Dry-run first:

```bash
python3 scripts/run_long_h120_v3a_combined_ablation.py --dry-run
```

Run all variants:

```bash
python3 scripts/run_long_h120_v3a_combined_ablation.py
```

Run only the main candidates:

```bash
python3 scripts/run_long_h120_v3a_combined_ablation.py \
  --ablations v3a_full,v3a_minus_compressed_activity,v3a_minus_compressed_activity_rejection_reversal,v3a_minus_compressed_activity_breakout,v3a_minus_compressed_activity_rejection_reversal_breakout
```

If features already exist and you only want to train:

```bash
python3 scripts/run_long_h120_v3a_combined_ablation.py \
  --skip-existing-features
```

## Compare

```bash
python3 scripts/compare_long_h120_v3a_combined_ablation.py
```

Output:

```text
outputs/valid/entry_grid/long_H120_v3a_combined_ablation_comparison.csv
```

## Interpretation rules

Do not select by mean q99 alone.

Prefer variants that:

- improve q95/q97 as well as q99,
- improve or at least do not destroy worst folds,
- keep q99 precision stable,
- reduce feature count without losing robust score,
- make domain sense for long_H120.

The current working hypothesis is:

```text
long_H120 = regime transition + trend continuation + volatility state
```

So variants that remove short-trigger or immediate-breakout noise while preserving regime/volatility/trend features are the main candidates.

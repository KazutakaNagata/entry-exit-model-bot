# Selected long_H120 feature sets

This document records the current reviewed long_H120 entry candidates after the
v3a combined-ablation and v3b structure-addition experiments.

These are **valid-fold candidates only**. They are not locked production or test
settings.

## Candidates

### `long_H120_v0`

Primary candidate:

```text
entry_feature_set_v3a - compressed_activity
```

Reason:

- Best robust valid-fold score among the current long_H120 experiments.
- Preserves the v3a long_H120 q99 strength while improving q95/q97 versus full
  v3a.
- Keeps `breakout`, `volume_price_pressure`, `rejection_reversal`, and
  `volume_spike`; combined ablation showed that blindly dropping them after
  removing `compressed_activity` can degrade ranking.

### `long_H120_tail_v0`

Tail comparison candidate:

```text
long_H120_v0 + compressed_price_geometry
```

Reason:

- Improved q99 metrics in the v3b addition experiment.
- Weakened q97, so it is not the primary candidate.
- Evaluate only as a strict-threshold / top-tail candidate and then verify with
  episode P/L.

## Commands

Write configs:

```bash
python3 scripts/build_selected_long_h120_feature_configs.py --force
```

Build features and train both selected candidates:

```bash
python3 scripts/run_selected_long_h120_entry.py
```

Run only one candidate:

```bash
python3 scripts/run_selected_long_h120_entry.py --candidates long_H120_v0
```

Compare selected candidates:

```bash
python3 scripts/compare_selected_long_h120_entry.py
```

## Downstream

After selecting the entry candidate to continue with, regenerate downstream data
from its OOF predictions:

```text
entry OOF -> make_exit_dataset.py -> train_exit_lgbm.py -> backtest_episode_valid.py
```

The previous exit pipeline expects OOF entry predictions, drops rows whose target
crosses valid-fold boundaries, and uses position features computed only from
entry execution through current time. Keep those constraints unchanged.

## Review rules

- Do not evaluate or tune on test.
- Do not treat `tail_v0` as better just because q99 is higher; check q95/q97,
  worst fold, and episode P/L.
- If features are rebuilt, run `audit_features.py --strict` before training.
- If entry feature set changes, regenerate OOF predictions and exit datasets.

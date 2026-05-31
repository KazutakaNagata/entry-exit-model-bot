# Entry Walk-Forward Retraining

This module evaluates the already-selected `long_H120_tail_v0` entry feature set
with model retraining.  It is not a new feature-selection or threshold-tuning
framework.

## Why this exists

The fixed-train research path found a promising valid-only strategy:

- entry model: `long_H120_tail_v0`
- selection: rolling quantile
- window: 60 days
- quantile: 0.995
- floor: 40 bps
- exit: fixed hold 120 minutes

That result is useful, but a final live-like check needs retraining because a
real bot can update the model over time.

## Training modes

`train_entry_walk_forward.py` supports:

- `fixed`: initial manifest train only, mainly for baseline parity.
- `expanding`: manifest train plus valid folds before the evaluated fold.
- `rolling`: fixed lookback window before each evaluated fold.

All training rows must be before the eval fold and label-overlap safe.  Test is
never read.

## Score history

Each fold model also scores the period immediately before that fold and writes:

```text
score_history.parquet
```

These rows are not tradable entries.  They are only used to warm rolling
quantile thresholds, which is closer to live operation than starting each fold
with no score history.

## Quick run

```bash
python3 scripts/run_selected_long_h120_walk_forward.py --dry-run
python3 scripts/run_selected_long_h120_walk_forward.py
```

Default runs:

- expanding train
- rolling 365d train
- rolling 730d train

and backtests each with:

- rolling quantile score history
- 60d q99.5
- floor 40bps
- fixed hold 120m

Compare episodes:

```bash
python3 scripts/compare_selected_long_h120_episode.py \
  --run-prefix selected_long_H120_walk_forward
```

## Interpretation

Do not compare to test.  This is still valid-only research.  The only question
is whether the fixed-train edge survives model retraining.

## Memory-safe walk-forward notes

`run_selected_long_h120_walk_forward.py` now defaults to the lighter comparison:

```text
--modes rolling
--rolling-train-windows 365
```

The previous default attempted `expanding` first.  Expanding mode is not an
N-day retrain; it means:

```text
fold_01 train = initial train
fold_02 train = initial train + fold_01
fold_03 train = initial train + fold_01 + fold_02
...
```

Because the train matrix grows every fold, it can be killed by the OS on a
local laptop.  This is usually reported as `SIGKILL: 9`, not as a Python
exception.

The walk-forward trainer now includes memory controls:

```bash
python3 scripts/run_selected_long_h120_walk_forward.py \
  --modes rolling \
  --rolling-train-windows 365
```

For a heavier comparison:

```bash
python3 scripts/run_selected_long_h120_walk_forward.py \
  --modes expanding,rolling \
  --rolling-train-windows 365,730
```

If the local machine still runs out of memory, cap each fold to the most recent
training rows:

```bash
python3 scripts/run_selected_long_h120_walk_forward.py \
  --modes rolling \
  --rolling-train-windows 365 \
  --max-train-rows 400000
```

`--max-train-rows` changes the effective training sample, so it should be
recorded in results and not compared blindly against uncapped runs.

By default the walk-forward runner:

```text
- downcasts numeric feature columns to float32
- skips feature importance output
- saves fold models
```

Feature importance can be enabled with:

```bash
--write-feature-importance
```


## Memory-safety defaults

The walk-forward runner downcasts numeric feature columns to `float32`, skips feature importance by default in the orchestration script, injects `force_col_wise=true` into LightGBM parameters by default, releases per-fold model objects with `gc.collect()`, and exposes `--max-train-rows` as an emergency laptop guardrail. These options are intended to prevent local `SIGKILL: 9` failures without changing the selected feature set or entry policy.

# Fixed-Hold Score Filters and Rolling Quantile Warmup

This document covers the leak-safe score filters added to the selected long_H120 fixed-hold research path.

## Why this exists

The currently promising baseline is:

- candidate: `long_H120_tail_v0`
- entry selection: past-only rolling score quantile
- rolling window: 60 days
- rolling quantile: 0.995
- score floor: 40 bps
- exit: fixed 120 minute hold
- cost: one 15 bps roundtrip per episode

The next question is whether the losing fold can be improved by requiring the entry score to be not just above the rolling quantile, but clearly above it.

## Leak-safe filters

The fixed-hold backtester supports three optional filters:

```text
--min-entry-pred-bps      require current predicted net bps >= value
--min-score-margin-bps   require current score - effective threshold >= value
--min-score-ratio        require current score / effective threshold >= value
```

These are live-safe because they use only:

- the current model score
- the effective entry threshold known at that timestamp
- the rolling threshold computed from past scores only

They do not use future prices, future labels, or full-fold score distributions.

## Rolling quantile warmup

By default, rolling quantile mode computes thresholds from past scores within the same valid fold only. Therefore, fold_01 does **not** use the final 60 days of train unless a score history file is supplied.

To warm up fold_01 with train-period scores, pass:

```bash
--score-history path/to/score_history.parquet
```

The score history file is used only for rolling threshold calibration. It is not used as tradable entries.

Required columns:

```text
timestamp
side
horizon_minutes
pred_entry_net_bps
```

Optional column:

```text
fold
```

If `fold` is present, history rows are matched to the same eval fold. This is the safest way to provide fold-specific train-tail scores, e.g. scores from the fold_01 entry model on the last 60 train days tagged `fold_01`.

If `fold` is absent, the history rows are shared as past warmup history for every fold.

## Example: score filter sweep

```bash
python3 scripts/sweep_selected_long_h120_score_filters.py
```

This runs the base strategy and small one-filter-at-a-time variants:

- `min_entry_pred_bps`: 60, 70
- `min_score_margin_bps`: 10, 20, 30, 40
- `min_score_ratio`: 1.25, 1.5, 2.0

Compare the output with:

```bash
python3 scripts/compare_selected_long_h120_episode.py \
  --run-prefix selected_long_H120_score_filter_sweep
```

## Interpretation

Do not pick the run with the highest mean only. Check:

- `mean_net_pl_bps_sum`
- `worst_net_pl_bps_sum`
- `episode_count`
- fold-level episode counts
- `mean_avg_net_pl_bps`
- `worst_avg_net_pl_bps`
- `mean_profit_factor`

The goal is to improve the worst fold without reducing trades to a meaningless count.

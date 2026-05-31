# Entry Overlap Diagnostics

This diagnostic compares two already-generated valid-fold episode backtests by
entry timestamp.  It is intended for questions like:

- Did the fixed-train model and rolling-retrained model enter at the same times?
- Are rolling-only entries bad?
- Are fixed-only entries where most of the P/L came from?
- Is the rolling model missing the profitable entries, or adding bad entries?

It does **not** train models, tune thresholds, run a backtest, or read test data.

## Usage

```bash
python3 scripts/compare_entry_episode_overlap.py \
  --left-run-name selected_long_H120_window_sweep_long_H120_tail_v0_hold120_rollQ0p995_win60d_continuous_valid_minP30000_floor40 \
  --right-run-name selected_long_H120_walk_forward_rolling180d_hold120_rollQ0.995_win60d_scoreHistory_floor40 \
  --left-label fixed_train \
  --right-label rolling180d
```

If the run names are long or ambiguous, pass explicit directories:

```bash
python3 scripts/compare_entry_episode_overlap.py \
  --left-run-dir outputs/valid/episode_backtest/<fixed_train_run> \
  --right-run-dir outputs/valid/episode_backtest/<rolling180d_run> \
  --left-label fixed_train \
  --right-label rolling180d
```

## Output

Default output:

```text
outputs/valid/entry_overlap/<left_label>_vs_<right_label>/
  entry_overlap_summary.json
  entry_overlap_by_group.csv
  entry_overlap_by_fold.csv
  entry_overlap_episodes.parquet  # falls back to csv if parquet is unavailable
```

## Main groups

- `both`: entries selected by both policies at the exact same `(fold, side, entry_time)`.
- `left_only`: entries selected only by the left policy.
- `right_only`: entries selected only by the right policy.

For fixed-vs-rolling diagnosis:

- If `left_only` is strongly positive, rolling is missing profitable entries.
- If `right_only` is strongly negative, rolling is adding bad entries.
- If `both` is strong but one side is weak overall, the difference is mostly in the unique entries.
- If overlap is very low, retraining changed the ranking/score distribution materially.

The match is intentionally exact.  Near-time cluster matching can be added later,
but the primary diagnostic should stay strict to avoid hiding differences in the
actual entry rule.

# Rolling Window Sweep for selected long_H120

This experiment tests whether the rolling score threshold should use a shorter
history than the earlier 60-day default.

The strategy under test is still:

```text
entry model: long_H120_tail_v0
entry gate : rolling quantile of past predicted net-return scores
exit       : fixed 120m hold
cost       : roundtrip cost charged once per episode
```

## Why this exists

A valid fold is only about three months.  With fold-local rolling thresholds,
`60d` spends a large part of each fold in a warmup / partially filled state.
That means a good result with `60d` does not prove that 60 days is the best live
lookback.  It may indicate that a shorter effective score distribution window is
sufficient or better.

This sweep uses only live-feasible rules.  It does **not** select entries by
fold-wide top quantiles.

## Run

Dry run:

```bash
python3 scripts/sweep_selected_long_h120_rolling_windows.py --dry-run
```

Default run:

```bash
python3 scripts/sweep_selected_long_h120_rolling_windows.py
```

Default grid:

```text
candidate:  long_H120_tail_v0
windows:    7,14,21,30,45,60 days
quantiles:  0.99,0.995
floors:     30,40,50 bps
```

Compare:

```bash
python3 scripts/compare_selected_long_h120_episode.py \
  --run-prefix selected_long_H120_window_sweep
```

## Focused run

To test the most relevant candidates first:

```bash
python3 scripts/sweep_selected_long_h120_rolling_windows.py \
  --window-days 14,30,60 \
  --quantiles 0.995 \
  --floors 40
```

## Min periods

The default uses `--rolling-score-min-periods auto`.

For q99/q99.5, tiny warmups are unstable.  Auto mode uses approximately half of
the requested window, with a floor and cap:

```text
7d  -> about 5,040 rows
14d -> about 10,080 rows
21d -> about 15,120 rows
30d -> about 21,600 rows
45d -> 30,000 rows
60d -> 30,000 rows
```

Use an explicit integer if you need exact comparability with older runs.

## Optional score history

`--score-history` can warm the rolling threshold with pre-fold score history.
When omitted, thresholds are computed from the current fold's past scores only.

## Rolling history mode

This sweep now accepts:

```bash
--rolling-history-mode fold_local|continuous_valid|score_history
```

Use `continuous_valid` to carry score history from previous valid folds into later folds without using future rows. This makes 90d / 120d windows meaningful when individual folds are only about three months.

Example:

```bash
python3 scripts/sweep_selected_long_h120_rolling_windows.py \
  --rolling-history-mode continuous_valid \
  --window-days 60,90,120 \
  --quantiles 0.995 \
  --floors 40,50
```

# Exit Dataset Builder

The exit model is trained on **position-state rows**, not ordinary timestamp rows.
Each row is identified by:

```text
entry_time, current_time, side
```

The initial dataset builder uses OOF entry predictions only. It refuses input where
`is_oof` is false, because exit training must not be conditioned on in-sample entry
scores.

## Timing

For an entry candidate at decision row `e`:

```text
entry executes at open[e + 1]
```

For an exit decision at row `t`:

```text
current exit price = open[t + 1]
future exit price  = open[t + 1 + K]
```

The target is:

```text
target_exit_hold_delta_bps = side * log(open[t+1+K] / open[t+1]) * 10000
```

A fresh roundtrip cost is **not** subtracted from every hold decision. Fixed
roundtrip cost is applied once per episode in episode backtests.

## Position features

The builder adds position-state features such as:

```text
age_minutes
unrealized_pl_bps
unrealized_pl_after_cost_bps
mfe_since_entry_bps
mae_since_entry_bps
giveback_from_mfe_bps
entry_pred_net_bps
current_pred_net_bps
score_decay_bps
```

`MFE`, `MAE`, and `giveback` are computed only from bars between entry execution
and `current_time`, inclusive. They are allowed in the exit dataset because they
summarize information observed after entry, not future information beyond the
current decision.

## Boundary rules

Rows are dropped when:

- the entry candidate is outside its valid fold,
- the target would cross the valid fold boundary,
- exact one-minute OHLCV timestamps needed for entry/current/future prices are missing,
- the entry score is not OOF.

The first implementation intentionally uses simple candidate filtering and simple
fixed-K hold-delta targets. Exit model training and episode backtesting are later
phases.

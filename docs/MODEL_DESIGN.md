# MODEL_DESIGN

## 全体設計

このリポジトリでは、bot を以下の状態機械として考えます。

```text
state = flat / long / short
```

flat 状態では entry model を使い、position 保有中は exit model を使います。

```text
flat:
    entry_model -> enter / no trade

position:
    exit_model -> hold / exit
```

## Entry model

### 目的

Entry model は「この episode を始める価値」を予測します。

短期方向予測ではなく、手数料後に残る固定保有 P/L を初期 target にします。

### Target

```text
side = +1 for long
side = -1 for short

entry_price = open[t+1]
exit_price  = open[t+1+H]

gross_return_bps = side * log(exit_price / entry_price) * 10000
net_return_bps   = gross_return_bps - roundtrip_cost_bps
```

### Model

初期モデル:

```text
LGBMRegressor
```

分類より回帰を優先します。理由は、手数料 15bps 環境では `+1bps` と `+80bps` が全く違うためです。

### Initial horizons

```text
H = 60m, 120m, 240m
```

初期モデル名:

```text
entry_long_H60
entry_long_H120
entry_long_H240
entry_short_H60
entry_short_H120
entry_short_H240
```

### Entry decision example

```text
enter long if:
    pred_long_H120_net_bps > entry_threshold
    and pred_long_H240_net_bps > secondary_threshold
    and no active cooldown
```

閾値は valid 5 fold のみで選びます。

## Exit model

### 目的

Exit model は「今 exit するより、もう少し hold する価値があるか」を予測します。

### Target

初期版では、position 保有中の row `t` について次を使います。

```text
current_exit_price = open[t+1]
future_exit_price  = open[t+1+K]

hold_delta_bps = side * log(future_exit_price / current_exit_price) * 10000
```

entry 価格を明示する場合:

```text
current_pl = side * log(current_exit_price / entry_price) * 10000 - cost
future_pl  = side * log(future_exit_price  / entry_price) * 10000 - cost
hold_delta = future_pl - current_pl
```

固定コストは差分でキャンセルされます。

### Model

初期モデル:

```text
LGBMRegressor
```

### Initial lookaheads

```text
long:  K = 30m, 60m
short: K = 15m, 30m
```

### Exit decision example

```text
hold if:
    pred_hold_delta_bps > hold_threshold

exit if:
    pred_hold_delta_bps <= hold_threshold for exit_confirm_bars
```

`hold_threshold` に 15bps を要求しないでください。15bps は新規 entry の問題であり、position 保有中は追加保有価値を見ます。

## Episode policy v0

初期の episode policy は単純にします。

```text
flat:
    if entry_pred > entry_threshold:
        enter

position:
    if exit_pred > hold_threshold:
        hold
    else:
        exit after confirmation

force exit:
    max_hold_minutes
    hard_stop_bps
    data_quality_error
```

churn 防止:

```text
entry_threshold > hold_threshold
cooldown_after_exit_minutes
same_side_reentry_block_minutes
exit_confirm_bars
max_round_trips_per_day
```

## Features

特徴量はすべて `t` 以前の 1分足から作ります。

初期 family:

```text
return_path
volatility
trend_persistence
micro_range
support_resistance
market_structure_void
rejection_reversal
pullback_geometry
breakout
volume_price_pressure
candle_shape
```

旧 long 60m q=0.95 で効いていた family は、entry の初期 feature set に優先して持ち込みます。

## Metrics

### Entry metrics

```text
top_q90_avg_net_bps
top_q95_avg_net_bps
top_q97_avg_net_bps
top_q99_avg_net_bps
worst_fold_top_q95_avg_net_bps
worst_fold_top_q99_avg_net_bps
score_decile_return
trade_count
MFE
MAE
```

### Exit metrics

```text
pred_hold_delta decile actual_hold_delta
exit_model vs fixed_exit episode P/L
giveback_before_exit
avg_hold_minutes
MAE / MFE during episode
```

### Episode metrics

```text
episode_net_pl_bps
episode_gross_pl_bps
round_trips_per_day
fee_paid_bps_per_day
profit_factor
win_rate
avg_win
avg_loss
avg_hold_minutes
median_hold_minutes
worst_fold_episode_net_pl
```


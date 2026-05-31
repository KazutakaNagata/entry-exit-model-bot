# LEAKAGE_RULES

この文書は、未来データリークを防ぐための強制ルールです。実装者は、モデル成績よりもまずこのルールを優先してください。

## 時刻定義

1分足の row `t` について、以下の時刻を区別します。

```text
bar t:
    open[t], high[t], low[t], close[t], volume[t]

feature_time[t]:
    feature row t が利用可能になる時刻

decision_time[t]:
    bar t が確定した直後

execution_time[t]:
    原則として bar t+1 の open
```

## Feature 生成ルール

`feature row at t` で使ってよいデータは、**t 以前に確定した 1分足のみ**です。

許可:

```text
close[t]
open[t]
high[t]
low[t]
volume[t]
rolling over <= t
expanding over <= t
```

禁止:

```text
close[t+1]
high[t+1]
low[t+1]
future rolling window
centered rolling window
shift(-1), shift(-H) を feature に使うこと
future MFE/MAE を feature に使うこと
label / target 列を feature に混ぜること
```

## Entry label 生成ルール

Entry label は未来を見て作ります。ただし、feature matrix と厳密に分離します。

row `t` の entry target は、原則として次の価格で作ります。

```text
entry_price = open[t+1]
exit_price  = open[t+1+H]
```

long:

```text
net_return_bps = log(exit_price / entry_price) * 10000 - roundtrip_cost_bps
```

short:

```text
net_return_bps = -log(exit_price / entry_price) * 10000 - roundtrip_cost_bps
```

## Exit label 生成ルール

Exit model は position 保有中の row に対して作ります。

row `t` で exit 判断する場合、約定は原則 `open[t+1]` とします。

シンプルな初期 target:

```text
current_exit_price = open[t+1]
future_exit_price  = open[t+1+K]
hold_delta_bps     = side * log(future_exit_price / current_exit_price) * 10000
```

固定コストは current と future の比較では多くの場合キャンセルされます。したがって exit model に毎回 15bps の追加利益を要求しないでください。

## Multi-timeframe 特徴量

1分足から 5m / 15m / 60m 相当の特徴量を作る場合も、必ず `t` 以前だけを使います。

禁止例:

```text
15分足がまだ確定していないのに、確定済み15分足として扱う
現在の60分足の最終 high/low を途中時点で使う
```

安全な例:

```text
rolling_15m_return = close[t] / close[t-15] - 1
rolling_60m_high   = max(high[t-59:t])
```

## 全期間fit禁止

以下は fold ごとに train のみで fit してください。

- scaler
- imputer
- encoder
- PCA / compression
- target-aware feature selection
- model calibration
- threshold selection

全期間や valid/test を含めて fit してはいけません。

## OOF ルール

二段目モデル、exit dataset、episode policy に model prediction を使う場合は、必ず OOF prediction を使います。

禁止:

```text
train 全体で fit した entry model の train prediction を exit dataset の特徴量に使う
```

許可:

```text
purged valid fold に対する OOF prediction を entry_score として使う
```

## Test 隔離

`test` は locked config で最後に 1回だけ評価します。

禁止:

```text
test metrics を見て feature set を変える
test metrics を見て threshold を変える
test metrics を見て horizon を変える
test metrics を見て short を無効化する
```

## レビュー用チェックリスト

PRごとに以下を確認してください。

- [ ] `shift(-...)` が feature 側にない。
- [ ] rolling window に `center=True` がない。
- [ ] feature matrix に `target`, `future`, `mfe`, `mae` などの label 系列が混ざっていない。
- [ ] entry price が `t+1 open` になっている。
- [ ] train / valid / test の分割が config に従っている。
- [ ] test を読んでいない。
- [ ] OOF が必要な場所で in-sample prediction を使っていない。


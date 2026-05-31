# RESEARCH_RULES

## 目的

このリポジトリの目的は、BTCJPY の 1分足データから、往復コスト 15bps 程度を考慮しても成立しうる swing / trend bot を研究することです。

短期の方向予測そのものではなく、**1回のポジションとして手数料後に残るか**を重視します。

## 戦略の基本単位

この研究では、1回のポジションを `episode` と呼びます。

```text
episode = entry してから exit するまでの一連の保有区間
```

評価は bar 単位ではなく episode 単位を主役にします。

## モデルの役割

### Entry model

Entry model は、flat 状態で「この episode を始める価値があるか」を予測します。

初期 target は以下です。

```text
side = +1 for long
side = -1 for short

gross_return_bps = side * log(exit_price / entry_price) * 10000
entry_target_bps = gross_return_bps - roundtrip_cost_bps
```

初期値:

```text
roundtrip_cost_bps = 15.0
H = 60m, 120m, 240m
```

### Exit model

Exit model は、position 保有中に「今 exit するより hold した方がよいか」を予測します。

初期 target は以下です。

```text
hold_delta_bps = future_value_bps - current_value_bps
```

シンプルな初期版では、`future_value` は `K` 分後に exit した場合の P/L です。

初期値:

```text
long:  K = 30m, 60m
short: K = 15m, 30m
```

## 旧研究の扱い

旧 tail classifier 研究は、以下の形で利用します。

- 旧 long 60m q=0.95 は、entry 特徴量・setup detector として参考にする。
- 旧 short tail は、そのまま entry には使わない。特徴量 family の知見だけ引き継ぐ。
- 旧 top95/top99 label は、新戦略の直接 target ではなく benchmark 扱いにする。

## Long / Short の扱い

Long と short は対称に扱いません。

- long は旧研究で一定の予測可能性があり、初期本命。
- short は top score が episode P/L に直結しにくく、慎重に扱う。
- short は no-trade が最良判断になる可能性を常に許容する。

## 評価の原則

- mean だけで採用しない。
- worst fold を必ず見る。
- q95/q97/q99 など複数の上位 bucket を見る。
- trade count が少なすぎる改善は疑う。
- gross ではなく net を主役にする。
- episode 単位の P/L を最終判断に近い指標とする。

## 禁止事項

- test 結果を見て調整する。
- feature selection に test を使う。
- threshold selection に test を使う。
- test 成績が悪いから valid へ戻って都合よく調整する。
- 未来データを特徴量に使う。
- in-sample prediction を二段目モデルや exit dataset の特徴量にする。


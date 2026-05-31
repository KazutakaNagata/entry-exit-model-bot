# DATA_CONTRACT

## 入力データ

このリポジトリは、まず BTCJPY の 1分足 OHLCV データだけを前提にします。

推奨カラム:

```text
timestamp
open
high
low
close
volume
```

任意カラム:

```text
quote_volume
trade_count
taker_buy_base_volume
taker_buy_quote_volume
```

## timestamp

- timezone を明示してください。
- 内部処理では UTC に揃えることを推奨します。
- 1分足は昇順、重複なしである必要があります。

## OHLCV quality checks

Data audit で最低限チェックする項目:

```text
timestamp monotonic increasing
timestamp duplicated rows
missing 1m bars
open/high/low/close positive
high >= max(open, close)
low <= min(open, close)
volume >= 0
large gaps
zero volume runs
```

## 欠損の扱い

初期方針:

- 小さな欠損は data audit で報告する。
- 欠損補完は勝手に行わない。
- 学習に使う前に、人間が欠損期間を確認する。
- gap をまたぐ label / episode は原則として除外する。

## 約定価格

初期の研究では、約定価格を以下に固定します。

```text
entry execution: next minute open
exit execution:  next minute open
```

Bid/ask や板データがない段階では、往復コスト 15bps に手数料・スプレッド・スリッページのバッファを含めます。

## 保存形式

推奨:

```text
raw:       data/raw/btcjpy_1m.csv or parquet
processed: data/processed/*.parquet
```

Parquet を推奨しますが、初期実装では CSV も許容します。


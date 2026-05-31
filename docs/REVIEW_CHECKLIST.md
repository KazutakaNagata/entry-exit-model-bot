# REVIEW_CHECKLIST

PR / task ごとの人間レビュー用チェックリストです。

## 共通

- [ ] 今回のタスク範囲を超えて実装していない。
- [ ] 不要なクラス階層や framework を追加していない。
- [ ] 新しい外部依存を勝手に追加していない。
- [ ] 生成物を git 管理対象にしていない。
- [ ] `python -m compileall swing_bot scripts` が通る。

## Data

- [ ] timestamp が昇順である。
- [ ] 重複 timestamp を検出している。
- [ ] missing bar を検出している。
- [ ] OHLC の整合性を検査している。
- [ ] gap をまたぐ label / episode を除外できる設計になっている。

## Split

- [ ] train / valid / test が config で固定されている。
- [ ] valid は 5 fold に分かれている。
- [ ] purge / embargo が config にある。
- [ ] test を通常 training script が読まない。

## Features

- [ ] feature row at `t` は `t` 以前のデータだけを使っている。
- [ ] `shift(-...)` が feature 側にない。
- [ ] centered rolling を使っていない。
- [ ] 全期間 fit の scaler / PCA / imputer がない。
- [ ] feature manifest に family / lookback / uses_future がある。
- [ ] target / label / future / MFE / MAE 列が feature matrix に混ざっていない。

## Labels

- [ ] entry label は `t+1 open` から作っている。
- [ ] fixed-hold exit は `t+1+H open` を使っている。
- [ ] long / short の符号が正しい。
- [ ] roundtrip cost が entry target から引かれている。
- [ ] exit label は hold_delta になっている。
- [ ] exit label に毎回 roundtrip 15bps を要求していない。

## Modeling

- [ ] fit は train のみで行っている。
- [ ] valid fold ごとの prediction が保存されている。
- [ ] OOF が必要な場所で in-sample prediction を使っていない。
- [ ] model selection に test を使っていない。

## Evaluation

- [ ] mean だけでなく worst fold を出している。
- [ ] q95/q97/q99 の top bucket を見ている。
- [ ] trade count を出している。
- [ ] gross だけでなく net を出している。
- [ ] episode 単位の metrics がある。

## Test audit

- [ ] locked config だけを読んでいる。
- [ ] test 中に threshold を変えていない。
- [ ] test 中に feature set を変えていない。
- [ ] test 中に horizon を変えていない。
- [ ] test 結果を valid 調整へ戻していない。


# IMPLEMENTATION_PLAN

AI 実装での事故を避けるため、作業を小さく分けます。各 task は人間レビュー可能なサイズにしてください。

## Task 0: docs / rules

作るもの:

```text
README.md
AGENTS.md
docs/RESEARCH_RULES.md
docs/LEAKAGE_RULES.md
docs/MODEL_DESIGN.md
docs/VALIDATION_PROTOCOL.md
docs/DATA_CONTRACT.md
docs/REVIEW_CHECKLIST.md
```

完了条件:

```text
まだ学習コードを書かない
研究ルールが明文化されている
```

## Task 1: repo skeleton

作るもの:

```text
swing_bot/ package
scripts/
configs/
tests/
data/
outputs/
models/
tmp/
```

完了条件:

```bash
python -m compileall swing_bot scripts
```

## Task 2: data loader / audit

作るもの:

```text
swing_bot/data/schema.py
swing_bot/data/load_ohlcv.py
swing_bot/data/quality.py
scripts/audit_data.py
```

完了条件:

```text
1分足データを読み込める
データ品質レポートを出せる
```

## Task 3: split manifest

作るもの:

```text
configs/splits/btcjpy_1m_split_v1.yaml
swing_bot/splits/split_manifest.py
swing_bot/splits/purged_walk_forward.py
```

完了条件:

```text
train / valid 5 fold / test が固定される
test は通常 script から読まれない
```

## Task 4: cost and labels

作るもの:

```text
swing_bot/labels/costs.py
swing_bot/labels/entry_net_return.py
swing_bot/labels/exit_hold_delta.py
swing_bot/labels/mfe_mae.py
```

完了条件:

```text
entry target が t+1 open から作られる
exit target が t+1 open から作られる
long/short の符号がテストされる
roundtrip cost が entry target から引かれる
```

## Task 5: minimal features

作るもの:

```text
swing_bot/features/registry.py
swing_bot/features/build_matrix.py
swing_bot/features/manifest.py
swing_bot/features/leakage_audit.py
swing_bot/features/families/price.py
swing_bot/features/families/return_path.py
swing_bot/features/families/volatility.py
swing_bot/features/families/trend_persistence.py
```

完了条件:

```text
feature_manifest.json が出る
全featureに family / lookback / uses_future=false が付く
```

## Task 6: legacy feature families

旧研究から有望 family を移植します。

優先順:

```text
return_path
volatility
trend_persistence
support_resistance
market_structure_void
rejection_reversal
pullback_geometry
micro_range
volume_price_pressure
breakout
candle_shape
```

完了条件:

```text
旧 long 60m q95 の知見を entry feature set に反映できる
各 family がリークレビュー可能
```

## Task 7: entry LGBM MVP

最初は `entry_long_H60` だけ実装します。

作るもの:

```text
swing_bot/models/lgbm_common.py
swing_bot/models/entry_lgbm.py
swing_bot/evaluation/topk.py
swing_bot/evaluation/fold_metrics.py
swing_bot/evaluation/entry_report.py
scripts/train_entry_lgbm.py
scripts/evaluate_entry_valid.py
```

完了条件:

```text
valid 5 fold の entry report が出る
test は読まない
```

## Task 8: entry multi-horizon

広げるモデル:

```text
entry_long_H60/H120/H240
entry_short_H60/H120/H240
```

完了条件:

```text
side × horizon の valid比較ができる
short は no-trade も選択肢として扱う
```

## Task 9: exit dataset

作るもの:

```text
swing_bot/backtest/episode.py
swing_bot/backtest/policy.py
scripts/make_exit_dataset.py
```

完了条件:

```text
OOF entry prediction から exit training dataset が作れる
in-sample entry score を使わない
```

## Task 10: exit LGBM MVP

最初は `exit_long_K30` だけ実装します。

作るもの:

```text
swing_bot/models/exit_lgbm.py
swing_bot/evaluation/exit_report.py
scripts/train_exit_lgbm.py
```

完了条件:

```text
exit valid report が出る
fixed exit baseline と比較できる
```

## Task 11: episode valid backtest

作るもの:

```text
swing_bot/backtest/metrics.py
swing_bot/evaluation/episode_report.py
scripts/backtest_episode_valid.py
```

完了条件:

```text
entry + exit の episode P/L が valid 5 fold で評価できる
threshold tuning は valid のみ
```

## Task 12: locked config

Valid で選んだ設定を保存します。

出力:

```text
outputs/valid/<run_id>/locked_config.json
```

完了条件:

```text
test audit に必要な設定がすべて固定される
```

## Task 13: final test audit

作るもの:

```text
scripts/evaluate_test_locked.py
```

完了条件:

```text
locked config を読むだけ
threshold / feature / horizon / policy を変更しない
```


# VALIDATION_PROTOCOL

## 基本方針

この研究では、test を最終監査専用にします。

```text
train:
    model fit

valid:
    feature set / horizon / threshold / policy / hyperparameter の調整
    valid 内 5 fold で評価

test:
    locked config で最後に 1回だけ評価
```

## Split manifest

分割は config で固定します。

```text
configs/splits/btcjpy_1m_split_v1.yaml
```

このファイルには以下を明記します。

```text
train period
valid period
valid fold 1-5
test period
purge_minutes
embargo_minutes
```

研究途中で split を変更する場合は、既存ファイルを書き換えず `split_v2` として新規作成してください。

## Valid 5 fold

Valid は 5 fold に分けます。

採用判断では、平均だけでなく以下を必ず見ます。

```text
mean across folds
worst fold
fold sign rate
trade count per fold
q95/q97/q99 stability
```

1 fold だけ良い改善は、原則として採用しません。

## Purge / embargo

Entry label は未来の `H` 分を使うため、train と valid の label window が重ならないように purge します。

初期ルール:

```text
purge_minutes >= max_entry_horizon_minutes
```

Exit / episode まで含める段階では、より保守的にします。

```text
purge_minutes >= max(max_entry_horizon, max_hold + max_exit_lookahead)
```

## Valid 出力

Valid run は以下へ保存します。

```text
outputs/valid/<run_id>/
```

最低限保存するもの:

```text
run_config.json
split_manifest_snapshot.yaml
feature_manifest.json
fold_metrics.csv
predictions_oof.parquet or .csv
summary.md
```

## Locked config

Valid で選んだ設定は locked config として保存します。

```text
outputs/valid/<run_id>/locked_config.json
```

含めるもの:

```text
split_version
feature_set
entry_model_name
exit_model_name
horizons
thresholds
cost_bps
cooldown
max_hold
hard_stop
selected_run_ids
```

## Test audit

Test は以下の script だけが読む想定です。

```text
scripts/evaluate_test_locked.py
```

Test audit は locked config を読み、何も調整しません。

出力先:

```text
outputs/test_audit/<locked_run_id>/
```

保存するもの:

```text
test_metrics.json
test_episodes.parquet or .csv
test_report.md
locked_config_snapshot.json
```

## 禁止事項

- test 成績を見て threshold を変える。
- test 成績を見て horizon を変える。
- test 成績を見て feature set を変える。
- test 成績を見て short を無効化する。
- test 成績を見て valid 評価へ戻る。

## 採用判断の例

Entry model の採用基準例:

```text
mean_top_q95_avg_net_bps > 0
worst_top_q95_avg_net_bps >= 0 or small negative tolerance
q95/q97/q99 が極端に乖離していない
trade_count が十分
MFE/MAE が許容範囲
```

Episode policy の採用基準例:

```text
mean_episode_net_pl_bps > baseline
worst_fold_episode_net_pl が悪化しない
round_trips_per_day が過剰でない
fee_paid_bps_per_day が過剰でない
avg_hold_minutes が短すぎない
```


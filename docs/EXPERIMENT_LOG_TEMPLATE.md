# EXPERIMENT_LOG_TEMPLATE

## Run

```text
run_id:
date:
owner:
status: draft / valid_selected / locked / test_audited
```

## Purpose

```text
何を確認したい実験か。
```

## Data / Split

```text
data_version:
split_version:
train_period:
valid_period:
test_period: not used / locked audit only
purge_minutes:
```

## Model

```text
model_type: entry / exit / episode_policy
side:
horizon_or_lookahead:
feature_set:
target:
cost_bps:
```

## Results by fold

| fold | metric_1 | metric_2 | metric_3 | notes |
|---|---:|---:|---:|---|
| fold_01 | | | | |
| fold_02 | | | | |
| fold_03 | | | | |
| fold_04 | | | | |
| fold_05 | | | | |

## Mean / Worst

```text
mean:
worst:
fold_sign_rate:
```

## Decision

```text
adopt / reject / quarantine / needs more evidence
```

## Notes

```text
気になった点、リーク疑い、fold failure、次に見ること。
```


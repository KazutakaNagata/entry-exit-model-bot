# Monthly family ablation: score-history quantile and combined-drop fix

This patch fixes four issues in the score-history-quantile monthly family ablation workflow.

## 1. Threshold source

`configs/rolling_protocol/long_H120_monthly_family_ablation_score_history_quantile_v0.yaml` now sets:

```yaml
training:
  score_history_days: train
```

For `entry_selection.mode: score_history_quantile`, this makes each valid/test segment compute the q99.5 threshold from the actual model-fit rows, i.e. the 9-month train window for that segment. It no longer uses only the previous 120 days.

The entry threshold is:

```text
effective_threshold = max(q995(train_score_history), score_floor_bps)
```

The default `score_floor_bps` remains `0.0` in this config.

## 2. Combined drop is integrated into the main monthly run

After the run evaluates:

```text
full
full - family_1
full - family_2
...
```

it now finds every single-family drop that improved over `full` under the configured metric and evaluates one additional candidate:

```text
full - all_improving_single_drop_families
```

This combined candidate is evaluated with a real model fit and backtest. It is not inferred from the single-drop results.

Config controls:

```yaml
family_ablation:
  combine_improving_single_drops: true
  combined_drop_metric_col: robust_score
  combined_drop_min_delta_bps: 0.0
```

## 3. Progress logging

`monthly_rolling.py` logs JSON lines to stdout and to:

```text
outputs/valid/rolling_monthly_research/<run_id>/progress.jsonl
```

Events include policy load, segment fit start/done, policy backtest done, combined-drop start/done, selected policy, and test backtest done.

## 4. Memory behavior

`cache_policy_data` defaults to `false`. Each policy feature matrix is loaded, evaluated, released, and garbage-collected before moving to the next policy. The combined-drop candidate is materialized once per cycle and then evaluated normally.

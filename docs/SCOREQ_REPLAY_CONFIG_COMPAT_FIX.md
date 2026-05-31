# scoreQ replay config compatibility fix

`replay_monthly_family_ablation_score_quantile.py` calls
`load_rolling_monthly_config(..., allow_empty_feature_policies=True)` because replay reconstructs
candidate feature policies from existing run directories rather than from the protocol config.

A later `monthly_rolling.py` memory/logging patch accidentally reverted the function signature to
`load_rolling_monthly_config(path)` and reintroduced a hard requirement for `feature_policies`.
That caused:

```text
TypeError: load_rolling_monthly_config() got an unexpected keyword argument 'allow_empty_feature_policies'
```

This patch restores the compatibility flag while preserving the memory/logging behavior.

Behavior:

- Default remains strict: `load_rolling_monthly_config(path)` still requires `feature_policies`.
- Replay can call: `load_rolling_monthly_config(path, allow_empty_feature_policies=True)`.
- No model/backtest behavior is changed.

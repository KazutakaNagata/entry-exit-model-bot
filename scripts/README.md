# scripts

CLI entrypoints will live here.

Rules:

- Keep scripts thin.
- Put reusable logic in `swing_bot/`.
- Training/validation scripts must not read test data.
- `evaluate_test_locked.py` is the only script that should read the test split, and only with a locked config.

Planned scripts:

```text
audit_data.py
build_features.py
train_entry_lgbm.py
evaluate_entry_valid.py
make_exit_dataset.py
train_exit_lgbm.py
backtest_episode_valid.py
evaluate_test_locked.py
```


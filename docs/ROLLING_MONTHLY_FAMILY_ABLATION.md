# Rolling Monthly Family Ablation Selector

This adds a mechanical family-level feature selection protocol for the monthly rolling research runner.

The intent is to avoid hand-picking families from the current validation period.  The runner starts from the full family universe and generates:

```text
full
full - family_A
full - family_B
...
```

Each monthly cycle evaluates these policies on the three prior validation months and selects a policy with a fixed robust score.  No family is globally deleted; each cycle starts again from the full universe.

## Score

The selector uses validation fold/month metrics:

```text
robust_score = total_net_pl_bps
             + 0.5 * worst_month_net_pl_bps
             + 0.25 * median_month_net_pl_bps
```

Constraints are applied as penalties:

```text
min_total_episode_count
min_active_valid_folds
max_worst_valid_loss_bps
```

## Build only

```bash
python3 scripts/build_family_ablation_policies.py \
  --features data/processed/binance_japan/BTCJPY/1m/features_entry_v3b.parquet \
  --manifest outputs/valid/features/entry_feature_manifest_v3b.json \
  --output-dir data/processed/binance_japan/BTCJPY/1m/family_policies/long_H120_monthly_v0 \
  --policy-prefix lofo_long_H120
```

## Run monthly LOFO protocol

```bash
python3 scripts/run_rolling_monthly_family_ablation.py \
  --first-test-month 2025-04-01 \
  --last-test-month 2026-04-01 \
  --run-id long_H120_monthly_family_ablation_v0
```

One cycle only:

```bash
python3 scripts/run_rolling_monthly_family_ablation.py \
  --first-test-month 2025-04-01 \
  --last-test-month 2026-04-01 \
  --max-cycles 1 \
  --run-id long_H120_monthly_family_ablation_v0_smoke
```

## Compare results

```bash
python3 scripts/compare_rolling_monthly_family_ablation.py \
  --run-prefix long_H120_monthly_family_ablation_v0
```

Outputs:

```text
outputs/valid/rolling_monthly_family_ablation/
  family_candidate_valid_metrics_all.csv
  family_ablation_deltas.csv
  family_selected_policy_summary.csv
  family_rolling_test_metrics_all.csv
  family_ablation_summary.json
```

## Interpretation

For `family_ablation_deltas.csv`:

```text
delta_score_vs_full > 0
  Removing that family improved the selector score in that cycle.

delta_score_vs_full < 0
  Removing that family hurt the selector score in that cycle, so the family was useful in that full-context comparison.
```

Do not permanently delete a family from one cycle.  The policy is selected per rolling cycle, and the next cycle starts from the full universe again.

# Family Ablation Combined-Drop Follow-up

The first monthly family ablation MVP evaluates:

```text
full
full - family_1
full - family_2
...
```

and selects the best single policy. That is intentionally simple, but it can miss
the case where many families individually hurt and should be dropped together.

This follow-up script reuses an existing completed single-month LOFO run:

1. Read `candidate_valid_metrics.csv`.
2. Compare each `full - family` against `full` using a metric, default
   `selector_raw_score`.
3. Collect all families whose single-drop metric improves full by more than
   `--min-delta-bps`.
4. Build a new combined feature policy:
   `full - all_improving_families`.
5. Optionally add add-back candidates:
   `combined + family_i`.
6. Run only these second-stage candidates for the same test month.

This avoids repeating the expensive full single-drop run. It does **not** avoid
training entirely: LightGBM must still be retrained for the combined feature set,
because combined-drop performance cannot be derived algebraically from the
single-drop outputs.

## Example: one month

```bash
python3 scripts/run_family_ablation_combined_followup.py \
  --source-run-name long_H120_family_ablation_202504_202507_202505 \
  --run-id long_H120_family_ablation_202505_combined_drop \
  --metric-col selector_raw_score \
  --min-delta-bps 0
```

With add-back candidates:

```bash
python3 scripts/run_family_ablation_combined_followup.py \
  --source-run-name long_H120_family_ablation_202504_202507_202505 \
  --run-id long_H120_family_ablation_202505_combined_drop_addback \
  --metric-col selector_raw_score \
  --min-delta-bps 0 \
  --include-addback
```

## Metric choice

Default is `selector_raw_score`, not `robust_score`, because the latter may
include activity constraints. Single-drop policies with too few trades can still
be useful signals for what to remove; the combined-drop candidate is evaluated
again with constraints.

Use `--metric-col robust_score` if you want the penalty-adjusted score.

## Important limitation

This script can identify `all_improving_families` from existing outputs without
retraining. But it cannot know the combined-drop P/L without retraining and
backtesting the combined feature set. LightGBM split structure changes when
families are removed together.

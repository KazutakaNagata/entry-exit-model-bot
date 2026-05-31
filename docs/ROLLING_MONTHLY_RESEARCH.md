# Rolling Monthly Research Protocol

This protocol tests the research process, not a single hand-picked feature set.
For each rolling test month:

1. Evaluate a fixed list of feature policy candidates on the three previous monthly validation folds.
2. Each validation month is trained on the immediately preceding nine months.
3. A fixed selector chooses one feature policy using valid-only metrics.
4. The chosen policy is trained on the nine months before the test month.
5. The next one month is evaluated with fixed-hold 120m and rolling quantile score selection.

The selector rules and candidate policies are fixed before running the rolling test series. Do not modify them after inspecting rolling test results without starting a new research process.

## Dry-run

```bash
python3 scripts/run_rolling_monthly_research.py \
  --first-test-month 2025-04-01 \
  --last-test-month 2025-09-01 \
  --dry-run
```

## Run

```bash
python3 scripts/run_rolling_monthly_research.py \
  --first-test-month 2025-04-01 \
  --last-test-month 2025-09-01
```

## Compare

```bash
python3 scripts/compare_rolling_monthly_research.py \
  --run-prefix rolling_monthly
```

## Important constraints

- Test months are evaluated only after feature policy selection on prior valid months.
- Feature policy candidates are domain-defined; policy selection is mechanical.
- This does not touch the separate final locked test audit.
- The default candidates are `long_H120_v0` and `long_H120_tail_v0`.

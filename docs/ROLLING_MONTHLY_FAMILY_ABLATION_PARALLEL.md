# Rolling Monthly Family Ablation Parallel Jobs

This layer splits monthly family-ablation research into one job per test month.
Each job runs the full monthly cycle for exactly one test month:

1. Build or reuse full + leave-one-family-out feature policies.
2. Evaluate candidate feature policies on the prior 3 valid months.
3. Select one feature policy using the fixed robust selector.
4. Train on the prior 9 months.
5. Backtest the selected policy on the test month.

The prepared job scripts write logs and status files so long runs can be monitored from tmux or cloud VMs.

## Important design

Family policy parquet files are prebuilt once before parallel jobs start. Each monthly job then calls:

```bash
python3 scripts/run_rolling_monthly_family_ablation.py ... --skip-policy-build
```

This avoids four parallel jobs all loading the huge base feature matrix to rebuild the same family-policy files.

## Prepare 4 monthly jobs

For four test months starting April 2025:

```bash
python3 scripts/prepare_rolling_monthly_family_ablation_jobs.py \
  --first-test-month 2025-04-01 \
  --last-test-month 2025-07-01 \
  --batch-id long_H120_family_ablation_202504_202507
```

This creates:

```text
outputs/valid/rolling_monthly_family_ablation_jobs/long_H120_family_ablation_202504_202507/
  prebuild_family_policies.sh
  job_manifest.csv
  jobs/test_202504.sh
  jobs/test_202505.sh
  jobs/test_202506.sh
  jobs/test_202507.sh
  logs/
  status/
  run_all_serial.sh
  run_all_parallel.sh
  launch_tmux.sh
```

## Run in tmux

```bash
bash outputs/valid/rolling_monthly_family_ablation_jobs/long_H120_family_ablation_202504_202507/launch_tmux.sh long_h120_family_ablation
```

Attach:

```bash
tmux attach -t long_h120_family_ablation
```

The launcher first runs `prebuild_family_policies.sh`, then opens one tmux window per month.

## Run with shell parallelism

For 4 parallel tasks:

```bash
bash outputs/valid/rolling_monthly_family_ablation_jobs/long_H120_family_ablation_202504_202507/run_all_parallel.sh 4
```

Use `2` on a smaller local machine. LightGBM is memory-heavy.

## Watch progress

```bash
python3 scripts/watch_rolling_monthly_family_ablation_jobs.py \
  --batch-dir outputs/valid/rolling_monthly_family_ablation_jobs/long_H120_family_ablation_202504_202507 \
  --watch \
  --interval-sec 30
```

## Aggregate results

```bash
python3 scripts/aggregate_rolling_monthly_family_ablation_jobs.py \
  --batch-dir outputs/valid/rolling_monthly_family_ablation_jobs/long_H120_family_ablation_202504_202507 \
  --run-prefix long_H120_family_ablation_202504_202507 \
  --output-dir outputs/valid/rolling_monthly_family_ablation_aggregate/long_H120_family_ablation_202504_202507
```

Outputs include:

```text
family_rolling_test_metrics_all.csv
family_rolling_cycles_all.csv
family_candidate_valid_metrics_all.csv
job_status_all.csv
aggregate_summary.json
```

## Cloud VM notes

- GPU is not required.
- Prefer CPU cores + RAM.
- Start with 4 parallel jobs only on a large VM.
- If jobs are killed, lower parallelism first.
- Failed monthly jobs can be rerun by executing their individual `jobs/test_YYYYMM.sh` script.

## Test discipline

This workflow uses each monthly test only once per fixed selector process. Do not inspect one test month, change selector logic, then claim the same batch remains out-of-sample.

## Memory/logging note

Monthly family-ablation runs now stream candidate policy matrices one at a time by default.  They no longer keep all candidate feature matrices in a Python cache.  Progress is written both to stdout and to:

```text
outputs/valid/rolling_monthly_research/<run_id>/progress.log
```

When launched through the prepared job scripts, those progress lines also appear in:

```text
outputs/valid/rolling_monthly_family_ablation_jobs/<batch>/logs/test_YYYYMM.log
```

# Rolling Monthly Research: Parallel Execution

Monthly rolling research can be split by `test_month`. Each test month is independent enough to run as a separate process, then aggregate results later.

## Prepare jobs

```bash
python3 scripts/prepare_rolling_monthly_jobs.py \
  --first-test-month 2025-04-01 \
  --last-test-month 2026-04-01 \
  --batch-id long_H120_monthly_v0_202604
```

This creates:

```text
outputs/valid/rolling_monthly_jobs/<batch_id>/
  job_manifest.csv
  job_manifest.json
  jobs/test_YYYYMM.sh
  logs/test_YYYYMM.log
  status/test_YYYYMM.json
  run_all_serial.sh
  run_all_parallel.sh
  launch_tmux.sh
```

Each job runs:

```bash
python3 scripts/run_rolling_monthly_research.py \
  --first-test-month YYYY-MM-01 \
  --last-test-month YYYY-MM-01 \
  --max-cycles 1
```

## Run locally with limited parallelism

```bash
bash outputs/valid/rolling_monthly_jobs/<batch_id>/run_all_parallel.sh 2
```

Use `2` or `3` on a local machine. LightGBM is CPU and memory hungry; too much parallelism can be slower or kill jobs.

## Run with tmux

```bash
bash outputs/valid/rolling_monthly_jobs/<batch_id>/launch_tmux.sh long_h120_monthly
```

Attach:

```bash
tmux attach -t long_h120_monthly
```

## Watch progress

```bash
python3 scripts/watch_rolling_monthly_jobs.py \
  --batch-dir outputs/valid/rolling_monthly_jobs/<batch_id> \
  --watch \
  --interval-sec 30
```

Each job writes a status JSON with `running`, `success`, or `failed`. Logs are in the batch `logs/` directory.

## Aggregate results

```bash
python3 scripts/aggregate_rolling_monthly_jobs.py \
  --job-root outputs/valid/rolling_monthly_jobs/<batch_id> \
  --output-dir outputs/valid/rolling_monthly_aggregate/<batch_id>
```

This collects available `rolling_test_metrics.csv`, `rolling_cycles.csv`, `candidate_valid_metrics.csv`, and job statuses.

## Cloud notes

Prefer CPU instances with enough RAM over GPU instances. Start with modest parallelism and increase only if memory headroom remains. The safest pattern is:

```text
one monthly job per tmux window or process
LightGBM threads kept moderate
parallelism limited by RAM, not CPU count
```

Keep every job isolated by test month. Failed months can be rerun without touching successful months.

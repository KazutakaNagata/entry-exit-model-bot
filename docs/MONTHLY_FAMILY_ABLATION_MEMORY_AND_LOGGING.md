# Monthly Family Ablation: memory and progress logging patch

This patch fixes a major memory issue in `run_rolling_monthly_research`.

## What changed

Before this patch, monthly family ablation kept every loaded candidate policy matrix in an in-process cache:

```text
full
full - family_1
full - family_2
...
```

That is dangerous for LOFO runs because each policy parquet can contain hundreds of columns and more than a million rows. Holding many of them simultaneously can exhaust VM RAM.

Now the default behavior is:

```text
load one policy matrix
fit/evaluate its 3 valid months
write outputs
delete the dataframe
gc.collect()
move to the next policy
```

A config flag still exists for tiny smoke tests:

```yaml
training:
  cache_policy_data: false  # default
```

Do not enable this for real family ablation jobs.

## Progress log

Each monthly research run now writes:

```text
outputs/valid/rolling_monthly_research/<run_id>/progress.log
```

The same progress lines are also printed to stdout, so the existing job logs capture them:

```text
outputs/valid/rolling_monthly_family_ablation_jobs/<batch>/logs/test_YYYYMM.log
```

Progress events include:

```text
monthly_research_start
cycle_start
policy_start
policy_data_loaded
valid_segment_fit_start
valid_segment_fit_done
policy_done
cycle_selected_policy
test_segment_fit_start
test_segment_fit_done
cycle_done
monthly_research_done
```

## Existing job scripts

Existing generated job scripts can be reused after applying this patch. They call the same Python entrypoint, so the new no-cache behavior and progress logging apply automatically.

If a job is already running with the old code, stop it and rerun it after applying this patch.


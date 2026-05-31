# score-history quantile replay config fix

`replay_monthly_family_ablation_score_quantile.py` reuses an existing monthly
family-ablation run.  In this replay path, candidate feature policies are read
from the source run directories, so the config file does not need to define a
non-empty `feature_policies` list.

The loader now supports `allow_empty_feature_policies=True`, and the replay
script uses that flag.  Normal rolling monthly runs still require explicit
feature policies unless the caller intentionally opts into this replay-only
mode.

Example:

```bash
python3 scripts/replay_monthly_family_ablation_score_quantile.py \
  --source-run-name long_H120_family_ablation_202504_202507_202504 \
  --run-id long_H120_family_ablation_202504_scoreQ_replay \
  --score-quantile 0.995 \
  --score-floor-bps 0
```

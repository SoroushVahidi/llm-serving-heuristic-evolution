# src/llmserveopt/evaluation

Run/compare/aggregate harness -- the glue between a policy and a metrics
table, and the unified runner for external baselines.

## Key files

- **`run_policy.py`** -- run a single policy on a single workload.
- **`compare.py`** -- run multiple policies on the same workload(s) and
  produce a comparison table.
- **`aggregate.py`** -- roll up multi-seed / multi-regime results.
- **`external_baseline_configs.py`** -- `TopologyDescription`: how many
  GPUs, what roles (prefill/decode/monolithic), for each external baseline.
- **`external_baseline_harness.py`** -- the unified runner for all 6
  faithful external baselines: validates topology before execution, wraps
  `select_action` non-invasively to count admit/preempt/swap/migrate
  events. Use this rather than hand-rolling a runner when evaluating an
  external baseline.

## What not to confuse

This package runs and compares policies -- it does not train anything. For
selector training/evaluation, see
[../selector/README.md](../selector/README.md) (v1) and
`selector/dataset_v2/calibrated_targeted_pilot.py` +
`scripts/train_selector_v2_calibrated_prototype.py` (v2, current).

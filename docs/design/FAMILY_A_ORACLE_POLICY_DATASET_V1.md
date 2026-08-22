# Family-A Oracle Policy Dataset V1

Date: 2026-08-21

## Purpose

Generate the first scaled Family-A oracle-labeled ESTF/WFS dataset after the
pilot verdict `PILOT_DATASET_READY_TO_SCALE`.

This is dataset generation only. It does not train a model, deploy a
controller, modify simulator/policy semantics, or use TEST.

## Sampling Design

The frozen scaled manifest is built from Family-A TRAIN/VAL configuration axes
already present in the repository:

- target utilization: values observed in the Family-A TRAIN/VAL manifest
- tenant weight skew: values observed in the Family-A TRAIN/VAL manifest
- favored tenant size: `long` and `short`
- prediction noise: values observed in the Family-A TRAIN/VAL manifest
- seeds: `20260816` through `20260837`

The resulting manifest has `704` scenarios across `32` configuration groups.
Each configuration group has `22` seeds. Split metadata is TRAIN/VAL only;
seeds divisible by 5 are marked `val`, all others `train`.

`favlong`/`favshort`, seed, split, scenario ID, and configuration group ID are
experimental metadata only and are never model features.

## Row Unit

One row is one eligible online Family-A scheduler decision state where ESTF and
WFS disagree from the same clean pre-decision state and produce exactly one
ESTF-only request and one WFS-only request.

Rows are retained using an outcome-blind rule:

- at most `3` events per scenario
- minimum `100` simulator steps between retained events in the same scenario
- exact duplicate sample IDs and state fingerprints are rejected

## Primary Label

The primary label uses whole-branch priority-weighted SLO-safe utility over the
bounded native counterfactual branch:

```
J_ESTF_whole = sum priority_i for requests completed without SLO violation
               in the ESTF-native branch window

J_WFS_whole = sum priority_i for requests completed without SLO violation
              in the WFS-native branch window

delta_J_whole = J_ESTF_whole - J_WFS_whole
```

Class:

- `ESTF` if `delta_J_whole > 0`
- `WFS` if `delta_J_whole < 0`
- `TIE_OR_UNCERTAIN` if `delta_J_whole == 0`

No practical-equivalence epsilon is invented. Exact numerical margins are
stored for later abstention analysis.

The branch horizon is the repaired Family-A continuation horizon:
`1500` extra steps. Future arrivals are included in both branches. The two
branches start from the same clean pre-decision simulator state.

## Compatibility Label

The pilot contested-pair utility is stored separately:

```
J_ESTF_contested
J_WFS_contested
delta_J_contested
oracle_label_contested
```

This preserves pilot compatibility while allowing future studies to compare
contested-pair labels against the stronger whole-branch target.

## Feature Schema

The model feature schema is pilot-compatible: `63` numeric causal `feat_*`
columns.

Groups:

- global online state
- ESTF-contested request
- WFS-contested request
- same-unit pairwise differences and ratios
- short causal history already frozen in the pilot schema

Forbidden as model features:

- scenario ID
- seed
- split
- favored-size stratum
- synthetic family label
- configuration group ID
- actual future output
- branch outcomes
- utility values
- deltas
- labels
- TEST indicators

The invalid mixed-unit `deadline_slack_if_admitted_now` feature is excluded.

## Grouping

Future cross-validation should use `configuration_group_id` when testing
strict generalization, not random row splits. This groups all seeds sharing
the same workload configuration.

## Parallelism

Parallelism is scenario-sharded and process-isolated:

- each worker is a separate Python process
- each worker owns disjoint manifest rows
- each worker writes only its own shard CSV, progress JSON, done marker, and
  shard log
- the master process merges only after all shard checksums verify
- no worker writes concurrently to the final merged CSV

Worker count for the first launch is `4`, conservative relative to `20`
logical CPUs and available memory.

## Resume Safety

Each completed shard has:

- immutable `shard_XXX.rows.csv`
- `shard_XXX.done.json`
- SHA-256 checksum of the shard rows

A rerun skips a shard only when the done marker checksum matches the shard CSV.
Merge refuses duplicate sample IDs and duplicate state fingerprints.

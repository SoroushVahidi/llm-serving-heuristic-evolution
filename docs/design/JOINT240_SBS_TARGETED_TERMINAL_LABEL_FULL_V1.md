# JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1

## Frozen Objective

Generate the complete targeted SBS-relative terminal state-action label dataset for all SBS-trajectory P6-vs-SBS disagreement states discovered by `JOINT240_SBS_DISAGREEMENT_SCAN_V1`.

This campaign does not train a learner, tune thresholds, build a live scheduler, or modify the manuscript.

## Primary Universe

Primary input:

`experiments/joint240_sbs_disagreement_scan_v1/full_v1/manifest_all_disagreements.csv`

The full manifest contains 8,888 SBS-trajectory decision states where at least one non-SBS P6 policy proposes a different full canonical action from SBS. It spans 236 scenarios and all 5 frozen folds. The 444,128 all-equal SBS trajectory states are not terminal-labeled because all six P6 actions equal SBS and therefore have analytically known zero override advantage under this deterministic one-action-then-SBS estimand.

Episode-thinned states are not the primary campaign because thinning saves only about 8.8%, most disagreement states are already isolated one-step episodes, and the pilot demonstrated beneficial and harmful effects across scenarios and folds.

## Causal Estimand

For predecision state `s` and canonical action `a`:

`Q_SBS(s,a)` is terminal utility after forcing action `a` once at state `s`, then continuing with fixed `kv_constrained_online`.

`a_SBS(s)` is the fresh native SBS action at `s`.

`A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,a_SBS)`.

There is no Alive continuation, no candidate-policy continuation, no H-step hold, no policy-forever switch, and no scenario-level selector interpretation.

Future arrivals, simulator configuration, and scenario seeds are held identical across branches from the same state.

## Unique Action Deduplication

For every state:

1. Compute native actions from all six P6 policies on copied predecision observable state.
2. Full-canonicalize each action.
3. Deduplicate identical canonical actions.
4. Execute one terminal branch per unique action, including SBS.
5. Serialize `state_action_rows` with one causal row per `(state_id, canonical_action_id)`.
6. Serialize `state_policy_action_map` with six rows per state mapping policy IDs to canonical actions.

Expected full branch count from the frozen scan is 30,746 unique action branches rather than 53,328 naive six-policy branches.

## Feature Contract

Primary future model representation remains `STATE_ACTION_V1`:

- 95 `CORE_PHYSICAL_STATE` features;
- candidate-action-vs-SBS `ACTION_DIFF` features;
- no `traj_` Alive/control-history fields in the primary representation.

Features are computed only from the predecision simulator state and pre-execution action differences. Terminal utility labels and provenance fields are kept separate from learnable features.

## Pre-Launch Compatibility Gate

Before the full array can launch:

- rerun targeted-terminal-label tests;
- regenerate the full manifest and verify 8,888 states and 30,746 expected unique branches;
- run the full runner on fixed pilot states;
- require exact match with existing local pilot rows for canonical action IDs, Q values, advantages, and shared features;
- stage exact source and inputs to Wulver;
- run a Wulver Slurm profile job on the same pilot-state subset and require exact reproduction.

If compatibility fails, the full campaign must not launch.

## Full Slurm Plan

Use Wulver Slurm on the CPU/general partition with account `ikoutis`. Shard by deterministic scenario groups so all target states for a scenario stay together in one task. Use resumable `DONE` and `FAILED` sentinels, per-task logs, and per-task provenance. Completed valid shards are not overwritten by the runner.

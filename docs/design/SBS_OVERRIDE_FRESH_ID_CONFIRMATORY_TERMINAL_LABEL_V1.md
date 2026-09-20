# SBS_OVERRIDE_FRESH_ID_CONFIRMATORY_TERMINAL_LABEL_V1

## Purpose

Generate terminal labels for the untouched fresh generator-holdout
confirmatory corpus only. This stage does not train, tune, threshold, analyze
selector utility, or run closed-loop scheduling.

## Confirmatory Blindness

Development data remain all joint240 labels/results, selector-development
outputs, nested-CV results, model-family choices, folds, and thresholds.

Confirmatory data are the fresh generator-holdout scenarios and their terminal
counterfactual outcomes. Before this label-generation campaign, permitted
information is limited to scenario identity, SBS trajectories, P6 action
support, online-safe features, and canonical actions.

Fresh `Q_SBS`, `A_SBS`, beneficial/harmful labels, and target distributions
must not be used for selector design in this task.

## Input Integrity Correction

The earlier support summary counted 2,918 disagreement rows. A pre-labeling
audit found 56 exact duplicate state rows from smoke-scan contamination in the
local aggregate. The clean Wulver full-support-only aggregate uses only the
80 completed `*-of-0080` support shards and contains:

- 2,862 unique fresh-ID disagreement states
- 6,996 unique non-SBS canonical action branches
- 2,862 SBS reference branches
- 9,858 total unique terminal continuations
- 78 fresh scenarios with at least one disagreement

The original contaminated files are preserved; labeling uses only:

- `fresh_state_manifest.full_support_only.csv`
- `fresh_state_policy_action_map.full_support_only.csv`

## Causal Semantics

For each state `s` and unique canonical action `a`:

`Q_SBS(s,a)` is terminal utility after forcing `a` once at `s` and then
following fixed `kv_constrained_online`.

The baseline branch is the fresh native SBS action at the same state. The
stored advantage is:

`A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,SBS)`

No Alive continuation, candidate-policy continuation, H-step hold,
policy-forever switching, or learned closed-loop policy is used.

## Implementation Reuse

This campaign reuses validated joint240 machinery:

- `run_one_step_then_sbs_terminal`
- `live_state_features_v1`
- `action_diff_features_v1`
- unique canonical action deduplication
- DONE/FAILED shard mechanics
- Wulver Slurm array pattern

The new code only adapts state IDs and scenario reconstruction for the fresh
generator-holdout manifest.

## OOD Null Result

Natural external OOD replay remains a separate action-support null result:
Azure 2023 code, Azure 2023 conversation, Azure 2024, Bailian/Qwen, and
BurstGPT v2 all produced zero canonical SBS-vs-P6 disagreement under frozen
native replay semantics. They are not terminal-labeled here.

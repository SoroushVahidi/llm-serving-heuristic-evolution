# JOINT240_SBS_TARGETED_TERMINAL_LABEL_PILOT_V1

## Frozen Purpose

Measure, on a preregistered pilot only, whether SBS-trajectory states with P6-vs-SBS action disagreement have nonzero and/or beneficial one-decision terminal effects. This is not a learning run and not the full 8,888-state campaign.

## Primary State Universe

Primary input is:

`/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-disagreement-scan-v1/experiments/joint240_sbs_disagreement_scan_v1/full_v1/manifest_all_disagreements.csv`

The all-disagreement manifest is primary because it contains 8,888 SBS-trajectory states, while episode-thinned contains 8,107. Thinning saves only about 8.8% and 85.49% of disagreement states are already isolated one-step episodes, so thinning risks removing rare action opportunities for modest savings.

## Causal Unit

The intervention is a unique canonical action, not a policy row.

For state `s` and canonical action `a`:

`Q_SBS(s,a)` is terminal utility after forcing `a` exactly once at predecision state `s`, then continuing with fixed `kv_constrained_online`.

`A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,a_SBS)`.

If multiple P6 policies produce the same canonical action at the same state, exactly one terminal branch is executed for that action, and every policy is mapped back to that canonical action.

## Trajectory Semantics

For every selected state:

1. Reconstruct the exact SBS trajectory from normal scenario initialization.
2. At the target predecision step, query all six P6 native policies on copied observable state.
3. Canonicalize full actions.
4. Deduplicate identical canonical actions.
5. Execute one terminal branch for SBS action and one branch for each unique non-SBS action.
6. In every branch, force the selected action once and then continue fixed SBS to terminal completion.
7. Keep future arrivals, simulator configuration, and scenario seed fixed.

No Alive continuation, no H-step hold, no candidate-policy continuation, and no learner training are used.

## Pilot Selection

Pilot selection is deterministic and outcome-blind, targeting approximately 300 states:

- approximately 60 states per frozen fold;
- maximum 2 states per scenario where possible;
- coverage across distinct canonical action counts, differing-policy signatures, early/middle/late trajectory bins, and isolated versus multi-step disagreement episodes;
- seed `20260919`;
- no terminal labels or previous terminal outcomes affect selection.

## Feature Schema

Primary future representation:

`STATE_ACTION_V1 = 95 CORE_PHYSICAL_STATE + 70 ACTION_DIFF`.

The three `traj_` Alive/control-history features are excluded from the primary representation. Features use only predecision observable simulator state plus candidate-vs-SBS action differences computed before executing either action.

## Output Artifacts

`state_action_rows`: one row per `(state_id, canonical_action_id)`, including SBS baseline and unique non-SBS actions.

`state_policy_action_map`: one row per `(state_id, policy_id)`, mapping all six P6 policies to canonical action IDs.

Shards are resumable and use `DONE`/`FAILED` sentinels. Completed shards are not overwritten when `resume=True`.

## Frozen Post-Pilot Analysis Plan

At the unique non-SBS action level report positive, zero, negative, and nonzero `A_SBS`. At the state level report `best_advantage(s) = max_a A_SBS(s,a)`, beneficial states, zero-best states, and states where every differing action is harmful. Report support by fold, scenario, policy mapping, distinct-action count, and disagreement episode type.

Classify the pilot as:

- `TARGET_SUPPORT_PROMISING` if positive and negative effects both exist across enough distinct states, scenarios, and folds to make later held-out learning plausible;
- `TARGET_SUPPORT_EXTREMELY_SPARSE` if effects exist but are concentrated in too few states or scenarios;
- `DIFFERENT_ACTIONS_BUT_CAUSALLY_NULL` if differing actions are overwhelmingly or exclusively zero effect;
- `PILOT_PENDING` while the launched pilot is incomplete.

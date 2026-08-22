# Family-A DAgger Active Oracle V1

Date: 2026-08-21

## Purpose

Define the closed-loop, on-policy data aggregation protocol for a learned
Family-A ESTF/WFS scheduler.

This is a design and repository-integration document only. It does not launch
rollouts, generate labels, train models, merge datasets, or alter the currently
running `family_a_oracle_dataset_v1_1k` job.

DAgger, AGGREVATE, SafeDAgger, active learning, abstention, group DRO,
decision-focused learning, and query-by-committee are established methods. The
project-specific question is how to adapt them to this repository's
LLM-serving scheduler interfaces without oracle leakage or label-definition
drift.

## Current State To Preserve

The scaled Family-A dataset design defines one supervised row as one online
Family-A scheduler decision state where native ESTF and native WFS propose
different admissible request sets from the same clean pre-decision state. The
primary target is whole-branch priority-weighted SLO-safe utility:

```
delta_J_whole = J_ESTF_whole - J_WFS_whole
```

Labels are:

- `ESTF` when `delta_J_whole > 0`
- `WFS` when `delta_J_whole < 0`
- `TIE_OR_UNCERTAIN` when `delta_J_whole == 0`

No practical-equivalence epsilon is part of the current label definition.
Later abstention may use a deployment margin, but that margin must not rewrite
the oracle label.

The first learner should use the existing pilot-compatible 63 numeric
`feat_*` causal feature columns. Scenario ID, seed, split, favored-size,
configuration group, branch outcomes, utility values, deltas, and labels are
metadata or labels, not model features.

## Repository Integration Points

Runtime scheduling:

- `src/llmserveopt/simulator/simulator.py::Simulator.run` is the closed-loop
  driver. Each step enqueues arrivals, builds an `ObservableState`, calls
  `policy.select_action(state)`, applies the returned `Action`, advances
  decode, and records metrics.
- `src/llmserveopt/policies/base.py::BasePolicy.select_action` is the policy
  insertion contract for any learned scheduler.
- `src/llmserveopt/core/types.py::ObservableState`,
  `ObservableRequest`, and `ObservableGPUState` define the online-visible
  state surface.
- `src/llmserveopt/core/action.py::Action` is the action returned to the
  simulator.

Native ESTF/WFS candidates:

- `src/llmserveopt/policies/estimated_service_time_first.py::EstimatedServiceTimeFirstPolicy`
  is native ESTF.
- `src/llmserveopt/policies/weighted_fair_share.py::WeightedFairSharePolicy`
  is native WFS.
- `src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py::build_native_policy_instances`
  constructs the native policies used by existing Family-A diagnostics.
- `src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py::REGIME_A`
  and `STAGE2_CANDIDATES[REGIME_A]` identify the ESTF/WFS native pair.

Disagreement detection and action comparison:

- `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py::canonical_action`
  canonicalizes admit-only actions.
- `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py::actions_disagree`
  tests whether two native actions differ.
- `scripts/generate_family_a_oracle_policy_v1.py::ScaledFamilyAObserver.select_action`
  already implements the exact Family-A pattern: snapshot pre-decision GPU
  counters, compute ESTF and WFS from the same state, detect disagreement,
  require one ESTF-only and one WFS-only request, then label.

Feature capture:

- `src/llmserveopt/analysis/family_a_observability_continuation_v1.py::extract_causal_features`
  extracts online causal state, pair, and history features.
- `scripts/generate_family_a_oracle_policy_v1.py::feature_columns`,
  `GLOBAL_FEATURES`, `SIDE_FEATURES`, `PAIR_FEATURES`, and `build_row`
  define the scaled dataset feature schema and row layout.

Pre-decision snapshot and branch isolation:

- `src/llmserveopt/analysis/family_a_observability_continuation_v1.py::snapshot_gpu_counters`
  and `restore_gpu_counters` protect the true pre-decision state from policy
  `select_action` bookkeeping mutation.
- `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py::fork_from_live_simulator`
  deep-copies live simulator mutable state for isolated counterfactual
  branches.
- `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py::LiveFork.advance_one_step`
  advances an isolated fork using simulator-native apply/decode logic.

Whole-branch oracle:

- `scripts/generate_family_a_oracle_policy_v1.py::run_weighted_branch`
  runs native bounded continuation and returns priority-weighted SLO-safe
  branch results.
- `scripts/generate_family_a_oracle_policy_v1.py::whole_branch_label` and
  `contested_pair_label` compute the primary and compatibility labels.
- `scripts/generate_family_a_oracle_policy_v1.py::stable_state_fingerprint`
  gives deterministic row-level deduplication keys.

Scenario metadata:

- `scripts/generate_family_a_oracle_policy_v1.py::scenario_from_manifest_row`
  rebuilds scaled Family-A scenarios from manifest rows.
- `scripts/generate_family_a_oracle_policy_v1.py::configuration_group_id`
  defines the configuration group as utilization, skew, favored size, noise,
  total jobs, and max active sequences. Seed is intentionally not in the
  group ID.

## DAgger Iteration Semantics

Let `D_0` be the immutable scaled offline oracle dataset:
`datasets/family_a_oracle_policy_v1/oracle_rows.csv` after the running job
finishes and merge/quality checks pass.

For iteration `k`:

1. Train `pi_k` on aggregated labeled dataset `D_k`, using only approved
   `feat_*` model features.
2. Run `pi_k` closed-loop on TRAIN collection scenarios through the normal
   `Simulator.run` / `BasePolicy.select_action` interface.
3. Log learner-induced visited states with online metadata, predictions,
   abstention decisions, fallback decisions, action candidates, OOD scores,
   and acquisition scores.
4. Split visited states into labelable policy-choice rows and unlabeled
   monitoring rows.
5. Select a small acquisition batch using a deterministic active-learning
   rule, before oracle labels are known.
6. For acquired labelable states, run the exact same bounded native ESTF/WFS
   whole-branch oracle used by `D_0`.
7. Reuse existing labels for duplicate `state_fingerprint` or `sample_id`.
   Do not rerun expensive counterfactual branches when semantics match.
8. Append new labeled rows to an immutable `D_{k+1}` directory with a manifest
   and quality summary.
9. Retrain to obtain `pi_{k+1}`.
10. Evaluate closed-loop on held-out validation groups without adding those
    validation labels to training.
11. Stop, continue, or redesign based on the stopping criteria below.

Oracle labels are collected offline. The oracle never controls an evaluation
policy.

## Learned Scheduler Shape

The first deployable learned scheduler should be a `BasePolicy` wrapper around
native ESTF and WFS:

1. Snapshot the true pre-decision GPU counters.
2. Compute `action_estf` and `action_wfs` from the same pre-decision state.
3. Restore the pre-decision counters between candidate calls.
4. If actions agree, return the agreed action and log an unlabeled support
   row if monitoring is enabled.
5. If actions disagree, extract the 63 `feat_*` features and score the model.
6. If the abstention rule fires, return WFS.
7. Otherwise return the native action selected by the learned model.
8. Restore the state to the selected action's post-admission bookkeeping
   shape before returning, matching the existing observer pattern.

This keeps all causal effects inside the existing simulator and policy
interfaces.

## Safe Policy Mixing

Recommended initial strategy: learner with WFS fallback on abstention. Do not
use oracle action mixing in evaluation.

Rationale:

- Fixed WFS is the strongest fixed parent on Family-A TRAIN/VAL mean ANWG in
  the current evidence.
- Prior controller failures show that `favlong` needs WFS's long-term
  priority/SLO protection.
- A classical beta schedule that mixes learner and WFS can be used only for
  TRAIN data collection if exploration is insufficient, and every mixed-run
  artifact must be marked `collection_policy = beta_mixture_wfs`. It must not
  be reported as the deployed policy's closed-loop performance.

Default collection policy:

```
if abstain:
    action = WFS
else:
    action = learner choice among ESTF/WFS
```

Optional TRAIN-only exploration:

```
beta_1 = 0.20, beta_2 = 0.10, beta_k = 0.0 for k >= 3
with probability beta_k choose WFS, otherwise choose abstaining learner
```

This is not the primary recommendation; it is a fallback if pure abstaining
learner collection yields too little support expansion.

## Abstention Rule

The first scheduler should train a utility-difference regressor for
`delta_J_whole`, plus a calibrated sign probability when available. Let:

- `m = predicted delta_J_whole`
- `p = P(ESTF better)`
- `u = epistemic uncertainty`, e.g. ensemble standard deviation or
  conformal interval half-width
- `d = OOD distance to training support`

Abstain to WFS when any condition holds:

- model cannot produce all required feature values
- OOD score exceeds the training-support threshold
- prediction interval contains zero with high confidence, e.g.
  `abs(m) <= c_u * u`
- calibrated sign probability is not decisive, e.g.
  `max(p, 1 - p) < 0.65`
- model emits `TIE_OR_UNCERTAIN` as the preferred class

Initial fallback policy: fixed WFS.

Do not redefine oracle ties with this rule. Abstention is a deployment safety
choice, not a label-generation change.

## Active Relabeling

Do not label every visited state. Whole-branch labels are expensive.

Candidate acquisition should use quantities available before labeling:

- uncertainty
- OOD distance to existing support
- predicted absolute utility margin
- disagreement between independently trained models
- group undercoverage
- abstention frequency
- duplicate/revisit status

First practical acquisition rule:

1. Keep only labelable ESTF/WFS disagreement states with a stable
   `state_fingerprint` not already labeled.
2. Reserve budget slices:
   - 35% highest uncertainty or query-by-committee disagreement
   - 25% highest OOD distance
   - 20% group-undercovered states, measured by configuration group and
     source iteration
   - 10% high predicted regret-risk states: high uncertainty and large
     predicted nonzero utility magnitude
   - 10% deterministic uniform random sample over the remaining pool
3. Deduplicate after each slice and fill unused quota from the next slice.

The random slice is important because `uncertainty * |predicted delta_J|` can
miss states where the model is confidently or incorrectly near zero.

Unlabeled support-monitoring rows should also be retained for states where
ESTF and WFS agree or where the learner abstains outside the labelable pair
structure, but they are not supervised ESTF/WFS choice rows.

## Label Budget

Conservative schedule:

- `D_0`: the current scaled offline dataset, approximately 1k target rows.
- Iteration 1: acquire at most 250 new oracle labels.
- Iteration 2: acquire at most 250 new oracle labels.
- Iteration 3: optional, at most 150 new oracle labels if iteration 2 shows
  closed-loop gain and persistent OOD/abstention.
- Hard cap before redesign: 650 on-policy labels.

Stop early when validation closed-loop utility saturates or when the new
label batch has more than 40% duplicates/revisits after acquisition.

## Eligible States

Three storage classes are needed:

1. `labelable_policy_choice`: ESTF and WFS disagree, the disagreement has the
   same one ESTF-only vs one WFS-only request structure as `D_0`, the feature
   schema is complete, split discipline permits labeling, and the state is not
   an existing duplicate.
2. `unlabeled_support_monitor`: ESTF and WFS agree, model abstains, state is
   OOD, pair structure is invalid, or the state is in validation and must not
   enter training. These rows support drift diagnostics but not supervised
   selector training.
3. `excluded`: TEST states, malformed states, feature-schema mismatches,
   non-TRAIN/VAL leakage, or rows whose action semantics are not comparable
   to ESTF/WFS native admission.

States where the learned policy differs from WFS are important acquisition
candidates, but the label remains the ESTF-vs-WFS whole-branch oracle, not
the learner-vs-WFS outcome.

## Whole-Branch Oracle Compatibility

Every on-policy label must match `D_0` semantics exactly:

- same clean pre-decision `ObservableState`
- same `snapshot_gpu_counters` / `restore_gpu_counters` discipline
- same native ESTF action and native WFS action from the identical state
- same bounded 1500-step native continuation
- same future-arrivals-included branch semantics
- same priority-weighted SLO-safe whole-branch utility
- same exact tie rule at zero
- same 63 causal feature columns
- same row-level fingerprinting
- same TEST exclusion

The implementation should factor the existing generator's branch labeling
helpers into reusable library code before any DAgger labeler is written.
Do not copy/paste a second oracle.

## Dataset Versioning And Provenance

Use immutable directories:

```
datasets/family_a_dagger_active_oracle_v1/D0/
datasets/family_a_dagger_active_oracle_v1/D1/
datasets/family_a_dagger_active_oracle_v1/D2/
...
```

Each `D_k` contains:

- `oracle_rows.csv`: aggregated labeled rows through iteration `k`
- `new_rows.csv`: rows newly labeled for this iteration
- `visited_states.csv`: unlabeled and labelable candidate states visited by
  `pi_{k-1}`
- `acquisition_manifest.csv`: selected candidate IDs and reasons
- `quality_summary.json`
- `provenance.json`
- `schema.json`
- `feature_classification.csv`

Additional metadata columns:

- `source_iteration`
- `source_policy_version`
- `source_policy_artifact`
- `collection_policy`
- `scenario_id`
- `canonical_scenario_id`
- `configuration_group_id`
- `split`
- `state_fingerprint`
- `sample_id`
- `acquisition_reason`
- `acquisition_score`
- `model_pred_delta_J_whole`
- `model_pred_prob_estf`
- `model_uncertainty`
- `ood_score`
- `abstained`
- `fallback_policy`
- `learner_selected_policy`
- `estf_action_canonical`
- `wfs_action_canonical`
- `oracle_label_version`
- `label_definition_version`
- `feature_schema_version`

All of these are metadata unless explicitly added to an audited feature list
in a later design. They are not model features in V1.

## Deduplication

Before labeling a candidate:

1. Compute `state_fingerprint` using the same normalized feature and
   contested-request payload as the scaled generator.
2. Check an index over all prior `D_0..D_k` fingerprints.
3. Check `sample_id` for exact duplicates.
4. Check `(scenario_id, step, estf_contested_request_id,
   wfs_contested_request_id)` for exact revisits.
5. Optionally check rounded feature-vector near duplicates for diagnostics.

If a duplicate has identical oracle semantics and label-definition version,
reuse its stored row. Do not rerun the expensive branch oracle.

## Group And Distribution Monitoring

For every iteration report:

- configuration groups represented
- visited states per group
- acquired labels per group
- label-producing scenarios per group
- whole-label counts: ESTF, WFS, TIE
- contested-label counts
- OOD rate by group
- abstention rate by group
- fallback rate by group
- learner ESTF/WFS occupancy
- switch count
- distance to `D_0` support
- overlap with prior iterations
- duplicate/revisit rate

Hidden regime metadata is diagnostic only. It must not become a model feature.

## Split Discipline

Do not collect labels from validation scenarios and then evaluate on those
same validation scenarios as unseen.

Recommended V1 split:

- TRAIN groups: closed-loop collection, acquisition, oracle labeling, and
  model fitting.
- DEV-VAL groups: threshold selection, abstention calibration, early stopping,
  and model selection. Labels may exist for DEV-VAL only if they are kept out
  of training and reported as validation-specific.
- FINAL-VAL groups: frozen closed-loop validation for reporting during this
  development phase. No acquisition labels from these groups enter any
  training set.
- TEST: untouched until a preregistered final evaluation.

Because `configuration_group_id` is the intended strict generalization key,
split by configuration group, not by row. If the current TRAIN/VAL manifest is
insufficiently separated, create a deterministic group split over the 32
configuration groups before DAgger begins.

## Closed-Loop Evaluation Protocol

Compare on the same frozen scenario set and report confidence intervals over
scenario or configuration-group units:

- fixed ESTF
- fixed WFS
- initial offline learner `pi_0`
- abstaining `pi_0`
- `pi_1` after first on-policy augmentation
- later `pi_k` only if previous iteration improved closed-loop validation
- existing `family_a_stateful_controller_v1`
- existing receding-horizon oracle controllers as diagnostic, non-deployable
  references
- native ESTF/WFS per-scenario envelope as context, not an available policy

Primary metric:

- whole-scenario priority-weighted SLO utility / ANWG-compatible objective

Secondary metrics:

- regret vs best fixed parent
- fraction of scenarios beating best fixed parent
- recovered oracle-envelope fraction
- worst configuration-group regret
- switch count
- policy occupancy
- abstention coverage
- selective regret on non-abstained decisions
- OOD rate
- fallback rate
- label/query efficiency per iteration

Evaluation policies must not call the oracle. Oracle labels may only be used
offline for training or post-hoc audits.

## Stopping Criteria

Success:

- mean validation ANWG improves over fixed WFS and over abstaining `pi_0`
  with a confidence interval that excludes negligible gain
- worst-group regret is non-positive or materially improved without creating
  a new severe tail
- abstention and OOD rates fall across iterations
- gains hold on configuration-held-out validation groups

Saturation:

- two consecutive iterations improve mean validation ANWG by less than
  0.002 absolute, or less than 10% of remaining gap to the native envelope
- new acquired labels are more than 40% duplicates/revisits
- abstention/OOD rates stop decreasing

Failure:

- offline accuracy or calibration improves while closed-loop ANWG does not
- learned policy loses to fixed WFS on mean or worst-group behavior
- acquisition repeatedly collapses into one configuration group
- DAgger repeatedly discovers large OOD regions without reducing OOD rate
- label budget grows without closed-loop value gain

## Failure Modes And Mitigations

- Oracle leakage: keep oracle calls out of evaluation policy classes; test
  source-text and runtime call paths.
- Validation leakage: split by `configuration_group_id`; never train on
  acquired validation labels.
- Policy mixing contamination: mark collection policy; evaluate only pure
  deployed policy variants.
- Repeated expensive labels: global fingerprint/sample index before branch
  rollout.
- Acquisition collapse into one regime: enforce group budget slices and
  report per-group quotas.
- Miscalibrated uncertainty: use ensembles plus calibration diagnostics;
  retain random acquisition slice.
- Fallback dominance: track fallback and abstention occupancy; tighten only
  after validation improvement.
- Learner induces non-disagreement states: store them as unlabeled support
  rows, not supervised ESTF/WFS labels.
- Whole-branch cost blowup: small per-iteration budgets and duplicate reuse.
- Fingerprint instability: deterministic tests over repeated extraction from
  the same pre-decision state.
- Semantic drift: one shared oracle library used by `D_0` and all DAgger
  iterations.
- Outcome-based sampling bias: acquisition scores computed before labels and
  logged in manifests.

## Minimal Implementation Plan

Proposed modules and scripts:

- `src/llmserveopt/learning/family_a_dagger_active_oracle_v1.py`
  - `FamilyALearnedSelectorPolicy`
  - abstention/fallback logic
  - on-policy state logger
- `src/llmserveopt/learning/family_a_oracle_labeling_v1.py`
  - refactored reusable helpers from
    `scripts/generate_family_a_oracle_policy_v1.py`
  - no duplicated oracle semantics
- `scripts/collect_family_a_dagger_candidates_v1.py`
  - closed-loop TRAIN collection only
  - writes visited/acquisition candidate manifests
- `scripts/acquire_family_a_dagger_labels_v1.py`
  - labels selected candidates only
  - reuses prior labels by fingerprint
- `scripts/aggregate_family_a_dagger_dataset_v1.py`
  - creates immutable `D_k` directories
- `scripts/train_family_a_learned_selector_v1.py`
  - trains `pi_k` from aggregated `D_k`
- `scripts/evaluate_family_a_learned_selector_closed_loop_v1.py`
  - closed-loop evaluation without oracle calls

Required tests:

- `tests/test_family_a_dagger_split_discipline_v1.py`
- `tests/test_family_a_dagger_provenance_v1.py`
- `tests/test_family_a_dagger_oracle_compatibility_v1.py`
- `tests/test_family_a_dagger_state_fingerprint_v1.py`
- `tests/test_family_a_dagger_dedup_reuse_v1.py`
- `tests/test_family_a_dagger_acquisition_v1.py`
- `tests/test_family_a_dagger_group_budget_v1.py`
- `tests/test_family_a_learned_selector_abstention_v1.py`
- `tests/test_family_a_learned_selector_fallback_v1.py`
- `tests/test_family_a_no_oracle_leakage_eval_v1.py`
- `tests/test_family_a_dagger_aggregation_reproducibility_v1.py`

## Design Decision

Classification: `DAGGER_ACTIVE_ORACLE_DESIGN_READY`

The repository already has the necessary simulator interface, pre-decision
snapshot discipline, ESTF/WFS disagreement detector, causal feature extractor,
branch fork machinery, and whole-branch oracle row builder. The DAgger work
should therefore be an integration/refactoring effort, not an oracle redesign.

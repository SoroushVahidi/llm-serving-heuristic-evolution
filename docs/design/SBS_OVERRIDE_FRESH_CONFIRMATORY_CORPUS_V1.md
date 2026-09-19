# SBS_OVERRIDE_FRESH_CONFIRMATORY_CORPUS_V1

## Purpose

Construct a fresh confirmatory scenario corpus for SBS-relative state-action
terminal labels after the joint240 learnability run ended as
`HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN`.

This stage is dataset construction only. It does not fit, tune, threshold, or
evaluate any learned selector on the fresh corpus.

## Development / Confirmatory Boundary

Development data are the entire existing joint240 targeted corpus and all
derived OOF outputs:

- 8,888 SBS-trajectory disagreement states
- 236 scenarios
- 21,858 non-SBS unique actions
- all existing model, threshold, fold, bootstrap, and OOF summaries

These data may be used to design the next selector, feature set, model family,
regularization, uncertainty/abstention logic, and thresholding strategy.

Confirmatory data are scenarios whose terminal state-action labels have not
been used to choose the next selector. Fresh terminal outcomes must not be
inspected until the development-side selector design is frozen.

## Duplication Gate

Prior search found related artifacts but no sufficient fresh untouched
SBS-relative targeted terminal-label corpus for this confirmatory stage.
Classification: `RELATED_BUT_NOT_EQUIVALENT`.

Related artifacts include public trace replay/stage0 windows, prospective
actionable-opportunity outputs, Azure 2024/Bailian/Qwen staging, and the
joint240 SBS terminal-label corpus. These are not a completed fresh
confirmatory SBS override label set.

## Frozen Candidate Corpus

The preregistered candidate manifest is:

`experiments/sbs_override_fresh_confirmatory_corpus_v1/fresh_candidate_scenario_manifest.csv`

It contains 240 scenarios:

- 80 fresh in-distribution generator-holdout scenarios from the joint
  multimechanism generator with IDs outside the joint240 development range.
- 80 Azure 2024 external/OOD scenarios: 40 staged windows x load factors
  `1.0` and `2.0`.
- 80 Bailian/Qwen external/OOD scenarios: 40 staged windows x load factors
  `1.0` and `2.0`.

Window/source selection is outcome-blind. Load factors are fixed before
support scanning. Terminal labels are not generated during manifest
preparation or support scanning.

## Policy-Separation Gate

For each candidate scenario, execute fixed SBS
`kv_constrained_online`. At each SBS predecision state:

1. Query all six P6 native policies on safe copied states.
2. Execute only the SBS action.
3. Record state-level and policy-level canonical action disagreement support.

This gate may reject or downweight sources for absent P6-vs-SBS action support.
It must not use terminal benefit, terminal Q-values, or learned selector
performance because those labels do not exist yet.

## Terminal Labeling Stop Rule

Only after `fresh_state_manifest.csv` and
`fresh_state_policy_action_map.csv` are frozen may targeted terminal labeling
be launched. The label estimand must match joint240:

`Q_SBS(s,a) = force a once, then fixed SBS continuation`

`A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,a_SBS)`

All labeling outputs must be sharded, resumable, atomic, and provenance
checked. After fresh terminal labels are produced, stop; do not fit or tune on
the fresh labels.

## Development-Side Selector Sketch

The next selector improvement should be chosen using development data only and
should prioritize downside control:

- conservative lower-confidence advantage gating;
- stricter abstention objectives that penalize harmful overrides and scenario
  losses;
- scenario-balanced training/evaluation;
- ExtraTrees/HGB only if selected by development-side nested CV;
- minimum predicted margin plus uncertainty gate.

Fresh confirmatory outcomes must not be used to choose among these designs.

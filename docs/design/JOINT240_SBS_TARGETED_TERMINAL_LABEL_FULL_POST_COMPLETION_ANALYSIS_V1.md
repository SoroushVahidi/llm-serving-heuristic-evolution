# JOINT240 SBS Targeted Terminal Label Full V1 - Post-Completion Analysis Plan

This document freezes the post-completion analysis for
`JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1` before interpreting the complete
target distribution.  The analysis reads completed shard outputs only.  It does
not rerun simulator branches, train a learner, tune thresholds, launch a live
scheduler, or modify manuscript claims.

## Inputs

- Full run root:
  `experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1`
- Expected source commit:
  `ff34f6fa0303b6f21e13277553f2d5da789b01d4`
- Expected state universe: 8,888 SBS-trajectory disagreement states.
- Expected causal rows: 30,746 unique `(state_id, canonical_action_id)` rows.
- Expected policy map rows: 53,328, exactly six P6 policy mappings per state.
- Cross-platform reproducibility contract:
  `CROSS_PLATFORM_NUMERIC_REPRODUCIBILITY_AMENDMENT_V1`.

## Integrity Gates

G1 completeness, G2 uniqueness, G3 SBS identity, G4 action deduplication,
G5 numeric validity, G6 cross-platform reproducibility, G7 feature leakage,
G8 fold/scenario conservation, and G9 provenance must pass before scientific
interpretation.  If any hard gate fails, the only allowed verdict is
`FULL_DATASET_INTEGRITY_FAIL`.

## Scientific Summaries

The primary action-level analysis uses unique non-SBS canonical actions only;
policy aliases mapping to the same action are not independent causal samples.
The primary state-level quantity is `best_advantage(s)`, the maximum ANWG
advantage over unique non-SBS actions at state `s`.

The analysis reports action, state, scenario, fold, policy-alias, temporal, and
feature-descriptive summaries.  Feature summaries are exploratory
characterization only and do not select a final model feature subset.

## Learning-Readiness Rule

Learning is permitted only as a future experiment if positive and negative
target support is broad across states, scenarios, and all frozen folds, with
scenario-group-safe evaluation possible.  Even then, no learner is trained in
this analysis task.

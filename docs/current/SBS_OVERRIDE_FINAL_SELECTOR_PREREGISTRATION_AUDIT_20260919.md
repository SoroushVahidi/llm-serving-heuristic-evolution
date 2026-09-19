# SBS Override Final Selector Preregistration Audit

Created: 2026-09-19.

## Verdict

`FINAL_CONFIRMATORY_SELECTOR_V1` is retained as:

`HISTORICAL_DEVELOPMENT_SELECTOR_WITH_AMBIGUOUS_FINAL_SELECTION_PREREGISTRATION`

This is a procedural preregistration issue, not a mathematical invalidation of
the V1 implementation or its development-only evidence.

Fresh confirmatory terminal outcomes remain scientifically blind:

`CONFIRMATORY_LABEL_ACCESS = NOT_ACCESSED`

## V1 Ambiguity

The V1 conservative selector line contained two conflicting final-selection
rules.

1. The implementation selected the final configuration from aggregated
   nested-inner candidate summaries, then refit the gate using
   full-development grouped crossfit predictions.
2. The V1 design markdown stated that the final V2 choice would be made from
   full outer-OOF development summaries using a scenario-bootstrap lower bound
   first.

The run-generated preregistration file cleanly froze the model families,
weighting choices, tau grid, residual gates, nested protocol, and inner
selection objective before outer outcomes were visible. It did not
unambiguously specify the final post-nested-CV confirmatory selector rule.

## Timeline Evidence

| Time / commit | Artifact | Evidence |
| --- | --- | --- |
| 2026-09-19 13:35:30 -0400 mtime | `scripts/sbs_override_conservative_selector_dev_v1.py` | Implements nested-inner summary final config selection plus full-development gate refit. |
| 2026-09-19 13:35:30 -0400 mtime | `docs/design/SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V1.md` | States an outer-OOF/bootstrap final-selection rule. |
| 2026-09-19 13:36:15 -0400 mtime | `experiments/sbs_override_conservative_selector_dev_v1/run_v1/PREREGISTERED_SEARCH_DESIGN.json` | Clean pre-outcome freeze of candidate families, weights, gates, tau grid, folds, and inner objective; silent on final post-CV selector choice. |
| 2026-09-19 13:46:09 -0400 mtime | `outer_fold_progress/outer_0.json` | First outer-fold progress artifact. |
| 2026-09-19 13:56:16 -0400 mtime | `outer_fold_progress/outer_1.json` | Second outer-fold progress artifact. |
| 2026-09-19 13:56:17 -0400 mtime | `SOURCE_FREEZE_QUERY3.json` | Freezes source hashes after two outer progress artifacts existed. |
| `b4c0971`, 2026-09-19 13:56:42 -0400 | V1 script/design/test committed | Commits both conflicting rules after `outer_0` and `outer_1` existed. |
| 2026-09-19 14:28:34 -0400 mtime | `FINAL_CONFIRMATORY_SELECTOR_V1.json` | Output provenance records `selected_by = mean_of_nested_inner_candidate_scores_refit_gate_on_full_development_oof`. |

Because the clean pre-outcome run-generated design does not resolve the final
selection rule, and the later committed source/design pair conflict, neither
rule is treated as the unambiguous V1 confirmatory preregistration.

## Why V1 Is Retained

V1 artifacts are development evidence and provenance. They should not be
deleted, overwritten, renamed in place, or silently corrected. In particular,
the following remain historical artifacts:

- `FINAL_CONFIRMATORY_SELECTOR_V1.json`
- `FINAL_CONFIRMATORY_SELECTOR_V1.joblib`
- `FINAL_CONFIRMATORY_SELECTOR_V1_DESIGN.md`
- `confirmatory_protocol_and_verdict_freeze.json`
- `development_oof_result.json`
- `final_development_oof_result.json`
- nested OOF prediction and decision files

V1 is not the confirmatory selector. It remains useful for auditing the
development search, understanding the conservative gate behavior, and preserving
the exact procedural history.

## V2 Remediation

`SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V2` supersedes the ambiguous V1 final
selection rule.

V2 freezes the following methodological decisions before any V2 full
development computation:

- nested outer OOF remains evaluation-only;
- outer held-out outcomes are not used for final configuration selection;
- final selection is performed in a separate full-development grouped-CV stage;
- every candidate is evaluated only with out-of-fold predictions;
- q90/q95 residual margins are calibrated only from out-of-fold residuals;
- the final all-development model is fitted only after the candidate and gate
  are frozen;
- fresh confirmatory outcomes are not used.

The superseded V1 markdown statement about selecting the final configuration
from outer-OOF bootstrap summaries is not rewritten as if it never happened. It
is documented here as a V1 ambiguity and is superseded only for V2.

## Confirmatory Boundary

The fresh confirmatory corpus may be inspected only for structural/provenance
metadata until a separate explicitly authorized one-shot confirmation task.

Forbidden before that task:

- reading fresh `Q_SBS`;
- reading fresh `A_SBS`;
- reading fresh terminal utilities;
- counting fresh positive, negative, or zero effects;
- evaluating V1 or V2 against fresh labels;
- using fresh outcomes for debugging, threshold selection, calibration, or
  interpretation.

`CONFIRMATORY_LABEL_ACCESS = NOT_ACCESSED`

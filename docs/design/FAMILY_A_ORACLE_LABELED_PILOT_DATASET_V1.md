# Family-A Oracle-Labeled Pilot Dataset V1

Date: 2026-08-21

## Purpose

Build a small, offline-only pilot dataset for deciding whether Family-A online
scheduler states can be labeled for ESTF-vs-WFS learning without using TEST,
training a controller, changing simulator semantics, or relying on the
previously diagnosed raw-completed-count target.

This pilot is a dataset quality study. It is not the final large dataset and
not a production controller.

## Row Unit

One row is one eligible online scheduler decision state where ESTF and WFS
disagree under symmetric evaluation from the same clean pre-decision state.

V1 uses the existing repaired 91 Family-A disagreement events:

- Source rows:
  `experiments/family_a_contested_request_value_diagnosis/contested_requests.csv`
  and
  `experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv`.
- Scope: TRAIN/VAL only.
- Structure: exactly one ESTF-only contested request and one WFS-only contested
  request in every event.
- Group key: `scenario_id` / `canonical_scenario_id`; all grouped-CV folds are
  scenario-atomic.

## Oracle Label Definition

The label uses priority-weighted SLO contribution for the two directly
contested requests under native continuation:

```
J_ESTF(s) = sum_{i in contested pair}
    priority_i * 1[request i completes and does not violate SLO under br_estf_estf]

J_WFS(s) = sum_{i in contested pair}
    priority_i * 1[request i completes and does not violate SLO under br_wfs_wfs]

delta_J = J_ESTF - J_WFS
```

Class:

- `ESTF` if `delta_J > 0`
- `WFS` if `delta_J < 0`
- `TIE_OR_UNCERTAIN` if `delta_J == 0`

No practical-equivalence threshold is invented in V1. The only tie threshold is
exact deterministic equality (`0.0`). This preserves all numerical utility
margins for future regression, ranking, or abstention work.

## Why This Avoids the Prior Bias

Prior reports established that raw completed-request count is ESTF-biased in
`favlong` and is a poor surrogate for the project objective. V1 therefore does
not use `delta_native_whole_branch_raw` as the class label.

Instead, V1 uses the numerator semantics of the corrected objective:
priority-weighted SLO-safe completion. A request contributes zero if it misses
its SLO, even if it completes. This preserves the completion/SLO distinction
that the prior constrained-formulation and contested-request studies found to
be central.

Known limitation: V1 uses contested-pair utility, not whole-branch
arrival-normalized weighted goodput. This is acceptable for a pilot quality
study because prior evidence shows each disagreement is a clean 1-vs-1
contested pair and the pair explains a substantial part of event-level value.
The larger scale-up should store whole-branch weighted/SLO branch utility too.

## Continuation Semantics

V1 uses existing bounded native continuation outcomes:

- ESTF branch: `br_estf_estf`
- WFS branch: `br_wfs_wfs`
- bound: 1500 extra steps, inherited from the repaired observability
  continuation diagnostic
- future arrivals: included, as in the existing bounded branch rollouts
- simulator/policy semantics: unchanged

The stored branch-order symmetry/non-interference invariants are tested by the
existing counterfactual suites and by the pilot artifact tests.

## Feature Schema

All model features are numeric columns prefixed with `feat_`.

Groups:

- Global online state: queue length, active count, capacity/KV state, waiting
  age distribution, service distribution, laxity distribution, admission
  geometry, short causal history trends.
- ESTF-contested request: priority, prompt tokens, predicted output tokens,
  predicted service proxy, remaining predicted service proxy, queue age,
  unit-consistent laxity.
- WFS-contested request: same fields.
- Pairwise features: ESTF-minus-WFS differences and same-unit ratios for
  priority, service, age, and laxity.

Invalid-unit exclusion: `deadline_slack_if_admitted_now` is not a model
feature because it mixes real time and raw token/service proxy.

## Leakage Rules

Excluded from model features:

- scenario ID
- seed
- split
- `favlong` / `favshort`
- synthetic family labels
- actual output length
- branch outcomes
- `J_ESTF`, `J_WFS`, `delta_J`
- final label
- raw-completion reference labels
- TEST indicators

`analysis_fav`, utilization, skew, noise, and seed are preserved only for
analysis and grouping audits.

## Storage Layout

Dataset path:

`datasets/family_a_oracle_policy_pilot_v1/`

Files:

- `README.md`
- `pilot_rows.csv`
- `schema.json`
- `feature_classification.csv`
- `provenance.json`
- `quality_summary.json`
- `model_sanity_summary.json`
- `delta_j_regression_summary.json`
- `representation_check_summary.json`
- `manifest.json`

Builder:

`scripts/build_family_a_oracle_policy_pilot_v1.py`

## Pilot Gates

The pilot is suitable to scale only if:

- labels use weighted/SLO utility, not raw completion count;
- extraction is deterministic and branch-order safe;
- both ESTF and WFS are represented;
- ties are preserved rather than forced;
- grouped-by-scenario sanity models show nontrivial signal;
- no hidden metadata is exposed as a model feature.

V1 intentionally does not rebalance rows or tune tie thresholds.

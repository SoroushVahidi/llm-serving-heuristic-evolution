# Family-A Oracle Policy Pilot Dataset V1 Analysis

Date: 2026-08-21

Offline pilot dataset quality study for learning when ESTF vs WFS should be
used in Family-A scheduling states. No large dataset was generated, no TEST
data was used, no simulator/policy semantics were modified, no production
controller was trained or deployed, no hyperparameter sweep was run, and
nothing was staged/committed/pushed.

## Executive Verdict

Classification: **`PILOT_DATASET_READY_TO_SCALE`**

Recommended next-stage modeling family:
**utility-difference regression/ranking with gradient-boosted trees**, keeping
logistic regression as the calibration and interpretability baseline.

The pilot is scientifically good enough to justify generating a larger
training dataset. The label is not raw completion count: it is
priority-weighted SLO contribution for the two directly contested requests
under native ESTF vs native WFS bounded continuation. The dataset is small
(`91` rows), but it is not class-collapsed (`32` ESTF / `22` WFS / `37` exact
ties), has `32` independent scenario groups, has zero exact or rounded
near-duplicate feature rows, and grouped-by-scenario sanity models show clear
cross-scenario signal above majority baseline.

The main scale-up requirement is to store a stronger whole-branch
priority-weighted SLO utility alongside the contested-pair utility. V1 is a
defensible seed-pilot target because prior evidence shows every repaired
Family-A disagreement is a strict 1-vs-1 contested pair and raw-count labels
are biased; it should not be mistaken for final full-scenario ANWG
supervision.

## Preflight

| Check | Value |
|---|---|
| branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `8e1223beb58fd4d296061b6b48e3ba493714108f` |
| upstream | `origin/contextual-compositional-heuristics-20260731` |
| ahead/behind | 0 / 0 |
| worktrees | 1 (main only) |
| lock files | none found |
| active scientific jobs | none found; only unrelated `unattended-upgrade-shutdown`, user `uvicorn`, and `update-manager` |
| RAM | 24Gi free / 59Gi available of 62Gi |
| disk | 638G available of 835G (20% used) |
| load average | 0.22 / 0.15 / 0.07 |

`git status --short` had one pre-existing modified test file
(`tests/test_decision_criticality_timescale_trainval_v1.py`) and many
pre-existing untracked Family-A artifacts. All unrelated local changes were
preserved. This task added/updated only the files listed in the artifacts
section.

## Oracle Label

For each identical pre-decision state `s`:

```
J_ESTF(s) = sum_{i in contested pair}
    priority_i * 1[completed_i and SLO_safe_i under br_estf_estf]

J_WFS(s) = sum_{i in contested pair}
    priority_i * 1[completed_i and SLO_safe_i under br_wfs_wfs]

delta_J = J_ESTF - J_WFS
```

Labels:

- `ESTF` if `delta_J > 0`
- `WFS` if `delta_J < 0`
- `TIE_OR_UNCERTAIN` if `delta_J == 0`

No practical-equivalence threshold existed in the repository for this target,
so V1 uses exact deterministic ties only and stores the numerical utility
margin for future thresholding/abstention work.

## Bias Audit

The label avoids the prior raw-completion bias because a completed-but-late
request receives zero credit. It uses priority-weighted SLO-safe contribution,
matching the numerator semantics of `arrival_normalized_weighted_goodput`.

Candidate target audit:

| Candidate reference | Decision |
|---|---|
| final/scenario ANWG | Best final objective, but not available per captured 91 decision state without new expensive branch extraction |
| priority-weighted SLO goodput | Adopted at contested-pair branch level |
| common-continuation utility | Retained as diagnostic context; prior reports showed local/common-continuation signal is often zero |
| native continuation utility | Used, but with weighted/SLO contested-pair utility rather than raw branch completion |
| completion + SLO decomposition | Stored as auxiliary labels, not collapsed into the class target |

Known limitation: V1 labels the two directly contested requests, not every
collateral request in the bounded branch. Scale-up should add whole-branch
weighted/SLO utility.

## Dataset Summary

Dataset path: `datasets/family_a_oracle_policy_pilot_v1/`

| Quantity | Value |
|---|---:|
| rows | 91 |
| independent scenarios | 32 |
| features | 63 numeric `feat_*` columns |
| exact duplicate feature rows | 0 |
| rounded-3 near-duplicate feature rows | 0 |
| samples/scenario mean / median / max | 2.84 / 3 / 3 |
| median temporal distance between samples in same scenario | 767 steps |
| minimum temporal distance | 150 steps |

Label balance:

| Label | Count | Fraction | Scenarios represented |
|---|---:|---:|---:|
| ESTF | 32 | 35.2% | 20 |
| WFS | 22 | 24.2% | 17 |
| TIE_OR_UNCERTAIN | 37 | 40.7% | 24 |

By split:

| Split | ESTF | WFS | TIE_OR_UNCERTAIN |
|---|---:|---:|---:|
| train | 29 | 13 | 27 |
| val | 3 | 9 | 10 |

By analysis stratum only:

| Stratum | ESTF | WFS | TIE_OR_UNCERTAIN |
|---|---:|---:|---:|
| favlong | 16 | 22 | 22 |
| favshort | 16 | 0 | 15 |

The absence of WFS labels in `favshort` is an important scale-up sampling
warning, not a leakage feature. The model never receives `favshort/favlong`.

## Delta-J Margins

| Metric | Value |
|---|---:|
| mean `delta_J` | -0.352 |
| median | 0.000 |
| p25 / p75 | 0.000 / 1.000 |
| min / max | -10.000 / 10.000 |
| p90 / p95 of `abs(delta_J)` | 9.000 / 10.000 |
| exact ties | 37 / 91 = 40.7% |

By class, non-tie margins are not tiny: ESTF rows have median `+2.5`; WFS
rows have median `-5.0`. Ties are common but do not dominate
catastrophically.

## Feature Schema And Leakage

Feature groups:

- Global online state: queue length, active count, GPU/KV capacity, queue age
  distribution, service distribution, laxity distribution, admission geometry,
  short causal history.
- ESTF-contested request: priority, prompt tokens, predicted output tokens,
  predicted service proxy, remaining predicted service proxy, queue age,
  unit-consistent laxity.
- WFS-contested request: same fields.
- Pairwise difference/ratio features: priority, service, age, and laxity
  differences/ratios.

Leakage audit:

- `ONLINE_CAUSAL_MODEL_FEATURE`: 63 `feat_*` columns.
- `METADATA_ONLY`: `sample_id`, `scenario_id`, `split`, `step`, request IDs,
  class IDs, provenance.
- `EXPERIMENT_METADATA`: `analysis_fav`, utilization, skew, noise, seed.
- `LABEL_OR_FUTURE_OUTCOME`: `J_ESTF`, `J_WFS`, `delta_J`, `oracle_label`,
  completion/SLO diagnostic labels, raw-completion reference.
- `INVALID_UNIT`: none included as model features.

Explicitly excluded from model features: scenario ID, seed, split,
favlong/favshort, synthetic family label, actual output length, branch
outcomes, `delta_J`, final label, TEST indicators, and
`deadline_slack_if_admitted_now`.

## Integrity

Tests run:

```
python3 -m pytest -q \
  tests/test_family_a_oracle_policy_pilot_v1.py \
  tests/test_family_a_observability_continuation_v1.py \
  tests/test_family_a_receding_horizon_oracle_v1.py
```

Result: `42 passed in 416.54s`.

Covered invariants:

- deterministic row/schema reproduction
- unique sample IDs
- no TEST rows
- no forbidden model features
- dimensional/unit validity for feature schema
- label/`delta_J` consistency
- group split integrity
- no row leakage across grouped folds
- stored branch-order symmetry for contested request outcomes
- no snapshot/restore interference via existing counterfactual suites

## Grouping

Split/fold key: `scenario_id`.

No random row split is used. All samples from one scenario stay in one fold.
The pilot uses 5-fold `GroupKFold` for sanity models. Multiple seeds of the
same parameter configuration still sit in separate groups in V1; scale-up
should consider a stricter configuration-level group key for final
generalization estimates.

## Classification Sanity Models

Binary sanity training excludes the 37 `TIE_OR_UNCERTAIN` rows but preserves
them in the dataset. No hyperparameter sweep was run.

| Model | BA mean±std | Macro F1 | ROC-AUC | PR-AUC(WFS) | Brier | Confusion `[ESTF,WFS]` |
|---|---:|---:|---:|---:|---:|---|
| majority | 0.500±0.000 | 0.368 | 0.500 | 0.409 | 0.409 | `[[32,0],[22,0]]` |
| logistic regression | **0.855±0.095** | **0.856** | 0.912 | 0.895 | 0.116 | `[[27,5],[2,20]]` |
| shallow tree depth 3 | 0.831±0.089 | 0.833 | 0.851 | 0.755 | 0.138 | `[[26,6],[2,20]]` |
| random forest modest | 0.834±0.088 | 0.835 | 0.925 | 0.914 | **0.095** | `[[27,5],[3,19]]` |
| XGBoost modest | 0.834±0.088 | 0.835 | **0.972** | **0.964** | 0.098 | `[[27,5],[3,19]]` |

Calibration summary for logistic: mean `P(WFS)` = 0.183 on true ESTF rows and
0.829 on true WFS rows. This is already useful for future abstention research,
though no calibration model was fit.

## Delta-J Regression

| Model | MAE mean±std | OOF Spearman | Mean fold Spearman | Non-tie sign accuracy |
|---|---:|---:|---:|---:|
| ridge regression | 3.113±1.042 | 0.508 | 0.499 | 0.778 |
| XGBoost regressor | **2.678±0.665** | **0.600** | **0.588** | **0.833** |

Regression is informative and should be prioritized at scale because it keeps
utility margin information and can naturally support abstention near zero.

## Pairwise Representation

| Representation | Classification BA | Regression OOF Spearman |
|---|---:|---:|
| concatenated global + ESTF/WFS + pairwise features | **0.855** | **0.508** |
| signed pairwise-difference/ratio only | 0.782 | 0.496 |

Pairwise features alone carry signal, but the concatenated representation is
better for classification and slightly better for regression in this pilot.

## Scenario-Level Generalization

Performance is grouped by held-out scenarios, not random rows. Fold BA ranges
from 0.708 to 1.000 for logistic, so the signal is real but still small-n.

Coverage:

| Parameter | Values represented |
|---|---|
| utilization | 1.1, 1.3, 1.5 |
| skew | 5.0, 10.0 |
| noise | 0.0, 0.3 |
| fav analysis stratum | favlong, favshort |

Label distribution by hidden analysis strata shows structure rather than pure
memorization: WFS appears only in `favlong`, but ESTF and tie labels appear in
both `favlong` and `favshort`; features exclude the stratum. A larger dataset
should enforce configuration-level grouped validation to separate state
learning from parameter-family memorization.

## Scale Cost Estimate

The pilot builder itself is cheap because it reuses existing oracle artifacts
(`~3s`, dataset size 192K). New labels at scale require rerunning/expanding the
bounded branch extraction. The closest measured cost is the repaired
observability continuation run: 12,303s for 91 events, or about 135s per label
on this machine, including scenario traversal.

| Labels | Estimated wall time / CPU time, single process | Disk for tabular outputs |
|---|---:|---:|
| 1,000 | ~37.6 hours | ~2 MB for pilot-style CSV/JSON; logs/branch traces extra |
| 5,000 | ~188 hours | ~10 MB for pilot-style CSV/JSON; logs/branch traces extra |
| 10,000 | ~376 hours | ~20 MB for pilot-style CSV/JSON; logs/branch traces extra |

Parallelism should be safe across scenarios because forks are isolated and
scenario groups are independent. Deduplicating adjacent/near-identical states
and capping samples per scenario can reduce cost substantially.

## Artifacts

- Design: `docs/design/FAMILY_A_ORACLE_LABELED_PILOT_DATASET_V1.md`
- Builder: `scripts/build_family_a_oracle_policy_pilot_v1.py`
- Dataset/data card: `datasets/family_a_oracle_policy_pilot_v1/README.md`
- Rows: `datasets/family_a_oracle_policy_pilot_v1/pilot_rows.csv`
- Schema: `datasets/family_a_oracle_policy_pilot_v1/schema.json`
- Feature audit: `datasets/family_a_oracle_policy_pilot_v1/feature_classification.csv`
- Quality summary: `datasets/family_a_oracle_policy_pilot_v1/quality_summary.json`
- Classification summary: `datasets/family_a_oracle_policy_pilot_v1/model_sanity_summary.json`
- Regression summary: `datasets/family_a_oracle_policy_pilot_v1/delta_j_regression_summary.json`
- Representation summary: `datasets/family_a_oracle_policy_pilot_v1/representation_check_summary.json`
- Tests: `tests/test_family_a_oracle_policy_pilot_v1.py`

## Confirmation

- pilot only
- no large dataset generation
- no TEST
- no controller deployment
- no neural-network training
- no hyperparameter sweep
- no simulator/policy semantic modification
- no commit/push/stage/stash/reset/clean

## Final Answer

Yes. The pilot oracle-labeled ESTF/WFS dataset is scientifically trustworthy
enough for a scale-up decision, sufficiently balanced for a seed pilot, and
learnable under grouped scenario validation. The next dataset should scale the
same state/feature protocol while adding whole-branch priority-weighted SLO
utility to strengthen the oracle target beyond contested-pair supervision.

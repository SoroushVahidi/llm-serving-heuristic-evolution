# Joint-240 Strong Learned Selector v1 — Design / Preregistration

**Status:** PREREGISTERED before held-out evaluation  
**Date:** 2026-08-25  
**Experiment dir:** `experiments/joint240_strong_learned_selector_v1/`  
**Parent (unmodified):** `experiments/joint240_same_distribution_adaptive_exploitability_v1/`  
**Workload matrix (unmodified):** `experiments/joint_multimechanism_generalization_v1/`

## 1. Scientific question

Does a stronger nonlinear, cost-sensitive learned selector recover materially more
of the joint-240 SBS→VBS headroom than the existing logistic `A_scen` selector?

This is **not** a test of all adaptive schedulers. It is a stronger
**learned portfolio-selection** baseline under the exact same-distribution
out-of-fold protocol used for `A_scen` / `A_live`.

## 2. Scope and non-goals

- Action portfolio remains **P6 only**.
- Same 240 scenarios, same frozen utility matrix, same outer 5 folds
  (`split_oof_folds.csv`, seed `20260825`).
- Do **not** overwrite parent `A_scen` / `A_live` artifacts.
- Do **not** edit the manuscript in this experiment.
- Do **not** claim this reproduces Fu et al. Learning-to-Rank (LTR learns
  request-level output-length ranking for SJF-like scheduling; this experiment
  learns scenario-level policy utility for portfolio selection).
- Do **not** retrain `A_live` or re-run terminal-criticality / vLLM studies.

## 3. Primary formulation (frozen)

**Direct utility prediction**, not 6-way winner classification.

For each scenario \(x\) and policy \(p \in P_6\), learn

\[
f(x,p) \approx \mathrm{ANWG}(x,p)
\]

using TRAIN folds only. At test time

\[
\hat p(x) = \arg\max_{p \in P_6} f(x,p)
\]

and evaluate using the **true frozen** matrix entry \(\mathrm{ANWG}(x,\hat p(x))\).

### Why pooled utility regression (not one-model-per-policy)

**Primary architecture:** one pooled regressor over scenario features plus a
one-hot policy indicator.

Rationale (frozen before evaluation):
- shares statistical strength across policies (\(n_{\mathrm{train}}\times 6\) rows);
- directly optimizes the portfolio utility surface rather than winner identity;
- matches the cost-sensitive nature of mistakes (large ANWG gaps matter more
  than classification accuracy).

## 4. Features (frozen)

Exactly the parent `A_scen` allowlist (17 generator parameters available at
scenario-selection time):

`offered_load`, `burstiness`, `long_fraction`, `prompt_scale`,
`prompt_heterogeneity_sigma`, `output_scale`, `output_heterogeneity_sigma`,
`tenant_weight_skew`, `class_share_skew`, `slo_tightness`,
`prediction_noise_sigma`, `kv_pressure_target`, `late_pressure`, `late_phase`,
`max_active_sequences`, `step_token_budget`, `n_requests`.

**Forbidden:** test policy outcomes, VBS identity, SBS labels at inference,
realized future output lengths, post-policy metrics, held-out utilities used
for hyperparameter selection.

## 5. Model families (maximum two)

1. **Primary:** `sklearn.ensemble.HistGradientBoostingRegressor`
2. **Secondary (robustness):** `sklearn.ensemble.ExtraTreesRegressor`

No new ML frameworks. No XGBoost/LightGBM install.

## 6. Outer folds (frozen)

Use the authoritative parent fold file:

`experiments/joint240_same_distribution_adaptive_exploitability_v1/split_oof_folds.csv`

- `SPLIT_SEED = 20260825`
- `N_FOLDS = 5`
- every scenario in exactly one outer test fold

## 7. Nested hyperparameter selection (frozen)

For each outer fold \(k\):

1. Hold out fold \(k\) entirely.
2. Among the remaining four folds, select hyperparameters by
   **leave-one-train-fold-out** inner CV:
   - for each candidate HP and each of the four train folds as inner val,
     fit on the other three, score mean selected-policy ANWG on the inner val
     (matrix lookup of \(\arg\max_p f(x,p)\)).
3. Choose the HP maximizing mean inner-val ANWG.
4. Refit on all four train folds with the chosen HP.
5. Predict on outer test fold \(k\).

**Never** use outer-test ANWG to choose hyperparameters, features, model
family, calibration, or thresholds.

### Primary HP grid (`HistGradientBoostingRegressor`)

| Parameter | Candidates |
|---|---|
| `learning_rate` | `{0.05, 0.1}` |
| `max_iter` | `{150}` (fixed) |
| `max_leaf_nodes` | `{15, 31}` |
| `min_samples_leaf` | `{10}` (fixed) |
| `l2_regularization` | `{0.0, 1.0}` |
| `random_state` | `20260825` |

Total: **8** candidates.

### Secondary HP grid (`ExtraTreesRegressor`)

| Parameter | Candidates |
|---|---|
| `n_estimators` | `{200}` (fixed) |
| `max_depth` | `{None, 8}` |
| `min_samples_leaf` | `{2, 5}` |
| `random_state` | `20260825` |
| `n_jobs` | `-1` |

Total: **4** candidates.

## 8. Comparators (frozen)

Report alongside:
- SBS (`kv_constrained_online` fixed)
- VBS
- Majority (parent OOF majority policy)
- existing frozen `A_scen`
- existing frozen `A_live`
- new primary strong selector (`A_hgb`)
- secondary strong selector (`A_et`)

Parent `A_scen` / `A_live` predictions are **read from**
`per_scenario_oof_results.csv` and **not** retrained.

## 9. Primary metrics (frozen)

For each method, on the 240 OOF scenarios:
1. mean terminal ANWG
2. Gain vs SBS: \(R_A - R_{\mathrm{SBS}}\)
3. GapClosure: \((R_A - R_{\mathrm{SBS}}) / (R_{\mathrm{VBS}} - R_{\mathrm{SBS}})\)
4. residual regret to VBS
5. catastrophic regressions: \(R_A < R_{\mathrm{SBS}} - 0.01\)
6. selected-policy accuracy vs VBS winner
7. mean regret when selection is wrong

Also report **fold-level** mean ANWG for each method.

## 10. Statistical inference (frozen)

- Paired scenario-level bootstrap on the 240 OOF predictions
- `B = 10_000`
- `BOOTSTRAP_SEED = 20260826`
- 95% percentile CIs for:
  - Gain vs SBS for `A_hgb` (and `A_et`)
  - paired `R_hgb - R_A_scen`
  - paired `R_hgb - R_A_live`
  - GapClosure for `A_hgb` (optional but reported)

## 11. Pre-specified interpretation criteria (frozen)

**PRIMARY — vs SBS**

- `STRONG_RECOVERY`: lower 95% CI of Gain vs SBS \(> 0\) **and** GapClosure \(\ge 0.50\)
- `PARTIAL_RECOVERY`: mean Gain vs SBS \(> 0\), but not `STRONG_RECOVERY`
- `NO_RECOVERY`: mean Gain vs SBS \(\le 0\)

**SECONDARY — vs A_scen**

- `CLEAR_IMPROVEMENT_OVER_A_SCEN`: lower 95% CI of paired
  \((R_{\mathrm{hgb}} - R_{A_{\mathrm{scen}}}) > 0\)

Labels are interpretive only; results are reported regardless.

## 12. Failure analysis (frozen checklist)

Regardless of outcome, compute:
1. winner-selection confusion matrix (`A_hgb` vs VBS identity)
2. regret distribution
3. performance by `n_elevated_mechanisms`
4. performance by VBS-gain magnitude tertiles
5. performance by VBS winner identity
6. catastrophic-regression scenarios

Any redesign based on these results is **out of scope** for this experiment
and must be labeled post-hoc if performed later.

## 13. Seeds and provenance

| Name | Value |
|---|---|
| Outer fold / parent split seed | `20260825` |
| Model `random_state` | `20260825` |
| Bootstrap seed | `20260826` |
| Bootstrap replicates | `10000` |
| Catastrophic ε | `0.01` |

## 14. Leakage safeguards

- Features: allowlist only
- Labels for fitting: TRAIN-fold matrix ANWG only
- Hyperparameters: inner CV on TRAIN folds only
- Test evaluation: frozen matrix lookup after prediction
- Parent `A_scen`/`A_live` used as frozen comparators only

## 15. Manuscript decision rule (for the final report; no edit now)

- `PROMOTE`: materially strengthens the paper (clear recovery or clear
  improvement that changes the narrative)
- `REPORT_NEGATIVE`: valid experiment; opportunity still not recovered
- `INVALID`: leakage / reproduction failure

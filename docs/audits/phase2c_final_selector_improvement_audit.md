# Phase 2C Final Selector Improvement Audit

**Date:** 2026-07-20/21  
**Branch:** `phase2c-final-selector-improvement`  
**Starting commit:** `8087ee1be3af2ef51e072ff37fb18617338ad10a`  
**Primary objective:** `arrival_normalized_weighted_goodput` (ANWG)  
**Primary runner:** `scripts/run_phase2c_final_selector_improvement.py`  
**Primary output:** `results/phase2c_final_selector_improvement/final_noaug_strict_20260720/` (gitignored)  

## Scope

This phase evaluated whether stronger causal selectors can close the remaining
gap to the per-window oracle/envelope over the existing deployable policy
portfolio. It did not generate new scheduling heuristics.

The evaluation uses only causal `feat_*` columns at prediction time. Training
uses simulator-derived policy outcome vectors to form labels/regret weights,
but reward, completion, selected-policy, oracle, and external-envelope columns
are rejected as features.

The newer Selector v2 calibrated targeted pilot is not used for final claims:
its non-OOD splits have confirmed row-range leakage, and OOD_TEST loses to
best fixed. Real-vLLM selector experiments are also not used as selector
superiority evidence because the selector arm is confounded by load shedding.

## Repository-State Findings

- Corrected system objective is ANWG, not old completed-request-only
  `weighted_goodput`.
- Phase 2B.16 synthetic fresh validation still has `regression_anwg` as the
  strongest corrected-objective selector: ANWG 0.9856 vs best fixed
  EDF/Orca/SLO-slack 0.9776 and oracle 0.9879.
- Phase 2C real-trace evaluation is a different domain. Prior best selector
  was Phase 2C.3 `native_non_oracle_dt`, ANWG 0.8063, while best fixed
  SCORPIO is 0.7963 and the all-policy envelope is 0.8298.
- Phase 2C train/val has **zero** `is_azure_conv_like` rows; all 135
  azure-conv-like rows are in eval. This makes learned Azure-conv gating
  impossible without additional non-eval training data.
- Orca beats SCORPIO pairwise in the labeled dataset, but in the final eval
  all-policy envelope is still usually achieved by SCORPIO, admission_control,
  or EDF; Orca is the oracle policy for only 2/325 final eval windows.

## Methods Implemented

New reusable selector logic is in `src/llmserveopt/selector/advanced.py`:

- per-policy reward regression with Random Forest, Extra Trees, and
  HistGradientBoosting regressors;
- regret/margin-weighted classification and reward regression;
- pairwise policy ranking over core pairs including Orca vs SCORPIO,
  SCORPIO vs admission_control, and SCORPIO vs WSP;
- prediction-margin uncertainty fallback to fixed policies;
- feature-only Azure-conv-like regime gate support.

New tests are in `tests/test_advanced_selector_models.py`.

## Experiments Run

Primary bounded final run:

```bash
python3 scripts/run_phase2c_final_selector_improvement.py \
  --skip-targeted-augmentation \
  --bootstrap 2000 \
  --out-dir results/phase2c_final_selector_improvement/final_noaug_strict_20260720
```

Runtime: 40.1s.

Targeted augmentation ablation:

```bash
python3 scripts/run_phase2c_final_selector_improvement.py \
  --bootstrap 500 \
  --out-dir results/phase2c_final_selector_improvement/final_aug_ablation_20260720
```

Runtime: 176.1s. Generated 50 train, 18 validation, and 52 fresh targeted
synthetic windows; 36/14/40 respectively were Azure-conv-like. This ablation
did not beat the primary no-augmentation selector on real eval.

## Main Final Evaluation

Final real-trace eval: 325 windows, all-non-oracle policy pool.

| Selector/policy | ANWG | CF | completed quality | gain vs best fixed | gap to oracle | gap closed |
|---|---:|---:|---:|---:|---:|---:|
| prior Phase 2C.3 DT | 0.8063 | 0.8126 | 0.9921 | +0.0100 | 0.0235 | 0.298 |
| `new_extra_reward_regression_weighted` (diagnostic, not validation-selected) | 0.8071 | 0.8121 | 0.9934 | +0.0109 | 0.0226 | 0.324 |
| `new_extra_reward_regression` (diagnostic, not validation-selected) | 0.8065 | 0.8121 | 0.9928 | +0.0103 | 0.0232 | 0.307 |
| `new_hgb_reward_regression` | 0.8055 | 0.8149 | 0.9890 | +0.0093 | 0.0242 | 0.277 |
| prior `dt_anwg` | 0.8021 | 0.8297 | 0.9708 | +0.0058 | 0.0276 | 0.174 |
| prior `rf_anwg` | 0.7995 | 0.8157 | 0.9821 | +0.0032 | 0.0303 | 0.096 |
| prior `knn_anwg` | 0.7977 | 0.8027 | 0.9934 | +0.0014 | 0.0321 | 0.043 |
| prior `regression_anwg` | 0.7974 | 0.8031 | 0.9927 | +0.0011 | 0.0324 | 0.033 |
| fixed SCORPIO | 0.7963 | 0.8009 | 0.9938 | baseline | 0.0335 | 0.000 |
| `new_pairwise_core_rf` | 0.7963 | 0.8009 | 0.9938 | +0.0000 | 0.0335 | 0.000 |
| validation-selected `new_rf_reward_regression` | 0.7926 | 0.8159 | 0.9750 | -0.0037 | 0.0372 | -0.110 |
| fixed WSP | 0.7061 | 1.0000 | 0.7061 | - | - | - |
| fixed multi-bin | 0.7034 | 1.0000 | 0.7034 | - | - | - |
| fixed ESTF | 0.6680 | 1.0000 | 0.6680 | - | - | - |
| fixed Orca | 0.6046 | 1.0000 | 0.6046 | - | - | - |
| fixed EDF | 0.6040 | 1.0000 | 0.6040 | - | - | - |

Best fixed policy: `scorpio_style_slo_guard`, ANWG 0.7963.  
All-policy oracle/envelope: 0.8298.  
Strict best selector: prior Phase 2C.3 `native_non_oracle_dt`, ANWG 0.8063.
The validation-selected new selector for the all-non-oracle pool was
`new_rf_reward_regression`; it failed to generalize (ANWG 0.7926). The
highest final-eval exploratory model,
`new_extra_reward_regression_weighted`, reached ANWG 0.8071 but was not
validation-selected and must be treated as diagnostic, not as the frozen
final selector.

Bootstrap CI for the best selector:

- mean ANWG CI: [0.7955, 0.8174]
- gap vs best fixed CI: [-0.0008, 0.0211]
- mean oracle-regret CI: [0.0183, 0.0291]

## Subgroups

For prior Phase 2C.3 `native_non_oracle_dt`:

| Subgroup | n | ANWG | best fixed | gap to oracle | gap closed | within 0.005 oracle |
|---|---:|---:|---:|---:|---:|---:|
| all | 325 | 0.8063 | 0.7963 | 0.0235 | 0.298 | 0.788 |
| Azure-derived | 141 | 0.8688 | 0.8447 | 0.0442 | 0.352 | 0.638 |
| BurstGPT-derived | 184 | 0.7584 | 0.7591 | 0.0076 | -0.112 | 0.902 |
| azure_2023_conv | 97 | 0.8092 | 0.8085 | 0.0642 | 0.011 | 0.474 |
| azure_conv_like | 135 | 0.8713 | 0.8474 | 0.0450 | 0.346 | 0.637 |
| external-loss analysis subset | 110 | 0.7959 | 0.7966 | 0.0531 | -0.012 | 0.536 |

The overall improvement hides weak Azure-conv performance. On
`azure_2023_conv`, the selector closes only 1.1% of the fixed-to-oracle gap.

## Regret Analysis

The final selector has 70 windows with positive regret to the all-policy
oracle/envelope.

- mean regret on failing windows: 0.1091
- p95 failing-window regret: 0.1871
- worst regret: 0.1950
- total regret: 7.6352 ANWG-window points

Regret by workload:

| Workload | failures | regret sum | mean | max |
|---|---:|---:|---:|---:|
| azure_2023_conv | 51 | 6.2292 | 0.1221 | 0.1950 |
| burstgpt_moderate_noise070 | 6 | 0.4447 | 0.0741 | 0.1500 |
| burstgpt_scaled_moderate | 5 | 0.4000 | 0.0800 | 0.1367 |
| burstgpt_scaled_high | 4 | 0.3145 | 0.0786 | 0.1316 |
| burstgpt_moderate_exact_prediction | 4 | 0.2469 | 0.0617 | 0.0750 |

Regret by oracle-best policy:

| Oracle-best policy | failures | regret sum | mean | max |
|---|---:|---:|---:|---:|
| admission_control | 32 | 3.9716 | 0.1241 | 0.1950 |
| edf | 36 | 3.5073 | 0.0974 | 0.1889 |
| orca_style | 2 | 0.1563 | 0.0781 | 0.1500 |

Top failures are almost all Azure-conv windows where the selector keeps
choosing SCORPIO at ANWG around 0.80-0.86 while admission_control or EDF
achieves near 1.0. These windows have causal features matching the known
failure regime: mean prompt around 1.1k-1.3k tokens and tight-SLO fraction
around 0.50.

Important nuance: many high-regret failures have oracle top-two margin 0
because admission_control and EDF tie at/near 1.0. They are near-ties among
the best policies, but not harmless selector mistakes because SCORPIO is far
below both.

## External/Full Envelope

On this dataset, the native all-non-oracle envelope and all-policy envelope
are identical at 0.8298. The external-style envelope is 0.8297, slightly
lower. No final eval window has external-style envelope greater than native
envelope; the remaining gap is selector-to-envelope, not missing external
policy actions in the candidate pool.

## Interpretation

Conventional model changes are not enough. Under strict validation-based
selection, the new all-policy selector did not beat the Phase 2C.3 prior
baseline and even lost to best fixed on final eval. The best final-eval
exploratory model (`new_extra_reward_regression_weighted`) was slightly
above the prior baseline, but it was not validation-selected and is not a
valid frozen-selector claim. Pairwise ranking did not recover the gap; it
collapsed to SCORPIO on this split. Prediction-margin fallback variants also
did not materially improve the strict result.

The dominant remaining failure is not Orca-vs-SCORPIO routing. In the final
regret table, only 2 failing windows have Orca as the oracle-best policy.
Most recoverable regret requires routing Azure-conv-like windows from
SCORPIO to admission_control or EDF.

The evidence points to training-data/formulation shift rather than proven
causal feature insufficiency:

- causal features clearly identify the failure regime (`is_azure_conv_like`);
- train/val has zero such rows in the original Phase 2C dataset;
- simple targeted synthetic augmentation created Azure-conv-like rows but did
  not improve real eval, suggesting the synthetic generator does not yet
  reproduce the real Azure-conv policy frontier.

## Status

`SELECTOR_STATUS = IMPROVABLE`

`BEST_SELECTOR_NAME = prior_phase2c3_native_non_oracle_dt`  
`BEST_SELECTOR_ANWG = 0.806265`  
`BEST_FIXED_POLICY = scorpio_style_slo_guard`  
`BEST_FIXED_ANWG = 0.796270`  
`ORACLE_ENVELOPE_ANWG = 0.829758`  
`GAP_CLOSED_FRACTION = 0.298456`  
`MEAN_ORACLE_REGRET = 0.023493`  
`MEANINGFUL_WINDOW_COUNT = 214` using `margin_best_all_non_oracle >= 0.005`  
`MAIN_REMAINING_FAILURE_MODE = azure_2023_conv long-prompt mixed-tight-SLO windows; selector chooses SCORPIO while admission_control/EDF tie near oracle`

Recommendation: do not spend more effort on generic tabular model swaps.
One focused selector-data experiment is justified before moving to new
heuristic generation: construct leakage-safe, real-trace-derived or
calibrated synthetic Azure-conv-like train/validation windows whose policy
frontier matches the final eval failure windows, then train a regime-aware
selector specifically for SCORPIO-vs-admission_control/EDF. If that does not
close the Azure-conv gap, selector research over the current portfolio should
be considered saturated and the project should move to verifier-constrained
generation of new scheduling heuristics.

## Tests

```bash
python3 -m pytest -q \
  tests/test_advanced_selector_models.py \
  tests/test_selector_models.py \
  tests/test_phase2c_labeled_selector_dataset.py \
  tests/test_phase2c3_external_aware_orca_recovery.py
```

Result: 66 passed in 4.02s.

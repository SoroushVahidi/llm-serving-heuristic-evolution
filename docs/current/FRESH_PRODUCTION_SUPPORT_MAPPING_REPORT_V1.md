# FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT

This report records the support-only stage of `FRESH_PRODUCTION_LATENCY_HEADROOM_CONFIRMATORY_V1`. It freezes fresh action-opportunity support and the later causal-labeling universe. No latency continuation, ANWG continuation, post-hoc Phase-D latency analysis, selector, or model was run.

## 1. PREFLIGHT

- Starting frozen design commit: `b4e6c603d0e4d8bbd4ad814341be97f361864a0b`.
- Support harness source commit: `3a2487ca62369addae5f4eaa9e856516506255e1`.
- Execution environment: `/home/soroush/modal-venv`, Python 3.12.3, NumPy 2.3.5, pandas 3.0.2, scikit-learn 1.8.0, pyarrow 24.0.0, pytest 9.0.3.
- Frozen design artifacts were hash-checked before execution.
- The Phase-A/B/D history and the frozen latency protocol were not modified.

## 2. FRESH_WINDOW_INTEGRITY

The run used exactly 60 fresh faithful windows, 20 each from `azure_2023_code`, `azure_2023_conv`, and `burstgpt`, with 12,000 request instances total. The frozen source-record overlap proof reports `0` overlapping request identities. No old Phase-A/B/D request was substituted.

## 3. RUN_COMPLETENESS

The exact matrix was `60 windows x (5 arrival + 7 KV + 6 active) = 1,080` conditions.

- Completed: 1,080
- Valid: 1,059
- Invalid: 21
- Failed: 0
- Missing: 0
- Duplicates: 0

The 21 invalid conditions are `INVALID_HORIZON_TRUNCATED`; they are retained in the support table and excluded from causal-regime selection. Valid conditions comprise 951 `VALID_UNCONSTRAINED`, 38 `VALID_PARTIALLY_CONSTRAINED`, and 70 `VALID_STRONGLY_CONSTRAINED` window-conditions.

## 4. PRESSURE_RESULTS

The complete per-window and workload-axis tables are frozen in `FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv` and `FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv`. The compact support map is:

| Workload | Axis | First valid disagreement | Fresh result |
|---|---|---|---|
| Azure code | arrival | none | zero across `0.5x` through `8x` |
| Azure code | KV | `kv_16000` | 431 states, rate 0.004561 |
| Azure code | active cap | `active_8` | 10 states, then 122 at `active_4` |
| Azure conversation | arrival | none | zero across `0.5x` through `8x` |
| Azure conversation | KV | `kv_16000` | 93 states, rate 0.000157 |
| Azure conversation | active cap | `active_4` | 64 states, rate 0.000108 |
| BurstGPT | arrival | none | zero across `0.5x` through `8x` |
| BurstGPT | KV | none | zero at every valid pressure point; `kv_8000` is invalid/truncated |
| BurstGPT | active cap | none | zero at every valid pressure point |

The frozen CSV retains achieved active/KV pressure, queue telemetry, binding counts, completion fraction, unfinished requests, and simulation end condition for every condition. No invalid condition is interpreted as scheduler opportunity.

## 5. ACTION_COLLAPSE

The scanner retained no-choice, policy-ranking difference with canonical collapse, and true canonical disagreement fields for every condition. The causal population uses only states with a unique non-SBS canonical executable action; ranking differences that collapse to SBS are not included.

## 6. ORIGINAL_VS_FRESH_SUPPORT_TRANSFER

The resource-pressure transition reproduces for Azure code and Azure conversation: Azure code first disagrees at `kv_16000` and `active_8`, while Azure conversation first disagrees at `kv_16000` and `active_4`. Fresh Azure code rates are close to the original Phase-B rates. Fresh Azure conversation retains the same onset points but has a larger `kv_16000` rate than the original sampled windows. BurstGPT differs materially: its original Phase-B low-capacity support does not reproduce in the selected fresh windows, which remain null at all valid points. Arrival scaling remains structurally null for all three workloads.

This is support-transfer evidence only. It contains no latency effect information.

## 7. MECHANICAL_REGIME_SELECTION

The frozen rule was applied using only validity, pressure order, disagreement support, and `windows_with_disagreement >= 2` for sustained support. Arrival axes and resource axes without valid disagreement support were excluded from the causal universe.

| Workload | Axis | Condition | Roles |
|---|---|---|---|
| Azure code | active cap | `active_8` | onset, sustained |
| Azure code | active cap | `active_4` | strongest-valid |
| Azure code | KV | `kv_16000` | onset, sustained, strongest-valid |
| Azure conversation | active cap | `active_4` | onset, sustained, strongest-valid |
| Azure conversation | KV | `kv_16000` | onset, strongest-valid |

No BurstGPT causal regime was manufactured. All regime rows explicitly record `latency_outcomes_used_for_regime_selection = false`.

## 8. FRESH_CAUSAL_UNIVERSE

The selected fresh manifest contains:

- 720 unique canonical disagreement states;
- 831 unique non-SBS state-action branches;
- 720 SBS reference branches;
- 1,551 planned terminal continuations;
- exhaustive labeling scope under the frozen `<= 50,000` continuation rule.

State counts are Azure code active 132 and KV 431; Azure conversation active 64 and KV 93. There are no duplicate state IDs or duplicate state/action branches. The universe is outcome-blind and includes live-state fingerprints, pressure telemetry, source-window identity, SBS action hashes, alternative action hashes, and P6 policy mappings.

## 9. LABELING_SCOPE

`FRESH_LATENCY_CAUSAL_CONFIRMATION = FEASIBLE_EXHAUSTIVE`, but causal labeling was not executed in this task. The final population is frozen for a later task.

## 10. LATENCY_BLINDNESS_CHECK

The support harness contains no continuation latency, ANWG, or causal advantage computation. Selection was run from the support table only. Existing Phase-D latency signs/magnitudes and the old post-hoc D2 table were not accessed. `PHASE_D_V1_POST_HOC_LATENCY_REANALYSIS_EXECUTED = NO` remains true.

## 11. FGCS_IMPLICATION

The fresh evidence strengthens pressure-to-action-opportunity reproducibility for Azure code and Azure conversation, while showing that support is workload/window dependent and did not transfer to the selected BurstGPT windows. It establishes feasibility for an exhaustive fresh latency-headroom confirmation, but provides no causal-headroom credit yet. Readiness remains 78/100 with 68% contribution-strength confidence, and the causal-headroom gate remains unresolved because the fresh latency objective has not been labeled.

## 12. RESULT_FREEZE

The following compact products are committed together with this report. Hashes are recorded after final artifact generation:

- `FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv`: see final commit hash manifest.
- `FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv`: see final commit hash manifest.
- `FRESH_SUPPORT_TRANSITIONS_V1.csv`: see final commit hash manifest.
- `FRESH_SUPPORT_RESULT_V1.json`: see final commit hash manifest.
- `FRESH_REGIME_SELECTION_V1.json`: see final commit hash manifest.
- `FRESH_ELIGIBLE_DISAGREEMENT_UNIVERSE_V1.json`: see final commit hash manifest.
- `FRESH_CAUSAL_LABELING_PLAN_V1.json`: see final commit hash manifest.

## 13. NEXT_TASK

Execute the frozen exhaustive fresh causal-latency campaign over the 720-state, 831-branch population, preserving the frozen one-step SBS-relative intervention and mean-latency endpoint. Do not inspect or use existing Phase-D latency outcomes to alter the population.

FRESH_SUPPORT_MAPPING_COMPLETE = YES
EXPECTED_SUPPORT_CONDITIONS = 1080
ALL_PREREGISTERED_SUPPORT_CONDITIONS_ACCOUNTED_FOR = YES
FRESH_REQUEST_OVERLAP_WITH_PHASE_A_B_D = 0
LATENCY_OUTCOMES_USED_FOR_REGIME_SELECTION = NO
PHASE_D_V1_POST_HOC_LATENCY_REANALYSIS_EXECUTED = NO
PHASE_D_V1_LATENCY_SIGNS_MAGNITUDES_ACCESSED_AFTER_AUDIT = NO
FRESH_CAUSAL_UNIVERSE_FROZEN = YES
FRESH_LATENCY_CAUSAL_CONFIRMATION = FEASIBLE_EXHAUSTIVE
CAUSAL_HEADROOM_GATE = UNRESOLVED_METRIC_SATURATION
FGCS_CONTRIBUTION_READINESS_SCORE = 78/100
FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 68%
READY_FOR_FRESH_LATENCY_CAUSAL_EXECUTION = YES

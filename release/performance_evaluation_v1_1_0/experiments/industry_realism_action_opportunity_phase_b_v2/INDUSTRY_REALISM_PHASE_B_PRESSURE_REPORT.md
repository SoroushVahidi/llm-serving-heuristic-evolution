# INDUSTRY_REALISM_PHASE_B_PRESSURE_REPORT

## 1. PREFLIGHT_AND_FREEZE

- Starting/preregistration execution HEAD: `4caed7fee92e052ec1de13d983ea3f8c03b523f1`
- Phase-A canonical result commit: `416e1403a5111ac0a421785b2c4a28ef2a485937`
- Historical Phase-B V1 status: `PHASE_B_V1_GRID = SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY`
- Historical Phase-B V1 grid SHA-256: `72eb3e7d6915e93c237cd721baec32e526d91c6ed4689da5fdc5b64e42965711`
- Phase-B V2 preregistration SHA-256: `13a0e927c4cc9f399aae43d7a0527e9c89d4d1e6b0dcff115f2ddba80b0b7782`
- Phase-B V2 grid SHA-256: `7beb6672183c4ce818a277be8a8085fb0e6387f0978b6427259a7cebad944191`
- Phase-B V1 outcomes: none; Phase-B V1 was not executed.

Final frozen grid:

- Arrival multipliers: `0.5, 1.0, 2.0, 4.0, 8.0`
- KV capacities: `8000000, 240000, 120000, 60000, 32000, 16000, 8000`
- Active-sequence caps: `512, 64, 32, 16, 8, 4`

## 2. PRESSURE_GRID_RATIONALE

The active-sequence grid retains native 512 and crosses the Phase-A observed maxima of 9, 7, and 31. The KV grid is absolute and calibrated to the Phase-A peak observed occupancy of about 30k KV tokens, rather than arbitrary fractions of the 8M native cap. The arrival grid spans underload, native load, moderate overload, and strong overload while preserving request identities, prompt/output lengths, and ordering.

## 3. RUN_COMPLETENESS

- Expected window-level conditions: 1080
- Completed window-level conditions: 1080
- Invalid window-level conditions: 20
- Failed window-level conditions: 0
- Validity counts: `{"INVALID_HORIZON_TRUNCATED": 20, "VALID_PARTIALLY_CONSTRAINED": 38, "VALID_STRONGLY_CONSTRAINED": 96, "VALID_UNCONSTRAINED": 926}`
- Missing results: none.

## 4. ARRIVAL_PRESSURE_RESULTS

| Workload | Condition | Achieved pressure | Binding states | Decision states | Disagreements | Rate | Validity |
|---|---:|---:|---:|---:|---:|---:|---|
| azure_2023_code | arrival_x0p5 | load 0.5; active 0.0175781; kv 0.00230588 | 0 | 107560 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | arrival_x1p0 | load 1; active 0.0175781; kv 0.0027155 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | arrival_x2p0 | load 2; active 0.0214844; kv 0.00318688 | 0 | 85507 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | arrival_x4p0 | load 4; active 0.0351562; kv 0.004212 | 0 | 67208 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | arrival_x8p0 | load 8; active 0.0898438; kv 0.00417888 | 0 | 49360 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | arrival_x0p5 | load 0.5; active 0.00976562; kv 0.001351 | 0 | 595106 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | arrival_x1p0 | load 1; active 0.0136719; kv 0.00172775 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | arrival_x2p0 | load 2; active 0.0195312; kv 0.00222113 | 0 | 306905 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | arrival_x4p0 | load 4; active 0.0332031; kv 0.0026015 | 0 | 172130 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | arrival_x8p0 | load 8; active 0.046875; kv 0.00402125 | 0 | 91989 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | arrival_x0p5 | load 0.5; active 0.0605469; kv 0.003802 | 0 | 441677 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | arrival_x1p0 | load 1; active 0.0605469; kv 0.003802 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | arrival_x2p0 | load 2; active 0.0605469; kv 0.003802 | 0 | 437586 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | arrival_x4p0 | load 4; active 0.0605469; kv 0.003802 | 0 | 424415 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | arrival_x8p0 | load 8; active 0.0839844; kv 0.00703238 | 0 | 385517 | 0 | 0 | VALID_UNCONSTRAINED |

## 5. KV_PRESSURE_RESULTS

| Workload | Condition | Achieved pressure | Binding states | Decision states | Disagreements | Rate | Validity |
|---|---:|---:|---:|---:|---:|---:|---|
| azure_2023_code | kv_8000000 | cap 8000000; kv 0.0027155 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | kv_240000 | cap 240000; kv 0.0905167 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | kv_120000 | cap 120000; kv 0.181033 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | kv_60000 | cap 60000; kv 0.362067 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | kv_32000 | cap 32000; kv 0.678875 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | kv_16000 | cap 16000; kv 1.35775 | 225 | 100115 | 596 | 0.00595315 | VALID_STRONGLY_CONSTRAINED |
| azure_2023_code | kv_8000 | cap 8000; kv 1.664 | 6273379 | 6340511 | 6270372 | 0.988938 | INVALID_HORIZON_TRUNCATED |
| azure_2023_conv | kv_8000000 | cap 8000000; kv 0.00172775 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | kv_240000 | cap 240000; kv 0.0575917 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | kv_120000 | cap 120000; kv 0.115183 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | kv_60000 | cap 60000; kv 0.230367 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | kv_32000 | cap 32000; kv 0.431937 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | kv_16000 | cap 16000; kv 0.716187 | 0 | 461985 | 18 | 3.89623e-05 | VALID_UNCONSTRAINED |
| azure_2023_conv | kv_8000 | cap 8000; kv 1.35862 | 6548 | 463652 | 10120 | 0.0218267 | VALID_STRONGLY_CONSTRAINED |
| burstgpt | kv_8000000 | cap 8000000; kv 0.003802 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | kv_240000 | cap 240000; kv 0.126733 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | kv_120000 | cap 120000; kv 0.253467 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | kv_60000 | cap 60000; kv 0.506933 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | kv_32000 | cap 32000; kv 0.9505 | 0 | 440461 | 0 | 0 | VALID_PARTIALLY_CONSTRAINED |
| burstgpt | kv_16000 | cap 16000; kv 1.90094 | 2 | 440464 | 12 | 2.7244e-05 | VALID_STRONGLY_CONSTRAINED |
| burstgpt | kv_8000 | cap 8000; kv 3.80175 | 388 | 440490 | 116 | 0.000263343 | VALID_STRONGLY_CONSTRAINED |

## 6. ACTIVE_SEQUENCE_PRESSURE_RESULTS

| Workload | Condition | Achieved pressure | Binding states | Decision states | Disagreements | Rate | Validity |
|---|---:|---:|---:|---:|---:|---:|---|
| azure_2023_code | active_512 | cap 512; active 0.0175781 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | active_64 | cap 64; active 0.140625 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | active_32 | cap 32; active 0.28125 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | active_16 | cap 16; active 0.5625 | 0 | 99992 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_code | active_8 | cap 8; active 1 | 86 | 99992 | 1 | 1.00008e-05 | VALID_STRONGLY_CONSTRAINED |
| azure_2023_code | active_4 | cap 4; active 1 | 3613 | 100127 | 100 | 0.000998732 | VALID_STRONGLY_CONSTRAINED |
| azure_2023_conv | active_512 | cap 512; active 0.0136719 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | active_64 | cap 64; active 0.109375 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | active_32 | cap 32; active 0.21875 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | active_16 | cap 16; active 0.4375 | 0 | 461985 | 0 | 0 | VALID_UNCONSTRAINED |
| azure_2023_conv | active_8 | cap 8; active 0.875 | 0 | 461985 | 0 | 0 | VALID_PARTIALLY_CONSTRAINED |
| azure_2023_conv | active_4 | cap 4; active 1 | 25935 | 462751 | 31 | 6.69907e-05 | VALID_STRONGLY_CONSTRAINED |
| burstgpt | active_512 | cap 512; active 0.0605469 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | active_64 | cap 64; active 0.484375 | 0 | 440461 | 0 | 0 | VALID_UNCONSTRAINED |
| burstgpt | active_32 | cap 32; active 0.96875 | 0 | 440461 | 0 | 0 | VALID_PARTIALLY_CONSTRAINED |
| burstgpt | active_16 | cap 16; active 1 | 1473 | 440691 | 14 | 3.17683e-05 | VALID_STRONGLY_CONSTRAINED |
| burstgpt | active_8 | cap 8; active 1 | 5673 | 444469 | 76 | 0.000170991 | VALID_STRONGLY_CONSTRAINED |
| burstgpt | active_4 | cap 4; active 1 | 25650 | 455652 | 244 | 0.000535496 | VALID_STRONGLY_CONSTRAINED |

## 7. TRANSITION_MAP

| Workload | Axis | First binding | First disagreement | Sustained disagreement |
|---|---|---|---|---|
| azure_2023_code | active_sequence_capacity | active_8 | active_8 | active_4 |
| azure_2023_code | arrival_pressure | NONE_OBSERVED | NONE_OBSERVED | NONE_OBSERVED |
| azure_2023_code | kv_capacity | kv_16000 | kv_16000 | kv_16000 |
| azure_2023_conv | active_sequence_capacity | active_4 | active_4 | active_4 |
| azure_2023_conv | arrival_pressure | NONE_OBSERVED | NONE_OBSERVED | NONE_OBSERVED |
| azure_2023_conv | kv_capacity | kv_8000 | kv_16000 | kv_8000 |
| burstgpt | active_sequence_capacity | active_16 | active_16 | active_8 |
| burstgpt | arrival_pressure | NONE_OBSERVED | NONE_OBSERVED | NONE_OBSERVED |
| burstgpt | kv_capacity | kv_16000 | kv_16000 | kv_16000 |

## 8. ACTION_COLLAPSE

Across valid workload-axis aggregates, canonical disagreements total 11328 states. Proxy/ranking differences total 29809 states, with 28941 collapsing to SBS-identical canonical actions. Including invalid horizon-truncated regimes, the retained totals are 6281700 canonical disagreements, 5341733 proxy/ranking differences, and 69334 collapsed states. The per-condition tables preserve both collapse and true canonical disagreement counts.

## 9. INDUSTRY_INTERPRETATION

Scheduler choice remains absent under native resources and under arrival scaling through 8x, because achieved active/KV pressure remains far below binding. Genuine canonical alternatives emerge when the same traces are replayed with tight KV or active-sequence resources. This supports a regime-dependent interpretation of action opportunity, with workload-specific onset thresholds. Interpret disagreement only in valid regimes and only as support/prevalence; no terminal benefit or causal headroom is inferred here.

## 10. NULL_AND_EXTREME_REGIMES

All zero-disagreement, zero-binding, non-monotonic, and invalid/extreme pressure points are retained in `PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv` and `PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv`.

## 11. PHASE_D_COVERAGE_PLAN

Coverage-based candidate regimes for later causal headroom labeling:

- azure_2023_code / active_sequence_capacity: binding=active_8, disagreement_onset=active_8, high_valid=active_4, null_control=active_512
- azure_2023_code / arrival_pressure: binding=NONE_OBSERVED, disagreement_onset=NONE_OBSERVED, high_valid=arrival_x8p0, null_control=arrival_x0p5
- azure_2023_code / kv_capacity: binding=kv_16000, disagreement_onset=kv_16000, high_valid=kv_16000, null_control=kv_8000000
- azure_2023_conv / active_sequence_capacity: binding=active_4, disagreement_onset=active_4, high_valid=active_4, null_control=active_512
- azure_2023_conv / arrival_pressure: binding=NONE_OBSERVED, disagreement_onset=NONE_OBSERVED, high_valid=arrival_x8p0, null_control=arrival_x0p5
- azure_2023_conv / kv_capacity: binding=kv_8000, disagreement_onset=kv_16000, high_valid=kv_8000, null_control=kv_8000000
- burstgpt / active_sequence_capacity: binding=active_16, disagreement_onset=active_16, high_valid=active_4, null_control=active_512
- burstgpt / arrival_pressure: binding=NONE_OBSERVED, disagreement_onset=NONE_OBSERVED, high_valid=arrival_x8p0, null_control=arrival_x0p5
- burstgpt / kv_capacity: binding=kv_16000, disagreement_onset=kv_16000, high_valid=kv_8000, null_control=kv_8000000

Do not label these in Phase B; Phase D should preregister causal-headroom sampling separately and include null-support controls.

## 12. FGCS_READINESS_CHECKPOINT

- Scientific novelty: 15/20
- Industry realism: 13/20
- Technical depth: 13/20
- Experimental rigor: 15/20
- Practitioner value: 6/10
- Reproducibility/community value: 8/10
- FGCS_CONTRIBUTION_READINESS_SCORE = 70/100
- FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 62%
- SYSTEM_REGIME_GATE = PASS

Remaining path to >90%: Phase-D causal headroom on coverage-selected regimes, optional Phase-C joint-regime mapping if justified, broader trace coverage or a precise scope claim, and a public reproducibility package.

## 13. NEXT_TASK

Preregister Phase D causal-headroom labeling over coverage-selected Phase-B regimes, including onset, moderate, high-valid, workload-diverse, and null-support cells.

PHASE_B_V1_EXECUTED = NO
PHASE_B_V2_COMPLETE = YES
REAL_TRACE_STRUCTURE_PRESERVED = YES
CAUSAL_LABELING_EXECUTED = NO
NEW_SELECTOR_TRAINING_EXECUTED = NO
SYSTEM_REGIME_GATE = PASS
FGCS_CONTRIBUTION_READINESS_SCORE = 70/100
FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 62%
READY_FOR_CAUSAL_HEADROOM_DESIGN = YES

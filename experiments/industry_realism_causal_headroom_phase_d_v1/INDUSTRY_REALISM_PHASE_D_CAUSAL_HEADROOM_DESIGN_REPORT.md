# INDUSTRY_REALISM_PHASE_D_CAUSAL_HEADROOM_DESIGN_REPORT

## 1. STARTING_STATE

- Branch: `contextual-compositional-heuristics-20260731`
- Starting HEAD: `d1228605e3e12ce0cdb89ccf9018e9c0721c3d5f`
- Phase-B result commit verified: `d1228605e3e12ce0cdb89ccf9018e9c0721c3d5f`
- Phase-B window conditions reported: `1080`
- Valid Phase-B canonical disagreement states reproduced: `11328`
- Phase-A and Phase-B result artifacts were read only; no prior result files were modified.

## 2. ELIGIBLE_CAUSAL_UNIVERSE

- Workloads: `azure_2023_code, azure_2023_conv, burstgpt`
- All valid Phase-B disagreement states: `11328`
- Unique non-SBS state-action branches across all valid support: `12169`
- Selected Phase-D disagreement states: `11328`
- Selected Phase-D non-SBS branches: `12169`
- Distribution by workload: `{"azure_2023_code": 697, "azure_2023_conv": 10169, "burstgpt": 462}`
- Distribution by axis: `{"active_sequence_capacity": 466, "kv_capacity": 10862}`
- Distribution by regime stage: `{"high_valid_pressure": 360, "onset": 33, "onset+sustained": 12, "onset+sustained+high_valid_pressure": 627, "sustained": 76, "sustained+high_valid_pressure": 10220}`

## 3. REGIME_SELECTION

Regimes are derived mechanically from the frozen Phase-B transition table: include onset, sustained, and highest valid pressure per workload/axis where support exists; merge duplicates; keep arrival-pressure zero-support cells as structural null controls.

| Workload | Axis | Condition | Stage labels |
|---|---|---|---|
| azure_2023_code | active_sequence_capacity | active_8 | onset |
| azure_2023_code | active_sequence_capacity | active_4 | sustained+high_valid_pressure |
| azure_2023_code | kv_capacity | kv_16000 | onset+sustained+high_valid_pressure |
| azure_2023_conv | active_sequence_capacity | active_4 | onset+sustained+high_valid_pressure |
| azure_2023_conv | kv_capacity | kv_16000 | onset |
| azure_2023_conv | kv_capacity | kv_8000 | sustained+high_valid_pressure |
| burstgpt | active_sequence_capacity | active_16 | onset |
| burstgpt | active_sequence_capacity | active_8 | sustained |
| burstgpt | active_sequence_capacity | active_4 | high_valid_pressure |
| burstgpt | kv_capacity | kv_16000 | onset+sustained |
| burstgpt | kv_capacity | kv_8000 | high_valid_pressure |

## 4. LABELING_SCOPE

- Scope: `EXHAUSTIVE_SELECTED_REGIME_LABELING`
- Exact selected state count: `11328`
- Exact selected branch count: `12169`
- Exact SBS reference count: `11328`
- Total terminal continuations planned: `23497`
- Deterministic rule: exhaustive selected-regime labeling because planned continuations are below the preregistered budget threshold.

## 5. CAUSAL_ESTIMAND

`Q_SBS(s,a)` forces exactly action `a` once at state `s`, then reverts immediately to fixed SBS / `kv_constrained_online`, preserving future arrivals and simulator semantics. `A_SBS(s,a)=Q_SBS(s,a)-Q_SBS(s,SBS)`.

## 6. METRICS

Primary metrics are `P(D)`, `P(B|D)`, `oracle_headroom(s)=max(0,max_a A_SBS(s,a))`, mean oracle headroom over all disagreement states, positive-headroom mean/median, and harm/zero/mixed state structure. Secondary action-level metrics report advantage sign fractions, moments, quantiles, worst harm, and best gain.

Opportunity-adjusted quantities: `P(D) x P(B|D)` and `P(D) x mean_oracle_headroom`, explicitly treated as oracle/local headroom rather than deployable learned-policy gain.

## 7. STATISTICAL_PROTOCOL

- Clustering unit: faithful window
- Bootstrap replicates: `2000`
- Bootstrap seed: `20260920`
- CI: percentile 95%
- Minimum clusters for CI: `5`
- Sparse regimes below the minimum cluster threshold get descriptive estimates only.
- Zero convention: exact floating zero unless pre-execution numerical repeatability testing freezes a tolerance.

## 8. IMPLEMENTATION_PLAN

Reuse `run_one_step_then_sbs_terminal`, `fork_from_live_simulator`, `Simulator.continue_run`, Phase-A canonicalization, and Phase-B pressure transformations. Required tests cover one forced intervention, SBS continuation, state immutability, arrival/random-state equivalence, reference replay, feasibility, deduplication, determinism, and pressure persistence.

## 9. COMPUTE_BUDGET

- Continuations: `23497`
- Recommended jobs/tasks: `94`
- Runtime estimate basis: `scaled from prior fresh targeted-label campaign size of 9,858 continuations; exact runtime depends on constrained-regime drain lengths`
- Relative to fresh confirmatory campaign: `2.384`
- Storage estimate: `93.99 MB`

## 10. PRACTITIONER_OUTPUT

Planned regime map columns: workload, axis, condition, achieved active/KV pressure, binding rate, `P(D)`, `P(B|D)`, mean oracle headroom, downside prevalence, and `ACTIONABLE_HEADROOM_INDEX=(P(D),P(B|D),mean_oracle_headroom)`.

## 11. CLAIM_BOUNDARIES

Phase D can establish causal simulator headroom under modeled resource-constrained replay of production-derived workload traces. It cannot establish real production deployment effects, measured production latency improvement, real-vLLM improvement, or learnability of a selector.

## 12. FGCS_READINESS_FORECAST

Current score remains `70/100` and confidence remains `62%`; the causal-headroom hard gate is pending until Phase-D execution. Strong practical headroom across multiple workloads/regimes would materially strengthen the contribution. Limited or harmful headroom would still support a rigorous characterization/negative-result paper. Broader workload coverage, literature positioning, and real-system validation would remain afterward.

## 13. PREREGISTRATION_FREEZE

- Local commit to be recorded after this report is committed.
- Universe manifest hash: `4b992e4dc7199147bd2e69615577bf0579baf82ae1f3e1f5aa7c60eace5c1b1d`
- Regime selection hash: `27f34ff44705de48a8ed08bf03838b122addfa47ba1986025edf4c060aebb316`
- Labeling plan hash: `53c757d7ea41bc0a29d298b9851f8c60eff185b6fc9b175c89857b143ee058b7`
- Metric protocol hash: `503dd9b427898c332457eb519ed17699ba8ab6806e55679259aea1c5aa86b1ad`
- Preregistration hash: `067ffc4aea089bb3e3c6d2d6db2ba5b089ef22269f89e5508bb0f06aef645e78`

## 14. EXACT_NEXT_TASK

`python3 scripts/industry_realism_causal_headroom_phase_d_v1_execute.py --input experiments/industry_realism_causal_headroom_phase_d_v1/PREREGISTRATION_V1.json --run-sharded --num-shards 96` after adding and passing the preregistered causal-correctness tests.

PHASE_D_DESIGN_COMPLETE = YES
PHASE_D_OUTCOMES_ACCESSED = NO
REGIME_SELECTION_COVERAGE_BASED = YES
NULL_SUPPORT_REGIMES_CAUSALLY_LABELED = NO
NEW_SELECTOR_TRAINING_AUTHORIZED = NO
CAUSAL_HEADROOM_GATE = PENDING
FGCS_CONTRIBUTION_READINESS_SCORE = 70/100
FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 62%
READY_FOR_PHASE_D_EXECUTION = YES

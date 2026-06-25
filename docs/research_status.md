# Research Status

**Last updated:** 2026-06-25  
**Current branch:** `phase2b6-fair-sweep-failure-audit`  
**Current phase:** Phase 2B.6 — Fair sweep, failure tracking, external baseline correctness audit

---

## Quick Facts

| Item | Value |
|---|---|
| Deployable scheduling policies | **19** |
| Non-deployable oracle policies | **1** (`oracle_srtf`) |
| Selector candidate policies | **19** (= deployable baselines) |
| Implemented selector models | 3 (`rule_based`, `decision_tree`, `random_forest`) |
| Test count | 710 passing, 0 failing |

---

## Important Branches / Commits

| Branch | Commit | Description |
|---|---|---|
| `phase2a4-2b4-final-eval` | `9ed8f71` | Final frozen evaluation (Phase 2A.4/2B.4) |
| `phase2b5-external-baselines` | `e1c6c01` | External baseline audit + completion_fraction metric |
| `phase2b5-admission-rule-selector-status` | `5d2afb6` | Admission control policy + feature rule selector |
| `phase2b6-fair-sweep-failure-audit` | current | Fair sweep, failure tracking, correctness audit |
| `main` | stale | Do not use; all work is on phase branches |

---

## Completed Work

### Phase 1 — Simulator + Baselines
- Deterministic step-based LLM serving simulator
- 10 Phase 1 baselines (FIFO, EDF, SOF, SPF, greedy fill, first/best fit, multi-bin, least loaded, random)

### Phase 1.5 — Serving-Style Baselines + Metrics
- 6 serving-style baselines: Orca-style, vLLM-style, Sarathi-style, SplitFuse-style, SLO-slack, WSPT
- TTFT / TPOT metrics
- BurstGPT + ShareGPT trace loading

### Phase 1.7 — Calibration
- RTX 5060 Ti calibrated with Qwen2.5-0.5B (prefill MAPE 9%, decode MAPE 12%)

### Phase 2A — Selector + Objective
- Selector dataset, features, labels, training
- RF selector WG=0.828 (+3pp vs. FIFO)
- Frozen shortlist and held-out CI computation

### Phase 2B — LLM Heuristic Search
- DSL verifier, LLM generation loop, controlled search
- Best generated heuristic: WG=0.9595 CI=[0.00, 0.27]

### Phase 2A.3B — Deadline/Laxity Baselines
- `least_laxity_first` (LLF) — 14 unit tests
- `estimated_service_time_first` (PARS-like SJF proxy) — 16 unit tests

### Phase 2B.5 — External Baseline Coverage
- External baseline audit (`docs/external_baseline_coverage_report.md`)
- Dataset/workload plan (`docs/dataset_workload_plan.md`)
- `completion_fraction` and `num_total` added to `RunMetrics`
- `admission_control` policy added (19th deployable baseline)
- `RuleBasedSelector` upgraded from FIFO placeholder to feature-based dispatch
- `scripts/report_research_status.py` for code-derived status
- 710 tests passing

### Phase 2B.6 — Fair Sweep + Failure Tracking (current)
- Experiment/accounting infrastructure: committed CSV templates + `docs/experiment_tracking.md`
- Per-policy correctness audit for 11 baselines (`docs/external_baseline_correctness_audit.md`)
- `AdmissionControlPolicy` threshold calibration (`docs/audits/admission_control_threshold_calibration_summary.md`)
- Fair sweep: all 19 deployable policies × 3 workloads × 2 seeds (`configs/phase2b6_fair_sweep.yaml`)
- Sweep finding: underloaded workloads (10s, 10-20 req/s) produce zero WG differentiation — all policies tie
- `report_research_status.py` updated: template existence checks + `--check` mode validates templates
- New tests: leakage/fairness audit + registry template tests

---

## Implemented Deployable Scheduling Policies (19)

All 19 are in `BASELINE_NAMES` and `SELECTOR_CANDIDATE_NAMES`.  
None have access to `actual_output_tokens` (oracle excluded).

| # | Policy name | Category | SLO-aware | KV-budget aware |
|---|---|---|---|---|
| 1 | `fifo` | Classical | No | No |
| 2 | `edf` | Classical | Yes | No |
| 3 | `shortest_output_first` | SRPT-style | No | No |
| 4 | `shortest_prompt_first` | Heuristic | No | No |
| 5 | `greedy_token_fill` | Packing | No | Yes |
| 6 | `least_loaded` | Load balancing | No | No |
| 7 | `multi_bin_batching` | Batching | No | No |
| 8 | `random_feasible` | Stochastic | No | No |
| 9 | `first_fit` | Packing | No | Yes |
| 10 | `best_fit` | Packing | No | Yes |
| 11 | `orca_style` | Serving-style (inspired) | Partial | Yes |
| 12 | `vllm_style_token_budget` | Serving-style (inspired) | No | Yes |
| 13 | `sarathi_style` | Serving-style (inspired) | No | Yes |
| 14 | `splitfuse_style` | Serving-style (inspired) | No | Yes |
| 15 | `slo_slack_score` | Composite | Yes | No |
| 16 | `weighted_shortest_processing` | WSPT | No | No |
| 17 | `least_laxity_first` | Deadline/laxity | Yes | No |
| 18 | `estimated_service_time_first` | SJF proxy | Partial | No |
| 19 | `admission_control` | Admission control | Yes | Yes |

---

## Non-Deployable Oracle Policies (1)

| Policy | File | Notes |
|---|---|---|
| `oracle_srtf` | `policies/oracle.py` | Hindsight SRTF oracle. Uses `actual_output_tokens`. **Never** in selector candidates. Access via `make_oracle_policy()` only. Always emits `UserWarning`. |

The selector must never choose `oracle_srtf`. This invariant is enforced by `selector/candidates.py` and tested by multiple tests.

---

## Selector Candidate Set

`SELECTOR_CANDIDATES` = `BASELINE_NAMES` minus `ORACLE_POLICY_NAMES`.

Currently: **19 candidates** (all 19 deployable policies above).

Verified at import time by `selector/candidates.py` assertion loop.

---

## Implemented Selector Models (3)

| Model | File | Type | Notes |
|---|---|---|---|
| `rule_based` | `selector/models.py:RuleBasedSelector` | Feature-based rules | No training needed. Dispatches to 7 different policies based on workload features. Was a FIFO placeholder before Phase 2B.5. |
| `decision_tree` | `selector/models.py:DecisionTreeSelector` | sklearn DT | max_depth=8, min_samples_leaf=20. Requires sklearn. |
| `random_forest` | `selector/models.py:RandomForestSelector` | sklearn RF | n_estimators=200, max_depth=10. Primary Phase 2A.4 selector. WG=0.828. |

### Rule-Based Selector Dispatch Logic (Phase 2B.5)

The selector picks policies based on these observable features (in priority order):

1. `fraction_tight_slo > 0.4` or `min_slack < 1.0` → `least_laxity_first`
2. `recent_slo_violation_rate > 0.3` → `admission_control`
3. `kv_utilization > 0.7` → `vllm_style_token_budget`
4. `mean_prompt_tokens > 512` or `p95_prompt_tokens > 1024` → `sarathi_style`
5. `mean_pred_output_tokens < 64` and `pred_output_cv < 0.5` → `estimated_service_time_first`
6. `burstiness_cv > 1.5` → `slo_slack_score`
7. default → `edf`

---

## Missing External Baselines

| Baseline | Status | Notes |
|---|---|---|
| CP-SAT / ILP oracle | ❌ Missing | Requires `ortools`; only useful for micro-traces (< 20 reqs) |
| True PARS (learning-to-rank) | ❌ Not implemented | Our `estimated_service_time_first` is a simplified proxy; true PARS needs learned ranking |
| Preemption-based SJF | ❌ Not implemented | Requires preemption support in simulator (Phase 2+ scope) |

---

## Missing Workloads / Datasets

| Dataset | Status | Notes |
|---|---|---|
| LMSYS-Chat-1M / ShareGPT raw | ❌ Not downloaded | Loader exists in `workloads/sharegpt.py`; data download pending |
| LongBench | ❌ Not downloaded | Synthetic long-context generator exists; LongBench data not needed yet |

All synthetic and BurstGPT workloads already exist in repo.

---

## Reporting Gaps

| Gap | Status | Notes |
|---|---|---|
| `completion_fraction` in reports | ✅ Added (Phase 2B.5) | `num_total` and `completion_fraction` now in `RunMetrics` and CSV output |
| Selective vs. forced-admission labels | Partial | Field added; report headers not yet updated |
| Multi-bin batching unit tests | ⚠️ Registry check only | Dedicated `test_multi_bin_batching_policy.py` not yet added |

---

## API-Credit Policy

**No paid API calls in Phase 2B.5 (or Phase 2B.5-admission).**

CloudRift, Cohere, OpenAI, Gemini, Mistral, and all other paid APIs are not called in:
- Any test
- Any script run during CI or development
- Any default script behavior

Paid API usage is gated behind explicit `--use-llm` flags or dedicated scripts.

---

## Phase 2B.6 Findings

| Finding | Detail |
|---|---|
| Cheap sweep (10s, 10-20 req/s): zero differentiation | All 19 policies achieve WG ≈ 0.9977-1.0 under underloaded conditions |
| AdmissionControl unit mismatch | `_laxity()` mixes seconds and steps; default `threshold=inf` is safe |
| Threshold calibration: `threshold=200.0` | Drops ~21% of requests, achieves WG=1.0, zero SLO violations |
| Historical rule_based failure (Phase 2A.4) | FIFO placeholder: WG=0.597 vs. best_fixed 0.798 (−0.20); now fixed in Phase 2B.5 |

---

## Next Planned Tasks

1. **Run overloaded baseline sweep** using existing overloaded configs (arrival_rate=60-90, duration=30s) to produce meaningful WG differentiation across all 19 policies.
2. **Add multi-bin batching unit tests** (`tests/test_multi_bin_batching_policy.py`).
3. **Re-evaluate rule_based selector** on Phase 2A.4 test split using updated feature-based dispatch (post Phase 2B.5 upgrade).
4. **Fix admission_control unit mismatch** — convert service proxy to seconds before laxity comparison.
5. **Update selector comparison table** in `docs/selector.md` with updated rule_based WG.
6. **Optional:** Add CP-SAT oracle for micro-benchmark traces.
7. **Write comparison paper section** after meaningful overloaded sweep results available.

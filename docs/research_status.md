# Research Status

**Last updated:** 2026-06-25  
**Current branch:** `phase2b9-selector-robustness-and-suite-freeze`  
**Current phase:** Phase 2B.9 — Selector robustness audit and comparison suite freeze

---

## Quick Facts

| Item | Value |
|---|---|
| Deployable scheduling policies | **19** |
| Non-deployable oracle policies | **1** (`oracle_srtf`) |
| Selector candidate policies | **19** (= deployable baselines) |
| Implemented selector models | 3 (`rule_based`, `decision_tree`, `random_forest`) |
| Test count | **787 passing**, 1 skipped (Phase 2B.8 base) + Phase 2B.9 tests added |

---

## Important Branches / Commits

| Branch | Commit | Description |
|---|---|---|
| `phase2a4-2b4-final-eval` | `9ed8f71` | Final frozen evaluation (Phase 2A.4/2B.4) |
| `phase2b5-external-baselines` | `e1c6c01` | External baseline audit + completion_fraction metric |
| `phase2b5-admission-rule-selector-status` | `5d2afb6` | Admission control policy + feature rule selector |
| `phase2b6-fair-sweep-failure-audit` | `a6df363` | Fair sweep, failure tracking, correctness audit |
| `phase2b7-overload-failure-mining` | `992fe11` | Unit fix, overloaded sweep, multi-bin tests, failure registry |
| `phase2b8-rule-selector-repair` | `429e96e` | Rule selector repair: KV-pressure guard, noise guard, slo_slack_score for tight SLO |
| `phase2b9-selector-robustness-and-suite-freeze` | current | Selector robustness audit, training data sufficiency, external baseline/dataset decisions |
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

### Phase 2B.6 — Fair Sweep + Failure Tracking
- Experiment/accounting infrastructure: committed CSV templates + `docs/experiment_tracking.md`
- Per-policy correctness audit for 11 baselines (`docs/external_baseline_correctness_audit.md`)
- `AdmissionControlPolicy` threshold calibration (`docs/audits/admission_control_threshold_calibration_summary.md`)
- Fair sweep: all 19 deployable policies × 3 workloads × 2 seeds (`configs/phase2b6_fair_sweep.yaml`)
- Sweep finding: underloaded workloads (10s, 10-20 req/s) produce zero WG differentiation — all policies tie
- `report_research_status.py` updated: template existence checks + `--check` mode validates templates
- New tests: leakage/fairness audit + registry template tests

### Phase 2B.9 — Selector Robustness Audit and Suite Freeze (current)
- **Selector training data audit**: Phase 2A.4 RF/DT use ~30 training windows (critically small for 19-class).
  KV-pressure and high-noise regimes missing from training data.
  See `docs/audits/phase2b9_selector_training_audit.md` for full analysis.
- **Broader robustness experiment**: 4 dev workloads (same as Phase 2B.7/2B.8, seeds 0–2) +
  5 held-out workloads (new regimes, seeds 3–5). Config: `configs/phase2b9_selector_robustness.yaml`.
  Runner: `scripts/run_phase2b9_selector_robustness.py`. Log: `logs/phase2b9/`.
- **External baseline decisions**: 5 must-add baselines identified (SCORPIO-style, KV-cache-aware,
  FairBatching, PARS-LTR, PROSERVE SlideBatching). See `docs/external_baseline_decision.md`.
- **Dataset/workload decisions**: BurstGPT (full), Azure LLM Inference 2023, LMSYS-Chat-1M,
  LongBench, and calibrated synthetics are must-use before publication.
  See `docs/dataset_workload_decision.md`.
- **Caveats and claims**:
  - Phase 2B.8 repaired rule selector matched best fixed baseline on 4 **development** workloads.
    This is **not** a final generalization claim (same workloads used for rule design and evaluation).
  - Phase 2B.9 evaluates the selector on held-out workloads (5 new + different seeds) for the first time.
  - RF/DT selectors have 30 training windows; insufficient for strong publication claims.
  - Final publication requires: ≥200 training windows, KV-pressure/noise regimes in training,
    real-trace BurstGPT/Azure data, and must-add external baselines.
- **Tests added**: `tests/test_phase2b9_selector_robustness.py`
  (rule selector dispatch table, oracle exclusion, config validation, doc existence checks)
- **Robustness results analyzed**: `docs/audits/phase2b9_selector_robustness_summary.md`
  - Dev WG=0.917, held-out WG=0.979, overall WG=0.951 (rule selector)
  - Beats best fixed on dev (+0.024) and held-out (+0.008); within 0.5 pp of per-window oracle on held-out
  - RF/DT not re-evaluated (Phase 2A.4 model artifacts absent in this run)
- **Failure cases**: `docs/audits/phase2b9_failure_cases_summary.md`
  - 1 unresolved: `heldout_very_high_noise_s4` (AC vs EDF, gap ≈ −0.023)
- Log: `logs/phase2b9/phase2b9_selector_robustness.log` (gitignored)

### Phase 2B.8 — Rule Selector Repair Under KV Pressure
- **Root cause**: Phase 2B.7 showed Rule 1 (`tight_slo/min_slack → least_laxity_first`) fired for
  all 3 differentiated workloads; LLF catastrophic under KV pressure (WG=0.101 vs WSP=0.477)
- **Repair**: Three changes to `RuleBasedSelector.predict_one()`:
  1. New Rule 1 (elevated): `mean_pred_output_tokens > 200 OR kv_utilization > 0.7 → weighted_shortest_processing`
     (KV pressure proxy: large outputs fill KV cache; WSP avoids urgency-induced cascade)
  2. New Rule 2: `pred_output_cv > 1.0 → admission_control`
     (High noise: laxity estimates unreliable under 70%+ prediction noise)
  3. Rule 4 (was Rule 1): tight SLO now → `slo_slack_score` instead of `least_laxity_first`
     (composite urgency+throughput score; avoids LLF throughput collapse under overload)
- **Policy choices reduced**: `_POLICY_CHOICES` from 7 to 6 (removed `vllm_style_token_budget` and `least_laxity_first`)
- Tests: 32 rule selector tests pass (including 3 failure-case regression tests)
- Total tests: 784 passing
- Sweep: same 4 workloads × 19 policies × 3 seeds as Phase 2B.7 (apples-to-apples)

### Phase 2B.7 — Unit Fix + Overloaded Failure Mining
- **Unit fix**: `AdmissionControlPolicy._laxity()` now unit-consistent (seconds); `laxity_threshold` in seconds
- New parameter `step_size=0.001` for service proxy → seconds conversion
- **Multi-bin unit tests**: `tests/test_multi_bin_batching_policy.py` (18 tests)
- **Overloaded sweep**: all 19 policies × 4 regimes × 3 seeds (`configs/phase2b7_overload_failure_mining.yaml`)
  - 3/4 workloads show substantial policy differentiation (WG range 0.43–0.56)
  - `prefill_heavy_small` still underloaded (all tie WG=1.0)
- **3 failure cases identified** in `results/failure_cases/failure_case_registry.csv`
  - Root cause: Rule 1 (`min_slack < 1.0s`) fires for all overloaded workloads → always picks LLF
  - LLF catastrophically bad in kv_pressure_decode_heavy (WG=0.101 vs best=0.477)
- Best fixed baseline overall: `weighted_shortest_processing` WG=0.827
- Rule-based selector overall WG ≈ 0.540 (−0.287 vs best fixed)
- Admission control (post unit-fix, threshold=inf): wins high_prediction_noise (rank 1), loses kv_pressure (rank last)

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

## Admission/Completion Accounting Limitation

The simulator tracks:
- `num_total` = all arrivals
- `num_completed` = requests serviced before drain
- `num_dropped` = arrivals − completed (never serviced OR timed out)
- `completion_fraction` = num_completed / num_total

**Not tracked:**
- `num_admitted` = requests explicitly accepted for service in some step
- `num_rejected` = requests explicitly filtered by admission control policy
- `admission_fraction` = num_admitted / num_total
- `conditional_completion_fraction` = num_completed / max(num_admitted, 1)

**Limitation:** In the current simulator, "dropped" conflates requests filtered by admission
control with requests that arrived but were never scheduled before simulation ended. The 
simulator does not distinguish these. For `admission_control` with `threshold=inf` (default),
`num_dropped` = requests that ran out of simulation time. For `threshold=0.0s`, dropped
includes genuinely-filtered infeasible requests — but the metrics don't distinguish them.

**TODO:** Add `num_rejected` tracking when `AdmissionControlPolicy` explicitly filters a request
(not just delays it). This requires simulator-level hooks or policy instrumentation.

---

## Next Planned Tasks

### Completed in Phase 2B.9 ✅
- Selector training data sufficiency and leakage audit (`docs/audits/phase2b9_selector_training_audit.md`)
- Broader robustness experiment (9 workloads: 4 dev + 5 heldout, seeds 0–5)
- Robustness results analysis (`docs/audits/phase2b9_selector_robustness_summary.md`)
- Failure case documentation (`docs/audits/phase2b9_failure_cases_summary.md`)
- External baseline decision document (`docs/external_baseline_decision.md`)
- Dataset/workload decision document (`docs/dataset_workload_decision.md`)
- Tests for oracle exclusion, rule dispatch, config validation, doc existence

### Remaining Before Submission
1. **Expand selector training data** — add KV-pressure + high-noise + real-trace windows to reach ≥200 training windows; re-train and re-evaluate RF/DT on Phase 2B.9 suite.
2. **Implement must-add baselines** (see `docs/external_baseline_decision.md` section B):
   - B.2 SCORPIO-style SLO guard (highest priority; directly addresses Phase 2B.7/2B.8 overload)
   - B.3 KV-cache-aware scheduler (addresses remaining KV WG gap)
   - B.4 FairBatching
   - B.1 PARS-style LTR (requires training data)
   - B.5 PROSERVE SlideBatching
3. **Ingest real-trace datasets** (see `docs/dataset_workload_decision.md` section B):
   - BurstGPT full dataset (currently using 10k subset only)
   - Azure LLM Inference 2023 / Splitwise trace
   - LMSYS-Chat-1M length statistics for calibrated synthetic
   - LongBench for long-context stress workloads
4. **LLM escalation** (CloudRift/Cohere): if further rule improvement needed after Phase 2B.9
   generalization analysis, synthesize updated rules. Log in API ledger. Limit: 1–2 calls per pattern.

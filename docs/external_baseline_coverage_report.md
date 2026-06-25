# External Baseline Coverage Report

**Branch:** phase2b5-external-baselines  
**Date:** 2026-06-25  
**Auditor:** automated audit (Phase 2B.5)

---

## Summary

| Category | Count |
|---|---|
| Baselines already implemented | 19 |
| Selectors already implemented | 3 |
| Newly implemented (this phase) | 0 |
| Still missing / not implemented | 3 |
| External author code used | 0 |
| All implementations | simulator-compatible approximations or internal-only |

All three "must-have" baselines from the task specification (LLF, PARS-like SJF, Multi-Bin Batching)
were already implemented in a prior phase (Phase 2A.3B).

---

## Implemented Baselines

### 1. FIFO / FCFS

| Field | Value |
|---|---|
| **Repo name** | `fifo` |
| **Canonical name** | FIFO / First-Come First-Served |
| **File** | `src/llmserveopt/policies/fifo.py` — `FIFOPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes (via `_feasible_on_gpu`) |
| **Batching / prefill / decode aware** | Partial (no prefill/decode split) |
| **Tests** | `test_policy_feasibility.py`, `test_simulator_basic.py` |
| **Implementation style** | Exact (standard algorithm) |
| **Literature source** | Standard; no specific paper |

---

### 2. EDF — Earliest Deadline First

| Field | Value |
|---|---|
| **Repo name** | `edf` |
| **Canonical name** | Earliest Deadline First |
| **File** | `src/llmserveopt/policies/edf.py` — `EDFPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | Yes (sorts by `slo_deadline`) |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Exact (standard algorithm) |
| **Literature source** | Classic real-time scheduling theory |

---

### 3. LLF — Least Laxity First

| Field | Value |
|---|---|
| **Repo name** | `least_laxity_first` |
| **Canonical name** | Least Laxity First |
| **File** | `src/llmserveopt/policies/least_laxity_first.py` — `LeastLaxityFirstPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No (uses `predicted_output_tokens` only) |
| **SLO-aware** | Yes (laxity = deadline − now − estimated service time) |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `tests/test_least_laxity_first_policy.py` (14 tests) |
| **Implementation style** | Exact (deterministic tie-breaking: laxity → deadline → priority → request_id) |
| **Literature source** | Classic real-time scheduling; Dertouzos 1974 |
| **Added in phase** | Phase 2A.3B |

---

### 4. SOF / SJF — Shortest (Predicted) Output First

| Field | Value |
|---|---|
| **Repo name** | `shortest_output_first` |
| **Canonical name** | SRPT-style Shortest Predicted Output First |
| **File** | `src/llmserveopt/policies/shortest_output_first.py` — `ShortestOutputFirstPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No (`predicted_output_tokens` only) |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Exact |
| **Literature source** | SRPT / SJF scheduling theory |

---

### 5. SPF — Shortest Prompt First

| Field | Value |
|---|---|
| **Repo name** | `shortest_prompt_first` |
| **Canonical name** | Shortest Prompt First |
| **File** | `src/llmserveopt/policies/shortest_prompt_first.py` — `ShortestPromptFirstPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No (prompt-length aware) |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Internal baseline |
| **Literature source** | N/A |

---

### 6. SLO Slack Score

| Field | Value |
|---|---|
| **Repo name** | `slo_slack_score` |
| **Canonical name** | Slack-based SLO composite scorer |
| **File** | `src/llmserveopt/policies/slo_slack_score.py` — `SloSlackScorePolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | Yes (deadline slack is primary component) |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Internal; inspired by urgency-based SLO literature |
| **Literature source** | General SLO-aware scheduling literature |

---

### 7. Greedy Token Fill

| Field | Value |
|---|---|
| **Repo name** | `greedy_token_fill` |
| **Canonical name** | Greedy token-fill batching |
| **File** | `src/llmserveopt/policies/greedy_token_fill.py` — `GreedyTokenFillPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | KV budget aware (fills to capacity) |
| **Token-budget / KV-cache aware** | Yes (primary mechanism) |
| **Batching / prefill / decode aware** | Batching-aware |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Internal baseline |
| **Literature source** | N/A |

---

### 8. First Fit

| Field | Value |
|---|---|
| **Repo name** | `first_fit` |
| **Canonical name** | First Fit bin-packing |
| **File** | `src/llmserveopt/policies/first_fit.py` — `FirstFitPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Exact (classic First Fit bin packing) |
| **Literature source** | Classic bin-packing; applied to GPU batch scheduling |

---

### 9. Best Fit

| Field | Value |
|---|---|
| **Repo name** | `best_fit` |
| **Canonical name** | Best Fit bin-packing |
| **File** | `src/llmserveopt/policies/best_fit.py` — `BestFitPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Exact (classic Best Fit bin packing) |
| **Literature source** | Classic bin-packing |

---

### 10. Multi-Bin Batching

| Field | Value |
|---|---|
| **Repo name** | `multi_bin_batching` |
| **Canonical name** | Multi-Bin-style batching |
| **File** | `src/llmserveopt/policies/multi_bin_batching.py` — `MultiBinBatchingPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No (`predicted_output_tokens` for binning) |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | Yes (bins by predicted output length) |
| **Tests** | `test_selector_candidates.py` (registry check) |
| **Implementation style** | Style/inspired approximation — NOT the official implementation of any published work |
| **Literature source** | Inspired by multi-bin batching ideas in LLM scheduling literature; no single canonical paper |
| **Added in phase** | Phase 1 |

> **Note:** The `multi_bin_batching` policy has no dedicated unit tests beyond registry presence.
> A targeted test file (`tests/test_multi_bin_batching_policy.py`) should be added in a follow-up.

---

### 11. Orca-style Iteration-level Batching

| Field | Value |
|---|---|
| **Repo name** | `orca_style` |
| **Canonical name** | Orca selective-batching (continuous-batching proxy) |
| **File** | `src/llmserveopt/policies/orca_style.py` — `OrcaStylePolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | Partial (class priority) |
| **Admission-control aware** | Yes (max_active_sequences, max_batch_tokens budgets) |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | Iteration-level batching |
| **Tests** | `test_policy_feasibility.py`, `test_sarathi_style_performance.py` |
| **Implementation style** | Style/inspired — NOT an official Orca reproduction |
| **Literature source** | Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022 |

---

### 12. vLLM-style Token-Budget / Paged-KV Proxy

| Field | Value |
|---|---|
| **Repo name** | `vllm_style_token_budget` |
| **Canonical name** | vLLM-inspired token-budget + paged-KV proxy |
| **File** | `src/llmserveopt/policies/vllm_style_token_budget.py` — `VLLMStyleTokenBudgetPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No (throughput-oriented) |
| **Admission-control aware** | Yes (KV block proxy) |
| **Token-budget / KV-cache aware** | Yes (primary mechanism) |
| **Batching / prefill / decode aware** | Partial (no preemption/eviction) |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Style/inspired — NOT a vLLM reproduction or performance benchmark |
| **Literature source** | Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," SOSP 2023 |

---

### 13. Sarathi-style Stall-Free Chunked Prefill

| Field | Value |
|---|---|
| **Repo name** | `sarathi_style` |
| **Canonical name** | Sarathi-Serve stall-free chunked-prefill proxy |
| **File** | `src/llmserveopt/policies/sarathi_style.py` — `SarathiStylePolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Yes (prefill chunk budget) |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | Yes (stall-free: decode priority, bounded prefill chunk) |
| **Tests** | `test_sarathi_style_performance.py` |
| **Implementation style** | Style/inspired — NOT an official Sarathi-Serve reproduction |
| **Literature source** | Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," OSDI 2024; arXiv 2023 |

---

### 14. SplitFuse-style / Dynamic-SplitFuse

| Field | Value |
|---|---|
| **Repo name** | `splitfuse_style` |
| **Canonical name** | Dynamic SplitFuse chunked-prefill proxy |
| **File** | `src/llmserveopt/policies/splitfuse_style.py` — `SplitFuseStylePolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Yes (step token budget) |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | Yes (fills to fixed step budget; decode + prefill composed) |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Style/inspired — NOT an official DeepSpeed-FastGen reproduction |
| **Literature source** | Holmes et al., "DeepSpeed-FastGen: High-Throughput Text Generation for LLMs via MII and DeepSpeed-Inference," arXiv 2024 |

---

### 15. PARS-like / Prompt-Aware SJF Proxy (ESTF)

| Field | Value |
|---|---|
| **Repo name** | `estimated_service_time_first` |
| **Canonical name** | PARS-like prompt-and-prediction-aware SJF proxy |
| **File** | `src/llmserveopt/policies/estimated_service_time_first.py` — `EstimatedServiceTimeFirstPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No (`predicted_output_tokens` only; `actual_output_tokens` never accessed) |
| **SLO-aware** | Partial (deadline as tie-breaker) |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `tests/test_estimated_service_time_first_policy.py` (16 tests) |
| **Implementation style** | Style/inspired — NOT a reproduction of PARS. PARS uses learning-to-rank; this uses deterministic token-length estimates only |
| **Literature source** | Inspired by: Zheng et al., "Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline," NeurIPS 2023 (PARS) |
| **Added in phase** | Phase 2A.3B |

---

### 16. WSPT — Weighted Shortest Processing Time

| Field | Value |
|---|---|
| **Repo name** | `weighted_shortest_processing` |
| **Canonical name** | WSPT — Weighted Shortest Processing Time |
| **File** | `src/llmserveopt/policies/weighted_shortest_processing.py` — `WeightedShortestProcessingPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Exact (classic WSPT rule) |
| **Literature source** | Classic scheduling theory; Smith 1956 |

---

### 17. Oracle SRTF (Hindsight Oracle)

| Field | Value |
|---|---|
| **Repo name** | `oracle_srtf` |
| **Canonical name** | Greedy hindsight SRTF oracle |
| **File** | `src/llmserveopt/policies/oracle.py` — `OracleShortestJobFirstPolicy` |
| **Online deployable** | **NO — hindsight oracle only** |
| **Uses future information** | **YES** (`actual_output_tokens` accessed) |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_oracle_not_deployable.py`, `test_oracle_srtf_wiring.py` |
| **Implementation style** | Internal oracle — greedy SRTF (globally optimal for mean completion time on single machine; approximate in multi-GPU batched setting) |
| **Literature source** | N/A |

---

### 18. Earliest Feasible GPU

| Field | Value |
|---|---|
| **Repo name** | `earliest_feasible_gpu` |
| **Canonical name** | First-fit dispatching / earliest-feasible-GPU |
| **File** | `src/llmserveopt/policies/earliest_feasible_gpu.py` — `EarliestFeasibleGPUPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Internal baseline |
| **Literature source** | N/A |

---

### 19. Least Loaded

| Field | Value |
|---|---|
| **Repo name** | `least_loaded` |
| **Canonical name** | Least Loaded / Least Utilized |
| **File** | `src/llmserveopt/policies/least_loaded.py` — `LeastLoadedPolicy` |
| **Online deployable** | Yes |
| **Uses future information** | No |
| **SLO-aware** | No |
| **Admission-control aware** | Feasibility check only |
| **Token-budget / KV-cache aware** | Yes |
| **Batching / prefill / decode aware** | No |
| **Tests** | `test_policy_feasibility.py` |
| **Implementation style** | Internal baseline |
| **Literature source** | N/A |

---

## Selector Models

### 20. Random Feasible

| Field | Value |
|---|---|
| **Repo name** | `random_feasible` |
| **File** | `src/llmserveopt/policies/random_feasible.py` |
| **Online deployable** | Yes |
| **Notes** | Stochastic baseline; deterministic under seed |

---

### 21. RF Selector — Random Forest

| Field | Value |
|---|---|
| **Repo name** | `random_forest` |
| **File** | `src/llmserveopt/selector/models.py` — `RandomForestSelector` |
| **Tests** | `test_selector_models.py`, `test_selector_evaluation.py` |
| **Notes** | sklearn RandomForestClassifier(n_estimators=200, max_depth=10) |

---

### 22. Decision Tree Selector

| Field | Value |
|---|---|
| **Repo name** | `decision_tree` |
| **File** | `src/llmserveopt/selector/models.py` — `DecisionTreeSelector` |
| **Tests** | `test_selector_models.py` |
| **Notes** | sklearn DecisionTreeClassifier(max_depth=8, min_samples_leaf=20) |

---

### 23. Rule-Based Selector

| Field | Value |
|---|---|
| **Repo name** | `rule_based` |
| **File** | `src/llmserveopt/selector/models.py` — `RuleBasedSelector` |
| **Tests** | `test_selector_models.py` |
| **Notes** | Placeholder only — always returns "fifo". A genuine hand-coded rule selector based on workload features is **missing** (see below) |

---

## Missing Baselines

### Missing 1: Explicit Admission Control Policy

**Status:** MISSING  
**Description:** A dedicated policy that explicitly drops (rejects) requests unlikely to meet their SLO based on laxity threshold or arrival-rate budget, before exhausting capacity. Current policies use passive feasibility checks but none actively drop near-deadline requests.  
**Why missing:** No simulator-level "explicit reject" mechanism was built; requests in the waiting queue are implicitly dropped at simulation end. Building this requires either a simulator-level reject action or an explicit filter policy that ignores requests with negative laxity (effectively marking them as permanently unschedulable).  
**Recommendation:** Implement `AdmissionControlPolicy` as a laxity-threshold wrapper: requests with `laxity < -threshold` are never scheduled (deliberately left to drop). This is a one-phase-up item.

---

### Missing 2: Genuine Hand-Coded Rule Selector

**Status:** MISSING (placeholder exists)  
**Description:** The `RuleBasedSelector` in `selector/models.py` always returns "fifo" and is a placeholder, not a real rule-based policy selector. A genuine version should dispatch to different scheduling algorithms based on observable workload features (e.g., "if queue_depth > 50 and mean_laxity < 2.0 → use LLF; else use EDF").  
**Recommendation:** Implement a `FeatureRuleBasedSelector` with a hand-coded decision tree over workload features. Useful as an interpretable baseline against the RF selector.

---

### Missing 3: CP-SAT / ILP / Tiny Optimal Oracle

**Status:** MISSING  
**Description:** No integer programming or constraint-programming oracle exists. The repo has a greedy hindsight oracle (`oracle_srtf`) but not a provably optimal solver.  
**Why not implemented:** CP-SAT/ILP require `ortools` or `pulp` and are computationally expensive for non-trivial traces. Only useful for very small traces (< 20 requests).  
**Recommendation:** Could be added as an optional `oracle_cpsat` policy gated behind `ortools` install, for micro-benchmark traces only.

---

## Cite-Only Systems (Not Re-implemented)

The following systems are cited in the literature. The repo has "style/inspired" approximations where noted:

| System | Paper | Repo Approximation |
|---|---|---|
| **Orca** | Yu et al., OSDI 2022 | `orca_style` (style/inspired) |
| **vLLM / PagedAttention** | Kwon et al., SOSP 2023 | `vllm_style_token_budget` (style/inspired) |
| **Sarathi-Serve** | Agrawal et al., OSDI 2024 | `sarathi_style` (style/inspired) |
| **DeepSpeed-FastGen / SplitFuse** | Holmes et al., arXiv 2024 | `splitfuse_style` (style/inspired) |
| **PARS** | Zheng et al., NeurIPS 2023 | `estimated_service_time_first` (style/inspired) |
| **DistServe** | Zhong et al., OSDI 2024 | None |
| **Tempo / JITServe** | — | None |
| **Apt-Serve** | — | None |
| **SCORPIO** | — | None |
| **TGI / TensorRT-LLM / SGLang / LMCache** | Various | None |
| **Ray Serve** | Moritz et al., OSDI 2018 | None |

**Wording policy:** We never claim exact reproduction. All serving-system-inspired policies use the "style/inspired" label with the caveat that Phase 1 simplifications (no preemption, instantaneous prefill, flat KV model) make them approximations.

---

## External Code Policy

No external author code was vendored or integrated in this or any prior phase.
All policies are simulator-compatible approximations written from scratch.
No license issues arise.

---

## Checklist Verification (Per Task Specification)

| Baseline from spec | Status | Repo name |
|---|---|---|
| FIFO / FCFS | ✅ Implemented | `fifo` |
| EDF | ✅ Implemented | `edf` |
| LLF / Least Laxity First | ✅ Implemented | `least_laxity_first` |
| SJF / SRPT-style shortest job first | ✅ Implemented | `shortest_output_first`, `weighted_shortest_processing` |
| Shortest predicted output first | ✅ Implemented | `shortest_output_first` |
| Slack-based SLO scheduler | ✅ Implemented | `slo_slack_score` |
| Admission-control baseline | ❌ Missing | — |
| Greedy token-fill batching | ✅ Implemented | `greedy_token_fill` |
| First-fit / Best-fit / Multi-bin batching | ✅ Implemented | `first_fit`, `best_fit`, `multi_bin_batching` |
| Orca-style continuous batching | ✅ Implemented (style/inspired) | `orca_style` |
| vLLM-style token-budget scheduler | ✅ Implemented (style/inspired) | `vllm_style_token_budget` |
| Sarathi-style chunked prefill | ✅ Implemented (style/inspired) | `sarathi_style` |
| SplitFuse-style prefill/decode handling | ✅ Implemented (style/inspired) | `splitfuse_style` |
| PARS-like prompt-aware SJF | ✅ Implemented (style/inspired) | `estimated_service_time_first` |
| KV-cache-constrained online scheduling | ✅ Approximated | `vllm_style_token_budget`, `greedy_token_fill` |
| RF selector | ✅ Implemented | `random_forest` (selector) |
| Decision Tree selector | ✅ Implemented | `decision_tree` (selector) |
| Hand-coded rule selector | ⚠️ Placeholder only | `rule_based` (always returns "fifo") |
| oracle_srtf / hindsight oracle | ✅ Implemented | `oracle_srtf` |
| CP-SAT / IP / tiny oracle | ❌ Missing | — |

---

## Recommended Next Steps

1. **Add dedicated multi-bin batching unit tests** (`tests/test_multi_bin_batching_policy.py`).
2. **Implement `AdmissionControlPolicy`** — laxity-threshold filter wrapping an inner scheduler.
3. **Implement `FeatureRuleBasedSelector`** — genuine hand-coded dispatch rules based on workload features.
4. **Add a `docs/baselines.md`** provenance file listing all wording conventions and cite-only systems.
5. **Evaluate all baselines** on a standard workload set (see `dataset_workload_plan.md`).
6. **Optional:** Add `oracle_cpsat` for micro-benchmark traces (requires `ortools`).

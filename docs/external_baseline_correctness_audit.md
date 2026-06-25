# External Baseline Correctness Audit

**Phase:** 2B.6  
**Date:** 2026-06-25  
**Auditor:** Soroush Vahidi  
**Scope:** Per-policy fidelity and correctness review for all external or literature-inspired baselines.

---

## Audit Criteria

For each policy, we assess:

| Criterion | Description |
|---|---|
| **Algorithm fidelity** | Does the implementation capture the key scheduling insight of the cited system? |
| **Oracle leak** | Does the policy access `actual_output_tokens` (non-deployable oracle field)? |
| **Determinism** | Is the policy deterministic under a fixed seed? |
| **Tie-breaking** | Is there a canonical, deterministic tie-breaking order? |
| **Unit consistency** | Are time/token units handled consistently (steps vs. seconds)? |
| **Safe wording** | Are claims in docs appropriate (no false reproduction claims)? |
| `completion_fraction` | Is the metric now emitted in CSV output? |

---

## Policy Audit Table

### 1. `orca_style`

**Claimed:** Orca-style iteration-level scheduler (Yu et al., OSDI 2022)  
**Key insight captured:** At every decode iteration, greedily admit as many waiting requests as fit within max active sequences, with priority-class ordering + FCFS within class.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Iteration-level greedy admission matches Orca's core loop |
| Oracle leak | ✅ Pass | Uses `predicted_output_tokens` only |
| Determinism | ✅ Pass | Priority class then request_id tie-break |
| Tie-breaking | ✅ Pass | Canonical order in code |
| Unit consistency | ✅ Pass | Sequence count only; no time units |
| Safe wording | ✅ Pass | Docs say "Orca-style" not "official Orca" |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

**Safe claim:** "Orca-style iteration-level admission policy"  
**Unsafe claim:** "Official Orca OSDI 2022 implementation"

---

### 2. `vllm_style_token_budget`

**Claimed:** vLLM-inspired token-budget / paged-KV proxy (Kwon et al., SOSP 2023)  
**Key insight captured:** Per-step token budget; block-granular KV allocation (block=16 tokens, approximating PagedAttention); shortest-predicted-output priority.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Token budget + paged KV proxy |
| Oracle leak | ✅ Pass | Uses `predicted_output_tokens` only |
| Determinism | ✅ Pass | Sorted by predicted output then request_id |
| Tie-breaking | ✅ Pass | Canonical order |
| Unit consistency | ✅ Pass | Token counts; no time units |
| Safe wording | ✅ Pass | Docs say "vLLM-inspired" |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

**Safe claim:** "vLLM-inspired token-budget and paged-KV proxy baseline"  
**Unsafe claim:** "vLLM scheduler" or "PagedAttention reproduction"

---

### 3. `sarathi_style`

**Claimed:** Sarathi-style stall-free chunked-prefill (Agrawal et al., OSDI 2024)  
**Key insight captured:** Decode throughput never blocked by prefill. Limits admitted prompt tokens per step; halves budget when decode work is present.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Chunked-prefill budget respected |
| Oracle leak | ✅ Pass | No actual_output_tokens access |
| Determinism | ✅ Pass | Fixed budget, deterministic sort |
| Tie-breaking | ✅ Pass | FCFS within budget |
| Unit consistency | ✅ Pass | Token counts only |
| Safe wording | ✅ Pass | "Sarathi-style"; O(N²) bug fixed in Phase 1.7C |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

**Performance note:** O(N²) `admitted_ids` set-comprehension bug was fixed in Phase 1.7C (commit 0afb014). Hoisted `admitted_ids: set[int]` outside the inner loop.

**Safe claim:** "Sarathi-style stall-free chunked-prefill baseline"  
**Unsafe claim:** "Official Sarathi-Serve OSDI 2024 implementation"

---

### 4. `splitfuse_style`

**Claimed:** Dynamic-SplitFuse-style chunked-prefill (Holmes et al., arXiv 2024 / DeepSpeed-FastGen)  
**Key insight captured:** Compose each forward pass to exactly fill a fixed token budget. Active decode requests each consume 1 token; remainder goes to new prefill admissions.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | SplitFuse composition logic present |
| Oracle leak | ✅ Pass | No oracle field access |
| Determinism | ✅ Pass | Budget-fill is deterministic |
| Tie-breaking | ✅ Pass | Request_id tie-break |
| Unit consistency | ✅ Pass | Token budget only |
| Safe wording | ✅ Pass | "Dynamic-SplitFuse-style inspired by DeepSpeed-FastGen" |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

**Safe claim:** "Dynamic-SplitFuse-style chunked-prefill baseline inspired by DeepSpeed-FastGen"  
**Unsafe claim:** "DeepSpeed-FastGen or MII reproduction"

---

### 5. `multi_bin_batching`

**Claimed:** Multi-Bin batching (groups by output-length bins)  
**Key insight captured:** Bin requests by predicted output length; prefer filling current-bin groups before starting new ones.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Bin grouping by predicted output length |
| Oracle leak | ✅ Pass | Uses `predicted_output_tokens` only |
| Determinism | ✅ Pass | Fixed bin boundaries, deterministic sort |
| Tie-breaking | ✅ Pass | Bin index then request_id |
| Unit consistency | ✅ Pass | Token count bins |
| Safe wording | ✅ Pass | No specific paper claim; "Multi-Bin-style" |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |
| Dedicated unit tests | ⚠️ Missing | Registry presence tested; no `test_multi_bin_batching_policy.py` |

---

### 6. `estimated_service_time_first`

**Claimed:** Prompt-and-prediction-aware SJF proxy (PARS-inspired)  
**Key insight captured:** Sort by estimated total service time (`α×prompt + β×output`); not a learning-to-rank system.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Est. service time formula is clean SJF proxy |
| Oracle leak | ✅ Pass | Uses `predicted_output_tokens` |
| Determinism | ✅ Pass | Est. time → deadline → priority → request_id |
| Tie-breaking | ✅ Pass | Canonical 4-key sort |
| Unit consistency | ⚠️ Caution | `α`, `β` weights produce step-counts; must match calibration |
| Safe wording | ✅ Pass | "Prompt-and-prediction-aware SJF proxy, not a PARS reproduction" |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

**Safe claim:** "Prompt-and-prediction-aware SJF proxy based on estimated prefill and decode service time. Not a reproduction of PARS, which uses prompt-aware learning-to-rank."

---

### 7. `least_laxity_first`

**Claimed:** Least Laxity First (LLF) deadline-aware baseline  
**Key insight captured:** Schedule by remaining laxity (`deadline − now − est_service_time`); handles preemption-risk cases that EDF misses.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Laxity formula correct |
| Oracle leak | ✅ Pass | Uses `predicted_output_tokens` |
| Determinism | ✅ Pass | Laxity → deadline → priority → request_id |
| Tie-breaking | ✅ Pass | Canonical 4-key sort |
| Unit consistency | ⚠️ Caution | Service proxy in steps; deadline in seconds — same issue as admission_control |
| Safe wording | ✅ Pass | "LLF deadline-aware baseline" |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

---

### 8. `admission_control`

**Claimed:** Laxity-based admission-control scheduling baseline (new in Phase 2B.5)  
**Key insight captured:** Filter requests with negative laxity beyond a threshold; sort survivors by urgency.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Laxity filter + urgency sort |
| Oracle leak | ✅ Pass | No actual_output_tokens |
| Determinism | ✅ Pass | Deterministic sort with canonical tie-break |
| Tie-breaking | ✅ Pass | laxity → priority → est → deadline → request_id |
| Unit consistency | ⚠️ **Known gap** | Service proxy in decode steps; SLO deadline in seconds. Default threshold=inf (no filtering) deliberately sidesteps mismatch. Calibration doc pending. |
| Safe wording | ✅ Pass | "Not a reproduction of Tempo, JITServe, or SCORPIO" |
| `completion_fraction` | ✅ Pass | Emitted via RunMetrics |

**Action item:** See `docs/audits/admission_control_threshold_calibration_summary.md` for threshold calibration guidance.

---

### 9. `greedy_token_fill`

**Claimed:** Best-fit KV capacity assignment  
**Key insight captured:** Greedily fill each GPU's KV capacity bin with highest-priority requests that fit.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Greedy best-fit across GPUs |
| Oracle leak | ✅ Pass | Token counts only |
| Determinism | ✅ Pass | Priority → request_id tie-break |
| Tie-breaking | ✅ Pass | Stable |
| Unit consistency | ✅ Pass | KV token counts |
| Safe wording | ✅ Pass | No false paper claims |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

---

### 10. `slo_slack_score`

**Claimed:** Urgency + service time + priority + wait composite  
**Key insight captured:** A composite score combining normalized SLO slack, estimated service time, priority class, and waiting time.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | Multi-factor composite |
| Oracle leak | ✅ Pass | Uses predicted values only |
| Determinism | ✅ Pass | Score → request_id tie-break |
| Tie-breaking | ✅ Pass | Canonical |
| Unit consistency | ✅ Pass | All terms normalized |
| Safe wording | ✅ Pass | No paper claim |
| `completion_fraction` | ✅ Pass | Added in Phase 2B.5 |

---

### 11. `oracle_srtf`

**Claimed:** Hindsight SRTF oracle — non-deployable upper bound  
**Key insight captured:** Shortest Remaining Time First using actual (not predicted) output tokens.

| Criterion | Status | Notes |
|---|---|---|
| Algorithm fidelity | ✅ Pass | True SRTF using actual remaining tokens |
| Oracle leak | **Expected — oracle only** | Access to `actual_output_tokens` is intentional and documented |
| Determinism | ✅ Pass | Deterministic under fixed trace |
| Tie-breaking | ✅ Pass | Canonical |
| Not in BASELINE_NAMES | ✅ Pass | Only in `ORACLE_POLICY_NAMES` |
| Never a selector candidate | ✅ Pass | Excluded from SELECTOR_CANDIDATES at import time |
| UserWarning at construction | ✅ Pass | Emits warning on instantiation |
| Access guard | ✅ Pass | Only via `make_oracle_policy()` |
| `completion_fraction` | ✅ Pass | Emitted via RunMetrics |

**Safe claim:** "Hindsight SRTF oracle — uses actual output tokens; non-deployable upper bound for performance analysis only."

---

## Summary

| Policy | Oracle Leak | Deterministic | Safe Wording | Unit Consistency | Completion Fraction | Overall |
|---|---|---|---|---|---|---|
| `orca_style` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Pass |
| `vllm_style_token_budget` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Pass |
| `sarathi_style` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Pass |
| `splitfuse_style` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Pass |
| `multi_bin_batching` | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ Missing unit tests |
| `estimated_service_time_first` | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ Pass (caution on α/β) |
| `least_laxity_first` | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ Pass (caution on units) |
| `admission_control` | ✅ | ✅ | ✅ | ⚠️ | ✅ | ⚠️ Threshold calibration needed |
| `greedy_token_fill` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Pass |
| `slo_slack_score` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Pass |
| `oracle_srtf` | expected | ✅ | ✅ | ✅ | ✅ | ✅ Pass (non-deployable) |

### Open action items

1. **Add `test_multi_bin_batching_policy.py`** — dedicated unit tests for `multi_bin_batching`.
2. **Calibrate `admission_control` threshold** — see `docs/audits/admission_control_threshold_calibration_summary.md`.
3. **Document α/β units** — add a note to `estimated_service_time_first` and `least_laxity_first` about step-unit service proxy.

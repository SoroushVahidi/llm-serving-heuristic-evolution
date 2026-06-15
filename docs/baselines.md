# Baseline Policies

All policies implement `BasePolicy` in `src/llmserveopt/policies/base.py`.
None require external dependencies beyond NumPy.

---

## Registered online baselines (18 policies)

These policies are registered in `src/llmserveopt/policies/registry.py` and are
used in all experiment comparisons. All are deployable in an online setting.
All 18 are also valid **selector candidates** (`SELECTOR_CANDIDATE_NAMES`).

| Policy | Category | Online deployable? | Uses prediction? | Uses SLO/deadline? | Uses KV/token budget? | Notes |
|---|---|---|---|---|---|---|
| `fifo` | Classical | Yes | No | No | No | Oldest request first; round-robin GPU |
| `edf` | Classical | Yes | No | Yes | No | Earliest Deadline First |
| `shortest_output_first` | SRPT-style | Yes | Yes (predicted) | No | No | Approximates SRPT; uses predicted output length |
| `shortest_prompt_first` | Heuristic | Yes | No | No | No | Shortest prompt = smallest KV footprint |
| `greedy_token_fill` | Packing | Yes | No | No | Yes (KV) | Best-fit KV capacity assignment |
| `least_loaded` | Load balancing | Yes | No | No | No | Assign to GPU with fewest active sequences |
| `multi_bin_batching` | Batching | Yes | Yes (predicted) | No | No | Groups by output-length bins; Multi-Bin-style |
| `random_feasible` | Stochastic | Yes | No | No | No | Random feasible admission; deterministic under seed |
| `first_fit` | Packing | Yes | No | No | Yes (KV) | First-fit bin packing across GPUs |
| `best_fit` | Packing | Yes | No | No | Yes (KV) | Best-fit bin packing (tightest-fit) across GPUs |
| `orca_style` | Serving-style | Yes | No | No | Yes (seq count) | Orca-style iteration-level scheduler |
| `vllm_style_token_budget` | Serving-style | Yes | Yes (predicted) | No | Yes (token budget + paged KV) | vLLM-inspired token-budget / paged-KV proxy |
| `sarathi_style` | Serving-style | Yes | No | No | Yes (chunk budget) | Sarathi-style stall-free chunked-prefill |
| `splitfuse_style` | Serving-style | Yes | No | No | Yes (token budget) | Dynamic-SplitFuse-style chunked-prefill |
| `slo_slack_score` | Composite | Yes | Yes (predicted) | Yes | No | Urgency + service time + priority + wait composite |
| `weighted_shortest_processing` | Composite | Yes | Yes (predicted) | No | No | WSPT priority × predicted processing time |
| `least_laxity_first` | Deadline/laxity | Yes | Yes (predicted) | Yes | No | LLF: deadline − now − estimated_service_time; handles preemption-risk cases that EDF misses |
| `estimated_service_time_first` | SJF proxy | Yes | Yes (predicted) | No | No | Prompt-and-prediction-aware SJF proxy (α×prompt + β×output). Not a PARS reproduction — no learning. |

---

## Non-deployable / oracle policies

The oracle is maintained separately in `ORACLE_POLICY_NAMES` and must never
appear in `BASELINE_NAMES` or `SELECTOR_CANDIDATE_NAMES`.

| Policy | File | Online deployable? | Notes |
|---|---|---|---|
| `oracle_srtf` | `oracle.py` | **No — hindsight oracle** | Uses actual (not predicted) output lengths. Non-deployable upper-bound candidate. Always emits `UserWarning` at construction. Use only as benchmark ceiling; label clearly as "hindsight upper bound" in all reports. Access via `make_oracle_policy()`, not `make_policy()`. |
| `earliest_feasible_gpu` | `earliest_feasible_gpu.py` | Yes (candidate) | Assign to the GPU that can start the request earliest; not yet registered |

---

## Phase 2A.3B hardened baselines

### Least Laxity First (`least_laxity_first`)

**Manuscript label:** "Least Laxity First (LLF) deadline-aware baseline"

Laxity is the remaining slack after accounting for estimated service time:

```
laxity_i = deadline_i − current_time − estimated_remaining_service_time_i
estimated_service_time_i = α × prompt_tokens_i + β × predicted_output_tokens_i
```

LLF is strictly more responsive to service-time uncertainty than EDF. A request
that will almost certainly miss its deadline (large service time, tight deadline)
gets higher priority even if its absolute deadline is later than another request.

- **actual_output_tokens**: never accessed — uses `predicted_output_tokens`.
- **Tie-breaking**: lower laxity → earlier deadline → higher priority → lower request_id.
- **Not an oracle**: relies only on online-observable estimates.

### Estimated Service Time First (`estimated_service_time_first`)

**Manuscript label:** "Prompt-and-prediction-aware SJF proxy"

A PARS-inspired baseline that approximates Shortest Job First using estimated
service time:

```
estimated_service_time_i = α × prompt_tokens_i + β × predicted_output_tokens_i
```

**IMPORTANT — do not conflate with PARS**: PARS (Prototype-Aware Request Scheduling)
uses prompt-aware learning-to-rank to estimate service time from prompt semantics.
This policy uses only token-length estimates and does not learn from data.

Safe wording: "prompt-and-prediction-aware SJF proxy based on estimated prefill and
decode service time. Not a reproduction of PARS, which uses prompt-aware learning-to-rank."

- **actual_output_tokens**: never accessed.
- **Tie-breaking**: lower estimated service time → earlier deadline → higher priority → lower request_id.

---

## Serving-style baseline provenance

Each serving-style baseline captures the **key scheduling insight** of the cited system.
None are reproductions of the original system's code.

### Orca-style (`orca_style`)

**Manuscript label:** "Orca-style iteration-level scheduler"

Reference: Yu et al., "Orca: A Distributed Serving System for Transformer-Based
Generative Models," OSDI 2022.

Key idea: at every decode iteration, greedily admit as many waiting requests as fit
within capacity, with priority-class ordering + FCFS within class.

- **Safe claim:** "Orca-style iteration-level admission policy"
- **Unsafe claim:** "Official Orca OSDI 2022 implementation"

### vLLM-inspired (`vllm_style_token_budget`)

**Manuscript label:** "vLLM-inspired token-budget / paged-KV proxy baseline"

Reference: Kwon et al., "Efficient Memory Management for Large Language Model Serving
with PagedAttention," SOSP 2023.

Key ideas: per-step token budget; block-granular KV allocation (default block 16 tokens,
approximating vLLM's page size); shortest-predicted-output priority within budget.

- **Safe claim:** "vLLM-inspired token-budget and paged-KV proxy baseline"
- **Unsafe claim:** "vLLM scheduler" or "PagedAttention reproduction"

### Sarathi-style (`sarathi_style`)

**Manuscript label:** "Sarathi-style stall-free chunked-prefill baseline"

Reference: Agrawal et al., "Sarathi: Efficient LLM Inference by Piggybacking Decodes
with Chunked Prefills," arXiv 2023; Sarathi-Serve, OSDI 2024.

Key idea: decode throughput is never blocked by prefill. Limits admitted prompt tokens
per step to `max_prefill_tokens_per_step`; halves the budget when decode work is present.

**Performance note:** The O(N²) set-comprehension bug was fixed in Phase 1.7C
(commit 0afb014). Hoisted `admitted_ids: set[int]` outside the inner loop.

- **Safe claim:** "Sarathi-style stall-free chunked-prefill baseline"
- **Unsafe claim:** "Official Sarathi-Serve OSDI 2024 implementation"

### Dynamic-SplitFuse-style (`splitfuse_style`)

**Manuscript label:** "Dynamic-SplitFuse-style chunked-prefill baseline inspired by DeepSpeed-FastGen"

Reference: Holmes et al., "DeepSpeed-FastGen: High-Throughput Text Generation for LLMs
via MII and DeepSpeed-Inference," arXiv 2024.

Key idea: compose each forward pass to exactly fill a fixed token budget. Active decode
requests each consume 1 token; remainder goes to new prefill admissions.

- **Safe claim:** "Dynamic-SplitFuse-style chunked-prefill baseline inspired by DeepSpeed-FastGen"
- **Unsafe claim:** "DeepSpeed-FastGen or MII reproduction"

---

## Dispatch vs. batching

Most policies handle both **dispatch** (which GPU) and **batching** (which requests
to admit in a single step) simultaneously. In the Phase 1/1.5 simulator, dispatching
and batching decisions are made atomically per step. Future phases may separate these.

---

## Missing from Phase 1 / 1.5

- Preemption-based policies (LAS, SJF-with-preemption)
- SLO-aware preemptive EDF
- Feedback-control policies (admission rate throttling)
- Prefix-cache-aware scheduling
- LLM-generated heuristics (Phase 2+)

# Baseline Policies

All policies implement `BasePolicy` in `src/llmserveopt/policies/base.py`.
None require external dependencies beyond NumPy.

---

## Registered online baselines (14 policies)

These policies are registered in `src/llmserveopt/policies/registry.py` and are
used in all experiment comparisons. All are deployable in an online setting.

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
| `orca_style` | Serving-style | Yes | No | No | Yes (seq count) | Orca-style iteration-level scheduler |
| `vllm_style_token_budget` | Serving-style | Yes | Yes (predicted) | No | Yes (token budget + paged KV) | vLLM-inspired token-budget / paged-KV proxy |
| `sarathi_style` | Serving-style | Yes | No | No | Yes (chunk budget) | Sarathi-style stall-free chunked-prefill |
| `splitfuse_style` | Serving-style | Yes | No | No | Yes (token budget) | Dynamic-SplitFuse-style chunked-prefill |
| `slo_slack_score` | Composite | Yes | Yes (predicted) | Yes | No | Urgency + service time + priority + wait composite |
| `weighted_shortest_processing` | Composite | Yes | Yes (predicted) | No | No | WSPT priority × predicted processing time |

---

## Unregistered / non-deployable policies

These exist in `src/llmserveopt/policies/` but are **not** in the standard
experiment comparison set.

| Policy | File | Online deployable? | Notes |
|---|---|---|---|
| `oracle_srtf` | `oracle.py` | **No — hindsight oracle** | Uses actual (not predicted) output lengths; non-deployable upper-bound candidate. Always emits a `UserWarning` at construction to prevent accidental use. **Do not include in online-policy comparison reports.** |
| `first_fit` | `first_fit.py` | Yes (candidate) | First-fit KV bin assignment; not yet registered |
| `best_fit` | `best_fit.py` | Yes (candidate) | Best-fit KV bin assignment; not yet registered |
| `earliest_feasible_gpu` | `earliest_feasible_gpu.py` | Yes (candidate) | Assign to the GPU that can start the request earliest; not yet registered |

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

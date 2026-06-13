# Baseline Policies

All policies implement `BasePolicy` in `policies/base.py` and are fully contained
within this repository.  None require external dependencies beyond NumPy.

---

## Phase 1 classical baselines

### 1. FIFO (`fifo`)

Admit requests in arrival order (oldest first).  Round-robin across GPUs.

- **Type**: Faithful classical baseline
- **Provenance**: Standard queue-theoretic policy; no external code

### 2. EDF (`edf`)

Earliest Deadline First: admit requests in order of increasing `slo_deadline`.

- **Type**: Faithful classical baseline
- **Provenance**: Classic real-time scheduling; standard algorithm
- **Note**: Optimal for single-machine feasibility testing but not for throughput

### 3. Shortest Output First (`shortest_output_first`)

Sort waiting requests by `predicted_output_tokens` ascending; admit shortest first.
Approximates SRPT (Shortest Remaining Processing Time).

- **Type**: Approximate SRPT variant (uses predicted, not actual, output length)
- **Provenance**: SRPT is a classical scheduling algorithm

### 4. Shortest Prompt First (`shortest_prompt_first`)

Sort by `prompt_tokens` ascending.  Prefers requests with smaller KV footprint.

- **Type**: Heuristic
- **Provenance**: Original implementation for this benchmark

### 5. Greedy Token Fill (`greedy_token_fill`)

For each waiting request (FIFO order), assign it to the GPU with the most
remaining KV capacity that can still admit it.

- **Type**: Heuristic (packing-style)
- **Provenance**: Original implementation; inspired by bin-packing ideas

### 6. Least Loaded (`least_loaded`)

Dispatch each request to the GPU with the fewest active sequences.

- **Type**: Load-balancing heuristic
- **Provenance**: Standard load-balancing strategy

### 7. Multi-Bin-style Batching (`multi_bin_batching`)

Group requests by predicted output length into discrete bins (e.g., ≤32, ≤64, ≤128,
≤256, >256 tokens).  Fill GPUs preferentially from one bin to reduce intra-batch
length variance.

- **Type**: Approximate style baseline — NOT an official implementation
- **Provenance**: Inspired by the Multi-Bin Batching idea from the LLM serving
  literature; this is our own simplified adaptation.  It is **not** the official
  implementation of any published paper.
- **Safe claim**: "Multi-Bin-style baseline implemented for comparison purposes"
- **Unsafe claim**: "Official Multi-Bin Batching from [paper]"

### 8. Random Feasible (`random_feasible`)

Randomly permute waiting requests (using a seeded RNG), then greedily admit.
Deterministic under fixed `seed`.

- **Type**: Stochastic baseline (deterministic under seed)
- **Provenance**: Original implementation

### 9. Oracle SRTF (`oracle_srtf`)

Uses **actual** output lengths (ground truth, not predictions) to sort requests
by true remaining work.  Greedy multi-GPU variant of SRTF.

- **Type**: Hindsight oracle — **NOT deployable**
- **Provenance**: Original implementation
- **Warning**: `OracleShortestJobFirstPolicy` always emits a `UserWarning` at
  construction time to prevent accidental use in production comparisons.
- **Safe claim**: "Hindsight greedy SRTF oracle for small-trace upper-bound estimation"
- **Unsafe claim**: "Globally optimal policy" or "deployable policy"

---

## Phase 1.5 serving-style baselines

These policies model scheduling behaviors inspired by published LLM serving systems.
Each is an **original implementation** designed to capture the key scheduling insight of
the referenced system; none are reproductions of the original system's code.

### 10. Orca-style iteration-level scheduler (`orca_style`)

**Manuscript label**: "Orca-style iteration-level scheduler"
**NOT** an official Orca reproduction.

Reference: Yu et al., "Orca: A Distributed Serving System for
Transformer-Based Generative Models," OSDI 2022.

Key idea: at every decode iteration, greedily admit as many waiting requests as fit
within capacity, with priority-class ordering + FCFS within class.  All admitted and
in-progress requests run together in the next step.

- **Safe claim**: "Orca-style iteration-level admission policy"
- **Unsafe claim**: "Official Orca OSDI 2022 implementation"

### 11. vLLM-inspired token-budget / paged-KV proxy (`vllm_style_token_budget`)

**Manuscript label**: "vLLM-inspired token-budget / paged-KV proxy baseline"
**NOT** a vLLM reproduction or vLLM performance benchmark.

Reference: Kwon et al., "Efficient Memory Management for Large Language
Model Serving with PagedAttention," SOSP 2023.

Key ideas captured: (1) per-step token budget caps active sequences; (2) KV cache
allocation uses block-granular rounding (default block size 16 tokens, approximating
vLLM's page size); (3) shortest-predicted-output priority within token budget.

- **Safe claim**: "vLLM-inspired token-budget and paged-KV proxy baseline"
- **Unsafe claim**: "vLLM scheduler" or "PagedAttention reproduction"

### 12. Sarathi-style stall-free chunked-prefill (`sarathi_style`)

**Manuscript label**: "Sarathi-style stall-free chunked-prefill baseline"
**NOT** an official Sarathi-Serve reproduction.

Reference: Agrawal et al., "Sarathi: Efficient LLM Inference by Piggybacking
Decodes with Chunked Prefills," arXiv 2023; Sarathi-Serve, OSDI 2024.

Key idea: decode throughput is never blocked by prefill.  This implementation
limits admitted prompt tokens per step to `max_prefill_tokens_per_step`, and halves
the budget when existing decode work is present (decode-first).

When `enable_prefill_modeling=True` in `ServiceModel`, pair with `decode_first=True`
for the most faithful simulation of stall-free chunked prefill.

- **Safe claim**: "Sarathi-style stall-free chunked-prefill baseline"
- **Unsafe claim**: "Official Sarathi-Serve OSDI 2024 implementation"

### 13. Dynamic-SplitFuse-style (`splitfuse_style`)

**Manuscript label**: "Dynamic-SplitFuse-style chunked-prefill baseline inspired by DeepSpeed-FastGen"
**NOT** an official DeepSpeed-FastGen reproduction.

Reference: Holmes et al., "DeepSpeed-FastGen: High-Throughput Text Generation
for LLMs via MII and DeepSpeed-Inference," arXiv 2024.

Key idea: compose each forward pass to exactly fill a fixed token budget.  Active
decode requests each consume 1 token; the remainder goes to new prefill admissions.

- **Safe claim**: "Dynamic-SplitFuse-style chunked-prefill baseline inspired by DeepSpeed-FastGen"
- **Unsafe claim**: "DeepSpeed-FastGen or MII reproduction"

### 14. SLO-slack composite score (`slo_slack_score`)

**Manuscript label**: "SLO-slack composite scoring policy"

Combines urgency (reciprocal of deadline slack), predicted service time, priority,
and waiting time into a single admission score.  Requests with the worst deadline
urgency are admitted first.

- **Type**: Original composite heuristic; no external paper claimed
- **Provenance**: Original implementation using `policies/scoring.py` utilities

---

## Dispatch vs. batching

Most policies handle both **dispatch** (which GPU) and **batching**
(which requests to admit in a single step) simultaneously.  In the Phase 1/1.5
simulator, dispatching and batching decisions are made atomically per step.
Future phases may separate these concerns.

## Missing from Phase 1 / 1.5

- Preemption-based policies (e.g., LAS — Least Attained Service)
- SLO-aware preemptive EDF
- Feedback-control policies (e.g., admission rate throttling)
- LLM-generated heuristics (Phase 2+)

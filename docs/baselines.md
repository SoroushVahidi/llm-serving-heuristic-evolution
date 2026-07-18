# Baseline Policies

All policies implement `BasePolicy` in `src/llmserveopt/policies/base.py`.
None require external dependencies beyond NumPy.

---

## Registered online baselines (20 policies)

These policies are registered in `src/llmserveopt/policies/registry.py` and are
used in all experiment comparisons. All are deployable in an online setting.
All 20 are also valid **selector candidates** (`SELECTOR_CANDIDATE_NAMES`).
Primary comparison metric: **ANWG** (`arrival_normalized_weighted_goodput`), with
weighted-goodput aliases retained only for compatibility with older outputs.

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
| `admission_control` | Admission control | Yes | Yes (predicted) | Yes | Yes | Laxity-based filter + urgency sort. See Phase 2B.5 note below. |
| `scorpio_style_slo_guard` | SCORPIO-inspired SLO guard | Yes | Yes (predicted) | Yes | Yes | SCORPIO-style TTFT/TPOT guard + credit throttling. See Phase 2B.10 note below. |

---

## Non-deployable / oracle policies

The oracle is maintained separately in `ORACLE_POLICY_NAMES` and must never
appear in `BASELINE_NAMES` or `SELECTOR_CANDIDATE_NAMES`.

`safe_fallback_wsp_margin*` variants are oracle-assisted analysis helpers, not
registered baselines, not selector candidates, and not deployable.

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
None are reproductions of the original system's code; all external-style policies in
this repository are internal simulator approximations built for controlled comparison.

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

## Faithful (non-proxy) baselines: `vllm_faithful`

This repository has **three distinct kinds of "vLLM" thing**, and they must
never be conflated:

| | `vllm_style_token_budget` | `vllm_faithful` | External real-vLLM HTTP harness |
|---|---|---|---|
| What it is | Lightweight proxy/inspired heuristic | Faithful independent reimplementation of a pinned vLLM scheduler version | Client-side admission control in front of a REAL running vLLM server |
| Where | `src/llmserveopt/policies/vllm_style_token_budget.py` | `src/llmserveopt/policies/vllm_faithful.py` | `scripts/run_vllm_external_baseline_comparison.py` |
| Fidelity claim | "vLLM-inspired" only | Algorithm/memory-semantics fidelity to a **named, pinned commit** | Real system, but vLLM's internal scheduler is a black box |
| Registered baseline? | Yes (one of the 20) | **No — see below** | N/A (not a simulator policy) |

### `vllm_faithful`

**Manuscript label:** "Faithful independent reimplementation of vLLM's
original (pre-chunked-prefill) FCFS scheduler and paged-KV block manager"

**Pinned reference:** vLLM commit `67d96c29fba9b72cb4c4edbc26211c208a00ebdd`
(tag `v0.1.0`), corresponding to Kwon et al., "Efficient Memory Management
for Large Language Model Serving with PagedAttention," SOSP 2023
(arXiv:2309.06180). Full source-provenance record, algorithm summary, and
explicit exclusions: `docs/vllm_faithful_scheduler_reference.md`.

Unlike `vllm_style_token_budget` (a heuristic *inspired by* the token-budget
idea), `vllm_faithful` reimplements the pinned reference's actual scheduling
algorithm: three request-group queues (waiting/running/swapped), FCFS
per-iteration admission and preemption bounded by fixed-size KV blocks with
a watermark reserve, and recompute-based preemption (discard-and-restart)
for the lowest-priority running sequence when block capacity runs out.

- **Safe claim:** "Faithful reimplementation of vLLM v0.1.0's FCFS scheduler
  and paged-KV block manager's scheduling/memory *decisions*."
- **Unsafe claim:** "Official vLLM code", "exact runtime reproduction", "a
  full vLLM performance/hardware-timing model", or a claim about vLLM's
  *current* scheduler (chunked prefill, prefix caching, the v1 engine, etc.
  postdate this pin and are not represented).

**Not currently a registered baseline.** `vllm_faithful` is fully
implemented, unit-tested, and directly importable
(`llmserveopt.policies.vllm_faithful.VLLMFaithfulPolicy`), but is
deliberately **not** added to `registry.py`'s `BASELINE_NAMES` /
`SELECTOR_CANDIDATE_NAMES` in the PR that introduced it. Doing so would
silently change the deployable-policy count (currently 20 — see
`docs/research_status.md`) and the selector's candidate pool, with real
downstream effects (selector retraining, evaluation-sweep counts, every doc
that states "20 deployable policies") that were out of scope for
introducing this baseline. Promoting it to a selectable/deployable baseline
is a deliberate follow-up decision for a future PR.

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

---

## Phase 2B.5: Admission Control Baseline

### `admission_control` — Laxity-filtered admission-control baseline

**Manuscript label:** "Laxity-based admission-control scheduling baseline"

**IMPORTANT:** This is NOT a reproduction of Tempo, JITServe, SCORPIO, or any
other published admission-control system.  It is a simple deterministic baseline
designed to isolate the admission-control effect in simulation.

**Algorithm (Phase 2B.7 unit-corrected):**
1. Compute estimated service time in seconds:
   `est_s = step_size × (α × prompt_tokens + β × predicted_output_tokens)`
2. Compute laxity in seconds: `laxity = slo_deadline − now − est_s`
3. Filter: skip requests with `laxity < −laxity_threshold`
4. Sort survivors: laxity ↑ → priority ↓ → est_s ↑ → deadline ↑ → request_id ↑
5. Greedily assign to GPUs with capacity

**Parameters:**
- `laxity_threshold` (seconds, default `float("inf")`): filter threshold. 0.0 = admit only
  requests whose estimated service time fits within remaining deadline.
- `step_size` (seconds/step, default 0.001): simulator step duration for unit conversion.
- `alpha`, `beta`: service proxy weights.

**Default threshold:** `float("inf")` (no filtering; acts as urgency-sorted admission).

**Unit fix (Phase 2B.7):** Prior to Phase 2B.7, laxity mixed seconds and decode steps.
This is now corrected: `est_s = step_size × est_steps` ensures all terms are in seconds.
`laxity_threshold=0.0` now correctly drops requests infeasible within their deadline.

**Safe claim:** "Laxity-based admission-control scheduling baseline"  
**Unsafe claim:** "Reproduction of Tempo, JITServe, or SCORPIO"

**Phase 2B.7 sweep result (laxity_threshold=inf):**
- Wins: `high_prediction_noise` workload (WG=0.988, rank 1/19)
- Loses: `kv_pressure_decode_heavy` (WG=0.051, rank 19/19 — urgency sorting ineffective under KV saturation)

---

## Phase 2B.10: SCORPIO-Style SLO Guard Baseline

### `scorpio_style_slo_guard` — SCORPIO-inspired SLO guard

**Manuscript label:** "SCORPIO-style SLO guard" or "SCORPIO-inspired TTFT/TPOT guard baseline"

**IMPORTANT:** This is NOT an official SCORPIO reproduction. It is a deterministic,
simulator-compatible approximation of SCORPIO's policy-level admission and guard ideas.

**Algorithm (summary):**
1. Filter infeasible requests by laxity and TTFT proxy slack (seconds).
2. Detect guard mode from KV utilization, decode pressure, queue overload, or negative mean laxity.
3. Under guard mode: throttle admissions via refilling credit budget; defer long decode under KV pressure.
4. Rank survivors by composite urgency score (laxity, priority, age, decode-pressure penalty).
5. Tie-break: laxity ↑ → priority ↓ → arrival ↑ → request_id ↑.
6. Greedily assign to feasible GPUs.

**TTFT/TPOT:** scheduling-time **proxies** only (see `docs/audits/phase2b10_scorpio_slo_guard_summary.md`).

**Safe claim:** "SCORPIO-inspired SLO guard baseline"  
**Unsafe claim:** "Official SCORPIO reproduction"

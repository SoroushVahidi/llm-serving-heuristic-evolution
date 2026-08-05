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

## Faithful (non-proxy) baselines: `vllm_faithful`, `vllm_chunked_prefill_faithful`, `sarathi_faithful`, `distserve_faithful`, `tetriinfer_paper_reimplementation`, `llumnix_faithful`

This repository has **four distinct kinds of "vLLM" thing**, an
analogous **two distinct kinds of "Sarathi" thing**, and a distinct
**"DistServe" thing**, **"TetriInfer" thing**, and **"Llumnix" thing** —
none of which may be conflated. These six span three structurally
different GPU topologies (monolithic, disaggregated prefill/decode,
multi-instance migratory) and are **not** all directly comparable to each
other without an explicit resource-normalization protocol — see
[external_baseline_integration.md](external_baseline_integration.md) for
the full topology-comparability matrix, resource-normalization protocols,
and selector-eligibility analysis across all six together.

| | `vllm_style_token_budget` | `vllm_faithful` | External real-vLLM HTTP harness |
|---|---|---|---|
| What it is | Lightweight proxy/inspired heuristic | Faithful independent reimplementation of a pinned vLLM scheduler version | Client-side admission control in front of a REAL running vLLM server |
| Where | `src/llmserveopt/policies/vllm_style_token_budget.py` | `src/llmserveopt/policies/vllm_faithful.py` | `scripts/run_vllm_external_baseline_comparison.py` |
| Fidelity claim | "vLLM-inspired" only | Algorithm/memory-semantics fidelity to a **named, pinned commit** | Real system, but vLLM's internal scheduler is a black box |
| Registered baseline? | Yes (one of the 20) | **No — see below** | N/A (not a simulator policy) |

| | `sarathi_style` | `sarathi_faithful` |
|---|---|---|
| What it is | Lightweight admission-rate proxy heuristic (no separate prefill phase modeled) | Faithful independent reimplementation of a pinned Sarathi-Serve scheduler version |
| Where | `src/llmserveopt/policies/sarathi_style.py` | `src/llmserveopt/policies/sarathi_faithful.py` |
| Fidelity claim | "Sarathi-style" only | Algorithm/memory-semantics fidelity to a **named, pinned commit** |
| Requires `enable_prefill_modeling=True`? | No (Phase 1 admission-rate heuristic) | Only to make chunked-prefill *behavior* observable; runs without crashing either way |
| Registered baseline? | Yes (one of the 20) | **No — see below** |

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

### `vllm_chunked_prefill_faithful`

**Manuscript label:** "Faithful independent reimplementation of pinned
vLLM v0.4.2 scheduler/chunked-prefill/block_manager_v1 decisions"

**Pinned reference:** vLLM commit `c7f2cf2b7f67bce5842fedfdba508440fe257375`
(tag `v0.4.2`), the first vLLM release whose own notes call chunked prefill
"ready for testing." Full source-provenance record, algorithm summary, and
explicit exclusions (`block_manager_v2`, prefix caching, speculative
decoding, LoRA, `delay_factor`):
`docs/vllm_chunked_prefill_faithful_scheduler_reference.md`. Preliminary
design audit (pin selection rationale, infrastructure survey):
`docs/vllm_chunked_prefill_faithful_design_audit.md`.

Separately pinned from `vllm_faithful` (v0.1.0, no chunked prefill at all)
— this baseline models vLLM's later `_schedule_chunked_prefill` path: a
shared per-iteration `SchedulingBudget`-equivalent, partial/chunked prompt
admission instead of all-or-nothing, and (verified against the pinned
source, not assumed) **no explicit decode-priority phase** — decode-phase
and continuing-prefill-phase requests share one FCFS-by-arrival-time budget
in `_schedule_running`, structurally different from `sarathi_faithful`'s
own explicit decode-first Phase 1a/1b split. `block_manager_v1` (the
default at this pin; `use_v2_block_manager=False`) is reused unchanged from
`vllm_faithful`'s own `KVBlockSpaceManager`.

- **Safe claim:** "Faithful reimplementation of vLLM v0.4.2's chunked-
  prefill scheduling *decisions* (admission, chunking, preemption)."
  Completes the `xlong_context_burst16` benchmark-pack fixture (16
  ~12,000-token-prompt requests) that `vllm_faithful` structurally cannot
  admit at all — this baseline's direct acceptance-test target.
- **Unsafe claim:** that this baseline alone reproduces the real, robust
  (N=5) Sarathi E2E advantage observed on `active_decode_plus_arriving_
  prefill`/`kv_pressure` — it does not; both remain a `TIE_NEAR_TIE` under
  a fair (equal-`ServiceModel`) comparison against `sarathi_faithful`, for
  a specific, root-caused reason unrelated to this baseline's own
  scheduling fidelity: see
  `docs/vllm_chunked_prefill_faithful_root_cause_analysis.md`.

**Not currently a registered baseline** — same rationale and mechanism as
`vllm_faithful`/`sarathi_faithful`: present in `EXTERNAL_BASELINE_REGISTRY`
(`llmserveopt.policies.vllm_chunked_prefill_faithful.
VLLMChunkedPrefillFaithfulPolicy`), never in `registry.py`'s
`BASELINE_NAMES`/`SELECTOR_CANDIDATE_NAMES`.

### `sarathi_faithful`

**Manuscript label:** "Faithful independent reimplementation of Sarathi-
Serve's stall-free chunked-prefill scheduler"

**Pinned reference:** microsoft/sarathi-serve, branch `osdi-sarathi-serve`,
commit `ceaa0660ea2487976101a8167aad5c8046e85b27`, corresponding to Agrawal
et al., "Taming Throughput-Latency Tradeoff in LLM Inference with
Sarathi-Serve," OSDI 2024 (arXiv:2403.02310). Full source-provenance
record, algorithm summary, existing-infrastructure audit, and explicit
exclusions: `docs/sarathi_faithful_scheduler_reference.md`.

Unlike `sarathi_style` (an admission-rate heuristic with no separate
prefill phase), `sarathi_faithful` reimplements the pinned reference's
actual scheduling algorithm: already-decoding sequences are reserved a
decode slot every iteration BEFORE any prefill work is considered
(stall-free / decode-first), continuing and new prefills share whatever
`chunk_size` budget remains, admission of new requests stops entirely (not
skip-and-continue) at the first request that cannot be allocated or would
get a 0-token chunk, and preemption uses the identical recompute/
victim-selection algorithm as `vllm_faithful` — because Sarathi-Serve's own
memory model literally reuses vLLM's `BlockSpaceManager` unchanged. This
baseline reuses this project's `KVBlockSpaceManager` and
`Action.preempt`/`GPUState.evict()` infrastructure (built for
`vllm_faithful`) rather than introducing anything new.

- **Safe claim:** "Faithful reimplementation of Sarathi-Serve's stall-free
  chunked-prefill scheduler's scheduling/memory *decisions*, as of commit
  `ceaa0660`."
- **Unsafe claim:** "Official Sarathi-Serve code", "exact runtime
  reproduction", "a full Sarathi-Serve performance/hardware-timing model",
  or a claim about the project's current scheduler (MoE support,
  pipeline-parallel fixes, etc. postdate this pin and are not represented).

**Not currently a registered baseline**, for the same reason as
`vllm_faithful` above: fully implemented and unit-tested
(`llmserveopt.policies.sarathi_faithful.SarathiFaithfulPolicy`), but
deliberately not added to `registry.py`'s `BASELINE_NAMES` /
`SELECTOR_CANDIDATE_NAMES` in the PR that introduced it.

### `distserve_faithful`

**Manuscript label:** "Faithful independent simulator-side reimplementation
of the pinned DistServe camera-ready FCFS online scheduling behavior, using
the repository's disaggregated prefill/decode infrastructure."

**Pinned reference:** LLMServe/DistServe, branch `camera-ready-simulator`,
commit `0ec355c8743d3fbd2d02f3cd62b5be6eae368f92`, corresponding to Zhong et
al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
Large Language Model Serving," OSDI 2024 (arXiv:2401.09670). Full
source-provenance record, architecture audit, and explicit exclusions:
`docs/distserve_faithful_scheduler_reference.md`.

Unlike `vllm_faithful`/`sarathi_faithful` (both single-pool schedulers),
`distserve_faithful` reimplements a **two-stage** scheduler on top of this
repository's disaggregated prefill/decode infrastructure (`GPUConfig.role`,
the bridge queue, `RequestPhase.MIGRATING`): a context (prefill) stage that
FCFS-admits whole prompts bounded by batch-size/token-budget/KV-block
capacity and stops entirely (not skip-and-continue) at the first
non-admittable request; a bridge-queue handoff that exposes a request for
decode-side admission only once its transfer delay has elapsed; and a
decode stage that FCFS-admits from the bridge queue (giving strict priority
to requests it previously swapped out over new migrations, and throttling
new-migration acceptance via a `waiting_block_prop_threshold` congestion
gate) and manages capacity via **swap** (`Action.swap` /
`GPUState.evict(preserve_progress=True)`) rather than vLLM/Sarathi-style
recompute preemption — confirmed genuinely core to the pinned reference's
`_step()`, not an optional mode. The pinned reference's core `LLMEngine` is
single-context-worker/single-decode-worker (not a multi-worker pool — that
exists only in DistServe's secondary `simdistserve` exploratory tool), so
this policy requires and validates exactly one `role="prefill"` GPU and one
`role="decode"` GPU, raising `ValueError` otherwise.

**Scope boundary:** this PR implements only the pinned reference's *online
request scheduling* (context-stage admission, decode-stage admission +
swap, bridge-queue handoff). It does **not** implement and does **not**
claim to reproduce DistServe's *offline* parallelism/placement planner (the
paper's goodput-optimized configuration-search algorithm) — that is a
separate, unimplemented capability.

**Disclosed, non-fabricated defaults:** `waiting_block_prop_threshold=0.05`
and `block_size=16` are verified values read directly from the pinned
source (`DecodingStageSchedConfig.__init__`; vLLM's own block-manager
default, which DistServe's block manager subclasses unchanged).
`context_max_batch_size`, `context_max_tokens_per_batch`,
`decode_max_batch_size`, and `decode_max_tokens_per_batch` have **no**
verified default in the pinned source or its evaluation scripts (unlike
Sarathi's clearly-sourced chunk size of 512) — the values shipped here are
this project's own conservative choices, exposed as explicit constructor
parameters precisely so they are never silently assumed as "the paper's."

- **Safe claim:** "Faithful reimplementation of the pinned DistServe
  camera-ready commit's online context-stage/decode-stage FCFS scheduling
  and swap-based decode-capacity *decisions*, on a single-prefill-worker/
  single-decode-worker topology."
- **Unsafe claim:** "Official DistServe code", "exact runtime/network
  reproduction", "a full DistServe performance/hardware-timing model", "a
  reproduction of DistServe's offline parallelism/placement planner", or
  any multi-worker routing/load-balancing claim (not part of the pinned
  reference's core scheduler).

**Not currently a registered baseline**, for the same reason as
`vllm_faithful`/`sarathi_faithful` above: fully implemented and
unit-tested (`llmserveopt.policies.distserve_faithful.DistServeFaithfulPolicy`),
but deliberately not added to `registry.py`'s `BASELINE_NAMES` /
`SELECTOR_CANDIDATE_NAMES` in the PR that introduced it — and it also
requires a disaggregated two-GPU (`role="prefill"`/`role="decode"`)
topology that ordinary single-pool experiment configs do not provide,
making blanket registration inapplicable without further config work.

### `tetriinfer_paper_reimplementation`

**This baseline is deliberately NOT labeled `_faithful`.** Every other
baseline in this section is pinned to a specific, author-maintained source
commit; TetriInfer has none. See
docs/tetriinfer_reference.md section 0 for the full reproducibility
determination — summarized here: **no official code repository, artifact,
or author-maintained implementation exists** for TetriInfer (verified live
via `gh api`/`gh search code`/web search), and it has **no peer-reviewed
venue** (arXiv preprint only, confirmed via the Semantic Scholar API and
DBLP's "CoRR" bucket). `tetriinfer_paper_reimplementation` is the
scientifically defensible label for a reimplementation built from the
paper's own prose description rather than a pinned commit.

**Manuscript label:** "Independent reimplementation of TetriInfer's core
two-level scheduling algorithm (as described in the paper), built on this
project's disaggregated prefill/decode infrastructure."

**Primary source:** arXiv:2401.11181, "Inference without Interference:
Disaggregate LLM Inference for Mixed Downstream Workloads" (Hu, Huang, Xu,
Chen, Xu, Chen, Feng, Wang, Wang, Bao, Sun, Shan; UCAS/ICT-CAS + Huawei
Cloud), v1 (only version), 2024-01-20, CC BY 4.0. Full source-provenance
record, exactly which algorithmic details are paper-specified vs. this
project's own disclosed adaptations, and the reproducibility determination:
docs/tetriinfer_reference.md.

Unlike `distserve_faithful` (exactly one prefill-role and one decode-role
GPU, per DistServe's own pinned single-worker-per-stage architecture),
this is **the first baseline in this project requiring genuine
multi-instance decode-side routing**: it accepts any number of
`role="prefill"` and `role="decode"` GPUs. It reimplements: a global
scheduler (least-loaded prefill-instance assignment, assigned once per
request); a local prefill scheduler (FCFS/SJF/LJF, non-preemptive,
batch-size/token-budget/KV-capacity bounded); a length-prediction
abstraction (`tetriinfer_length_prediction.py` — bucketed, deterministic,
no ML model, no external/paid API, with an optional configurable noise
path to study prediction-error sensitivity, and a structural guarantee
that it is never given ground-truth output length); an inter-instance
decode dispatcher (`tetriinfer_routing.py` — power-of-two random candidate
sampling from resource-eligible instances, tie-broken by the paper's own
stated "spread heavy decode requests evenly" objective); and a local
decode scheduler (greedy / reserve-static / reserve-dynamic admission
gates). Unlike `distserve_faithful`, this policy **never uses swap** —
TetriInfer's own decode-side story is admission-time avoidance of
thrashing, not runtime eviction.

- **Safe claim:** "Independent reimplementation of TetriInfer's
  paper-described two-level scheduling algorithm (dispatcher routing,
  reserve-static/reserve-dynamic admission gates, chunked-prefill-
  compatible context stage), evaluated against this project's own
  disclosed, non-ML length-prediction abstraction."
- **Unsafe claim:** "Official TetriInfer code" (none exists), "verified
  against TetriInfer's source" (no source exists to verify against), "an
  OSDI/peer-reviewed TetriInfer paper" (arXiv preprint only), a
  reproduction of the paper's own OPT-125M length predictor or its 74.9%
  empirical accuracy figure (this project's predictor is a disclosed,
  deterministic, non-ML substitute), or a claim about instance-flip
  elasticity (not implemented — see docs/tetriinfer_reference.md §E).

**Not registered as a deployable baseline or selector candidate**: fully
implemented and unit-tested
(`llmserveopt.policies.tetriinfer_paper_reimplementation.TetriInferPaperReimplementationPolicy`),
but deliberately not added to `registry.py`'s `BASELINE_NAMES` /
`SELECTOR_CANDIDATE_NAMES` — both because of the lower source-confidence
label (a paper reimplementation, not a pinned-commit-verified baseline)
and because it requires a multi-GPU disaggregated topology that ordinary
single-pool experiment configs do not provide. Historical policy counts
(currently 20 deployable/selector-candidate policies, see
docs/research_status.md) are unaffected.

### `llumnix_faithful`

**Manuscript label:** "Faithful independent reimplementation of Llumnix's
core cluster scheduling algorithm (initial dispatch, periodic migration-
pair selection, LCFS migration-candidate selection, destination admission
gating), composing with `vllm_faithful` for local per-instance scheduling."

**Pinned reference:** the OSDI 2024 artifact repository
`alibaba/llm-scheduling-artifact`, commit
`a90824307249573f9c7548645c22994c65f83a08` (pushed 2024-06-05, the same
day as the paper's arXiv v1 submission), corresponding to Sun et al.,
"Llumnix: Dynamic Scheduling for Large Language Model Serving," OSDI 2024
(arXiv:2406.03243). **Not** `AlibabaPAI/llumnix` ("Llumnix v0", a separate,
continuously-evolving Ray-based project) or `llumnix-project/llumnix`
("Llumnix v1", a March-2026 architecture rewrite) — both post-date and
diverge from the pinned artifact; see
docs/llumnix_faithful_scheduler_reference.md §3 for the full relationship.
Full source-provenance record, algorithm summary, and explicit exclusions:
docs/llumnix_faithful_scheduler_reference.md.

Unlike the other faithful baselines here (each single- or fixed-role-
topology), Llumnix requires genuinely **independent, role-identical
instances** (every GPU runs the full request lifecycle, no prefill/decode
split) with a NEW live-migration primitive: `Action.migrate` moves an
already-active (already-decoding) request from one instance to another,
preserving decoded progress and its overall service-time anchor
(admission_time), via a dedicated in-flight table
(`Simulator._relocating`) deliberately kept separate from the DistServe/
TetriInfer bridge queue (`Simulator._migrating`) — a bridge-queue request
has produced zero output tokens and may be claimed by any decode-role GPU;
a Llumnix migration candidate has always decoded at least one token and
has one fixed, policy-chosen destination. This baseline implements the
pinned reference's **verified defaults only**: `dispatch_strategy='naive'`
(round-robin, session-sticky — degenerates to plain round-robin in this
simulator, which has no session/multi-turn concept), `migrate_strategy=
'LCFS'` (migrate the most-recently-admitted decoding, non-priority-exempt
request from an overloaded instance), `load_metric='consumed_speed'`, and
`enable_load_control_prefill=False`'s migration-pair selection (including
its exact migration-benefit condition and its unconditional migrate-out
trigger for any instance with a currently-preempted/stalled request).
Local per-instance scheduling is composed directly with
`VLLMFaithfulPolicy`'s own per-GPU worker (not duplicated) — see the
reference doc's §E for the one disclosed divergence this composition
carries forward (the pinned Llumnix source's own local scheduler uses
swap preemption; `vllm_faithful` models only recompute, a difference
already disclosed in `vllm_faithful`'s own reference doc, not new here).

- **Safe claim:** "Faithful reimplementation of the pinned Llumnix OSDI
  2024 artifact's default cluster-scheduling *decisions* — dispatch,
  migration triggering, migration-pair selection, migration-candidate
  selection, and destination admission — composed with `vllm_faithful`'s
  existing local scheduler."
- **Unsafe claim:** "Official Llumnix code", "exact Ray/vLLM runtime
  reproduction", a claim about `AlibabaPAI/llumnix` ("v0") or
  `llumnix-project/llumnix` ("v1") behavior (both post-date and diverge
  from this pin), a reproduction of any non-default dispatch/migration
  strategy (`'balanced'`/`'load'`/`'block'`/global `FFIT`/`FCFS`/`BE`/
  `SJF`/`LJF` dispatch, `'SJF'`/`'LJF'` migration), auto-scaling, or a
  claim about session/multi-turn behavior (this simulator has no session
  concept at all).

**Not registered as a deployable baseline or selector candidate**: fully
implemented and unit-tested
(`llmserveopt.policies.llumnix_faithful.LlumnixFaithfulPolicy`), but
deliberately not added to `registry.py`'s `BASELINE_NAMES` /
`SELECTOR_CANDIDATE_NAMES` — it requires a genuine multi-instance
(N independent, role=None GPU) topology that ordinary single-pool
experiment configs do not provide. Historical policy counts (currently 20
deployable/selector-candidate policies, see docs/research_status.md) are
unaffected.

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

---

## Baseline-integration scaffolds (evaluation-ready, offline-scored — not yet a live sweep entry)

Distinct from every policy above: these are not `BasePolicy` entries usable
in a normal experiment sweep today. They live outside `src/llmserveopt`
entirely, are never imported by it, and are never selector candidates.

### vLLM-LTR (`baselines/vllm_ltr/`) — added 2026-08-04, completed same day

Official source: https://github.com/hao-ai-lab/vllm-ltr, pinned commit
`13bbf6ff3dab661791d41362551b089e5f77c91c` (Apache-2.0). Paper: Fu et al.,
*"Efficient LLM Scheduling by Learning to Rank"*, **NeurIPS 2024 (main
conference)** (arXiv:2408.15792 as supplementary preprint id). Fulfills/
supersedes the "Prompt-Aware LTR / PARS-style Scheduler" item in
`docs/external_baseline_decision.md` §B.1.

**Status: evaluation-ready external baseline (offline-scored; official
checkpoint verified).** The real official checkpoint
(`LLM-ltr/OPT-Predictors`, both a classification and a regression variant)
was downloaded, hash-recorded (`CHECKPOINT_PROVENANCE.md`), and verified:
exact state-dict key/shape match against the checkpoint's own declared
`OPTForSequenceClassification` architecture, and bit-exact agreement
between the adapter's scoring path and an independent from-scratch
recomputation of the pinned source's score formula, on real ShareGPT text.
A complete offline scoring pipeline (`adapter/offline_scoring.py`) turns
real prompt text into a cached, integrity-hashed `{request_id: score}` map
consumed by `VLLMLTRSemanticReferencePolicy` — no fallback heuristic if a
score is missing, no simulator request objects modified. **Still not a live
per-step simulator policy**: the official predictor requires tokenized
prompt *text*, and this simulator's `Request`/`ObservableRequest`
deliberately was not modified to carry it (out of scope by design, not a
gap in this baseline's completeness) — so it needs an externally-supplied
score map, exactly as built. Full verification results, real overhead
numbers, and the one disclosed-but-infeasible check (a live differential
against the actual served vLLM-fork engine, which would require building
its CUDA extensions from source): `docs/audits/vllm_ltr_baseline_audit_20260804.md`.

**Not** in `BASELINE_NAMES`, `SELECTOR_CANDIDATE_NAMES`,
`POLICY_LIBRARY_V2_NAMES`, or `EXTERNAL_BASELINE_NAMES` (locked by
`tests/test_vllm_ltr_baseline_adapter.py::TestSelectorScopeInvariants`) —
still deliberate, per the original task's "do not add to the main selector
candidate set yet."

**Comparative evaluation completed 2026-08-04.** A first head-to-head
simulator comparison ran on real WildChat-1M prompt/response text (300
requests, 3 seeds, 10 policies including vLLM-LTR evaluation-only), after
recovering from an initial run that never finished (selector performance
bug, fixed; see `docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md`).
Full results, independent re-verification, and classification:
`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`.
**Bottom line:** in this workload regime, `vllm_ltr_semantic_reference` tied
FIFO/EDF/EST/SOF/WSP/`oracle_srtf` exactly (ANWG=0.9957 all three seeds) —
the oracle itself also tied FIFO, meaning this specific regime has no
reorderable headroom for *any* ordering policy to demonstrate, so vLLM-LTR
neither beat nor lost to anything. It does make genuinely different
per-request ranking decisions (Spearman agreement with EST ≈0.35–0.40, with
SOF ≈0.43–0.48 — moderate, not near-1.0), so it is not behaviorally
redundant with the existing SJF-proxy policies; whether that distinct
ranking translates into measurable ANWG benefit remains untested and needs
a higher-contention regime. Classification: **EVALUATION_ONLY** (still not
registered as a selector candidate); foundational-library eligibility not
established by this run.

**Safe claim:** "vLLM-LTR offline-scored external baseline (official
checkpoint downloaded, hash-verified, and architecturally/numerically
verified against an independent recomputation; ranking rule reproduced
exactly; evaluated head-to-head against 9 other policies on real WildChat-1M
text across 3 seeds — tied FIFO/oracle exactly in this (uncontended)
workload regime, with moderate-but-not-near-1.0 ranking agreement with the
existing SJF-proxy policies)"  
**Unsafe claim:** "vLLM-LTR beats/loses to policy X" (it was statistically
tied with FIFO and the theoretical oracle in the one regime tested so far;
see the audit doc for the full result set and its regime-specific caveat)

### PARS-Serve-2026 (`baselines/pars/`) — integration complete, **EVALUATED, INDEPENDENTLY VERIFIED, EVALUATION_ONLY**

Official source: `https://github.com/SPEAR-UIC/PARS` (pinned commit
`fd4e125b65bb73aef5eccafa79c2509434be61ec`). Paper: Tao et al., *"Ranking
Before Serving: Low-Latency LLM Serving via Pairwise Learning-to-Rank,"*
ISC High Performance 2026 (arXiv:2510.03243). **Named "PARS-Serve-2026" in
this project's prose to disambiguate from an unrelated, earlier "PARS"
already referenced in `docs/external_baseline_coverage_report.md`
(Zheng et al., NeurIPS 2023, now called "PARS-2023" there) — see
`baselines/pars/PROVENANCE.md`'s naming-disambiguation note.**

**Status: official code integrated and verified; a real checkpoint was
trained locally** (the official repository ships no pretrained
checkpoint — only training code — a real `bert-base-uncased`-based
pairwise ranker was trained here with the official, unmodified training
script: 3 epochs, `best_val_accuracy=0.9141`, checkpoint SHA256
`d54be0871ebc9f2c2538b4e53da7f45cb57ae678563488822cdc1694bc33eb27`).
Every official script (preprocessing, training) runs completely
unmodified; the adapter (`baselines/pars/adapter/`) dynamically imports
the official `PairwiseRanker` class from a local, non-committed clone at
runtime rather than duplicating it. 22/22 adapter unit tests and 10/10
real-checkpoint fidelity tests pass. **Known license gap, disclosed not
hidden:** the official repository has no LICENSE file at all — see
`baselines/pars/PROVENANCE.md` for the full explanation and the explicit,
user-directed decision to proceed with local, non-commercial research
use.

**Comparative evaluation complete and independently verified**
(WildChat control + all 7 accepted canonical-suite families, 8 workloads
× 3 seeds × 10 policies, 60,830 request-level rows, zero unexplained
mismatches): PARS never ranks above 5th of 10 policies in any family,
records zero unique wins across all 8 families, and is statistically
significantly worse than the best policy in 5 of 8 families — while
being significantly better than FIFO/EDF in 3 burst/long-tail-heavy
families, showing its length-prediction signal is real but consistently
dominated by simpler heuristics (`shortest_output_first`,
`estimated_service_time_first`) and by this project's best
fixed/adaptive policies (`scorpio_style_slo_guard`,
`regression_anwg_selector`). **Final classification: EVALUATION_ONLY** —
not promoted to any selector-candidate or deployable-policy list. Full
implementation record: `docs/audits/pars_baseline_implementation_20260804.md`.
Full evaluation, recovery, and verification record:
`docs/audits/pars_first_comparative_evaluation_20260804.md`. Also see
`docs/BASELINE_STATUS.md` for the single-table cross-baseline status
index.

**Safe claim:** "PARS-Serve-2026 official code (training/scoring
pipeline) integrated as an evaluation-only external baseline with a
locally-trained, hash-verified checkpoint; evaluated head-to-head against
9 other policies across WildChat control and all 7 accepted
canonical-suite families; classified EVALUATION_ONLY — zero unique wins,
consistently dominated by simpler/existing policies in discriminative
regimes, though statistically better than FIFO/EDF in burst-heavy
regimes."  
**Unsafe claim:** "PARS beats/matches vLLM-LTR" (only one non-
discriminative workload, WildChat control, is directly comparable between
the two — see the evaluation doc §6 for the full scope caveat) or any
claim that PARS is a candidate for selector/deployable-policy inclusion.

### VTC (`baselines/vtc/`) — added 2026-08-05, fairness-validated comparative sweep complete, **EVALUATION_ONLY** (FOUNDATIONAL_CANDIDATE, not registered)

Official source: `https://github.com/Ying1123/VTC-artifact` (pinned
commit `192c2e2014c69c8c6c699d7113c3822e4db632e6`, Apache-2.0). Paper:
Sheng et al., *"Fairness in Serving Large Language Models,"* OSDI 2024
(arXiv:2401.00588). Full audit: `docs/audits/
vtc_official_artifact_audit_20260805.md`. Initial smoke-evaluation record
(superseded by the repair below): `docs/audits/
vtc_initial_integration_20260805.md`. Repair methodology: `docs/audits/
vtc_fairness_benchmark_repair_20260805.md`. Final comparative results +
independent verification + scientific decision: `docs/audits/
vtc_fairness_comparative_evaluation_20260805.md`. Full provenance:
`baselines/vtc/PROVENANCE.md`.

**Status: official fairness-scheduling algorithm integrated, verified,
and evaluated in a fairness-validated comparative sweep.** The official artifact is a
full S-LoRA-based GPU serving engine with custom CUDA kernels, requiring
CUDA 11.8/PyTorch ≤2.1.2/`triton==2.1.0` — this development machine's GPU
(RTX 5060 Ti, Blackwell) cannot build those kernels at all (a
compiler-generation gap, not a version to pin around). VTC's actual
fairness **algorithm** (`slora/server/router/vtc_req_queue.py`), however,
is pure Python/NumPy with zero GPU dependency, and was verified to import
and run correctly, completely unmodified, via
`baselines/vtc/adapter/official_loader.py`. `baselines/vtc/adapter/
simulator_policy.py`'s `VTCFairnessPolicy` drives the real,
unmodified `VTCReqQueue.append`/`generate_new_batch`/`update_counter`
methods directly — nothing about the fairness-selection logic is
reimplemented. Classification: **official policy reused with simulator
adapter** (not "official VTC," not a proxy — see the audit doc §4-5 for
the full, disclosed deviation list). 25/25 fidelity tests pass
(`tests/test_vtc_baseline_adapter.py`), including tests against the raw,
unmodified official class directly (min-served selection, insertion-order
tie-breaking, the counter-lift-on-return rule, aborted-request zero-charge,
the exact linear cost formula).

**Tenant semantics:** this project's accepted canonical suite has no
tenant/client concept at all (`Request`/`ObservableRequest` carry no such
field). `baselines/vtc/fairness_workloads.py` adds six clearly-labeled,
separate fairness-extension workload families (`balanced_tenants`,
`one_heavy_hitter`, `heterogeneous_token_sizes`, `bursty_tenant`,
`returning_inactive_tenant`, `priority_fairness_conflict`), reusing
`Request.class_id` as the tenant id rather than touching core types.

**Initial smoke evaluation found a methodological confound, which was then
diagnosed and repaired (2026-08-05, same day).** The first smoke pass
(`docs/audits/vtc_initial_integration_20260805.md`) found 5 of 6 families
showed no policy divergence at all (insufficient backlog contention), and
the one that diverged (`heterogeneous_token_sizes`) looked confound-driven
rather than fairness-driven. `docs/audits/
vtc_fairness_benchmark_repair_20260805.md` diagnosed this precisely: a
**units mismatch** — this simulator's native `_feasible_on_gpu` reads
`GPUConfig.max_batch_tokens` as a per-step ACTIVE-REQUEST-COUNT cap
(Phase-1 simplification), while the official `VTCReqQueue`/`ReqQueue` code
reads the same field as a real cumulative PROMPT-TOKEN budget — feeding
one number into both interpretations created a confound unrelated to
VTC's fairness mechanism (confirmed directly: `MatchedAdmissionFIFOPolicy`,
FCFS ordering under VTC's exact same official admission gate, reproduced
VTC's identical 0.036 completion fraction in that family, proving the
collapse was 100% admission-driven, not ordering-driven).

**Repair:** three labeled comparison variants
(`baselines/vtc/adapter/variants.py`) — **A** official VTC, **B**
matched-admission FIFO (via the official, unmodified `ReqQueue` FCFS base
class VTCReqQueue itself subclasses), **C** fairness-isolation VTC
(capacity rescaled to avoid the units mismatch, still 100% unmodified
official code). All six fairness-extension workloads were retuned for
genuine, verified backlog contention and gated by
`scripts/check_vtc_fairness_headroom.py` (all 6 pass every threshold) before
any comparative sweep ran. 20 additional tests (16 headroom + 4
hand-verifiable micro-traces) lock in the repair.

**Fairness-validated comparative sweep** (6 policies × 6 families × 3
seeds = 108 runs, independently re-verified with **zero unexplained
mismatches** via a from-scratch recomputation,
`scripts/verify_vtc_fairness_sweep.py`): VTC achieves the strictly-best or
tied-best checkpoint Jain's-index in **17 of 18** family×seed combinations
(13 outright wins, 4 ties), losing only in `bursty_tenant` (a small, real,
disclosed negative result). `official_vtc` and `fairness_isolation_vtc`
are numerically indistinguishable throughout the repaired sweep,
confirming the fairness wins are an ordering effect, not an admission-gate
artifact. The trade-off is real and bounded: in
`priority_fairness_conflict` (engineered to expose VTC's blindness to
`priority`/`slo_deadline`), VTC has the WORST ANWG of all six policies
(0.680 vs. `scorpio_style_slo_guard`'s 0.984) and a 38.1% tight-SLO
violation rate (SCORPIO: 0.0%) in exchange for near-perfect fairness
(Jain 1.000).

**Final classification: EVALUATION_ONLY** (deployment status, unchanged —
still a wrapped external adapter, single-GPU only). **Scientific
classification: FOUNDATIONAL_CANDIDATE**, scoped specifically to VTC's
fairness objective as a candidate primitive for future fairness-aware
composition, not as a general ANWG-maximizing policy replacement — see
`docs/audits/vtc_fairness_comparative_evaluation_20260805.md` for the full
decision record. **Not registered** as a selector-candidate or deployable
policy this task, per explicit instruction (eligibility and registration
are deliberately kept separate).

**Safe claim:** "VTC's official fairness-scheduling algorithm
(`VTCReqQueue`) is integrated and executed verbatim, unmodified, inside
this project's simulator; a headroom-gated, independently-verified
comparative sweep across 6 dedicated fairness workloads shows VTC achieves
the best-or-tied fairness outcome in 17/18 family×seed combinations, with
a real, bounded, well-understood throughput/SLO trade-off in the one
scenario designed to expose it; classified a scientific
FOUNDATIONAL_CANDIDATE for a future fairness-aware composition context,
not registered this task."
**Unsafe claim:** "VTC is ready for foundational-library registration"
(explicitly deferred — native, non-wrapped reimplementation is the
disclosed next step) or "VTC ran as the official GPU serving system" (it
did not; only the scheduling algorithm ran, the GPU engine layer is
currently unbuildable on this hardware).

# Pinned DistServe Reference — Disaggregated Prefill/Decode Infrastructure

This document records the exact upstream reference used to design this
project's disaggregated prefill/decode simulator infrastructure, and
defines the boundary between (A) behavior directly supported by the pinned
source, (B) simulator abstractions required to reproduce it, (C) existing
infrastructure reused unchanged, (D) new shared infrastructure added in
this stage, and (E) differences that cannot be faithfully represented in
this simulator.

> **Update (distserve_faithful implementation stage):** the source was
> re-read directly (not from memory) before implementing the
> `distserve_faithful` policy. Two corrections/additions from that
> re-verification are folded in below: (1) the CORE production reference
> (`distserve/engine.py`'s `LLMEngine`, `distserve/single_stage_engine.py`)
> is a **single context worker + single decode worker** system — the
> "multiple workers with load-balancing routing" architecture mentioned
> below under "Worker/resource partition assumption" is confirmed to exist
> **only** in the secondary `simdistserve` tool, not in the core FCFS
> baseline this project pins to; see the corrected note in that section.
> (2) Swap-based decode-side capacity management is confirmed to be part
> of the **always-active** decode-stage step execution (not an optional/
> secondary path) — see the new "Swap-based decode-side capacity
> management" subsection under A, and its corresponding entries in C/D.

## Paper

- **Title:** "DistServe: Disaggregating Prefill and Decoding for
  Goodput-optimized Large Language Model Serving"
- **Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu,
  Xuanzhe Liu, Xin Jin, Hao Zhang
- **Venue:** 18th USENIX Symposium on Operating Systems Design and
  Implementation (OSDI 2024)
- **arXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

## Official repository

- **Repo:** [LLMServe/DistServe](https://github.com/LLMServe/DistServe)
- **License:** Apache License 2.0

## Pinned commit/branch

**Branch `camera-ready-simulator`, commit
`0ec355c8743d3fbd2d02f3cd62b5be6eae368f92`** (commit message "Address
comments", dated 2024-04-29 — squarely within the OSDI 2024 camera-ready
window; OSDI 2024 was held July 10–12, 2024). Fetched live via the GitHub
API, not from memory. The repository's `main` branch has continued to
evolve substantially since (most recent commit as of this audit:
2025-04-06) and was deliberately NOT used, per the same "paper-era, not
current `main`" principle applied to `vllm_faithful` and
`sarathi_faithful`.

Two other branches were considered and rejected: `feat-add-simulator`
(earlier, 2024-04-25, superseded by `camera-ready-simulator`) and
`hotfix-issue-13` (2024-06-14, post-dates the camera-ready window, not
paper-era).

Source files read directly (via `gh api repos/LLMServe/DistServe/contents/...
?ref=0ec355c8743d3fbd2d02f3cd62b5be6eae368f92`):

- `distserve/context_stage_scheduler.py` (211 lines) — `ContextStageFCFSScheduler`
  (prefill-side scheduler; DistServe calls prefill the "context stage")
- `distserve/decoding_stage_scheduler.py` (274 lines) — `DecodingStageFCFSScheduler`
  (decode-side scheduler)
- `distserve/request.py` — `MigratingRequest` ("bridge queue" element)
- `distserve/config.py` — `ContextStageSchedConfig`, `DecodingStageSchedConfig`
- `distserve/simulator/simulated_worker.py` (188 lines) — DistServe's own
  reference discrete-event simulator's KV-migration cost model
- `distserve/simulator/estimator.py` (35 lines) — DistServe's own
  profiled compute-time model (prefill/decode), for contrast with the
  migration cost model
- `simdistserve/base/scheduler.py` (88 lines) — a second, independent
  SimPy-based simulation tool in the same repo (see the corrected "Worker/
  resource partition assumption" note below: this tool's multi-worker
  routing is NOT part of the core FCFS baseline pinned here)
- `distserve/engine.py` (347 lines) — `LLMEngine`, the actual production
  orchestrator class; its own docstring is the authoritative description of
  the single-context-worker/single-decode-worker architecture
- `distserve/single_stage_engine.py` (710 lines) — `ContextStageLLMEngine`
  and `DecodingStageLLMEngine`, confirming exactly one instance of each
  runs per `LLMEngine`, and that swap in/out is invoked unconditionally
  inside `DecodingStageLLMEngine._step()`, not behind any optional flag

## A. Behavior directly supported by the pinned source

### Terminology

DistServe calls prefill the **"context stage"** and decode the **"decoding
stage."** A request's KV cache, once its context stage finishes, must be
handed off ("migrated") to a decoding-stage worker before decoding can
begin.

### Context (prefill) stage scheduler — `ContextStageFCFSScheduler`

FCFS, no chunking (the entire prompt is one unit — unlike Sarathi-Serve,
there is no partial-prefill-per-iteration concept here at all). A request
is admitted into the next batch if all of:
- batch size < `max_batch_size`,
- batch's total input tokens + this request's tokens ≤ `max_tokens_per_batch`,
- GPU blocks needed for (current batch + still-unaccepted-by-decode
  requests + on-the-fly requests) ≤ `max_num_gpu_blocks`.

Once a request finishes the context stage, it does **not** immediately
continue anywhere — it becomes a `MigratingRequest` and sits in an
**`unaccepted_queue`** ("finished the context stage but not yet accepted
by the decoding stage"), and its KV blocks remain reserved on the context
worker's memory (`num_on_fly_request_block` accounting) until the decode
side actually accepts it.

### Decoding stage scheduler — `DecodingStageFCFSScheduler`

FCFS. Requests newly migrated in start in the decode side's own
`unaccepted_queue` too (`add_request` for a `MigratingRequest` = "accept
any request that comes in" at the API level — but see the *admission gate*
in `post_process`, below). Once accepted:

1. `get_next_batch()` builds the batch, and if blocks are insufficient,
   **swaps out** (not recompute!) the last request in the current batch,
   putting it in a `swapped_queue`.
2. Swapped-out requests are re-admitted (swapped back in) with strict
   priority over the plain `waiting_queue`.
3. New `waiting_queue` admissions happen FCFS, subject to the same
   batch-size/token/block checks as the context stage.

**Why swap instead of recompute (a genuine, motivated difference from
vLLM/Sarathi-Serve):** by the time a request reaches the decode stage in a
disaggregated system, its (potentially expensive, potentially
cross-machine) context-stage work and migration are already sunk costs.
Recompute-preemption would mean re-running context-stage processing AND
re-migrating — vastly more expensive than in a colocated system where
"recompute" just means redoing local prefill. Swap avoids re-paying that
cost.

### Migration ("bridge queue") admission gate — `DecodingStageFCFSScheduler.post_process`

Called once per iteration. For the request at the front of the decode
side's `unaccepted_queue`, it is accepted (migrated) only if **both**:
- the decode side's current `waiting_queue` block demand is below
  `waiting_block_prop_threshold` (default **0.05**, i.e. 5%) of total
  decode-side GPU block capacity — a congestion-avoidance gate, not a hard
  capacity check;
- enough free GPU blocks currently exist on the decode side for this
  specific request's KV cache.

If accepted: `engine_migrate_block_callback(migrating_req)` performs the
actual KV transfer, then the request moves into the decode side's
`waiting_queue`. If not accepted, admission from `unaccepted_queue` stops
for this iteration (FCFS: only the front is checked).

### KV-transfer cost model (from DistServe's own reference simulator)

`distserve/simulator/simulated_worker.py`'s `SimulatedWorker.migrate_blocks()`
does **not** implement a bandwidth/size-dependent transfer cost. It incurs
only `_simulate_ray_overhead()` — a small **fixed** constant
(`RAY_OVERHEAD_BLOCKING = 1` ms / `RAY_OVERHEAD_NONBLOCKING = 2.2` ms).
This is read directly from the pinned reference's own simulator, not
inferred or assumed: **DistServe's own reference simulation treats KV
migration as a small fixed overhead, separate from and far cheaper than
compute time**, which is instead modeled via
`distserve/simulator/estimator.py`'s profiled quadratic curves
(`a + b·tokens + c·tokens²`) per prefill/decode phase. This directly
justifies this project's own choice (see D below) of a simple, configurable
fixed-delay transfer model rather than inventing a bandwidth-based one.

### Worker/resource partition assumption (corrected)

Prefill and decode run as **separate GPUs/resource pools** — a hard
architectural assumption of the paper (the entire contribution is *not*
colocating the two phases), confirmed by the scheduler code's clean
separation into two classes with no shared state.

**Correction from the original infrastructure-stage audit:** the CORE
production reference is a **single context worker + single decode worker**
per `LLMEngine`, not a multi-worker pool with load-balancing routing.
`distserve/engine.py`'s `LLMEngine` docstring describes exactly one
`ContextStageScheduler` and one `DecodingStageScheduler`, and
`distserve/single_stage_engine.py` instantiates exactly one
`ContextStageLLMEngine` and one `DecodingStageLLMEngine` per `LLMEngine`.
The "`_prefill_heads`/`_decode_heads` worker-pool least-loaded routing"
architecture previously cited here is real, but exists **only** in
`simdistserve/base/scheduler.py` — a second, independent SimPy tool in the
same repository, not the core FCFS baseline's production code path. Since
this project pins to the production scheduler classes
(`distserve/context_stage_scheduler.py` / `decoding_stage_scheduler.py`),
`distserve_faithful` targets exactly one prefill-role GPU and one
decode-role GPU, consistent with the actual `LLMEngine` architecture —
using only verified behavior rather than substituting a generic
load-balancer the pinned core reference does not itself use.

### Swap-based decode-side capacity management (always-active, not optional)

Re-verified directly in `distserve/single_stage_engine.py`:
`DecodingStageLLMEngine._step()` unconditionally calls into swap in/out
logic every iteration ("this may trigger swap_in if some requests have
been swapped out to CPU... this may also trigger swap_out if GPU blocks
are not enough" — read directly from the source comment). This is *not*
behind any optional flag or ablation switch — it is how the core FCFS
decode-stage scheduler manages capacity, full stop. `distserve_faithful`
therefore implements it (see D and C below), rather than deferring it as
optional.

## B. Simulator abstractions required to reproduce this

1. A way to tag a GPU as belonging to the prefill pool or the decode pool
   (or neither, for legacy/colocated behavior).
2. An explicit state representing "this request has finished prefill and
   is in transfer/handoff, not yet eligible for decode admission" — global
   simulator state, not something hidden inside one policy (per this
   stage's explicit requirement).
3. A separate scheduling queue for transfer-ready requests, distinct from
   the ordinary (needs-prefill) waiting queue, so a policy can tell the two
   apart.
4. A configurable, parameterized transfer delay, defaulting to zero (for
   structural/fidelity tests) — not a bandwidth model (see above).
5. A way for a request migrated onto a decode-role GPU to skip prefill
   entirely (it was already done, on a different worker).

## C. Existing infrastructure reused unchanged

- `GPUConfig`/`GPUState` themselves — a prefill-role GPU and a decode-role
  GPU are each just an ordinary `GPUConfig`/`GPUState` instance with their
  own independent capacity. Capacity isolation between the two pools falls
  out for free from having two separate instances; nothing new was needed
  for this.
- `KVBlockSpaceManager` (built for `vllm_faithful`) — DistServe's own block
  accounting (`_get_block_needed`, `max_num_gpu_blocks`) is structurally
  the same fixed-size-block model. A future `distserve_faithful` policy can
  instantiate one `KVBlockSpaceManager` per GPU (prefill-side and
  decode-side each get their own), exactly as `vllm_faithful` and
  `sarathi_faithful` already do. No changes needed to that module.
- `ServiceModel`'s existing prefill/decode split
  (`enable_prefill_modeling`, `InternalRequest.prefill_remaining`) —
  prefill execution on a prefill-role GPU is unchanged; only what happens
  *once prefill finishes* differs when disaggregation is enabled.
- `CompletedRequest.ttft`/`.tpot` — unchanged; verified (see infrastructure
  tests) to remain correct when a request's decode phase runs on a
  *different* `GPUState` instance than its prefill phase.

## D. New shared infrastructure added in this stage

All of the following are additive and default to fully backward-compatible
values (disaggregation is opt-in and off by default):

- `GPUConfig.role: Optional[str] = None` (and the matching
  `ObservableGPUState.role`) — `"prefill"`, `"decode"`, or `None` (legacy/
  colocated, the only value any existing config uses).
- `RequestPhase.MIGRATING` — a new enum member (`simulator/request.py`).
  Existing members (`WAITING`/`ACTIVE`/`COMPLETED`) and all existing
  transitions between them are unchanged.
- `InternalRequest.transfer_ready_time: float = -1.0` — when a migrating
  request's transfer delay elapses. `-1.0` (unused) for every request that
  never migrates.
- `ServiceModel.enable_disaggregation: bool = False` and
  `ServiceModel.migration_transfer_delay: float = 0.0` — opt-in switch and
  the fixed transfer delay (see KV-transfer cost model above; `0.0` gives
  the zero-cost mode the task requires).
- `GPUState.pop_pending_handoff()` — drains requests this GPU just handed
  off this step (only ever non-empty for a `role="prefill"` GPU with
  disaggregation enabled).
- `Simulator._migrating` / `_migrating_map` — the "bridge queue" (named
  after DistServe's own terminology), holding `InternalRequest`s that have
  finished prefill and are awaiting transfer completion.
- `ObservableState.migrating_queue: List[ObservableRequest]` — exposes
  only **transfer-ready** (delay already elapsed) requests to policies,
  distinct from `waiting_queue` (which is unchanged and holds only
  genuinely-new, needs-prefill requests).
- `Simulator._apply_action`'s admission lookup is extended to check
  `_migrating_map` (in addition to `_waiting_map`) when resolving an
  `Action.admit` request ID — **no new `Action` field or verb was
  introduced**; `admit`/`preempt` remain the only two action verbs.
  `GPUState.admit()` skips prefill entirely (`prefill_remaining = 0`) for
  a request coming from `RequestPhase.MIGRATING`, regardless of what
  `ServiceModel` would otherwise compute for it.

**Added in the `distserve_faithful` implementation stage** (swap support,
confirmed genuinely core — see above):

- `Action.swap: Dict[int, List[int]]` — a new, narrowly-scoped third action
  verb. Unlike `preempt` (discard-and-restart, vLLM/Sarathi's recompute
  semantics), `swap` evicts an active DECODING request while **preserving**
  its progress (`tokens_decoded`, `first_token_time` untouched) and routes
  it into the bridge queue as **immediately transfer-ready**
  (`transfer_ready_time = now`), matching DistServe's own "swapped requests
  re-admitted with priority over ordinary waiting" behavior — the policy is
  responsible for that priority ordering (it controls what it puts into
  `admit` and in what order), not the simulator. Defaults to empty; every
  pre-existing policy (including `vllm_faithful`/`sarathi_faithful`) never
  sets it, so this is fully backward compatible.
- `GPUState.evict(request_id, preserve_progress=False)` — the existing
  eviction method gains an optional parameter (default `False` = the
  original recompute behavior, unchanged) that, when `True`, skips
  resetting `tokens_decoded`/`first_token_time`.

## E. Differences that cannot be (or are not) faithfully represented

- **Pipeline parallelism** (`parallel_config.pipeline_parallel_size`,
  multiple in-flight batch queues per stage): no analogue in this
  simulator's single-decision-per-step execution model, same exclusion
  already made for `vllm_faithful`/`sarathi_faithful`.
- **Advanced decode-stage policies** (`srpt`, `mlfq`, `sj-mlfq`): the
  pinned `DecodingStageSchedConfig` supports these, but `fcfs` is the
  paper's core/baseline algorithm and the only one implemented in the
  context-stage scheduler at all; `DistServeFaithfulPolicy` pins `fcfs` on
  both sides, for the same "core algorithm, not every ablation" reasoning
  already applied to Sarathi-Serve's dynamic chunking schedule.
- **Bandwidth/network-topology-aware KV-transfer cost**: the paper
  discusses placement-aware bandwidth optimization, but DistServe's own
  reference *simulator* (the primary source for timing assumptions, as
  opposed to the real Ray/NVLink system) does not model this — it uses a
  flat constant. This project follows the reference simulator's own
  choice rather than inventing a bandwidth model with no primary-source
  basis.
- **`num_min_free_blocks_threshold`, `use_skip_join`, `proactive_offloading`,
  `num_queues_for_prediction`** (all present in `DecodingStageSchedConfig`):
  secondary tuning knobs not part of the core algorithm description;
  excluded, same reasoning as above.
- **Representative parameter values**: unlike Sarathi-Serve's OSDI
  evaluation scripts (which gave a clear "512" chunk-size signal), no
  equivalent single evaluation script setting `max_batch_size`/
  `max_tokens_per_batch` for the two stages was found in the pinned
  commit's `distserve/evaluation/` scripts in the time available for this
  audit. `waiting_block_prop_threshold=0.05` **is** a verified default
  (from `DecodingStageSchedConfig.__init__`) and `block_size=16` is
  inherited from vLLM's own block-manager default (reused unchanged by
  DistServe's block manager). `DistServeFaithfulPolicy`'s
  `context_max_batch_size=32`, `context_max_tokens_per_batch=4096`,
  `decode_max_batch_size=128`, and `decode_max_tokens_per_batch=4096` are
  this project's own conservative choices, documented in the policy's
  module docstring as *not* sourced from a specific paper evaluation run,
  unlike `sarathi_faithful`'s `chunk_size=512` — exposed as explicit
  constructor parameters precisely so they are never silently assumed as
  "the paper's."

### Implementation note: `waiting_block_prop_threshold`'s per-round semantics

`DecodingStageFCFSScheduler.post_process`'s `should_accept` check compares
the pinned reference's own decode-side `waiting_queue` (requests already
accepted from the bridge/`unaccepted_queue` but not yet batched) against
the threshold, and is invoked in a `while` loop that re-checks the
condition **after each acceptance** — so the gate is against a backlog
that starts small and grows as more requests are pulled in during the same
round, not a one-shot check against the bridge queue's full, static
backlog. This simulator has no separate persistent "accepted but not yet
batched" tier distinct from the bridge queue itself, so
`DistServeFaithfulPolicy._run_decode_stage` reconstructs the same
semantics directly: it walks the bridge queue's new-migration candidates
FCFS, maintaining an `accepted_blocks` counter that starts at **zero every
call** and stops accepting the instant `accepted_blocks >=
waiting_block_prop_threshold * total_decode_blocks`. Implementing this as
a single check against the whole static backlog instead (an earlier,
incorrect draft of this policy did so) causes permanent starvation as soon
as the backlog exceeds a few requests, since `waiting_block_prop_threshold`
defaults to a small fraction (0.05) of capacity.

## Do not use current `main` blindly

`LLMServe/DistServe`'s `main` branch has continued to evolve substantially
past the OSDI 2024 camera-ready commit (as of this audit, over a year of
subsequent commits). Any claim made about this infrastructure, or about a
future `distserve_faithful` policy built on it, is a claim about DistServe
**as of commit `0ec355c8`**, not about the project today.

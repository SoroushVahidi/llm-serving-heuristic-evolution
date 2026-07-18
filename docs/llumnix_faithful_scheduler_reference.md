# Pinned Llumnix Reference — Cluster Scheduling, Migration Trigger, and Live-Migration Mechanism

This document records the exact primary source used to build the
`llumnix_faithful` baseline, and defines the boundary between (A) behavior
directly supported by the pinned source, (B) simulator abstractions
required to reproduce it, (C) existing infrastructure reused unchanged,
(D) new shared infrastructure added for this baseline, and (E) differences
that cannot be faithfully represented.

**Read this document before touching
`src/llmserveopt/policies/llumnix_faithful.py`.**

## 1. Paper

- **Title:** "Llumnix: Dynamic Scheduling for Large Language Model Serving"
- **Authors:** Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi
  Zhang, Yong Li, Wei Lin (Alibaba Group)
- **Venue:** 18th USENIX Symposium on Operating Systems Design and
  Implementation (OSDI 2024)
- **arXiv:** [2406.03243](https://arxiv.org/abs/2406.03243), v1 (only
  version), submitted 2024-06-05
- **arXiv license:** arXiv non-exclusive-distribution license (paper
  text). **Not** CC-BY (differs from the TetriInfer pin's license — do not
  conflate the two when quoting/reusing text).

## 2. Official OSDI 2024 artifact repository (THE pin for this baseline)

- **Repo:** [alibaba/llm-scheduling-artifact](https://github.com/alibaba/llm-scheduling-artifact)
  — "Artifact of OSDI '24 paper, 'Llumnix: Dynamic Scheduling for Large
  Language Model Serving'"
- **License:** Apache License 2.0 (code license — distinct from the
  paper's arXiv license above)
- **Pinned commit:** `a90824307249573f9c7548645c22994c65f83a08`
  (`main` branch, HEAD as of this audit; last pushed 2024-06-05 —
  the SAME DAY as the paper's arXiv v1 submission. No tags exist in this
  repo; this is the only reasonable pin.)
- This repository is a **frozen, standalone artifact snapshot** — it is
  its own fork of vLLM with Llumnix's changes applied directly into a
  `vllm/` source tree (not a separate package importing vLLM), created
  2024-04-26 specifically for OSDI Artifact Evaluation. It has **never
  been touched by any post-OSDI Llumnix development** (see §3 below) --
  this is exactly the "paper-era, not current main" guarantee this
  project's other faithful baselines rely on, and here it is unusually
  strong: this repo isn't just an old commit of an evolving project, it
  is a repository that was **never subsequently modified for anything
  other than its own OSDI-era artifact concerns**.

## 3. Relationship to `AlibabaPAI/llumnix` (v0) and `llumnix-project/llumnix` (v1)

**Do not confuse either of these with the pinned artifact above.** Both
post-date the OSDI 2024 submission and have continued to diverge:

- **`AlibabaPAI/llumnix`** ("Llumnix v0", Ray-based architecture): created
  2024-05-20 (before the artifact's last push, but a genuinely different,
  ongoing production project, not the AE snapshot), still active as of
  this audit (pushed 2026-03-12). Its own README explicitly calls this
  "Llumnix v0" and states it "is a better choice for local deployments and
  quick prototyping" — i.e., it is presented as the more research-facing
  fork, but it is still a **separately maintained, continuously evolving**
  codebase (v0.1.0 launched Nov 2024, vLLM backend updated to v0.6.3.post1
  in Jan 2025, etc.) — not the artifact.
- **`llumnix-project/llumnix`** ("Llumnix v1"): per `AlibabaPAI/llumnix`'s
  own README, "a new architecture designed to be more modular and
  cloud-native" — a March 2026 refactor, explicitly NOT what OSDI 2024
  described.
- **This project pins the OSDI 2024 artifact repository directly**
  (`alibaba/llm-scheduling-artifact`), not either of the above. This
  avoids the "does the current repo materially change scheduling
  semantics" question entirely for the CORE algorithm, since the artifact
  predates every subsequent architectural change by construction. Any
  future baseline wanting to model Llumnix v1's cloud-native
  re-architecture would need its own separate provenance audit.

## 4. Exact source files read (via `gh api ...contents/...?ref=a908243...`, live, not from memory)

- `vllm/core/request_scheduler.py` (674 lines) — the cluster-level
  `RequestScheduler`: initial dispatch policies, migration-pair selection
  (`need_migrate_balanced`/`need_migrate_prefill_v1`/`_v2`), instance-load
  computation entry points, auto-scaling.
- `vllm/core/scheduler.py` (558 lines) — the **per-instance** local
  scheduler (`Scheduler`): FCFS prompt admission, swap-based preemption,
  and `get_migrating_seq_groups`/`allocate_migrate_seq_groups` — the
  actual migration-candidate-selection and destination-admission logic.
- `vllm/core/policy.py`, `vllm/core/block_manager.py` — local scheduler's
  priority-sort and paged-KV block manager (vLLM-derived).
- `vllm/engine/llm_engine_manager.py` (714 lines) — the top-level
  orchestrator (Ray/asyncio plumbing): periodic migration-trigger check,
  `_migrate()` execution, auto-scaling triggers.
- `vllm/engine/llm_engine.py` (852 lines) — per-instance "Llumlet":
  `migrate_out`/`migrate_in` (GPU-to-GPU block transfer), `step()`
  (records `num_block_last_running_request`, used by the migration-benefit
  projection).
- `vllm/instance_info.py` (178 lines) — `InstanceInfo`/`InstanceLoadInfo`:
  the exact instance-load metric formulas.
- `vllm/config.py`, `vllm/engine/arg_utils.py` — verified default values
  for every dispatch/migration/scaling knob (see §A).
- `vllm/worker/cache_engine.py` — the migration executor (`send_gpu_cache`/
  `recv_gpu_cache`), confirming request-level (whole block-table) KV
  transfer, not chunked/layer-wise.
- Code-organization map from the artifact's own `README.md`, confirming
  the above file list is exactly what the artifact's authors themselves
  identify as "the key files that Llumnix creates or changes."

## A. Behavior directly supported by the pinned source

### A.1 System architecture

- **`LLMEngineManager`** (top-level orchestrator, Ray/asyncio): routes
  `generate()` calls through one of four modes (`original`/`loop`/
  `callback`/`global`); **`callback` is the default** (`dispatch_mode`
  default is `'local'`, which selects `generate_mode = 'callback'`, not
  `'global'`). Owns the periodic migration-trigger check and executes
  approved migration pairs.
- **`RequestScheduler`** (a Ray remote actor, logically the "cluster
  scheduler"): owns per-instance `InstanceInfo` state, initial dispatch
  policy, migration-pair selection, and auto-scaling decisions.
- **`Scheduler`** (one per instance, i.e., per "Llumlet"): a near-verbatim
  vLLM v0.1.x-era local scheduler (FCFS prompt admission bounded by
  `max_num_batched_tokens`/`max_num_seqs`, then swap-based preemption when
  out of blocks) — see §A.5 and §E for the exact relationship to
  `vllm_faithful`.

### A.2 Instance-load metric (`vllm/instance_info.py`) — VERIFIED default: `'consumed_speed'`

Two metrics exist (`load_metric` config, default `'consumed_speed'`):

- **`'consumed_speed'`** (default): `instance_load = -1 * (num_available_gpu_block / num_request)`
  where `num_available_gpu_block = num_free_gpu_block - num_watermark_block`
  and `num_request = num_running_request` (when prefill load-control is
  disabled, which is itself the default — see below). **Negated**: fewer
  available blocks per running request → more negative (= more loaded).
  An instance with zero running requests has `instance_load = -inf`
  (maximally *available*, i.e., idle).
- **`'used_ratio'`** (non-default alternative): `num_used_gpu_block / num_total_gpu_block`
  — simpler, unused by the default configuration.
- `enable_load_control_prefill` (default `False`) gates a more complex
  variant of `'consumed_speed'` that additionally accounts for waiting-time
  SLO pressure and reserves blocks for priority requests
  (`priority_reserved_blocks`) — **not** part of the core default path;
  see §E.

### A.3 Initial placement / dispatch — VERIFIED default: `'naive'` (round-robin, session-sticky)

`dispatch_strategy` config, **default `'naive'`**:

- `dispatch_naive`: the FIRST request of a new session is assigned via
  simple round-robin (`instance_ptr`, incremented mod `num_instance`);
  every SUBSEQUENT request in the SAME session goes to the SAME instance
  (`session_instance` sticky map). No load information is consulted at
  all for the default policy.
- Other dispatch strategies exist in the source (`'unbalanced'`,
  `'balanced'` — fewest dispatched requests, `'load'` — instance-load-
  based, `'block'` — KV-capacity-based) and a separate `'global'`
  dispatch mode with its own sub-strategies (`FFIT`/`FCFS`/`BE`/`SJF`/
  `LJF`) — these are real, implemented, alternative policies, **not** the
  verified default, and are not implemented here (see §E: "core algorithm,
  not every ablation," the same reasoning already applied to every prior
  faithful baseline in this project).

### A.4 Migration trigger, candidate selection, and destination admission — the core of this baseline

**Trigger (periodic, not continuous or per-arrival):**
`LLMEngineManager._update_instance_info` checks
`self.num_instance_info_update % (self.num_instance * need_migrate_frequency) == 0`
(`need_migrate_frequency` default `4`) — i.e., roughly once every 4
scheduling rounds *per instance*, aggregated across all instances. A
`self.migrating` guard prevents overlapping migration-decision rounds.

**Migration-pair selection (`RequestScheduler.need_migrate`,
default path `need_migrate_balanced` since `enable_load_control_prefill`
defaults to `False`):**
1. Sort all instances by `instance_load` descending (most loaded first).
2. **Migrate-out candidates**: instances with `num_killed_request > 0`
   (i.e., an instance with ANY currently-preempted/stalled request is
   ALWAYS a migrate-out candidate, regardless of its load number) **OR**
   `instance_load > migrate_out_load_threshold` (default threshold
   derived from CLI `migrate_out_threshold=1.5`, stored internally
   negated as `-1.5`).
3. **Migrate-in candidates**: instances with `num_killed_request == 0`
   AND `instance_load < migrate_out_load_threshold` — **the exact same
   threshold value gates both sides** in the default `need_migrate_balanced`
   path (a separate `migrate_in_threshold=3.0` config value exists but is
   not referenced by this function in the pinned source — it is used only
   by other, non-default code paths).
4. Pair the i-th most-loaded migrate-out instance with the i-th
   least-loaded migrate-in instance (for `i` in `range(min(len(out), len(in)))`).
5. **Migration-benefit condition** (this is the specific, quotable
   condition the task instructions warn not to approximate away):
   simulate load after moving exactly ONE request (`_get_instance_load_after_migrate`,
   using `num_block_last_running_request` — the block footprint of the
   LAST-scheduled running request from that instance's most recent `step()`
   — as the assumed migrated request's size). The pair is approved only if
   `right_load_after_mig <= migrate_out_load_threshold` **and**
   (`load_diff_after_mig > 0` and `load_diff_before_mig > load_diff_after_mig`)
   **or** the migrate-in instance was completely idle (`instance_load == -inf`).
   This is a genuine **benefit check**: migration is rejected if it would not
   actually improve (or would overshoot) the load balance between the pair.

**Migration candidate selection at the SOURCE instance
(`Scheduler.get_migrating_seq_groups`, `migrate_strategy` config,
**default `'LCFS'`**):**
- **`'LCFS'`** (Last-Come-First-Served, the default): scans the `running`
  list from the END (most-recently-admitted-into-running request first),
  picks the first request found that (a) has decoded at least one output
  token already (`get_output_len() > 0` — i.e., **only requests already in
  the decoding phase are migration candidates; a request still mid-prefill
  is never migrated**) and (b) is NOT a priority request
  (`priority_type == 0` — **priority requests are never migration
  candidates as a source**). Exactly ONE request is selected per migration
  event.
- Non-default alternatives `'SJF'`/`'LJF'` exist (shortest/longest current
  total sequence length among decoding-phase requests) — not implemented
  here (see §E).

**Destination admission (`Scheduler.allocate_migrate_seq_groups`,
run on the DESTINATION instance):**
- **Rejects ALL incoming migrations outright if the destination currently
  has ANY request in its own local `waiting` queue** (`if len(self.waiting): return all-False`)
  — a destination mid-prefill-admission-backlog never accepts a migration.
- Otherwise, for each candidate: if the request is itself still
  `WAITING`-status (can happen for `'SJF'`/`'LJF'`/re-tried candidates,
  though not for the default `'LCFS'` path per the decoding-phase-only
  filter above), it is simply appended to the destination's waiting queue
  (always accepted). If `RUNNING`-status: accepted only if
  `block_manager.can_allocate(seq_group)` (destination has KV capacity for
  the request's CURRENT total footprint) **and**
  `num_curr_seqs + 1 <= max_num_seqs` (destination has sequence-count
  capacity) — on KV-capacity failure the loop `continue`s to the next
  candidate (harmless for `'LCFS'`'s single-candidate-per-event default,
  relevant for other strategies), but on sequence-count-cap failure the
  loop `break`s (stops entirely).

### A.5 KV migration semantics — request-level (whole block-table), not chunked

`LLMEngine.migrate_out`/`migrate_in` (per-instance) transfer a request's
**entire current block table in one shot** via
`cache_engine.send_gpu_cache`/`recv_gpu_cache` (direct GPU-to-GPU tensor
copy, NCCL/gloo depending on `migrate_backend`). There is a "multi-stage"
variant (`migrate_out_multistage`) that overlaps the SOURCE's ongoing
computation with the transfer of already-stable blocks (the paper's
"KV cache is append-only" optimization, §4.2 of the paper) — this is a
**performance optimization of the transfer mechanism**, not a change to
*which* blocks eventually move or *when* the migration is logically
complete; it is out of scope here for the same reason DistServe's
bandwidth-aware transfer-cost model was excluded (network/timing detail,
not a scheduling decision) — see §E.

### A.6 Priority / SLO behavior (secondary, not the default path)

`priority_type` exists on every request (0 = normal, 1 = priority).
Priority requests: (a) are never selected as a migration source under the
default `'LCFS'` strategy (§A.4); (b) receive a reserved block quota
(`priority_reserved_blocks`) when `enable_load_control_prefill` is enabled
(non-default); (c) get a dedicated dispatch tie-break in `dispatch_load`
(non-default dispatch strategy). **Priority handling is layered on top of
the default `'naive'`+`'LCFS'`+`'consumed_speed'` path, not a replacement
for it** — this baseline implements the migration-source exclusion (a),
since it is intrinsic to the default `'LCFS'` candidate-selection function
itself, and documents (b)/(c) as excluded non-default extensions.

### A.7 Fragmentation and preemption-stall handling

- **Fragmentation** is handled *implicitly* by the load-balancing
  migration mechanism itself (moving requests off a fragmented/high-usage
  instance) — the pinned source does **not** define a separate,
  fragmentation-specific metric or trigger distinct from the
  `instance_load` computation above. There is no dedicated
  "defragmentation" code path to pin.
- **Preemption-stall handling**: `num_killed_request > 0` unconditionally
  qualifies an instance as a migrate-out candidate (§A.4, step 2) —
  regardless of its `instance_load` value. This is the pinned source's
  actual, specific mechanism for what the task calls "preemption-stall
  handling": an instance that has had to preempt (swap out) any request is
  always considered for migration relief, prioritized ahead of the
  ordinary load-threshold check.

## B. Simulator abstractions required to reproduce this

- A way to move a **running (already admitted, already decoding)**
  request from one independent GPU/instance to another, preserving decode
  progress, with a configurable transfer delay, KV-block release on the
  source and allocation on the destination, and the possibility of
  destination-side rejection. See §D — this does not exist yet, and is
  NOT the same primitive as the DistServe-style bridge queue (see §C).
- A per-instance load metric matching `'consumed_speed'` exactly, and a
  periodic (not continuous) trigger evaluated at the cluster level.
- Deterministic tie-breaking for dispatch (round-robin pointer) and for
  the source-instance/destination-instance pairing sort.

## C. Existing infrastructure reused unchanged

- `GPUState`, `ObservableGPUState`, per-GPU `active_request_ids`/
  `current_kv_tokens` — each Llumnix "instance" maps to one GPU/GPUState
  in this simulator, exactly as `distserve_faithful`/`tetriinfer_paper_reimplementation`
  map DistServe/TetriInfer "instances" onto GPUs.
- `KVBlockSpaceManager` (this project's own paged-KV-block accounting,
  built for `vllm_faithful`) — reused for each instance's local block
  bookkeeping, exactly as `vllm_faithful`/`sarathi_faithful`/
  `distserve_faithful`/`tetriinfer_paper_reimplementation` already do.
- `arrival_then_id` deterministic tie-breaking (`policies/tie_breaking.py`).
- `GPUConfig.role` is **not** reused for Llumnix instances — Llumnix
  instances are NOT prefill/decode-disaggregated in the pinned source
  (each instance runs BOTH phases, exactly like `vllm_faithful`'s
  single-pool model); every Llumnix instance is `role=None`
  (legacy/colocated), matching ordinary `GPUConfig` usage.

**Not reused: the DistServe/TetriInfer bridge queue
(`Simulator._migrating`/`_migrating_map`, `RequestPhase.MIGRATING`,
`ObservableState.migrating_queue`).** This is a deliberate, documented
choice (per the task's explicit instruction), not an oversight:

- The bridge queue models **prefill-done → decode-not-yet-started**
  handoff between two GPUs with **disjoint roles** (a request is never
  simultaneously prefilling and decoding). Its `transfer_ready_time` gate
  and `RequestPhase.MIGRATING` state specifically represent "prefill
  finished, KV in transit, decode has not begun."
- Llumnix migration is fundamentally different: it moves an
  **already-decoding** request (per §A.4's `'LCFS'` filter, which requires
  `get_output_len() > 0`) between two **role-identical, general-purpose**
  instances, purely for load-balancing/fragmentation/priority reasons —
  there is no phase transition at all. Reusing the bridge queue's
  semantics here would conflate "this request finished stage 1, hasn't
  started stage 2" with "this request is mid-stage-2, being relocated,"
  which are genuinely different states with different invariants (a
  bridge-queue request has produced zero output tokens by construction;
  a Llumnix migration candidate has produced at least one, by the same
  `'LCFS'` filter). See §D for the new, distinct primitive this requires.

## D. New shared infrastructure added in this stage

- **`Action.migrate: Dict[int, List[Tuple[int, int]]]`** — a new,
  fourth action verb (alongside `admit`/`preempt`/`swap`), mapping a
  source `gpu_id` to a list of `(request_id, destination_gpu_id)` pairs.
  Deliberately **not** reusing `Action.swap` (DistServe's swap moves a
  request to the SAME logical bridge queue it would use if it had simply
  finished prefill, always destined for re-admission on demand by the
  SAME policy that swapped it out; Llumnix's migration has an explicit,
  policy-chosen DESTINATION instance at the moment of migration, and
  moves between two ordinary running-instance slots, not into any bridge
  queue).
- **`Simulator._relocating: Dict[int, InternalRequest]`** in-flight
  tracking (new, separate from the bridge queue's `_migrating`/
  `_migrating_map`): holds requests that have left their source GPU's
  `_active` dict but have not yet been admitted onto their destination,
  reusing `InternalRequest.transfer_ready_time` for the readiness
  timestamp (same concept as the bridge queue's own field — see
  `RequestPhase.RELOCATING`'s docstring in simulator/request.py for why
  sharing the field name is fine even though the queue mechanism differs)
  plus a new `InternalRequest.migration_destination_gpu_id` field (the
  bridge queue has no destination concept at all — any decode-role GPU
  may claim a bridge-queue request; a migration's destination is fixed by
  the policy at the moment of migration and enforced by the simulator:
  admitting onto any other `gpu_id` is rejected).
- **`ObservableGPUState.incoming_migrations`**: per-destination-GPU view
  of transfer-ready relocating requests (mirrors `ObservableState.
  migrating_queue`'s "only transfer-ready are visible" rule, but scoped
  per-GPU since a relocation has exactly one destination rather than
  being claimable by any decode-role GPU).
- **`GPUState.evict(..., preserve_progress=True)`** is reused, with one
  extension needed specifically for live migration: it now ALSO preserves
  `admission_time` under `preserve_progress=True` (previously always
  reset to -1.0 even under swap) — required so a migrated request's
  overall service-time anchor (used for queuing-delay/TTFT/latency
  metrics) survives the relocation unchanged, matching the paper's own
  "near-zero overhead" framing. Verified harmless for `distserve_faithful`'s
  existing swap use: its own re-admission path never uses the new
  `is_relocation=True` bypass (see `GPUState.admit`), so it always
  overwrites `admission_time` with the current time regardless of what
  `evict()` leaves there — this change is invisible to it.
- **`GPUState.admit(..., is_relocation=True)`**: a new parameter that
  skips reassigning `admission_time`/`prefill_remaining` on admission,
  since a relocating request is resuming already-in-progress service, not
  being admitted for the first time.

### Implementation note: destination admission must re-check `vllm_faithful`'s OWN block-manager capacity, not just the simulator's raw GPUConfig

An early draft of this policy admitted transfer-ready incoming migrations
directly (adding them to `Action.admit` unconditionally once
transfer-ready), relying only on the *simulator's* own `GPUConfig`-based
admission check (`check_admission`, a raw token/sequence-count check).
This crashed under a randomized multi-config stress harness: `vllm_faithful`'s
own `KVBlockSpaceManager` (block-size-rounded, and populated only via its
own admission decisions) had no record of the migrated-in request at all,
and its defensive `_adopt_untracked_active` step (meant only for a fresh
policy instance inheriting pre-existing state) tried to retroactively
allocate blocks that no longer existed once the destination filled up in
the interim between migration approval and actual transfer completion.

The fix (`LlumnixFaithfulPolicy._migration_footprint`): the exact KV-token
footprint of a candidate is recorded at the moment migration is approved
(from the source's own block manager, before it is freed there), and
re-checked against the *destination's* block manager — with an explicit
`can_allocate`/`allocate` call performed by this policy itself — at the
moment the request actually becomes transfer-ready. A request that no
longer fits by then is simply left in `incoming_migrations` for a later
round (matching the pinned reference's own `allocate_migrate_seq_groups`
per-candidate rejection, not a crash or a dropped request). This is
exactly the pinned source's own two-stage design (migration approved based
on a load *projection*; actual admission re-checked, separately, at
transfer-completion time) — the crash came from skipping the second stage
entirely, not from a fidelity gap in the algorithm itself.

## E. Differences that cannot be (or are not) faithfully represented

- **Local per-instance scheduler uses swap-based preemption, not
  recompute.** The pinned `Scheduler._schedule` (§A.1) preempts via
  `blocks_to_swap_out`/`_preempt`, matching vLLM's SWAP preemption mode —
  this project's existing `vllm_faithful` baseline (pinned to vLLM
  v0.1.0) models only vLLM's RECOMPUTE preemption mode (vLLM v0.1.0's own
  default), a difference already disclosed in `vllm_faithful`'s own
  reference doc as an intentional, documented scope choice. This baseline
  reuses `vllm_faithful`'s recompute-based local scheduler as-is (per
  task instruction: "reuse or compose with vllm_faithful semantics where
  possible; do not duplicate scheduler logic") rather than implementing a
  second, swap-based local scheduler variant — a disclosed simplification
  at the LOCAL level, orthogonal to the CLUSTER-level migration fidelity
  this baseline is actually about.
- **Multi-stage/overlapped migration transfer** (§A.5): modeled as a
  single configurable transfer delay, not a two-phase overlap-with-compute
  optimization — a timing/performance detail, not a scheduling decision.
- **Non-default dispatch strategies** (`'unbalanced'`/`'balanced'`/
  `'load'`/`'block'`/global `FFIT`/`FCFS`/`BE`/`SJF`/`LJF`) and non-default
  migration strategies (`'SJF'`/`'LJF'`) are not implemented — only the
  verified DEFAULT (`'naive'` dispatch, `'LCFS'` migration) is, per the
  "core algorithm, not every ablation" principle already applied
  throughout this project.
- **Auto-scaling** (elastic instance count) is excluded — this simulator
  has no dynamic-instance-provisioning concept for ANY baseline (same
  exclusion reasoning as TetriInfer's instance-flip).
- **`enable_load_control_prefill`'s more complex load formula and
  `need_migrate_prefill_v1`/`_v2`** are excluded — non-default,
  priority/SLO-specific extensions layered on top of the core path (§A.6).
- **Fault tolerance** (Ray actor failure recovery) — infrastructure
  concern, not a scheduling decision.

## Do not use `AlibabaPAI/llumnix` or `llumnix-project/llumnix` blindly

Both have continued to evolve substantially past the OSDI 2024 artifact
(v0.1.0 launched Nov 2024; vLLM backend bumped to v0.6.3.post1 Jan 2025;
a full v1 architecture rewrite as of Mar 2026). Any claim made about this
baseline, or about the pinned artifact's behavior, is a claim about the
`alibaba/llm-scheduling-artifact` repository at commit
`a90824307249573f9c7548645c22994c65f83a08`, not about either currently
evolving project.

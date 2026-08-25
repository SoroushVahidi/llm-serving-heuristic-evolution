# Pinned vLLM Reference — `vllm_chunked_prefill_faithful` Baseline

This document records the exact upstream reference used to build the
`vllm_chunked_prefill_faithful` simulator-side scheduling baseline (see
`src/llmserveopt/policies/vllm_chunked_prefill_faithful.py`). It is the
full-rigor follow-up to `docs/vllm_chunked_prefill_faithful_design_audit.md`
(a preliminary audit that explicitly deferred this document), matching the
source-read rigor of `docs/vllm_faithful_scheduler_reference.md` and
`docs/sarathi_faithful_scheduler_reference.md`. Every claim below was
checked live against the actual vLLM repository at the pin in question via
`gh api repos/vllm-project/vllm/contents/...?ref=<sha>` (not recalled from
training-data memory), during this session.

## Paper / project

vLLM has no single paper covering chunked prefill specifically (chunked
prefill originates from Sarathi-Serve's OSDI 2024 paper — see
`docs/sarathi_faithful_scheduler_reference.md` — and was upstreamed into
vLLM separately). This baseline is pinned directly to vLLM source code, the
same way `vllm_faithful` is.

## Official repository

- **Repo:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **License:** Apache License 2.0

## Pinned commit/tag

**Tag `v0.4.2`, commit `c7f2cf2b7f67bce5842fedfdba508440fe257375`**
(committer date 2024-05-05T04:28:58Z, confirmed live via
`gh api repos/vllm-project/vllm/git/refs/tags/v0.4.2` and
`gh api repos/vllm-project/vllm/commits/c7f2cf2b7f67bce5842fedfdba508440fe257375`).
Rationale carried over unchanged from
`docs/vllm_chunked_prefill_faithful_design_audit.md`: `v0.4.2` is the first
tagged release whose own release notes state "Chunked prefill is ready for
testing" (the earlier `v0.4.0` release notes call it "progress", and its
source still has scaffolding-era `TODO`s); `v0.4.2` is the first point
chunked prefill is a complete, documented, citable feature rather than
in-progress work — the same kind of criterion `vllm_faithful` used to prefer
`v0.1.0` over the `submission` tag.

Source files read directly (via `gh api repos/vllm-project/vllm/contents/...
?ref=c7f2cf2b7f67bce5842fedfdba508440fe257375`) to derive this baseline:

- `vllm/core/scheduler.py` (1163 lines) — `SchedulingBudget`,
  `Scheduler._schedule_running`, `_schedule_swapped`, `_schedule_prefills`,
  `_schedule_default`, `_schedule_chunked_prefill`, `_get_num_new_tokens`
- `vllm/core/block_manager_v1.py` (625 lines) — `BlockSpaceManagerV1`:
  `can_allocate`, `allocate`, `can_append_slots`, `append_slots`
- `vllm/core/interfaces.py` (113 lines) — `AllocStatus`,
  `BlockSpaceManager.get_block_space_manager_class`
- `vllm/config.py` (1225 lines) — `SchedulerConfig` (chunked-prefill
  defaults, `use_v2_block_manager`)
- `vllm/engine/arg_utils.py` (649 lines) — `EngineArgs` CLI defaults
  (`block_size=16`, `max_num_seqs=256`, `use_v2_block_manager=False`,
  `enable_chunked_prefill=False`)
- `vllm/sequence.py` (766 lines) — `SequenceGroup`/`SequenceStatus`
  (referenced for `is_prefill()`/`get_max_num_running_seqs()` semantics,
  not separately reproduced here since this project's `Request`/
  `InternalRequest` model already covers the single-sequence-per-group case)

## Chunked prefill enablement: opt-in, and its own default budget

Confirmed live in `vllm/config.py` (`SchedulerConfig.__init__`,
around line 607-634): `enable_chunked_prefill: bool = False` — still opt-in
at this pin, exactly matching what every real vLLM 0.24.0 job in this whole
investigation does (`--enable-chunked-prefill` passed explicitly). When
enabled **and** `max_num_batched_tokens` is not explicitly set, the
scheduler config's own default becomes **512**, not the non-chunked
default (`max(max_model_len, 2048)`):

```python
if max_num_batched_tokens is not None:
    self.max_num_batched_tokens = max_num_batched_tokens
else:
    if enable_chunked_prefill:
        # For chunked prefill, choose the well-tuned batch size.
        self.max_num_batched_tokens = 512
    else:
        self.max_num_batched_tokens = max(max_model_len, 2048)
```

This is why `vllm_chunked_prefill_faithful`'s own standalone default is
**512** (`DEFAULT_MAX_NUM_BATCHED_TOKENS`), not `vllm_faithful`'s inherited
**2560** (`vllm_faithful`'s value is the *unrelated* non-chunked-mode
default computed a different way at its own v0.1.0 pin — see that
document). `max_num_seqs` default is unchanged (`256`, `arg_utils.py`
line 46), as is `block_size` (`16`) and the block manager's `watermark`
(`0.01`, `block_manager_v1.py` line 220).

## `SchedulingBudget`: the per-iteration shared token/sequence budget

`vllm/core/scheduler.py` lines 41-99. A per-`schedule()`-call object with
`token_budget` (= `max_num_batched_tokens`) and `max_num_seqs`, tracking
`num_batched_tokens`/`num_curr_seqs` incrementally as sequence groups are
scheduled across **all three** phases below (running, swapped, prefills) —
one shared budget object threaded through the whole iteration, not a
per-phase budget. `can_schedule()` checks both dimensions at once;
`remaining_token_budget()` is what chunk-size computations (`_get_num_new_
tokens`) read to cap a chunk. A `TODO` comment on the class itself
(`"...meaning it could be updated more than once when scheduling RUNNING
requests. Since this won't happen if we only have chunked prefill
scheduling, we can remove this feature... when chunked prefill is enabled
by default"`) confirms this class carries request-id-aware de-duplication
specifically to support **both** `_schedule_default` (non-chunked) and
`_schedule_chunked_prefill` sharing one implementation — a v0.4.2-era
implementation detail, not a chunked-prefill-specific design choice this
baseline needs to reproduce.

This project's policies (`vllm_faithful`, `sarathi_faithful`) don't build a
separate budget object either — they track `num_batched_tokens`/
`num_curr_seqs` as plain local integers inline in the scheduling loop.
`vllm_chunked_prefill_faithful` follows the same established, simpler
convention rather than introducing a new `SchedulingBudget`-equivalent
class purely for its own sake.

## Algorithm summary: `_schedule_chunked_prefill` (as implemented at this pin)

Read directly from `scheduler.py` lines 805-890 (`_schedule_chunked_prefill`),
805-712 (`_schedule_running`/`_schedule_prefills`, shared with the
non-chunked path via the `enable_chunking` parameter), and `_schedule`
(line 892: `if self.scheduler_config.chunked_prefill_enabled: return
self._schedule_chunked_prefill() else: return self._schedule_default()`
— confirming `_schedule_chunked_prefill` is a **structurally different**
method from `vllm_faithful`'s pinned `_schedule_default`-equivalent
algorithm, not a variant of it).

### Queues

Three queues, same names as `vllm_faithful`'s pin: `waiting`, `running`,
`swapped`. **Structural difference from `vllm_faithful`'s v0.1.0 pin**:
at v0.1.0, `running` contains only fully-admitted (decode-phase or
about-to-decode) sequences — a waiting group's entire prompt is admitted
in one shot, so nothing "partially prefilling" ever sits in `running`. At
this v0.4.2 chunked-prefill pin, `running` can contain **both** decode-phase
sequences **and** sequences still mid-prefill from a previous iteration
(their prompt admitted, KV blocks reserved, but not yet fully computed) —
confirmed directly by `SchedulerRunningOutputs`'s own fields
(`decode_seq_groups` **and** `prefill_seq_groups`, scheduler.py lines
161-194) and by `_schedule_chunked_prefill`'s own docstring: "schedule as
many decoding requests as possible... schedule chunked prefill requests
that are not finished... schedule swapped requests... schedule new prefill
requests" (lines 806-817).

### Per-iteration scheduling order (`_schedule_chunked_prefill`, lines 819-890)

1. **`_schedule_running(self.running, budget, curr_loras, fcfs_policy,
   enable_chunking=True)`** — a **single combined FCFS-by-arrival-time loop**
   over `self.running`, which (per the structural point above) mixes
   decode-phase and continuing-prefill-phase sequences together. For each
   sequence group, in arrival order:
   - Compute `num_new_tokens` via `_get_num_new_tokens` (lines 1139-1163):
     1 for a decode-phase sequence; for a still-prefilling sequence,
     `min(remaining_prompt_tokens, budget.remaining_token_budget())` (a
     genuine chunk, capped by whatever budget is left after earlier
     candidates in this same loop already consumed some).
   - **If `num_new_tokens == 0` (budget exhausted): `break` — stop
     scheduling the *rest* of the running queue entirely this iteration**,
     read directly from source (line 415-416: `if num_running_tokens == 0:
     break`), not `continue`-and-try-the-next-one. Confirmed exact
     structural match to `vllm_faithful`'s and `sarathi_faithful`'s own
     "close admission entirely, don't skip-and-retry" convention, just
     applied to the *running* phase here instead of only `waiting`.
   - Otherwise, `_can_append_slots`/`_append_slots` (KV-slot capacity check)
     is called **unconditionally for both decode and continuing-prefill
     candidates** — if it fails, preempt the **lowest-priority remaining
     candidate in this same running-queue pass** (`running_queue.pop()`,
     i.e. from the *back*, the most-recently-arrived remaining one; if none
     left, preempt the current candidate itself) — **identical
     victim-selection algorithm** to `vllm_faithful`'s own pinned running-
     queue loop (see `docs/vllm_faithful_scheduler_reference.md`), just over
     a queue that can now contain continuing-prefill candidates as
     preemption victims too, not only decode ones.
   - **This is the single most important structural fact this reference
     doc exists to record**: unlike Sarathi-Serve's own scheduler (which
     explicitly reserves decode budget in a dedicated first pass — Phase
     1a — before any prefill chunk gets a look at the budget; see
     `docs/sarathi_faithful_scheduler_reference.md`), **v0.4.2's chunked-
     prefill `_schedule_running` has no separate decode-priority phase at
     all** — decode-phase and continuing-prefill-phase sequences compete
     for the *same shared budget in one FCFS-by-arrival-time pass*. A
     still-prefilling sequence that happens to sort earlier by arrival time
     can consume the entire shared per-step token budget before a
     later-arriving decode-phase sequence gets its turn, causing that
     decode sequence to receive `num_new_tokens == 0` and `break` out
     un-scheduled this iteration (i.e., **it stalls**) — the exact failure
     mode Sarathi-Serve's stall-free design explicitly targets. This is not
     an inference from behavior; it is read directly from the fact that
     `_schedule_running`'s FCFS sort (`policy.sort_by_priority`, line 409)
     is applied to the *entire* `self.running` queue with no decode/prefill
     partitioning anywhere in this function, contrasted with Sarathi's own
     `SarathiScheduler._schedule`, which explicitly loops decode-phase
     sequences (Phase 1a) to completion *before* touching any
     continuing-prefill sequence (Phase 1b) at all.
2. **`_schedule_swapped`** — not modeled, same exclusion as `vllm_faithful`
   (every request here is single-sequence, so the pinned scheduler's own
   preemption-mode selection always picks `RECOMPUTE`, never `SWAP`; see
   Exclusions below).
3. **`_schedule_prefills(self.waiting, budget, curr_loras,
   enable_chunking=True)`** — admits new requests from `waiting`, FCFS,
   using whatever budget phases 1-2 left over. For each waiting group, in
   arrival order: `can_allocate` (KV-capacity/watermark check) — stop
   admitting entirely (`break`) on the first `AllocStatus.LATER`
   (line 660-661); otherwise compute
   `num_new_tokens = min(prompt_tokens, budget.remaining_token_budget())`
   (line 640-642) and stop admitting entirely if that chunk would be 0 or
   `max_num_seqs` would be exceeded (line 688-691: `if (num_new_tokens == 0
   or not budget.can_schedule(...)): break`). Otherwise allocate blocks for
   the **full** prompt up front (`_allocate_and_set_running` →
   `block_manager.allocate(seq_group)`, which reserves blocks for
   `len(seq.logical_token_blocks)` — the sequence's full known prompt
   length, since the prompt is entirely known at admission time even though
   only the first chunk will be *computed* this iteration) and admit it
   with a budget-bounded first chunk.

### Why continuing-prefill sequences never actually fail `can_append_slots`

Traced directly from `block_manager_v1.py`'s `append_slots` (lines 385-429):
it asserts `len(block_table) == len(logical_blocks) - 1` — i.e. it only
ever supports growing by **exactly one** new physical block per call,
which would be wrong for a multi-hundred-token prefill chunk **if** blocks
were reserved incrementally. They are not: `logical_token_blocks` reflects
the sequence's **full** known prompt length from the moment it is created
(the prompt is entirely known up front, unlike output tokens), and
`can_allocate`/`allocate` at admission (`_schedule_prefills`) already
reserve blocks for that full length. So by the time a continuing-prefill
sequence reaches `_schedule_running` in a later iteration,
`len(block_table) == len(logical_blocks)` already — `append_slots` takes
its early `last_block.ref_count == 1` return-`{}` path, a structural no-op.
This independently confirms (not just by analogy to
`sarathi_faithful`'s own reference doc) that this project's existing
`KVBlockSpaceManager` convention — reserving a request's full prompt token
count at admission, never incrementally during prefill chunks — is
faithful to the real v0.4.2 mechanics, not merely a simulator convenience.
Consequently, `vllm_chunked_prefill_faithful` (like `sarathi_faithful`)
never calls `can_append_slot`/`append_slot` for a continuing-prefill
candidate — only for decode-phase ones — while still letting
continuing-prefill candidates be selected as preemption **victims** by a
decode candidate's slot search (see the algorithm point above): the
capacity check is a structural no-op for them, but their eligibility to be
evicted is not.

### KV-cache block management: `block_manager_v1.py` — v1 is the default

Confirmed live (`gh api repos/vllm-project/vllm/contents/vllm/core?ref=
c7f2cf2b7f67bce5842fedfdba508440fe257375`): at this pin, `vllm/core/`
contains both `block_manager_v1.py` and `block_manager_v2.py`, selected in
`scheduler.py`'s own `__init__` (lines 274-276):

```python
BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
    version="v2" if self.scheduler_config.use_v2_block_manager else "v1")
```

`use_v2_block_manager: bool = False` is the confirmed default in both
`vllm/config.py` (`SchedulerConfig.__init__`) and `vllm/engine/arg_utils.py`
(`EngineArgs.use_v2_block_manager`, line 42) — **`block_manager_v1` is the
reference behavior at this pin**, exactly matching what
`docs/vllm_chunked_prefill_faithful_design_audit.md` recommended and what
this project's existing `KVBlockSpaceManager` already models (built against
v0.1.0's own, single, unversioned block manager, whose `can_allocate`/
`allocate`/`can_append_slots`/`append_slots` semantics at line 259-338 of
`block_manager_v1.py` are read-verified to be unchanged from the v0.1.0
pin: same watermark-guarded `AllocStatus.OK/LATER/NEVER` three-way check,
same free-block-count heuristic for `can_append_slots`). `block_manager_v2.py`
exists at this pin and is an **explicit, named exclusion** below — not
silently ignored.

### Preemption (`Scheduler._preempt`, lines 1036-1064)

Unchanged from the `vllm_faithful` v0.1.0 pin: `RECOMPUTE` (discard blocks,
reinsert at front of `waiting`) if the sequence group has exactly one
sequence, `SWAP` otherwise. Every request in this project is single-sequence,
so `RECOMPUTE` is always selected — same exclusion rationale as
`vllm_faithful`. Verified this did not change between the two pins by
reading both `_preempt` implementations directly.

## Differences from historical `vllm_faithful` (pinned v0.1.0)

| | `vllm_faithful` (v0.1.0) | `vllm_chunked_prefill_faithful` (this baseline, v0.4.2) |
|---|---|---|
| Scheduler entry point | `_schedule()` → `_schedule_default()`-equivalent only | `_schedule()` → `_schedule_chunked_prefill()` (chunked prefill always enabled for this baseline — it exists specifically to model that path) |
| Prompt admission | All-or-nothing per iteration | Partial/chunked, budget-bounded (`_get_num_new_tokens` with `enable_chunking=True`) |
| Long-prompt (`prompt_tokens > max_num_batched_tokens`) requests | Never admitted (confirmed empirically by this project's own long-context benchmark-pack fixture) | Admitted incrementally over multiple iterations |
| `running` queue contents | Decode-phase sequences only | Decode-phase **and** continuing-prefill-phase sequences, mixed in one FCFS pass |
| Decode-priority mechanism | N/A (no prefill/decode split exists in this pin's scheduling model at all) | **None** — no explicit decode-first phase; decode and continuing-prefill compete for shared budget in pure arrival order (see algorithm section above) |
| `max_num_batched_tokens` default | 2560 (this pin's own non-chunked default) | 512 (this pin's own chunked-prefill-enabled default) |
| `max_num_batched_tokens` role | Hard per-iteration ceiling on *admission* only | Hard per-iteration ceiling on *all* scheduled work (running + swapped + new), shared across all three phases |
| Preemption mechanics | Recompute, `Action.preempt`/`GPUState.evict()` | Unchanged — confirmed by direct source comparison, not assumed |
| Block manager | `block_manager_v1`-equivalent (only version at v0.1.0) | `block_manager_v1` (default; `block_manager_v2` exists at this pin but is explicitly excluded — see below) |

## What is explicitly excluded from this baseline

- **`block_manager_v2`.** Exists at this pin (`vllm/core/block_manager_v2.py`),
  selected only when `use_v2_block_manager=True`, which is not the default
  and is not used by any real vLLM job referenced by this investigation.
  Excluded per explicit instruction; `block_manager_v1` (the default) is
  what this baseline models, via the existing `KVBlockSpaceManager`.
- **Prefix caching** (`enable_prefix_caching` / `--enable-chunked-prefill`
  does not imply it). No real vLLM job in this investigation's benchmark
  pack enabled it; `KVBlockSpaceManager` does not model content-addressed
  block sharing at all. Excluded, matching the design audit's scope.
- **Speculative decoding, disaggregated prefill/decode for this baseline
  specifically, the `v1` engine rewrite, and any scheduler feature added
  after this pin.** None of these exist in the v0.4.2 source read for this
  document. (This project separately has `distserve_faithful` for
  disaggregation, pinned to its own, different upstream reference — not
  related to this baseline.)
- **Copy-on-write / sequence forking (beam search)** and **swap-based
  preemption** — same exclusion and rationale as `vllm_faithful` (every
  request here is single-sequence; the pinned scheduler's own logic always
  selects `RECOMPUTE`).
- **LoRA-aware scheduling** (`curr_loras`, `lora_int_id` gating in
  `_schedule_running`/`_schedule_prefills`) — this project has no LoRA
  concept; every `curr_loras`-gated branch in the pinned source is a no-op
  for a LoRA-free workload, so it is omitted rather than modeled as
  always-trivially-true infrastructure.
- **`delay_factor`/`_passed_delay`** (an optional scheduling-delay knob to
  let the waiting queue fill up before admitting) — defaults to `0` (always
  "passed") in `EngineArgs`, not exercised by any real vLLM job in this
  investigation, and not modeled by either existing faithful baseline
  either. Omitted for the same reason.
- **Hardware/runtime performance modeling.** Same as both existing
  reference docs: this pin's scheduler makes *admission, chunking, and
  preemption decisions*; it says nothing about how long a prefill chunk or
  decode step actually takes on real hardware. That remains
  `ServiceModel`/`CalibratedServiceModel`'s job, unchanged by this baseline.

## Known simulator-execution-layer limitation (disclosed, not fixed here)

This baseline's *admission-time* accounting faithfully mirrors v0.4.2's
lack of an explicit decode-priority phase (see the algorithm section above)
— but the simulator's own shared execution step,
`GPUState._step_phase15` (`src/llmserveopt/simulator/gpu.py`), computes
`prefill_budget` from `service_model.step_token_budget - len(decoding)`
**identically regardless of `ServiceModel.decode_first`'s value** — both
branches of its `if service_model.decode_first: ... else: ...` perform the
exact same computation (verified by reading the code and empirically, by
running the same scenario with `decode_first=True` and `decode_first=False`
and observing byte-identical output — see
`docs/vllm_chunked_prefill_faithful_root_cause_analysis.md`). This means
**every** policy using Phase-1.5 execution — `sarathi_faithful`,
`vllm_chunked_prefill_faithful`, or any future one — gets Sarathi's own
"decode is unconditionally protected from prefill" property *for free* at
the execution layer, regardless of what that policy's own scheduler
actually decided about ordering. `vllm_chunked_prefill_faithful`'s
admission-side accounting is faithful to the pinned source's lack of a
decode-priority guarantee, but the simulator cannot currently *observe* the
consequence of that lack (a genuinely stalled decode sequence) the way real
hardware can. This is a pre-existing simulator-infrastructure property, not
something introduced by or specific to this baseline, and is intentionally
left unmodified here — see
`docs/vllm_chunked_prefill_faithful_root_cause_analysis.md` for the full
analysis and why fixing it is out of scope for this change (changing
shared `GPUState` execution semantics would affect `sarathi_faithful`'s
existing, already-relied-upon numbers).

## Do not use current `main` blindly

vLLM's scheduler has changed substantially since this pin (the `v1` engine
rewrite, prefix caching becoming default in later versions, further
chunked-prefill tuning, speculative decoding integration, etc.). None of
that is represented here. Any claim made about
`vllm_chunked_prefill_faithful` is a claim about vLLM **as of commit
`c7f2cf2b7f67bce5842fedfdba508440fe257375` (tag `v0.4.2`)**, not about vLLM
today.

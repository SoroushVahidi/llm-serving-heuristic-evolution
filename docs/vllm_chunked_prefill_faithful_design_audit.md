# Design/Provenance Audit: `vllm_chunked_prefill_faithful` (not yet implemented)

This is a design and provenance audit for the new, separately-pinned
baseline recommended in `docs/wulver_vllm_kv_pressure_results.md`'s
"Recommended vLLM simulator strategy: B" section, written across two
2026-07-19 sessions (the Sarathi float16 runtime-validation task, and a
follow-up CPU-only finalization pass). **No code was written for this
baseline.** `vllm_faithful` was not modified. This document only
determines what a future implementation would need, per explicit
instruction not to implement it yet — a full faithful scheduler
reimplementation (see how much primary-source work
`docs/vllm_faithful_scheduler_reference.md` and
`docs/sarathi_faithful_scheduler_reference.md` each represent) is not
something to do as a side effect of an unrelated GPU validation task.
Every specific commit, config default, and file layout claim below was
checked live via `gh api` against the actual vLLM repository at the pin in
question, not recalled from training-data memory of vLLM's history.

## Recommended pin: vLLM tag `v0.4.2`, commit `c7f2cf2b7f67bce5842fedfdba508440fe257375`

Determined by live GitHub API lookup during this session (`gh api
repos/vllm-project/vllm/releases --paginate`, `gh api repos/vllm-project/
vllm/git/refs/tags/v0.4.2`), not from memory, matching the methodology of
the two existing pinned-reference docs.

- **`v0.4.0`** (2024-03-30, commit `51c31bc10ca7c48b580cd58fcd741ba4d6db4447`)
  is the earliest tagged release whose own notes mention chunked prefill at
  all ("Progress in chunked prefill scheduler (#3236, #3538)") — but its own
  wording ("progress") and an in-code `TODO` comment about removing a
  workaround "when chunked prefill is enabled by default" indicate the
  feature is still incomplete/scaffolding at this point, not a clean
  reference to pin to.
- **`v0.4.2`** (2024-05-05) is the first release whose own notes explicitly
  state **"Chunked prefill is ready for testing"**, linking a dedicated
  performance-docs page, alongside prompt-logprob support for chunked
  prefill and doc cleanup PRs. This is the same kind of criterion
  `vllm_faithful` used to justify pinning `v0.1.0` over the earlier
  `submission` tag ("the cleaner, citable reference point") — `v0.4.2` is
  the first point where chunked prefill is a complete, documented,
  citable feature rather than in-progress scaffolding.
- Confirmed via the live repo contents at this pin
  (`vllm/core/scheduler.py`): `chunked_prefill_enabled` is a real
  `SchedulerConfig` field (`enable_chunked_prefill: bool = False` in
  `vllm/config.py` at this pin, still opt-in — matching every real vLLM
  0.24.0 job in this whole investigation, which all pass
  `--enable-chunked-prefill` explicitly rather than relying on a default).

This is a recommendation, not a final decision — picking the version
"maximally relevant" to the real vLLM 0.24.0 runtime used on Wulver
(per `docs/wulver_vllm_kv_pressure_results.md`'s Part 8 note) versus the
version that most cleanly represents chunked prefill's *initial* stable
form (this pin) is a real tradeoff a human should weigh, the same way
`vllm_faithful`'s own v0.1.0-vs-`submission` choice was weighed and
recorded explicitly rather than assumed.

## Chunked-prefill scheduler code paths at this pin (`vllm/core/scheduler.py`)

Read live via `gh api repos/vllm-project/vllm/contents/vllm/core/
scheduler.py?ref=v0.4.2`. Key structures, by name:

- **`SchedulingBudget`**: a per-iteration token/sequence budget object
  (`token_budget: int`, `num_batched_tokens`, `num_curr_seqs`), with
  `can_schedule()` and `remaining_token_budget()` — this is the direct
  analogue of `ServiceModel.step_token_budget` (already used by
  `sarathi_faithful`) and of Sarathi-Serve's own `chunk_size`-bounded
  admission (`docs/sarathi_faithful_scheduler_reference.md`'s Phase 1b/2).
  A `TODO` comment on this class notes it is "request_id-aware" as a
  workaround that "can be removed... when chunked prefill is enabled by
  default" — a v0.4.2-specific implementation detail future archaeology
  should re-check against whatever exact commit is ultimately pinned.
- **`enable_chunking` parameter**, threaded through `_schedule_running()`
  and `_schedule_swapped()`/`_schedule_prefills()`: when true, a
  `SequenceGroup` can be scheduled with `token_chunk_size < ` its full
  remaining prompt, gated by `budget.can_schedule()` rather than an
  all-or-nothing admission check — this is the structural difference from
  `vllm_faithful`'s pinned v0.1.0 behavior (`docs/
  vllm_faithful_scheduler_reference.md`: "a waiting group's entire prompt
  is admitted in one iteration ... or not admitted at all").
- Running (already-admitted, possibly-still-prefilling) sequences are
  scheduled before new admissions from `waiting`, consistent with the
  release notes' "prioritizes decode" framing and with `vllm_faithful`'s
  existing Phase-1-running-first structure — this part likely does *not*
  need to change relative to the existing pattern.

## KV block manager at this pin: two implementations, v1 is the default

Checked live (`gh api repos/vllm-project/vllm/contents/vllm/core?ref=v0.4.2`):
at v0.4.2, `vllm/core/` contains **both** `block_manager_v1.py` and
`block_manager_v2.py`, selected by `SchedulerConfig.use_v2_block_manager`
(`vllm/config.py`), which **defaults to `False`** — so the reference
behavior at this pin is `BlockSpaceManagerV1`, whose `allocate`/
`can_allocate`/`append_slots` method signatures match what
`vllm_faithful`'s existing `KVBlockSpaceManager` already models (built
against v0.1.0's single, unversioned block manager). This is a real,
concrete complication a v0.1.0-only pin doesn't have, and one worth
recording plainly: it means the existing shared `KVBlockSpaceManager`
infrastructure is likely still directly reusable for a v0.4.2-pinned
policy (since v1 is the default), but `block_manager_v2.py`'s existence at
this pin should be an explicit, named exclusion in the eventual reference
doc (mirroring how the two existing reference docs each carry an explicit
exclusions section) rather than silently ignored.

## Differences from historical `vllm_faithful` (pinned v0.1.0)

| | `vllm_faithful` (v0.1.0) | `vllm_chunked_prefill_faithful` (proposed, v0.4.2) |
|---|---|---|
| Prompt admission | All-or-nothing per iteration (`docs/vllm_faithful_scheduler_reference.md`) | Partial/chunked, budget-bounded (`SchedulingBudget.can_schedule`) |
| Long-prompt (>`max_num_batched_tokens`) requests | Never admitted (this session's audit finding) | Admitted incrementally over multiple iterations |
| Preemption mechanics | Recompute or swap, existing `Action.preempt`/`GPUState.evict()` | Expected unchanged (chunking is an admission/scheduling change, not a preemption-mechanics change) — to be confirmed against the pinned source, not assumed |
| `max_num_batched_tokens` role | Hard per-iteration ceiling on *admission* only | Hard per-iteration ceiling on *all* scheduled work (running + swapped + new), shared via `SchedulingBudget` |

## Required shared simulator primitives

Per this session's read of `docs/sarathi_faithful_scheduler_reference.md`'s
own infrastructure audit table, most of what a chunked-prefill vLLM policy
needs **already exists**, built for `sarathi_faithful`:

| Capability | Status for a new `vllm_chunked_prefill_faithful` |
|---|---|
| Chunked prefill execution | Reuse as-is — `GPUState._step_phase15` (built for `sarathi_faithful`, itself reused from Phase 1.5) |
| Chunked prefill admission decision | Reuse the *mechanism* `sarathi_faithful` added (policy-side partial-chunk admission), but with v0.4.2's specific `SchedulingBudget` accounting and decode-priority order, not Sarathi-Serve's `chunk_size` static value |
| KV/paged-block memory | Reuse as-is — same `KVBlockSpaceManager` already shared by both existing faithful baselines |
| Preemption | Reuse as-is, pending confirmation the pinned v0.4.2 source didn't change preemption semantics from v0.1.0 |
| TTFT/TPOT definitions | Reuse as-is (same "first decode step" / per-step decode framing already used) |

**Net expectation**: this should be achievable as a single new policy file
analogous to `sarathi_faithful.py`, not a new simulator-infrastructure
commit — the same conclusion `sarathi_faithful`'s own audit reached.

## Backward-compatibility strategy

- New file: `src/llmserveopt/policies/vllm_chunked_prefill_faithful.py`.
- New reference doc: `docs/vllm_chunked_prefill_faithful_scheduler_reference.md`
  (this document is a preliminary audit, not that doc — the real one needs
  the same full source-read rigor as the two existing reference docs).
- New registry entry in `EXTERNAL_BASELINE_REGISTRY`
  (`src/llmserveopt/policies/external_baselines_registry.py`):
  `"vllm_chunked_prefill_faithful"`, `pinned_source="vLLM commit
  c7f2cf2b7f67bce5842fedfdba508440fe257375 (tag v0.4.2)"` (or whatever
  commit is finally chosen), `requires_chunked_prefill_scheduling=True` —
  a field the schema already supports (currently `True` only for
  `sarathi_faithful`).
- **Zero changes** to `vllm_faithful.py`, its registry entry, its
  reference doc, or any test that depends on v0.1.0 semantics.
- `run_gpu_external_validity_audit.py` / `run_sarathi_gpu_smoke_and_validation.py`'s
  `run_simulator_scenario()` functions would gain a third policy entry
  (`"vllm_chunked_prefill_faithful"`) alongside the existing two — additive,
  not a change to the existing `vllm_faithful`/`sarathi_faithful` calls.

## Tests needed

Mirroring `tests/test_vllm_faithful_scheduler.py` and
`tests/test_sarathi_faithful_scheduler.py`'s existing patterns:

- Construction/reset/basic-scheduling smoke tests (same shape as both
  existing files).
- **A chunked-admission test that is the direct negative-image of the
  long-context-drop finding in this investigation**: a request whose
  `prompt_tokens > max_num_batched_tokens` should be admitted *partially*
  and complete over multiple steps, where the equivalent `vllm_faithful`
  test would show it never admitted at all. This is the single most
  important new test — it's the concrete, checkable claim that motivated
  this whole design audit (job 1111572's xlong scenarios, and this
  session's `vllm_faithful` 100%-drop confirmation on the same shape).
- A decode-priority test confirming already-running sequences are
  scheduled before chunking in new admissions, matching the "prioritizes
  decode" release-note framing.
- A regression test asserting `vllm_faithful`'s own test suite and
  behavior are byte-for-byte unchanged after this addition (import-only
  check plus rerunning the existing suite is likely sufficient, given no
  shared file is modified).

## What this document does not do

- Does not implement `VLLMChunkedPrefillFaithfulPolicy`.
- Does not modify `vllm_faithful.py`, its tests, or its registry entry.
- Does not commit to `v0.4.2` as final — flags the tradeoff against a
  version closer to the real 0.24.0 runtime used on Wulver, for a human to
  weigh.
- Does not re-verify preemption-mechanics parity between v0.1.0 and v0.4.2
  at the source level — flagged above as a required check before
  implementation, not assumed.

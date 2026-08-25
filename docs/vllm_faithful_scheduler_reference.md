# Pinned vLLM Reference — `vllm_faithful` Baseline

This document records the exact upstream reference used to build the
`vllm_faithful` simulator-side scheduling baseline (see
`src/llmserveopt/policies/vllm_faithful.py` and
`src/llmserveopt/simulator/kv_block_manager.py`). It exists so the fidelity
claims made about that baseline can be checked against a specific, named
version of the real system rather than "vLLM" as a moving target.

## Paper

- **Title:** "Efficient Memory Management for Large Language Model Serving
  with PagedAttention"
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin
  Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Venue:** ACM SIGOPS 29th Symposium on Operating Systems Principles
  (SOSP 2023)
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

## Official repository

- **Repo:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **License:** Apache License 2.0

## Pinned commit/tag

Two candidate references were checked (both fetched live via the GitHub API
during this work, not from memory):

| Ref | Commit | Committer date | Project name at this ref |
|---|---|---|---|
| `submission` | `aa50b17ca776f8c69a793787d0ce06dfa4671884` | 2023-04-17 | `cacheflow` (pre-rebrand) |
| `v0.1.0` | `67d96c29fba9b72cb4c4edbc26211c208a00ebdd` | 2023-06-20 | `vllm` |

**`v0.1.0` is the pinned reference for this baseline.** Rationale:

- It is the first public, PyPI-tagged release under the actual `vllm`
  project name/layout (`vllm/core/scheduler.py`,
  `vllm/core/block_manager.py`), matching what the community and later
  vLLM releases themselves treat as "the original scheduler" when
  comparing against newer chunked-prefill-era versions.
- The `submission` tag is an internal pre-rebrand (`cacheflow`) snapshot
  from before the paper's own SOSP submission cycle; it is not a tagged
  release and its exact relationship to the published paper text is less
  clear than a real, versioned release.
- Both refs implement the same algorithm (FCFS scheduling, block-based KV
  allocation with a watermark, recompute/swap preemption); `v0.1.0` is the
  cleaner, citable reference point.

Source files read directly (via `gh api repos/vllm-project/vllm/contents/...
?ref=67d96c29fba9b72cb4c4edbc26211c208a00ebdd`) to derive this baseline:

- `vllm/core/scheduler.py` (422 lines) — `Scheduler._schedule()`,
  `PreemptionMode`, `SchedulerOutputs`
- `vllm/core/block_manager.py` (255 lines) — `BlockAllocator`,
  `BlockSpaceManager`
- `vllm/sequence.py` (245 lines) — `SequenceGroup`, `SequenceStatus`
- `vllm/config.py` / `vllm/engine/arg_utils.py` — `SchedulerConfig`,
  `CacheConfig`, and default values (`block_size=16`,
  `max_num_batched_tokens=2560`, `max_num_seqs=256`, `watermark=0.01`)

## Algorithm summary (as implemented at this pin)

### Queues

Three request-group queues: `waiting`, `running`, `swapped`. A request enters
`waiting` on arrival and leaves the system only from `running` (on
completion) or transiently through `waiting`↔`running`↔`swapped` via
preemption.

### Per-iteration scheduling order (`Scheduler._schedule`)

1. **Running sequences first.** Iterate `running` (sorted by the scheduling
   policy — FCFS by default) and try to reserve one new KV-cache slot per
   sequence for the next decode token (`block_manager.can_append_slot`). If
   a slot cannot be reserved, **preempt** the lowest-priority running
   sequence group (`running.pop(-1)`, i.e. the most recently
   admitted/lowest-priority one) to free blocks; repeat until the current
   group fits or it is itself preempted.
2. **Swap in, if any swapped groups exist and no swap-out just happened this
   iteration.** Swapped groups are strictly prioritized over new admissions
   from `waiting` ("to bound the amount of CPU memory taken by swapped
   sequence groups" — direct quote from the source comment).
3. **Admit from `waiting`, FCFS, only if `swapped` is empty.** For each
   waiting group, in arrival order: skip if preempted this iteration; stop
   if `block_manager.can_allocate` fails (not enough free blocks, respecting
   the watermark); stop if admitting this group's prompt tokens would
   exceed `max_num_batched_tokens` for this iteration; stop if the resulting
   running-sequence count would exceed `max_num_seqs`.

### KV-cache block management (`BlockSpaceManager` / `BlockAllocator`)

- Fixed-size blocks (`block_size` tokens each, default 16).
- A free list per device (GPU/CPU); `allocate()` pops one block and sets its
  reference count; `free()` decrements the ref count and returns the block
  to the free list only at ref count 0 (supports copy-on-write sharing for
  beam search — not applicable here, see Exclusions below).
- `can_allocate`/`can_swap_in` reserve a **watermark** fraction of GPU
  blocks (default 1%) to avoid thrashing from too-frequent eviction.
- `can_append_slot`: a running sequence can always get a new decode-token
  slot if there is at least one free block per sequence in the group
  (allocating a new logical block only when the current last block is full).

### Preemption (`Scheduler._preempt`)

Two modes, chosen automatically:

- **Recompute** (default, used when the sequence group has exactly one
  sequence — true for every non-beam-search request): discard all of the
  victim's KV blocks and move it back to the **front** of the `waiting`
  queue. It re-enters as if it were a brand-new prompt; all decode progress
  is lost and will be redone from scratch.
- **Swap** (used only for multi-sequence groups, e.g. beam search): move
  the victim's KV blocks from GPU to CPU memory and place it in `swapped`;
  resumed later via step 2 above, with its progress intact.

### Continuous batching

`schedule()` is called once per engine iteration and returns metadata for
every sequence now in `running` (both sequences just admitted this
iteration, marked `is_prompt=True`, and ones continuing decode from a prior
iteration). There is no separate "batch" concept beyond "whatever is in
`running` this iteration" — this **is** continuous batching.

## What is explicitly excluded from this first faithful baseline

- **Chunked prefill.** At this pin, a waiting group's entire prompt is
  admitted in one iteration (subject to the `max_num_batched_tokens`
  budget check) or not admitted at all — there is no partial/chunked
  admission of a single prompt across iterations. Chunked prefill (the
  Sarathi-Serve-inspired scheduler change) was added to vLLM in later
  releases and is a materially different scheduling algorithm. It is
  intentionally out of scope here; this repository already has a
  Sarathi-inspired chunked-prefill baseline (`sarathi_style`,
  `src/llmserveopt/policies/sarathi_style.py`) built independently of this
  pin, and a future `vllm_faithful`-style Sarathi-Serve baseline (per the
  roadmap) should pin its own separate upstream reference rather than being
  folded into this one.
- **Copy-on-write / sequence forking (beam search).** The block manager's
  ref-counting exists specifically to let multiple sequences in a group
  share prompt blocks (beam search) and copy-on-write a block when they
  diverge. Nothing in this repository's `Request`/simulator model has more
  than one sequence per request, so this is a no-op here by construction —
  implemented as ref-counting infrastructure (for correct free/double-free
  semantics) but never exercised beyond ref count 1.
- **Swap-based preemption.** Since every request here is single-sequence,
  the pinned scheduler's own logic always selects **recompute** mode
  (`if len(seqs) == 1: RECOMPUTE else: SWAP`) — swap mode is simply never
  reached for our workload shape. Modeling a CPU swap space would add a
  second memory pool with no behavior this baseline can ever exercise, so
  it is omitted rather than built and left dead.
- **Hardware/runtime performance modeling.** This pin's scheduler makes
  *admission and preemption decisions*; it says nothing about how long a
  prefill or decode step actually takes on real hardware. That remains the
  job of this simulator's existing `ServiceModel`/`CalibratedServiceModel` —
  unchanged by this work.

## Do not use current `main` blindly

vLLM's scheduler has changed substantially since this pin (chunked prefill,
prefix caching, `v1` engine rewrite, speculative decoding, disaggregated
prefill/decode, etc.). None of that is represented here. Any claim made
about `vllm_faithful` is a claim about vLLM **as of commit `67d96c29`**,
not about vLLM today.

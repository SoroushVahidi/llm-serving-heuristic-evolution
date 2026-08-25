# Pinned Sarathi-Serve Reference — `sarathi_faithful` Baseline

This document records the exact upstream reference used to build the
`sarathi_faithful` simulator-side scheduling baseline (see
`src/llmserveopt/policies/sarathi_faithful.py`). It exists so the fidelity
claims made about that baseline can be checked against a specific, named
version of the real system rather than "Sarathi-Serve" as a moving target.

## Paper

- **Title:** "Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve"
- **Authors:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree
  Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran
  Ramjee
- **Venue:** 18th USENIX Symposium on Operating Systems Design and
  Implementation (OSDI 2024)
- **arXiv:** [2403.02310](https://arxiv.org/abs/2403.02310)

## Official repository

- **Repo:** [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve)
- **License:** Apache License 2.0

## Pinned commit/branch

**Branch `osdi-sarathi-serve`, commit `ceaa0660ea2487976101a8167aad5c8046e85b27`**
(fetched live via the GitHub API, not from memory). Commit message: "Add
OSDI Experiment Folder" (2024-06-04) — this branch is the repository's own
designated snapshot for reproducing the OSDI 2024 paper's experiments
(`osdi-experiments/` directory with per-figure/per-table run scripts),
making it the most defensible "paper-era" reference, analogous to how
vLLM's `v0.1.0` was chosen for the `vllm_faithful` baseline (see
`docs/vllm_faithful_scheduler_reference.md`).

Source files read directly (via `gh api repos/microsoft/sarathi-serve/contents/...
?ref=ceaa0660ea2487976101a8167aad5c8046e85b27`):

- `sarathi/core/scheduler/sarathi_scheduler.py` (238 lines) — `SarathiScheduler._schedule()`,
  `_get_seq_next_num_prefill_tokens()`, chunk-size schedule
- `sarathi/core/scheduler/base_scheduler.py` (134 lines) — shared
  `_allocate`/`_append_slot`/`_preempt`/pipeline-stage throttling
- `sarathi/core/block_space_manager/sarathi_block_space_manager.py` (8 lines)
  — `class SarathiBlockSpaceManager(VLLMBlockSpaceManager): pass`
- `sarathi/config.py` — `SarathiSchedulerConfig` fields
- `osdi-experiments/table-6/scheduling_ablation.sh`,
  `osdi-experiments/figure-9/prefill_chunking_overhead_runs.sh` — the
  paper's own evaluation scripts, used here to source representative
  parameter values (see below), since `SarathiSchedulerConfig` itself
  declares `chunk_size` as a required `Optional[int]` with no built-in
  default.

## Key finding: Sarathi-Serve's memory model IS vLLM's, unchanged

`SarathiBlockSpaceManager` is a literal no-op subclass of
`VLLMBlockSpaceManager` — Sarathi-Serve does not modify vLLM's paged-KV
block allocation, watermark reserve, or free-list mechanics at all; it only
changes **scheduling** (what work gets batched together, and when). This is
why `sarathi_faithful` reuses this project's existing
`src/llmserveopt/simulator/kv_block_manager.py` (built for `vllm_faithful`)
directly rather than introducing new memory infrastructure — the pinned
reference itself draws no distinction here.

## Algorithm summary (as implemented at this pin)

### Queues

Two request queues: `waiting` and `running` (no `swapped` queue at all —
unlike vLLM's own scheduler, Sarathi-Serve's `_preempt` always discards the
victim's blocks and reinserts it at the front of `waiting`; there is no
swap-to-CPU path in this scheduler).

### Per-iteration scheduling order (`SarathiScheduler._schedule`)

**Phase 1a — reserve decode slots for already-prefilled running sequences
first (the stall-free/decode-first property).** Iterate `running` sorted by
priority (FCFS); for each sequence whose prompt processing is already
finished (i.e. currently decoding), try to reserve its next decode-token
slot (`block_manager.can_append_slot`). If it cannot get a slot, preempt the
lowest-priority running sequence (evict from the back of what remains),
repeating until it fits or it is itself preempted — **identical
victim-selection algorithm to vLLM's own scheduler** (see
`docs/vllm_faithful_scheduler_reference.md`). Each decoding sequence kept
contributes exactly 1 to `num_batched_tokens`.

**Phase 1b — resume sequences still mid-prefill from a previous
iteration**, using whatever `chunk_size` budget Phase 1a left over:
`next_num_prefill_tokens = min(remaining_prompt_tokens, chunk_size -
num_batched_tokens_so_far)`. Memory for these was already allocated when
they were first admitted, so no capacity check is needed here — only the
token-budget check. If the leftover budget is 0, the sequence simply isn't
scheduled this iteration (no progress, no preemption) — the source comment
notes this "should always be false" in the non-pipelined case, which
matches this simulator's own execution model (no pipeline parallelism).

**Phase 2 — admit new requests from `waiting`, FCFS, chunked**: for the
request at the front of `waiting` (skip if its arrival time hasn't come
yet), check `block_manager.can_allocate`; if it fails, **stop admitting
entirely** (`break`, not `continue` — read directly from the pinned source;
a source comment nearby claims "different from vllm scheduler" but the
actual code still breaks on the first non-allocatable request, so the code,
not the comment, is what this baseline reproduces). Check `max_num_seqs`.
Compute its own `next_num_prefill_tokens` chunk against the remaining
budget; if that chunk would be 0 (no budget left), stop admitting. Otherwise
allocate blocks for the full prompt up front and admit it with a
`chunk_size`-bounded first chunk.

### Chunk sizing

- **Static (paper-evaluated) mode:** `chunk_size` is a single fixed value
  used every iteration. The paper's own OSDI evaluation scripts use
  `chunk_size ∈ {512, 1024, 2048, 8192, 16384}` across ablations, with
  **512 the most frequently used value** (`table-6/scheduling_ablation.sh`,
  `figure-9/prefill_chunking_overhead_runs.sh`) — used as this baseline's
  default.
- **Dynamic mode (`enable_dynamic_chunking_schedule`):** an optional,
  profiling-adjacent feature that linearly interpolates chunk size between
  `low_chunk_size` and `high_chunk_size` over `chunk_schedule_stages`,
  keyed off total prompt tokens processed cluster-wide
  (`chunk_schedule_max_tokens`). **Every OSDI evaluation script in the
  pinned commit passes `--sarathi_scheduler_enable_dynamic_chunking_schedule
  false`** — this feature was not used to produce any of the paper's
  reported results. Excluded from `sarathi_faithful` for the same reason
  chunked prefill's own eventual vLLM upstreaming was excluded from
  `vllm_faithful`: it is a real but secondary, non-headline feature of the
  reference, not exercised by the paper's own evaluation.

### Representative parameters used in the paper's own scripts

| Parameter | Value(s) seen in `osdi-experiments/` | Used as `sarathi_faithful` default |
|---|---|---|
| `chunk_size` | 512, 1024, 2048, 8192, 16384 | 512 |
| `max_num_seqs` (`replica_scheduler_max_batch_size`) | 128 (table-6), 1 (figure-9, single-request microbenchmark) | 128 |
| `block_size` | inherited from vLLM's own default (16) — `SarathiBlockSpaceManager` adds nothing | 16 |

## What is explicitly excluded from this first faithful baseline

- **Dynamic chunking schedule** (`enable_dynamic_chunking_schedule`): see
  above — never used in the paper's own reported evaluation.
- **Pipeline parallelism throttling** (`num_pipeline_stages`,
  `num_running_batches`): this simulator has no pipeline-parallel execution
  model at all (one scheduling decision per step, not micro-batched across
  pipeline stages), so this concept has no analogue here.
- **Prompt-length rejection** (`_check_request_prompt_length` /
  `FINISHED_IGNORED`): the pinned scheduler silently drops requests longer
  than `max_model_len`. This project's `GPUConfig` has no separate
  "model context length" distinct from KV capacity, so there is nothing to
  map this onto; omitted rather than approximated.
- **Swap-based preemption**: never present in Sarathi-Serve's own scheduler
  in the first place (unlike vLLM's own scheduler, which supports but
  rarely uses it) — `_preempt` here is unconditionally recompute-only, so
  there is nothing to exclude beyond what the reference itself already
  excludes.
- **Hardware/runtime performance modeling**: this pin's scheduler makes
  *chunking, batching, and preemption decisions*; it says nothing about how
  long a chunk or decode step actually takes on real hardware. That remains
  the job of this simulator's existing `ServiceModel`/
  `CalibratedServiceModel`, unchanged by this work.

## Existing simulator infrastructure audit

Before writing any new code, the existing simulator was audited against
what the pinned reference needs. Conclusion: **no new shared infrastructure
was required** — everything needed either already existed (built for Phase
1.5's own "Sarathi-style stall-free principle," per `service_model.py`'s own
docstring) or was already added for `vllm_faithful`
(`docs/vllm_faithful_scheduler_reference.md`) and is directly reusable,
since Sarathi-Serve's own memory model is literally vLLM's, unchanged (see
above).

| Capability | Status | Where |
|---|---|---|
| Explicit prefill phase | **Sufficient as-is** | `InternalRequest.prefill_remaining`, `.is_prefilling`/`.is_decoding` |
| Chunked prefill (execution) | **Sufficient as-is** | `GPUState._step_phase15`: `chunk = min(max_prefill_chunk_tokens, prefill_remaining, prefill_budget)`, budget decremented per request in order — structurally the same FCFS-shared-budget mechanic as `_get_seq_next_num_prefill_tokens` |
| Chunked prefill (admission decision) | **Small extension (policy-side only)** | The pinned reference refuses to admit a new request whose first chunk would be 0 (`break` in Phase 2); no existing policy shadows this check before returning `Action.admit`. Implemented in `sarathi_faithful` itself — no simulator change |
| Per-step token budget | **Sufficient as-is** | `ServiceModel.step_token_budget` (≈ `chunk_size`) |
| Decode-first execution | **Sufficient as-is** | `ServiceModel.decode_first` |
| Active-decode preservation under memory pressure (preemption) | **Reuse existing (built for `vllm_faithful`)** | `Action.preempt`, `GPUState.evict()`, `Simulator._apply_preemptions()` — identical recompute semantics, identical victim-selection algorithm |
| KV/paged-block memory | **Reuse existing (built for `vllm_faithful`)** | `src/llmserveopt/simulator/kv_block_manager.py` — Sarathi's own `SarathiBlockSpaceManager` is a no-op subclass of vLLM's, so reusing the same manager is itself faithful, not a shortcut |
| TTFT | **Sufficient as-is** | `CompletedRequest.ttft` / `InternalRequest.first_token_time`, set the first time decode begins (i.e. once ALL of a request's prefill chunks — however many steps they took — are done), which is the correct definition under chunked prefill |
| TPOT / TBT | **Sufficient as-is** | `CompletedRequest.tpot` |
| Per-request SLOs | **Sufficient as-is, but unused by design** | `Request.slo_deadline`/`priority` exist, but the pinned reference's own scheduler has **no SLO/deadline-aware logic at all** — pure FCFS. `sarathi_faithful` must not use these fields for scheduling decisions, exactly mirroring the reference |

Net result: `sarathi_faithful` is implemented as a single new policy file
reusing 100% of the memory/preemption infrastructure already built for
`vllm_faithful`, plus the simulator's own existing Phase 1.5 prefill/decode
execution machinery. No new shared infrastructure commit was needed for
this baseline.

## Do not use current `main` blindly

The `microsoft/sarathi-serve` repository has continued to evolve since this
pin (MoE support, pipeline-parallel fixes, a `vidur` simulator branch, an
`niyama_asplos2026` branch, OSS-release cleanups, etc.). None of that is
represented here. Any claim made about `sarathi_faithful` is a claim about
Sarathi-Serve **as of commit `ceaa0660`**, not about the project today.

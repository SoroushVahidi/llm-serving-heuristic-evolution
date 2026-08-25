# Pinned SLAI Reference — `slai_faithful` Baseline

This document records the exact upstream reference used to build the
`slai_faithful` simulator-side scheduling baseline (see
`src/llmserveopt/policies/slai_faithful.py`). It exists so the fidelity
claims made about that baseline can be checked against a specific, named
version of the real system rather than "SLAI" as a moving target — same
convention as `docs/sarathi_faithful_scheduler_reference.md`,
`docs/distserve_faithful_scheduler_reference.md`, etc.

> **Relationship to `slai_style_phase_aware`:** this repository already had a
> `slai_style_phase_aware` policy (Policy Library v2) before this baseline was
> added. That policy is an **explicit non-reproduction** ("simulator-level
> approximation... not an SLAI reproduction" — `docs/current/POLICY_LIBRARY.md`)
> and cites no specific commit. `slai_faithful` is a new, separate, additive
> policy; `slai_style_phase_aware` is unmodified and unremoved.

---

## A. Source provenance

- **Paper:** "Optimal Scheduling Algorithms for LLM Inference: Theory and
  Practice"
- **Authors:** Agrim Bari, Parikshit Hegde, Gustavo de Veciana (University of
  Texas at Austin)
- **Venue:** ACM SIGMETRICS 2026 / *Proc. ACM Meas. Anal. Comput. Syst.*, Vol.
  9, No. 3, Article 59 (published December 2025)
- **arXiv:** [2508.01002](https://arxiv.org/abs/2508.01002) (v1 2025-08-01, v2
  2025-12-01)
- **License (paper text):** CC BY 4.0

### Official repository

- **Repo:** [github.com/agrimUT/SLAI](https://github.com/agrimUT/SLAI)
- **License:** **Apache License 2.0**
- **Pinned commit:** `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380` (`main`,
  "Update README.md", 2025-08-14) — the repository's HEAD at the time this
  baseline was built; cloned and read directly (not from memory).
- **Relationship to Sarathi-Serve:** this repo is a fork/derivative of
  Sarathi-Serve — its own README states "This project builds on the
  Sarathi-Serve codebase... [and] uses the same setup as Sarathi-Serve (OSDI
  branch)." Confirmed directly in source: `SLAIBlockSpaceManager` (
  `sarathi/core/block_space_manager/slai_scheduler_space_manager.py`) is a
  literal no-op subclass of `SarathiBlockSpaceManager`, and
  `_get_seq_next_num_prefill_tokens()` in `slai_scheduler.py` is structurally
  identical to `sarathi_scheduler.py`'s own version. SLAI's memory model and
  chunked-prefill mechanic **are** Sarathi-Serve's, unchanged — exactly the
  same relationship this repo already documented between `sarathi_faithful`
  and `vllm_faithful`'s `KVBlockSpaceManager`.

### Source files read directly (cloned locally, inspected line-by-line)

- `sarathi/core/scheduler/slai_scheduler.py` (423 lines) —
  `SLAIScheduler._schedule()`, `_post_batch_processing()`, `_tbt_for()`,
  `_slack()`/`_deadline()` (present but **dead code**, never called by
  `_schedule()` — see §C), `_get_seq_next_num_prefill_tokens()`
- `sarathi/core/block_space_manager/slai_scheduler_space_manager.py` (7
  lines) — `class SLAIBlockSpaceManager(SarathiBlockSpaceManager): pass`
- `sarathi/config.py` — `SLAISchedulerConfig` field list (`token_budget`,
  `fcfs`, `fixed_offset`, `below_memory_limit_offset`,
  `above_memory_limit_offset`, `memory_limit`, `user_priority`,
  `time_between_tokens`, `limit_total_decodes`)
- `sarathi/core/datatypes/sequence.py` — `time_between_tokens`,
  `is_strict_tbt`, `prefill_e2e_time_deadline`, `last_schedulable_time`
  fields on `Sequence`
- `sarathi/core/scheduler/hold_n_scheduler.py` (135 lines) — see §RAD below;
  this is **not** the SLAI or RAD scheduler
- `sarathi/core/scheduler/scheduler_registry.py` — confirms the full set of
  registered scheduler types (`VLLM`, `ORCA`, `FASTER_TRANSFORMER`,
  `SARATHI`, `SIMPLE_CHUNKING`, `HOLD_N`, `SLAI_SCHEDULER`) — **no `RAD`
  scheduler type exists in the code at all**
- Paper Section 6 ("SLO Aware LLM Inference Scheduler"), Eq. 8, and the
  "Batch construction" 4-step description — cross-checked against the code
  above (they match)
- Paper Section 7 ("Experimental setup" / "Impact of different scheduler
  parameters") — representative parameter values (token budget 512, active
  request cap and decode limit both 128, offset Θ=5/10 with 96% memory
  threshold, TBT 0.1s/0.5s for paying/free-tier users)
- Paper Section 4 / Algorithm 1 ("RAD: A throughput optimal scheduler") —
  read in full to determine RAD's relationship to SLAI and its
  implementation status (§RAD below)

---

## B. Directly reproduced behavior

Every important behavior below is cited to the specific upstream
file/function it is grounded in.

### B.1 Last-schedulable-time (LST) — the central mechanism (Eq. 8)

> `Y_{i,j} = s_{i-1,j} + TBT_j - Θ · b_batch`

Reproduced in `SlaiFaithfulPolicy._run_gpu_schedule`'s top-of-function LST
pass, mirroring `slai_scheduler.py`'s `_post_batch_processing()`
(lines 107–126): **any** request that transitions to decode-ready
(`remaining_prefill == 0`) since the last call — whether via a continuing
prefill chunk finishing or a brand-new admission whose entire prompt fits in
one chunk — gets its first LST assigned on the **next** call, using that
call's decision instant as the `s_{i-1,j}` anchor. This exactly reproduces
the pinned source's one-iteration lag: prefill completes in iteration *N* →
LST assigned and critical/non-critical judged starting iteration *N+1*
(`slai_scheduler.py` never assigns LST inline in the same iteration a
request's prefill happens to finish — only `_post_batch_processing`, at the
top of the *next* `_schedule()` call, does).

### B.2 Critical vs. non-critical classification and batch construction order

Reproduced faithfully as the paper's own 4-step "Batch construction"
(Section 6) and the corresponding code (`slai_scheduler.py:134-422`):

1. **Critical decodes** (`now >= LST`), scheduled first, in increasing-LST
   order, capped by `decode_limit` (`limit_total_decodes` /
   `max_critical_decodes_allowed`, lines 150-185).
2. **Prefill**, non-preemptive: already-active (`paused_prefills`)
   continuing prefills get priority (lines 186-203); then new requests from
   `waiting`, FCFS or SPF/tiered-SPF (lines 204-263).
3. **Leftover budget → additional non-critical decodes**, increasing-LST
   order, capped by the *same* combined `decode_limit` (lines 264-298).
4. Every decode actually served this step (critical + any extra
   non-critical granted leftover budget) gets its LST refreshed — mirrors
   `_post_batch_processing`'s re-derivation-on-service rule exactly.

### B.3 Offset (Θ) — fixed and dynamic memory-pressure-adaptive modes

Reproduced exactly, including the paper's own headline numbers (Section 7.1):
`below_memory_limit_offset=5`, `above_memory_limit_offset=10`, switching
threshold `memory_limit_fraction=0.96` — "SLAI (SPF, dynamic offset)" is the
paper's flagship, best-performing configuration and this baseline's default.
`_offset()` uses this simulator's `current_kv_tokens / max_kv_tokens` as the
direct analogue of the pinned source's
`block_manager.get_num_used_gpu_blocks() / num_total_gpu_blocks`
(`slai_scheduler.py:118-124`).

### B.4 Tiered-SPF prefill-admission ordering

Reproduced (generalized — see §C) from `slai_scheduler.py:204-223`: when
`fcfs=False`, arrived waiting requests are reordered — `user_priority=True`
sorts strict-TBT-tier requests before relaxed-tier ones, then by prompt
length within each tier; `user_priority=False` is plain SPF (prompt length
only).

### B.5 Chunked-prefill token accounting

`_get_seq_next_num_prefill_tokens`-equivalent chunk computation
(`min(remaining_prompt, budget_left)`) is structurally identical between
`slai_scheduler.py` and `sarathi_scheduler.py` in the pinned source (SLAI
did not modify this) — reproduced identically to how `sarathi_faithful`
already reproduces it in this repo, reusing the same `KVBlockSpaceManager`.

---

## C. Simulator adaptations (disclosed)

These are places where a faithful reproduction of the *algorithm* required
an explicit, disclosed choice, because this simulator's abstractions differ
from the pinned reference's real async serving engine.

1. **TBT-per-request ("user tiers").** The pinned reference's benchmark
   harness assigns each request an explicit `time_between_tokens` /
   `is_strict_tbt` pair, set externally by
   `synthetic_request_generator.py` (paying users get a strict TBT +
   `is_strict_tbt=True`; free-tier users get a relaxed TBT). This project's
   `Request` has no such field. `slai_faithful` maps `class_id → TBT` via a
   configurable dict — the two extremes (0.1s / 0.5s) are the paper's own
   experimental values (Section 7); 0.3s for the middle tier is a
   disclosed, **non-paper-sourced** interpolation, needed only because this
   project's `class_id` convention has three tiers where the paper's
   benchmark had exactly two.
   >
   > **Dataset-audit finding (2026-07-22):** this project has *two*
   > independently-authored 3-tier `class_id` vocabularies in active use,
   > not one — `"tight"/"medium"/"loose"` (`Request.class_id`'s own
   > docstring convention, `workloads/synthetic.py`, and the SwissAI/
   > TraceLab external sweep scripts' `assign_slo()`) and
   > `"interactive"/"standard"/"batch"` (`workloads/augmentation.py`'s
   > `DEFAULT_SLO_AUG`, used by the in-repo BurstGPT and Azure loaders).
   > Both assign the *same* priority values per tier (3.0/2.0/1.0), and
   > `selector/dataset_v2/features.py` already treats
   > `{"tight", "interactive", "critical"}` as one equivalence class for its
   > own feature engineering. The default mapping was extended to
   > `{"tight": 0.1, "interactive": 0.1, "critical": 0.1, "medium": 0.3,
   > "standard": 0.3, "loose": 0.5, "batch": 0.5}` so the **same** TBT
   > applies to a request regardless of which of the two vocabularies —
   > and therefore regardless of which of Azure/BurstGPT/SwissAI/TraceLab —
   > it came from. Before this fix, every BurstGPT/Azure request (which use
   > `"standard"`, not `"medium"`) would have silently fallen through to
   > `default_tbt` (0.5s) rather than being classified by its true tier —
   > this was audited and corrected, not discovered and left in place.
   > Regression-tested in `test_cross_vocabulary_class_id_equivalence`
   > (`tests/test_slai_faithful_scheduler.py`).
2. **N-tier SPF-priority generalization.** The paper's `is_strict_tbt` is a
   strict boolean (exactly two tiers). This baseline generalizes it to sort
   admission candidates on `(tbt(req), prompt_tokens, request_id)`, which
   reduces to the paper's exact two-tier behavior whenever there are only
   two distinct TBT values, and extends cleanly to this project's
   three-tier `class_id` convention without inventing a new mechanism.
3. **Batch execution time (`b_batch`).** The pinned reference tracks a
   running average of **real, GEMM-cost-dependent, variable** batch
   execution time (`_mean_batch_dur`/`_max_batch_dur` in `slai_scheduler.py`),
   used as the unit for the offset safety margin. This simulator's discrete
   step model has **no such variance by construction** — every step is
   exactly `step_size` wall-clock seconds regardless of batch composition.
   `slai_faithful` therefore uses `step_size` directly wherever the
   reference would use `b_batch`. The offset-margin **formula** (Eq. 8) is
   reproduced exactly; the real-world phenomenon it exists to absorb
   (execution-time variability) simply does not exist in this simulator's
   execution model. This is disclosed here rather than papered over with a
   fake "running average" that would trivially converge to the constant
   `step_size` anyway.
4. **Discrete-step LST anchor.** The pinned reference is an async,
   continuous-time engine where a batch's own wall-clock end time is a
   distinct instant from the next scheduling decision. This simulator's
   `select_action()` is called once per discrete step at that step's
   **start** time (`ObservableState.time`); `slai_faithful` uses that same
   instant as its "batch end" anchor, for both simplicity and consistency —
   a constant, disclosed `step_size`-scale timing choice, not a source of
   unbounded drift.
5. **Dead-code exclusion (`_slack()`/`_deadline()` and
   `prefill_e2e_time_deadline`).** The pinned source defines
   `_slack()`/`_deadline()` helpers referencing
   `seq.prefill_e2e_time_deadline`, but **`_schedule()` never calls either
   of them** — confirmed by reading the full 423-line file. This appears to
   be a legacy/experimental hook from development that the shipped
   scheduler does not use. `slai_faithful` correctly omits it: reproducing
   dead code would not be "more faithful," it would fabricate behavior the
   pinned commit's actual scheduler does not exhibit.
6. **Disabled experimental orderings.** The pinned source contains large
   commented-out blocks (lines 299-405) implementing alternative
   non-critical-decode orderings (longest-first, and a shortest-remaining
   variant that would read `seq.get_oracle_decode_tokens()` — i.e., an
   oracle-leakage path). These are **inactive in the shipped commit** (the
   live code path uses plain LST-ascending order for non-critical decodes,
   §B.2 step 3). `slai_faithful` reproduces only the live, active code path
   — not the commented-out alternatives, and in particular never touches
   the oracle-leakage variant, which was disabled in the pinned commit
   itself.

---

## D. Reused infrastructure

- **`KVBlockSpaceManager`** (`simulator/kv_block_manager.py`) — built for
  `vllm_faithful`, already reused by `sarathi_faithful`. Reused again here
  unchanged, since SLAI's own memory model is Sarathi-Serve's (§A).
- **Chunked-prefill token-budget accounting** — the same
  `min(remaining, budget_left)` mechanic `sarathi_faithful` already uses;
  no new simulator-side execution-model change was needed for this part.
- **`ObservableRequest.slo_deadline`** — exists but is **not** used by
  `slai_faithful` for the TBT mechanism (the pinned reference's own TBT
  concept is a per-token metric, not the end-to-end deadline this field
  represents); left unused, exactly mirroring how `sarathi_faithful`
  correctly leaves `slo_deadline`/`priority` unused because its own pinned
  reference has no SLO-aware logic at all.
- **`ObservableGPUState.current_kv_tokens` / `.max_kv_tokens`** — reused
  directly as the memory-utilization signal for the dynamic offset (§B.3).

## Simulator extension required (new, not reused)

**`Action.hold_decode`** (see `core/action.py`'s docstring and
`src/llmserveopt/policies/slai_faithful.py`'s module docstring) — a new,
narrowly-scoped Action verb: maps `gpu_id -> [request_id, ...]`, telling the
simulator to skip a specific active, currently-decoding request's token
production for exactly one step, while it remains fully active (KV/slot
reservation untouched, no eviction, no re-queueing).

**Why this was necessary, not optional:** SLAI's central mechanism is
per-request decode deferral based on that request's own LST. Neither of the
simulator's two pre-existing GLOBAL execution models could express this:

| Existing model | Behavior | Why insufficient for SLAI |
|---|---|---|
| Decode-protected (`enable_decode_prefill_contention=False`, or `True`+`decode_first=True`) | **Every** decoding request advances, unconditionally, every step | SLAI must be able to defer *specific* decodes while serving others — this model has no such lever at all |
| Shared-contention (`enable_decode_prefill_contention=True`, `decode_first=False`) | Decode and prefill compete in a single FCFS-by-**arrival-time** pass; a request can get zero progress if the combined budget runs out before reaching it | The exclusion is an accidental side effect of arrival-order + budget exhaustion, not a request's own TBT deadline — wrong selection criterion, and (for requests placed late in arrival order) can silently exclude a *critical* decode instead of a safely-deferrable one |

`Action.hold_decode` was added as the minimum change to close this gap:
threaded through `Simulator._advance_decode()` → `GPUState.step()` →
`_step_phase1()`/`_step_phase15()` → both `_advance_decode_protected()` and
`_advance_shared_contention()` (by simply excluding held requests from the
`decoding` list passed into either method — their existing
`len(decoding)`/`for req in decoding` logic then automatically frees the
held request's budget slot with no further code changes needed inside
either method). Defaults to an empty dict; every pre-existing policy leaves
it empty, so behavior is bit-identical for them (regression-tested in
`tests/test_simulator_decode_hold.py`, and confirmed by full non-hardware
test-suite runs before and after this baseline's own addition — see the
implementation report's TEST_RESULTS section for exact pass counts).

---

## E. Explicit exclusions

Everything from the real SLAI implementation **not** reproduced here:

- **RAD** — the *other* scheduler in the same paper/repo. See the
  dedicated §RAD section below; not part of `slai_faithful` at all.
- **Real GPU kernel/hardware timing** — this baseline makes *scheduling
  decisions* (what to admit, defer, chunk, and when); actual step/batch
  duration remains this simulator's `ServiceModel`'s job, unchanged.
- **Pipeline parallelism** (`num_pipeline_stages`) — same rationale as
  `sarathi_faithful`'s identical exclusion: no pipeline-parallel execution
  model exists in this simulator at all.
- **Multi-GPU / tensor-parallel model sharding** — the pinned reference's
  own benchmark harness runs single-GPU (Mistral-7B on one RTX ADA 6000);
  `slai_faithful`'s multi-GPU support (independent engines sharing one
  global waiting queue, ascending `gpu_id` order) is this policy's **own**
  extension, identical in spirit to `sarathi_faithful`'s multi-GPU
  extension, not part of the pinned single-engine reference.
- **The disabled experimental non-critical-decode orderings** and the
  **dead `_slack()`/`_deadline()` helpers** — see §C items 5-6; these are
  inactive in the pinned commit itself, so excluding them is *increasing*
  fidelity to the actual shipped scheduler, not reducing it.
- **Prompt-length rejection** — same rationale/exclusion as
  `sarathi_faithful`: this project's `GPUConfig` has no separate "model
  context length" distinct from KV capacity.

## §RAD — why RAD is not implemented as a baseline here

RAD (Resource-Aware Dynamic scheduler, paper Section 4, Algorithm 1) is
the **other** scheduler this paper introduces — provably throughput-optimal
under stated conditions, but by the paper's **own explicit statement**
(Section 5): *"The throughput-optimal RAD scheduler described in Section 4
focuses on maximizing throughput, but it does not consider latency SLOs...
due to the challenges mentioned in Section 5."* RAD and SLAI are separate
mechanisms — SLAI is not a special case or an extension of RAD; SLAI was
motivated specifically because RAD does *not* handle SLOs.

**RAD has no continuously-running reference implementation in the pinned
repo.** `SchedulerRegistry` (`scheduler_registry.py`) lists exactly 7
scheduler types — `VLLM`, `ORCA`, `FASTER_TRANSFORMER`, `SARATHI`,
`SIMPLE_CHUNKING`, `HOLD_N`, `SLAI_SCHEDULER` — none named RAD, and a
whole-repo grep for `RAD`/`resource.aware` (case-insensitive) returns zero
hits outside the paper text itself. `Hold_NScheduler`
(`hold_n_scheduler.py`) shares RAD's `N`-parameter naming (RAD's Algorithm 1
also parameterizes on "max prefills per cycle N"), but it is a **single-shot
microbenchmark probe** — "wait until `hold_n` decode-ready sequences exist,
then run them together with one filler prefill request **once**"
(`hold_n_scheduler.py:38-39`), used only to generate one Figure-6c data
point (batch execution time vs. number of decode tokens), not a continuously
-operating scheduler. It never cycles, never repeats, never alternates
Prefill/Decode Mode the way Algorithm 1 actually describes RAD operating.

RAD's own design principle (optimal GEMM tiling: batching exactly
`b_col`-many decode-iterations, or prefill chunks sized to
`LCM(b_row, b_col, b_red)`) is fundamentally about **tensor-core-tile-level
GPU compute efficiency** — a hardware/kernel-execution concern this
simulator does not model at all (batch/step duration here is a function of
token counts via `ServiceModel`, not GEMM tile-dimension alignment). A
faithful RAD reproduction's *entire value proposition* (provable throughput
optimality *because of* optimal tiling) cannot be meaningfully verified in
a simulator that has no notion of tile-dimension-dependent execution time.
Implementing RAD here would necessarily reduce to reproducing only its
*cycle structure* (alternate prefill/decode phases up to N requests per
cycle) without any way to verify the actual claim that makes RAD RAD. See
the final report's `RAD_NEXT_STEP` field for the recommendation this
finding drives.

---

## F. Safe manuscript wording

> "SLAI-inspired last-schedulable-time-gated decode scheduler, faithfully
> reimplementing the SLAI algorithm from Bari, Hegde, and de Veciana,
> 'Optimal Scheduling Algorithms for LLM Inference: Theory and Practice'
> (ACM SIGMETRICS 2026 / arXiv:2508.01002), pinned against the official
> implementation at github.com/agrimUT/SLAI, commit `5098a7a` (Apache-2.0).
> Faithful to the paper's last-schedulable-time formula (Eq. 8),
> critical/non-critical batch-construction ordering, and fixed/dynamic
> offset memory-pressure adaptation. Adapted, disclosed simulator-level
> choices: per-request TBT is derived from this project's `class_id`
> (not an explicit per-request field, as in the pinned reference's
> benchmark harness); the offset safety margin's unit is this simulator's
> fixed `step_size` rather than the pinned reference's variable,
> GEMM-cost-dependent batch-execution-time average, since this simulator's
> discrete-step execution model has no such variance by construction. Does
> **not** reproduce RAD, the other scheduler introduced in the same paper
> (see this baseline's reference doc for why: RAD has no continuously-
> running reference implementation, and its throughput-optimality claim
> depends on GEMM-tile-level hardware assumptions outside this simulator's
> scope)."

---

## Do not use current `main` blindly

The `agrimUT/SLAI` repository may continue to evolve after this pin. Any
claim made about `slai_faithful` is a claim about the pinned repository as
of commit `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380`, not about the
upstream project today.

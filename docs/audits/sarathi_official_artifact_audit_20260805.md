# Sarathi-Serve Official Artifact Audit — 2026-08-05

Audit-only pass, per this task's explicit instructions: no simulator
semantics, CC5/CC6, canonical benchmark suite, VTC, or Algorithm
Stress-Test Library files were modified. This document is the
consolidated scientific/engineering audit of Sarathi-Serve as an
external baseline for this project, following the same audit structure
used for `docs/audits/vtc_official_artifact_audit_20260805.md`.

**Headline finding, stated up front:** this is not a cold start. This
project already has (a) a rigorous, line-cited independent faithful
reimplementation of Sarathi-Serve's scheduler inside the simulator
(`sarathi_faithful.py`, pinned to the OSDI-paper-era commit), and (b)
real, repeated-trial, statistically-characterized GPU execution of the
actual official code on Wulver (A100), compared head-to-head against
real vLLM. `docs/BASELINE_STATUS.md`'s Sarathi-Serve row does not
reflect either of these — it was last written when only the older,
coarser `sarathi_style` proxy existed. That staleness is corrected by
this pass (§11).

## 0. Repository state at start (task step 1)

- Branch: `contextual-compositional-heuristics-20260731`
- Starting SHA: `71d07af7d6a6b467b6940f3fb4ea10a1cac9ac38`
- Upstream: `origin/contextual-compositional-heuristics-20260731`, 0 ahead / 0 behind, working tree clean
- `python scripts/check_contextual_composition_status.py` — passed
- `python scripts/check_contextual_composition_status.py --resume-readiness` — passed
- VTC: confirmed `FOUNDATIONAL_CANDIDATE` (`docs/BASELINE_STATUS.md`)
- PARS-Serve-2026: confirmed `EVALUATION_ONLY` (`docs/BASELINE_STATUS.md`)
- Algorithm Stress-Test Library: committed and clean — `71d07af` ("research:
  add literature-grounded algorithm stress-test library") is itself the
  branch tip; `configs/stress_tests/`, `scripts/stress_tests/`,
  `tests/stress_tests/`, `docs/research/algorithm_stress_tests/` all
  present and tracked, nothing outstanding

## 1. Official artifact identification (task step 2)

| Field | Value |
|---|---|
| Paper | Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee, *"Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"* |
| Venue | 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2024) |
| arXiv | [2403.02310](https://arxiv.org/abs/2403.02310) |
| Official repo | `microsoft/sarathi-serve` (verified live via `gh api repos/microsoft/sarathi-serve`) |
| Default branch | `main`, not archived, last pushed 2026-01-08 |
| License | Apache-2.0 (`license.spdx_id` confirmed via API) |
| OSDI paper-era snapshot | branch `osdi-sarathi-serve`, commit `ceaa0660ea2487976101a8167aad5c8046e85b27` ("Add OSDI Experiment Folder", 2024-06-04) — repo's own designated reproduction snapshot (`osdi-experiments/` per-figure/table scripts) |
| Current `main` tip (as of this audit) | `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` ("minor (#59)", 2026-01-08) |
| Docker images | None found |
| Checkpoints | None — pure scheduler, no learned component; standard HF model weights (Llama/Mistral/Falcon/original-Qwen) loaded at runtime |
| Datasets | Paper's own OSDI evaluation traces, sourced per-experiment in `osdi-experiments/` (not re-derived in this pass) |
| Reproducibility/AE instructions | `README.md`'s "Reproducing Results" section: "Refer to readmes in individual folders corresponding to each figure in `osdi-experiments`." No separate formal AE-committee page found — same pattern as this project's VTC audit (author's own artifact-evaluation material, not a third-party AE archive) |
| Official setup | mamba/conda env, Python 3.10/3.11, `pip install -e .`; README states "tested with CUDA 12.3 on H100 and A100 GPUs" |
| PyPI package | None (`pip install sarathi-serve` fails — confirmed by a prior pass in this repo, `docs/gpu_external_validity_audit.md`) |
| Benchmark scripts | `osdi-experiments/<figure>/*.sh` (e.g. `table-6/scheduling_ablation.sh`, `figure-9/prefill_chunking_overhead_runs.sh`) |

No competing "official" repository exists — `microsoft/sarathi-serve` is
the author-owned, single canonical artifact, directly linked from the
paper and OSDI proceedings page.

### The repository has continued to evolve since the paper-era pin

`ceaa0660` (osdi-sarathi-serve) and `96f9911` (current main) have
**diverged** histories (`git compare`: 20 commits ahead, 2 behind — not a
simple fast-forward). Files changed between the two include the exact
scheduler files this project's faithful reimplementation is built from:
`sarathi/core/scheduler/sarathi_scheduler.py` (+18/−9 of 238 lines, ~11%
touched), `sarathi/core/scheduler/base_scheduler.py`, and `sarathi/config.py`.
This is a **new finding this pass**, not previously recorded: the commit
used to build/run the *real GPU* validation on Wulver (`96f9911`, see §3)
is not the same scheduler code the *simulator's* faithful reimplementation
(`sarathi_faithful.py`) was built from (`ceaa0660`). The divergence is
moderate, not a rewrite, but it means the Wulver real-hardware results are
not a strict apples-to-apples validation of `sarathi_faithful.py`'s exact
modeled algorithm — see §6 and §9's scientific risks.

## 2. Reproducibility audit (task step 3)

| Dependency | Official requirement | This workstation |
|---|---|---|
| Python | 3.10/3.11 (mamba env) | 3.12.3 (project venv) |
| CUDA | "tested with CUDA 12.3 on H100 and A100 GPUs" (README) | Driver reports CUDA 13.0; no `nvcc`/CUDA toolkit installed at all |
| GPU | H100/A100 tested envelope | RTX 5060 Ti, 16 GB, Blackwell (`sm_120`), driver `580.173.02` |
| PyTorch/NCCL/Triton/FlashAttention | Not pinned in the top-level README; resolved via the fork's own `setup.py`/`requirements` (not independently re-derived this pass) | PyTorch 2.12.0+cu130 present, but this is this project's own environment, not a Sarathi-Serve-specific install |
| Custom CUDA kernels | `pos_encoding_ops`, `layernorm_ops`, `activation_ops`, `moe_ops` (compiled by `setup.py`, targets `sm_70/75/80/86/89/90` by default) | No `sm_120` (Blackwell) target in that list; no local CUDA toolchain to attempt a build at all |
| Multi-GPU / distributed | Not required for base operation; used for pipeline-parallel scale-out experiments (e.g. Falcon-180B) | N/A — single GPU here |

**Can this workstation faithfully execute official-code experiments?**
No, for the same categorical reason established for VTC
(`baselines/vtc/PROVENANCE.md`'s "hardware blocker" section): the
official artifact's tested CUDA generation (12.3) predates public
Blackwell (`sm_120`) support, and this host additionally has no `nvcc`
at all to attempt a build with. This is not a version to pin around —
it is a compiler/architecture generation gap, confirmed empirically by
this project's own prior probe (`docs/gpu_external_validity_audit.md`,
July 2026): no PyPI package exists, and a source install was already
recorded as blocked on this exact host.

**Wolverine (Wulver) requirement: YES, and it has already been used,
successfully, for exactly this purpose.** Real official-code GPU
execution of Sarathi-Serve has already happened on the Wulver cluster
(A100, via Slurm) — not hypothetically, but with a working build and
multiple completed validation jobs:

- **Build (job 1111574, compute node, 1× A100): SUCCEEDED.** Root cause
  of an earlier login-node failure (OOM from a 6x-wider default codegen
  target list when no GPU is visible) diagnosed and worked around by
  building on a node with an A100 actually visible (`sm_80` only).
  `import sarathi` (v0.1.8) and all four compiled CUDA extensions
  confirmed working. Vendored source tree itself not modified; one
  local monkeypatch applied only in the *harness script*, for a
  RoPE-scaling validation false-positive on newer HF config conventions.
- **GPU smoke+validation (jobs 1111576, 1111663): initially blocked**
  by a genuine model-architecture-support gap in this vendored commit
  (`Qwen2ForCausalLM` unsupported; only original `QWenLMHeadModel`,
  Llama, Falcon, Mistral, Mixtral, InternLM, Yi are registered) —
  resolved for later runs by switching to a supported model
  (`mistralai/Mistral-7B-Instruct-v0.1`).
- **Repeated-trial validation (jobs 1111988/1111989/1111990,
  N=5 trials × 2 systems × 5 scenarios): SUCCEEDED**, all 11 Slurm jobs
  exit `0:0`. This is real, official, unmodified Sarathi-Serve code
  (vendored fork commit `96f9911`) running head-to-head against real
  vLLM 0.24.0 on `mistralai/Mistral-7B-Instruct-v0.1`/A100-SXM4-80GB.
  Full record: `docs/wulver_sarathi_vllm_repeated_validation.md`.

**Result summary (bootstrap-CI, 5 trials/scenario, paired):**

| Scenario | Winner (robust, 5/5, CI excludes 0) | Mean diff (vLLM−Sarathi) |
|---|---|---|
| `active_decode_plus_arriving_prefill` | **Sarathi** | +1.017s [0.990, 1.036] |
| `kv_pressure` | **Sarathi** | +0.836s [0.769, 0.903] |
| `long_prompt_moderate_output` | **vLLM** | −0.256s [−0.298, −0.213] |
| `prefill_heavy_burst` | **vLLM** | −0.147s [−0.157, −0.137] |
| `mixed_prompt_lengths` | **vLLM** | −0.205s [−0.257, −0.161] |

This is genuinely strong evidence — real hardware, matched prompts,
deterministic decoding, tight per-system variance, 5 independent
process-level trials per system per scenario — and it is **not**
"Sarathi always wins" or "Sarathi never wins"; it is scenario-dependent,
with both directions robustly confirmed. This becomes the empirical
backbone of §6 (integration strategy) and §8 (stress-test mapping)
below.

## 3. Architectural analysis (task step 4)

Already performed with full source citations against the `ceaa0660`
pin in `docs/sarathi_faithful_scheduler_reference.md` (fetched via
`gh api ...?ref=ceaa0660...`, not from memory or a guess). Summary,
re-verified as still accurate for that pin:

- **Scheduler:** `sarathi/core/scheduler/sarathi_scheduler.py`
  (`SarathiScheduler._schedule()`, `_get_seq_next_num_prefill_tokens()`)
  — three-phase per-iteration loop: (1a) reserve decode slots for
  already-prefilling running sequences first, preempting lowest-priority
  running sequences if needed (stall-free/decode-first property,
  identical victim-selection to vLLM's own scheduler); (1b) resume
  mid-prefill sequences with whatever chunk-size budget remains;
  (2) admit new requests from `waiting`, FCFS, chunked, stopping
  entirely (not skipping) on the first non-allocatable request.
- **Chunked prefill:** static `chunk_size` (paper evaluates
  512/1024/2048/8192/16384; 512 most common) is a hard per-step token
  budget shared between resuming-prefill and newly-admitted work. A
  `enable_dynamic_chunking_schedule` mode exists but every OSDI
  evaluation script disables it — excluded from the faithful
  reimplementation for the same reason.
- **Queue management / admission:** two queues only (`waiting`,
  `running`) — no `swapped`/CPU-offload path; `_preempt` is
  unconditionally recompute-only.
- **KV cache interaction:** `sarathi/core/block_space_manager/
  sarathi_block_space_manager.py` is an 8-line **no-op subclass** of
  `VLLMBlockSpaceManager` — Sarathi-Serve changes *scheduling*, not
  memory management, at all. This is the single most consequential
  architectural fact for simulator compatibility (§5).
- **Priority handling:** none — pure FCFS, no SLO/deadline-aware logic
  anywhere in the pinned scheduler.
- **Pipeline stages:** `num_pipeline_stages`/`num_running_batches`
  throttling exists in the base scheduler for pipeline-parallel
  deployments (e.g. the paper's Falcon-180B results) but is not
  exercised by the single-GPU scenarios this project's faithful
  reimplementation or the Wulver validation runs use.
- **Config:** `sarathi/config.py`'s `SarathiSchedulerConfig` declares
  `chunk_size` as a required field with no built-in default — values
  are sourced from the paper's own evaluation scripts, not invented.

**Drift since the pin (new this pass, not previously audited):**
`sarathi_scheduler.py`, `base_scheduler.py`, and `config.py` all
changed between `ceaa0660` and the Wulver-validated `96f9911`
(§1). The change to `sarathi_scheduler.py` is moderate (~11% of lines),
not architecturally re-derived line-by-line in this pass — flagged as
follow-up work if a byte-for-byte re-validation against current `main`
is ever wanted (§9).

## 4. Scientific analysis (task step 5)

**Main assumptions** (paper + this project's independent technical
review of the paper): LLM inference has two bottleneck regimes —
compute-bound prefill, memory-bound decode; a single static, offline-
calibrated per-step token budget can sit near the hardware's
arithmetic-intensity "knee" and remain simultaneously throughput-optimal
and SLO-respecting; chunking overhead (extra attention recomputation
across chunk boundaries) is amortizable at moderate context lengths.

**Strengths:** conceptually simple ("two ideas, two parameters");
substantial measured gains (2.6× capacity for Mistral-7B, 5.6× for
Falcon-180B with pipeline parallelism, per the paper); graceful
degradation under load ("knee shifts right, slope after it is
gentler"); influential enough that vLLM adopted chunked prefill as a
default scheduling feature shortly after.

**Documented limitations (author-acknowledged or reviewer-identified):**
long-context attention-recompute overhead grows non-linearly (reviewer
estimate: <3% at 8K tokens, potentially 10-15% at 64K) — the paper
sidesteps this by evaluating traces with prompt lengths ≤ ~13K tokens;
static token budget is calibrated offline and may drift under workload
composition shift; speculative decoding and multi-modal models are not
addressed; chunk-size sensitivity has a "1.5-2× wide" sweet spot
(below C≈128 attention-recompute overhead "kills throughput," above
C≈2048 TBT inflates); per-user/per-request fairness is not separately
measured — the reviewer explicitly notes "a pathological workload with
one very long prompt...can starve short prompts' first tokens" because
chunked prefills monopolize the shared per-step budget.

**Failure modes / stress cases used by the authors:** the paper's own
ablations (`osdi-experiments/table-6`, `figure-9`) vary chunk size,
batch size, and scheduling policy, but do not include an explicit
decode-only-saturation regime or a formal fairness/starvation
measurement.

**Follow-up papers criticizing or extending Sarathi-Serve** (via
targeted web search, not memory):

- *"From Tokens to Layers: Redefining Stall-Free Scheduling for MoE
  Serving with Layered Prefill"* (arXiv:2510.08055) — argues
  Sarathi-Serve's stall-free scheduling operates at token granularity
  without regard to per-layer transformer structure, which is
  suboptimal specifically for MoE serving; proposes layer-granularity
  scheduling instead.
- *"Beyond the Buzz: A Pragmatic Take on Inference Disaggregation"*
  (arXiv:2506.05508) — argues chunked-prefill co-location (Sarathi's
  approach) still requires "careful rate matching" between the
  fundamentally imbalanced compute-bound-prefill/memory-bound-decode
  phases, and that full prefill/decode disaggregation (DistServe-style)
  sidesteps that coupling entirely at the cost of network overhead —
  directly relevant since this project already has a faithful DistServe
  baseline (`distserve_faithful.py`) to compare against.

**Does this project's stress-test library already cover these
situations?** No. `grep -i sarathi
configs/stress_tests/algorithm_stress_test_catalog.yaml` returns zero
matches — Sarathi-Serve is currently entry #15 in
`ALGORITHM_INVENTORY_20260805.md`, explicitly scoped there as
"planned — spec-only, no implementation this task," and was never
carried into the catalog itself. §7 identifies exactly which target and
counter workloads should be added (not built now, per this task's
scope).

## 5. Integration strategy (task step 6)

**Decision: keep the existing dual-track approach; no strategy change
recommended.**

The preference order given by this task (official code > adapter >
proxy > rewrite) was already applied, correctly, split across two
different venues for two different purposes:

1. **Inside the simulator (in-repo baseline): Strategy D — independent
   faithful reimplementation** (`sarathi_faithful.py`, pinned to
   `ceaa0660`). This is *not* a fallback taken because A/B were never
   tried — it is the *correct* choice for this artifact structurally,
   for a reason VTC's audit makes visible by contrast: VTC's entire
   fairness algorithm (`VTCReqQueue`) is pure Python/NumPy with zero
   GPU dependency, so it could be dynamically imported and run for real
   inside this project's CPython simulator (VTC's Strategy B/adapter).
   Sarathi-Serve's scheduler has no such standalone form — it is
   inseparable from a full serving engine: custom compiled CUDA
   kernels, a Ray-based worker/executor stack, and real GPU memory
   management. There is nothing to `import` into a pure-Python
   discrete-event simulator the way `VTCReqQueue` could be. Strategies
   A (run official code unchanged, in-repo) and B (in-repo adapter) are
   not mechanically available for this artifact inside a
   non-GPU-executing simulator, at any fidelity — this is a structural
   fact about what Sarathi-Serve *is* (a serving engine), not a
   corner cut.
2. **Outside the simulator (external, real-hardware ground truth):
   Strategy A — official code, unmodified, actually run.** This *was*
   achieved (§2), on Wulver, and is the correct and sufficient venue
   for it: real GPU execution needs a real GPU serving stack, which a
   discrete-event simulator by design does not provide. The Wulver
   repeated-trial results (§2) are used as external validation evidence
   for the simulator's claims, not folded into the simulator's own
   execution path — exactly analogous to how the project's real-vLLM
   Wulver runs validate `vllm_faithful`/`vllm_chunked_prefill_faithful`
   without vLLM itself running inside the simulator.

**What would change this recommendation:** if a future goal specifically
required per-request-level exact numerical parity with the *current*
official scheduler (`96f9911`) rather than a faithful mechanism-level
model of the *paper-era* scheduler (`ceaa0660`), the commit drift found
in §1 would need to be resolved first (either re-pin the faithful
reimplementation to `96f9911` and re-audit the ~11% scheduler diff, or
explicitly bound the claim to "paper-era Sarathi-Serve," which is what
`docs/sarathi_faithful_scheduler_reference.md` already does in its own
"Do not use current `main` blindly" section).

## 6. Simulator compatibility (task step 7)

Re-verified against current HEAD (not just cited from the existing
reference doc): the shared execution machinery `sarathi_faithful.py`
depends on has changed since it was written (`git log 1e5cc74..HEAD --
gpu.py simulator.py kv_block_manager.py` shows 7 intervening commits,
including "Make mixed prefill/decode execution semantics load-bearing,"
which fixed a dead-branch bug so Sarathi-style decode-protected
execution is now the actually-active code path, not a no-op flag). All
61 Sarathi-adjacent and neighboring tests
(`pytest tests/ -k sarathi` plus co-located external-baseline/prefill-model
tests) pass against current HEAD — this strengthens, rather than
undermines, the existing compatibility table.

| Mechanism | Status | Basis |
|---|---|---|
| Explicit prefill phase | **FULLY REPRESENTABLE** | `InternalRequest.prefill_remaining`, `.is_prefilling`/`.is_decoding` |
| Chunked prefill (execution) | **FULLY REPRESENTABLE** | `GPUState._step_phase15` shared per-step budget mechanic, structurally matches `_get_seq_next_num_prefill_tokens` |
| Chunked prefill (admission decision) | **FULLY REPRESENTABLE** | Implemented policy-side in `sarathi_faithful.py`, no simulator change needed |
| Decode-first / stall-free execution | **FULLY REPRESENTABLE, and now load-bearing** | `ServiceModel.decode_first`; previously a dead branch, fixed in commit `79095c1` |
| Preemption under memory pressure | **FULLY REPRESENTABLE** | `Action.preempt`, `GPUState.evict()` — identical recompute/victim-selection semantics to the pinned reference |
| KV/paged-block memory | **FULLY REPRESENTABLE** | Reuses `vllm_faithful`'s `kv_block_manager.py` — faithful because Sarathi's own memory manager is a no-op vLLM subclass |
| TTFT / TPOT | **FULLY REPRESENTABLE** | `CompletedRequest.ttft`/`.tpot`, correct under chunking (first-token time set once all prefill chunks complete) |
| Per-request SLOs | **PARTIALLY REPRESENTABLE** | Fields exist but correctly unused, matching the reference's own pure-FCFS behavior |
| Pipeline-parallel throttling | **NOT REPRESENTABLE** | No pipeline-parallel execution model in this simulator at all (one scheduling decision per step) |
| Prompt-length rejection (`FINISHED_IGNORED`) | **NOT REPRESENTABLE** | `GPUConfig` has no separate "model context length" distinct from KV capacity |
| Long-context (≥32K) quadratic attention-recompute cost | **NOT REPRESENTABLE** | Simulator's timing model has no notion of attention cost scaling separately from token count at extreme lengths — this is exactly the regime §4's literature critique targets |
| Real GPU kernel/instruction-level execution overlap | **NOT REPRESENTABLE** | Simulator is a discrete-event scheduling abstraction, not a hardware execution model — by design, same as every other baseline in this project |

**Extensions that would eventually be required** (identified, not
implemented): (1) a long-context-aware timing term so the simulator can
represent the quadratic attention-recompute regime the literature
critique targets — currently the simulator would silently model long
contexts as if their per-token cost were flat; (2) a pipeline-parallel
scheduling model, only relevant if Falcon-180B-scale claims are ever
targeted; (3) a per-request-class TTFT-variance metric hook to make the
"short prompts starved by long-prompt chunk monopolization" fairness
critique directly measurable (today only mean/p95 latency by class
exists, not a dedicated starvation-detection metric).

## 7. Stress test mapping (task step 8)

Using the now-hardened catalog schema (`evidence_class`,
`test_role: TARGET|COUNTER`, `source_citations`,
`real_system_followup_required`) from
`configs/stress_tests/algorithm_stress_test_catalog.yaml`:

**Target workloads** (Sarathi should win — grounded in real Wulver
hardware evidence, not hypothesis):

- Mirror `active_decode_plus_arriving_prefill` — an already-decoding
  sequence competing with a newly arriving long prefill for the next
  scheduling slot. `evidence_class: INTERNAL_EMPIRICAL_FINDING`
  (robust 5/5 real-hardware win, CI [0.990s, 1.036s], §2).
- Mirror `kv_pressure` — long context + long decode, concurrency 12.
  `evidence_class: INTERNAL_EMPIRICAL_FINDING` (robust 5/5, CI
  [0.769s, 0.903s]).

**Counter workloads** (Sarathi should genuinely lose — equally real,
not hedged):

- Mirror `long_prompt_moderate_output`, `prefill_heavy_burst`,
  `mixed_prompt_lengths` — all three showed a **robust vLLM advantage**
  (5/5 wins, tight CIs excluding zero) in the Wulver comparison.
  `evidence_class: INTERNAL_EMPIRICAL_FINDING`, `test_role: COUNTER`.

**Missing / literature-motivated additions** (not yet real-hardware
validated in this project — would need `evidence_class:
PAPER_MOTIVATING_STRESS_CASE`, honestly labeled per the catalog's own
discipline, matching how `estf_counter_reasoning_prompt_length_
misprediction` and `llf_counter_laxity_instability_under_prediction_
error` were "genuinely revised" rather than force-fit in the existing
validation pass):

- **Short-prompt starvation under long-prompt monopolization**: the
  reviewer's explicit critique — one very long prompt repeatedly
  consumes the shared per-step chunk budget, delaying short prompts'
  TTFT. Directly testable with existing simulator infrastructure
  (§6 marks per-request TTFT as fully representable); needs the
  starvation-detection metric noted in §6 to be measured cleanly rather
  than inferred from aggregate p95.
- **Long-context (≥32K) attention-bound degradation**: the paper's own
  sidestepped regime. **Not currently executable** per §6's NOT
  REPRESENTABLE finding — would need to be logged as a documented,
  non-executable catalog entry (same disclosed-scope pattern already
  used for the 6 out-of-scope `regression_anwg`/vLLM-LTR/PARS entries
  in the existing catalog), not silently omitted.

None of the five Wulver-mirrored entries or either literature-motivated
entry currently exist in the catalog (`grep -ci sarathi
algorithm_stress_test_catalog.yaml` → 0).

## 8. Implementation roadmap (task step 9) — estimate only, not built

| Item | Estimate |
|---|---|
| New catalog entries | 5 Wulver-mirrored (2 target, 3 counter) + 2 literature-motivated (1 likely non-executable) = 7 entries, ~80-120 YAML lines, matching existing per-entry density |
| New generator functions | `scripts/stress_tests/generators.py`, ~150-250 LOC, same pattern as existing single-algorithm sections |
| Difficulty | Low-moderate — `sarathi_faithful.py` and `vllm_chunked_prefill_faithful.py` (needed as `comparison_algorithms`) both already exist and are tested; no new policy code required |
| Expected fidelity | High for the 5 Wulver-mirrored entries (parameter-matched to real-hardware-validated scenarios); exploratory/hypothesis-stage for the 2 literature-motivated entries until independently confirmed |
| New tests | ~10-15, mirroring `tests/stress_tests/test_stress_test_generators.py`'s existing structure |
| Validation steps | `scripts/stress_tests/run_stress_test_smoke.py` at both smoke and `--full` scale, matching the existing 16/16-gate bar; the long-context entry would be logged `NOT_AUTO_EVALUABLE`/disclosed-scope rather than force-evaluated |
| Expected runtime | Minutes — CPU-only, simulator-only, no GPU required (same as the rest of the stress-test library) |
| Expected GPU use | None for this step |
| Expected Wolverine/Wulver use | None for this step; a full re-validation of `sarathi_faithful.py` against current `main` (`96f9911`, per §1's drift finding) would be a separate, larger effort at roughly the scale of the original Sarathi Wulver campaign (build + repeated-trial jobs, ~1 day Slurm turnaround based on prior job IDs) |
| **Scientific risks** | (1) the 2 literature-motivated workloads are hypotheses, not yet empirically confirmed in this project — must be labeled `PAPER_MOTIVATING_STRESS_CASE`, not upgraded to `INTERNAL_EMPIRICAL_FINDING`, until independently run; (2) the `ceaa0660`/`96f9911` scheduler drift (§1) means "the simulator predicts the Wulver result" is not yet a strict validation claim — it should be stated as "the simulator's target/counter split for these 5 scenarios is *motivated by* real-hardware evidence from a closely related but not identical scheduler version," not as confirmed simulator accuracy |
| **Engineering risks** | Low — squarely inside the already-proven catalog/generator pattern; the main known risk class (class-qualified-metric-vs-scalar gate bugs) was already found and fixed project-wide in the existing validation pass, so the hardened evaluator should catch a recurrence rather than silently mis-grade a new entry |

## 9. Optional smoke check (task step 10)

**No local clone-and-build was attempted.** Justification, stated
explicitly rather than silently skipped:

1. This exact workstation's incompatibility with Sarathi-Serve's tested
   envelope was already established by a prior pass in this repo
   (`docs/gpu_external_validity_audit.md`, July 2026): no PyPI package
   exists, and a source install requiring custom CUDA-kernel compilation
   was already probed and found blocked on this host's CUDA/GPU
   generation. This audit's own environment check (§2) reconfirms the
   same blocker still holds today (no `nvcc` present at all, CUDA 13.0
   driver on Blackwell `sm_120`, official envelope is CUDA 12.3 on
   H100/A100).
2. A full official-code build **has already been achieved** on
   appropriate hardware (Wulver A100, jobs 1111574 through 1111990,
   §2) — repeating a build attempt already known to be architecturally
   blocked on this host would consume time for zero new information.
3. Per this task's own preference for evidence over motion, the
   "optional" smoke check was performed in the form that actually adds
   information on this host: read-only GitHub API queries (no local
   disk/compute cost, no repository modification) that (a) reconfirmed
   the repo is live, unarchived, Apache-2.0; (b) discovered and
   quantified the `ceaa0660`/`96f9911` divergence (§1) — a genuine new
   finding this pass, not previously recorded anywhere in this repo's
   docs; (c) confirmed the scheduler files the faithful reimplementation
   depends on are among those that changed.

**Recorded outcome:** success (metadata/comparison queries), not
applicable (no build attempted, with reasons above), no new dependency
or CUDA conflicts discovered beyond what was already documented, no
missing assets found.

## 10. Documentation (task step 11)

- This document: `docs/audits/sarathi_official_artifact_audit_20260805.md`
- `docs/BASELINE_STATUS.md`'s Sarathi-Serve row updated (§11 below) —
  it was stale, referencing only `sarathi_style` and omitting
  `sarathi_faithful.py` and all Wulver real-hardware validation work
  entirely.

## 11. `docs/BASELINE_STATUS.md` update record

The Sarathi-Serve row previously read (in relevant part): "Not
integrated (official code)... `sarathi_style` implemented, style/
inspired only... Proxy/inspired, explicitly NOT an official
reproduction... None planned." This was accurate for the state of the
project when it was written, but predates `sarathi_faithful.py`
(commit `1e5cc74`) and all of the Wulver Sarathi-vs-vLLM validation
work. Updated to reflect: the official repo is identified and its
paper-era commit is pinned for the faithful in-simulator
reimplementation; the official code has genuinely been run, unmodified,
on real GPU hardware externally (not integrated into the simulator, for
the structural reasons in §5); and the exact next action is the
stress-test catalog gap identified in §7, not "none planned." See the
diff applied alongside this document.

## 12. Validation (task step 12)

No simulator, policy, CC5/CC6, canonical-suite, VTC, or stress-test
library source files were modified — this pass is documentation-only
(this audit doc + a `docs/BASELINE_STATUS.md` row edit), consistent
with the task's explicit scope. Per task step 12's own instruction,
`compileall`/`pytest --collect-only` are unnecessary since no `.py`
files changed; they were not run for that reason, not skipped by
oversight. What *was* run, as part of this audit's own fact-finding
(§6), already re-confirms project health at current HEAD:
`python scripts/check_contextual_composition_status.py` (pass),
`--resume-readiness` (pass), and `pytest tests/ -k sarathi` plus
co-located external-baseline/prefill-model tests — 61 passed, 0
failed.

## 13. Commit and push (task step 13)

Recorded after commit — see the Final Output block below for the
verified before/after SHAs and sync status.

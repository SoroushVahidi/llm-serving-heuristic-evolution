# Project Handoff — 2026-07-23

**Purpose of this document:** a durable, self-contained handoff for a future
researcher or agent who knows nothing about this project except the
repository. The project is pausing for potentially several months after this
document is written. Read this document first, before any other doc in
`docs/current/`.

**Machine-readable companion:** `docs/current/project_handoff_state.json`.

---

## A. Project Objective

This repository builds a deterministic, discrete-event **LLM-serving
scheduling simulator** and uses it to study how to choose, combine, or
synthesize request-scheduling policies (admission order, batching, KV/cache
management) for LLM inference serving.

The long-term goal is **not** just "pick the best of N fixed policies." It is
a **state-conditioned mechanism** that estimates the suitability/module-credit
of scheduling components for a given serving scenario, then uses that
evidence to **combine or synthesize** a new deployable policy — compared
fairly against fixed, adaptive, and external (real-system-derived) baselines.

The primary objective metric is **ANWG** — arrival-normalized weighted
goodput (`RunMetrics.arrival_normalized_weighted_goodput` in
`src/llmserveopt/core/metrics.py`): `sum(priority_i * 1[completed & on-time]) /
sum(priority_i over all arrivals)`. This is the metric essentially every
result cited below is expressed in.

---

## B. Current Research Directions

### B.1 Policy selection (the "selector")
Given full-information reward vectors (every candidate policy's ANWG on a
window), learn a model that picks a good policy per window/context using only
online-observable features. Concerns in active use: regret-aware/listwise
objectives (not plain top-1 classification), near-tie handling (many windows
are near- or exact-ties — treating them as hard-classification targets
creates noisy labels), OOD robustness (in-distribution wins routinely fail to
transfer to held-out/OOD splits), and the current selector status (§D) is
"useful, not solved" — frozen for retraining until simulator calibration
improves (see §J/K).

### B.2 Policy composition / structural synthesis
A parallel, **not-yet-decisively-validated** research direction:
module-level composition (mixing sub-behaviors of different policies),
state-dependent utility (deciding what to combine based on observed
pressure), pairwise/single-module "donor" advantages, a symbolic DSL
(`SchedulerGenomeV1` in `src/llmserveopt/policies/genome.py`) for
representing and mutating scheduler structure, and QD/evolutionary framing
for structural synthesis. **Status: implemented as infrastructure
(composition harness, structural synthesis harness, parent selection), but
explicitly NOT validated as scientifically superior** — the native
composition pilot returned `NATIVE_COMPOSITION_PILOT_DECISION = NO_GO` (see
§I). See `docs/current/COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md` for the
full architecture writeup.

### B.3 External-baseline evaluation
Faithful baselines matter because they let the project claim comparisons
against real, published, pinned-commit scheduler implementations rather than
only internal/synthetic policies. As of this handoff there are **7 faithful
external baselines** (§C, §G), the newest being **`slai_faithful`** —
implemented, tested, and put through a bounded realistic-data pilot in this
same work session (§H). All faithful external baselines are deliberately
excluded from selector training (`selector_eligible=False` for every entry in
`EXTERNAL_BASELINE_REGISTRY`).

---

## C. Current Policy Library (exact counts, from live code — verified 2026-07-23)

Verified by importing `llmserveopt.policies.registry` and
`llmserveopt.policies.external_baselines_registry` directly (not from
docs) in this checkout:

| Group | Count | Source |
|---|---|---|
| Historical/internal (`BASELINE_NAMES`) | **20** | `src/llmserveopt/policies/registry.py` |
| Policy Library v2 additions (`POLICY_LIBRARY_V2_NEW_NAMES`) | **7** | same file |
| Total deployable v1+v2 (`POLICY_LIBRARY_V2_NAMES`) | **27** | same file |
| Selector v1 candidates (`SELECTOR_CANDIDATE_NAMES`) | **20** (= `BASELINE_NAMES`) | same file |
| Faithful external baselines (`EXTERNAL_BASELINE_NAMES`) | **7** | `src/llmserveopt/policies/external_baselines_registry.py` |
| — of which `selector_eligible=True` | **0** | same file |
| Oracle/reference-only (`ORACLE_POLICY_NAMES`) | **1** (`oracle_srtf`) | `registry.py` |
| **Grand total registered scheduling policies** | **34** (27 + 7) **+ 1 oracle** | |

The 7 faithful external baselines (exact list from `EXTERNAL_BASELINE_NAMES`):
`vllm_faithful`, `vllm_chunked_prefill_faithful`, `sarathi_faithful`,
`distserve_faithful`, `tetriinfer_paper_reimplementation`,
`llumnix_faithful`, **`slai_faithful`** (new this session).

**Do not trust any doc that says "6 faithful baselines" or "33 total
policies"** — those were correct before this session and are now stale by
exactly one (`slai_faithful`). `docs/current/BASELINES.md` and
`docs/current/POLICY_LIBRARY.md` were updated this session and are current.

---

## D. Current Selector Status

(Full detail: `docs/current/SELECTOR_STATUS.md`, dated 2026-07-22, unchanged
this session.)

- **Strongest known selector result:** the 27-policy V2 selector/regret
  benchmark (`SELECTOR_V2_27_STATUS = STRONG` in its own report — see
  `docs/current/EXPERIMENT_INDEX.md` row "27-Policy Selector/Regret
  Benchmark", root `/mmfs1/project/ikoutis/sv96/llmserveopt-data/v2_selector_regret_benchmark_20260722T134925Z`).
- **What it beats:** some fixed baselines on some splits (e.g. RF per-policy
  regressor beat WSP in-distribution, `0.559481` vs `0.527113`).
- **Where it stays below the oracle envelope:** it lost to WSP
  out-of-distribution (`0.247707` vs `0.256383`), and — the most important
  caveat — **the learned top-1 selector did not meaningfully capture the
  V1-to-V2 oracle-envelope gain on held-out OOD**, despite that gain being
  real (`+0.008904` ANWG, `docs/current/PROJECT_STATUS.md`).
- **OOD limitations:** consistent across three generations of selector work
  (v2 Overnight → v2 OOD investigation → v3 multi-domain) — in-distribution
  wins do not reliably transfer.
- **Faithful external baselines are excluded from selector training**:
  enforced by `selector_eligible=False` on every entry in
  `EXTERNAL_BASELINE_REGISTRY`, verified by
  `tests/test_faithful_baseline_scope_audit.py` and
  `tests/test_external_baseline_integration.py`. This includes the new
  `slai_faithful`.
- **Frozen selector artifact on disk: NONE.** Verified this session —
  `find` for `*.joblib` across both the git repo and the entire shared data
  root (`/mmfs1/project/ikoutis/sv96/llmserveopt-data/`) returned zero
  results. Any future selector evaluation must retrain from
  `scripts/persist_corrected_selector_artifact.py`'s recipe (or equivalent),
  using `src/llmserveopt/selector/models.py`'s
  `PerPolicyRegressionAnwgSelector` (`name="regression_anwg"`, described in
  its own docstring as "the strongest deployable selector under
  arrival-norm WG"), and will need the underlying labeled CSVs (also not
  present in this checkout — check
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/` for them first).
- **Explicit stop/go position (2026-07-22, unchanged):** freeze broad
  generic selector-model sweeps; do **not** retrain the 27-policy selector
  as the next major step; resume only after simulator calibration (§E, §K).

---

## E. Current Simulator Status

- **Known discriminative-power concern (the project's #1 ranked bottleneck,
  `docs/current/ROADMAP_GAP_ANALYSIS.md`):** the simulator/ANWG objective
  often collapses diverse workload regimes to near-identical policy rewards.
  Confirmed by the dedicated
  `simulator_discriminative_audit_20260722T223236Z` (root:
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/simulator_discriminative_audit_20260722T223236Z`):
  `KV_CACHE_COUPLING_VERDICT = WEAK_DIRECT_COUPLING`,
  `PREFILL_DECODE_COUPLING_VERDICT = PARTIAL_AND_WEAK_UNDER_CURRENT_WORKLOADS`,
  `COMBINER_TRAINING_SIGNAL = WEAK`, `COMBINER_EVALUATION_READINESS =
  NEEDS_SIMULATOR_FIX`.
- **This session's SLAI bounded pilot independently reproduced the exact
  same symptom** on 3 of 4 real datasets (BurstGPT/Azure/SwissAI all tied at
  ANWG=1.0 across every tested policy — see §H, §I) — fresh, independent
  corroboration of the pre-existing diagnosis, not a new problem.
- **What `Action.hold_decode` added** (this session, see §H): a new,
  narrowly-scoped simulator primitive letting a policy defer a *specific*
  active decoding request's token production for one step without eviction —
  required because neither pre-existing global execution model
  (decode-protected / shared-contention) could express a per-request
  deferral decision. Fully backward-compatible (empty by default for every
  pre-existing policy); the only consumer today is `slai_faithful`.
- **What remains unresolved:** everything in the discriminative-power audit
  above is still open. `Action.hold_decode` is a new *mechanism*, not a fix
  to the discriminative-power problem — it does not by itself change how
  strongly KV/cache, prefill/decode contention, or SLO pressure couple to
  ANWG.

---

## F. Dataset Status

| Dataset | Prepared artifacts | Loader fidelity | Underloaded/saturated? | Notes |
|---|---|---|---|---|
| **Azure** (2023 conv/code traces) | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/selector_v2_overnight_20260720T235405/{raw,processed}/azure/` — already-converted JSONL, exact `Request` schema | High — `class_id ∈ {"interactive","standard","batch"}` via `workloads/augmentation.py`'s `DEFAULT_SLO_AUG`, direct token counts | **Underloaded** in this session's 300-request pilot windows (ANWG=1.0 for every policy) | In-repo loader (`scripts/data/convert_azure_llm_trace.py`) |
| **BurstGPT** | same root, `.../burstgpt/` | High — same augmentation pipeline, `src/llmserveopt/workloads/burstgpt.py` | **Underloaded** in this session's pilot | In-repo loader |
| **SwissAI** | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_trace_staging_20260722T172215Z/` (raw/processed/windows) + `swissai_v2_policy_sweep_20260722T184451Z/combined/policy_vectors.csv` (512-window × 27-policy outcomes, does NOT include `slai_faithful`) | **Low for prompt/output tokens** — 0/20,000 audited rows have `prompt_tokens` or `output_tokens` populated; only `total_buckets` (KV-block-reuse proxy) is available. This session's pilot used a disclosed 80/20 `total_buckets×64` prompt:output split — **not a faithful reconstruction** | **Saturated** — `swissai_v2_policy_sweep` full run: mean ANWG `0.991726`, zero strict V2 marginal oracle gain; this session's smaller pilot: also ANWG=1.0 ties | External to the git repo (see §L on why); not tracked by git |
| **TraceLab** | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/tracelab_staging_20260722T192050Z/` + `tracelab_v2_policy_sweep_20260722T214129Z/combined/` | High for prompt/output (100% populated: `input_context_tokens`/`output_tokens`) | **Saturated in the full 512-window sweep** (`TRACELAB_V2_SWEEP_STATUS = REDUNDANT`) **but the only dataset showing real differentiation in this session's small 3-window pilot** (§H) — the discrepancy is plausibly window-size/selection, not settled | Windows are naturally small: max `n_requests`=128 (session-scoped agent traces) vs. SwissAI's uniform 512 — a genuine, disclosed ceiling, not a bug |

**Should natural workload distributions remain the primary evaluation
basis?** Yes, per every current doc (`PROJECT_STATUS.md`,
`ROADMAP_GAP_ANALYSIS.md`) — the repeated finding is that raw workload
novelty is necessary but not sufficient; the simulator/objective needs
calibration (§E, §K) before natural-distribution evaluation will be reliably
discriminative. Synthetic SLO augmentation is explicitly "training/regime-
probing evidence only," not a substitute for natural data.

---

## G. External Baseline Status

**Faithful (pinned commit, `selector_eligible=False`):**
`vllm_faithful` (v0.1.0), `vllm_chunked_prefill_faithful` (v0.4.2),
`sarathi_faithful` (osdi-sarathi-serve), `distserve_faithful`
(camera-ready-simulator), `tetriinfer_paper_reimplementation` (paper-only,
no official code exists), `llumnix_faithful` (OSDI 2024 artifact repo),
**`slai_faithful`** (new — `github.com/agrimUT/SLAI`, commit `5098a7a`).

**Style/inspired only (NOT faithful — do not conflate, see §L):**
`orca_style`, `vllm_style_token_budget`, `sarathi_style`, `splitfuse_style`,
`scorpio_style_slo_guard`, `sola_style_state_aware`,
`slai_style_phase_aware` — all in `registry.py`/`POLICY_LIBRARY_V2`, none
pinned to a commit, none verified against source.

**Planned/missing important baselines** (per
`docs/external_baseline_decision.md`, still open): PARS-style
learning-to-rank scheduler, WAIT/Nested-WAIT KV-cache-aware scheduler
(Jaillet et al., arXiv:2502.07115 — real citation year is Feb 2025, not the
doc's stated "2024"), FairBatching, PROSERVE SlideBatching (real citation is
arXiv:2512.12928, Dec 2025, not "2024" as the doc states — both citation-year
errors were found and flagged, not corrected, this session).

**Deliberately not representable in this simulator:** prefix-cache/cache-
reuse-aware scheduling, disaggregated prefill/decode routing beyond what
`distserve_faithful`/`tetriinfer_paper_reimplementation` already model,
request splitting, heterogeneous GPU affinity/routing, exact chunk-size
prefill actions, durable multi-tenant credit accounting beyond class-level
approximation (`docs/current/POLICY_LIBRARY.md`'s "Unsupported or Deferred
Families").

**SLAI/RAD evidence (this session, verified by cloning and reading the
actual upstream source, not just the paper):** SLAI and RAD are the two
schedulers introduced by the **same** paper — Bari, Hegde, de Veciana,
"Optimal Scheduling Algorithms for LLM Inference: Theory and Practice" (ACM
SIGMETRICS 2026, arXiv:2508.01002). `slai_faithful` is implemented, tested
(22 dedicated tests + full-suite regression), and run through a bounded
realistic-data pilot (§H). **RAD should remain theoretical/reference-only**:
it has **zero continuously-running reference implementation upstream**
(`SchedulerRegistry` lists 7 types, none named RAD; the only RAD-adjacent
code, `Hold_NScheduler`, is a single-shot microbenchmark probe for one
figure's data point, confirmed by reading its full 135-line source) and its
entire throughput-optimality claim depends on GEMM-tile-dimension-optimal
batching — a hardware/kernel-execution concern this simulator's token-count-
based execution model cannot represent. Full reasoning:
`docs/slai_faithful_scheduler_reference.md` §RAD.

---

## H. SLAI Implementation State

- **Upstream repo:** `github.com/agrimUT/SLAI`
- **Pinned commit:** `5098a7aba05e3edbcfa3a509d6cc9cd248fc4380` (main,
  2025-08-14)
- **License:** Apache License 2.0
- **Faithful behaviors:** last-schedulable-time formula (Eq. 8) with the
  pinned source's one-iteration lag; 4-step batch construction order
  (critical decodes → prefill → leftover-budget non-critical decodes →
  LST refresh); fixed/dynamic memory-pressure offset (Θ=5/10, 96% threshold,
  the paper's own flagship "SPF, dynamic offset" configuration); chunked-
  prefill/KV reuse of Sarathi-Serve's own memory model (`SLAIBlockSpaceManager`
  is a no-op subclass upstream, confirmed by reading the source).
- **Disclosed adaptations:** per-request TBT derived from `class_id`
  (mapped consistently across **both** of this project's independently-
  authored 3-tier `class_id` vocabularies — see the fix in §L); N-tier
  generalization of the paper's exactly-2-tier SPF priority; `step_size`
  used in place of the paper's variable, GEMM-cost-dependent `b_batch`
  (this simulator has no batch-duration variance by construction).
- **Test results:** 22 dedicated tests (`tests/test_slai_faithful_scheduler.py`)
  + 15 for the new `Action.hold_decode` primitive
  (`tests/test_simulator_decode_hold.py`) + updated scope-audit/pinned-count
  tests, all passing. Full non-hardware suite (pre-`slai_faithful`, testing
  only the `hold_decode` primitive in isolation): **2501 passed, 88 skipped,
  26 deselected, 0 failed**, exit code 0. **Durable record and exact
  reproduction command:** `docs/current/PAUSE_PROVENANCE_2026-07-23.md`
  (the original log lived at a session-scratch `/tmp` path that will not
  survive logout/reboot — do not rely on it; the provenance doc is the
  durable source now).
- **Bounded pilot results:** experiment root
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/slai_faithful_bounded_pilot_20260723T033609Z/`
  (job **1129769**, COMPLETED, exit 0:0, 00:01:47 elapsed). 12 windows (3
  each: Azure, BurstGPT, SwissAI, TraceLab) × 5 policies (`slai_faithful`,
  `sarathi_faithful`, `vllm_chunked_prefill_faithful`,
  `weighted_shortest_processing`, `scorpio_style_slo_guard`). Full results:
  `results/pilot_results.json`, `results/pilot_summary.json`.
  - **Oracle-envelope gain from adding SLAI: exactly 0.0 in every one of the
    12 windows.**
  - Azure/BurstGPT/SwissAI: every policy tied at ANWG=1.0 (underloaded pilot
    windows — consistent with §E/§F's known saturation issue, not a
    slai_faithful-specific finding).
  - TraceLab (the only loaded regime tested): slai_faithful **lost clearly**
    to WSP/SCORPIO in all 3 windows (ANWG 0.51/0.36/0.47 vs. 0.80/0.68/0.79),
    and roughly tied-or-slightly-worse vs. the other two chunked-prefill
    faithful baselines — suggesting a shared property of the whole chunked-
    prefill-admission family under extreme long-context load, not something
    unique to SLAI's decode-deferral mechanism.
  - Decode-hold **does activate meaningfully** on real data: 0%–16.4% of
    steps across the 12 windows (mean 8.2%) — confirms the mechanism is
    live, not dead code.
- **`FULL_SWEEP_RECOMMENDATION = NO_GO`**
- **Reasons for NO_GO:** zero oracle-envelope gain across every tested
  window, plus a clear loss on the one genuinely load-differentiated
  dataset in the pilot. This is a negative *scientific* result on realistic
  data as currently configured, not an implementation-fidelity failure —
  the implementation itself is source-grounded and well-tested. Recommended
  before reconsidering: (1) recalibrate BurstGPT/Azure/SwissAI windows to
  genuinely loaded regimes; (2) re-run TraceLab with a larger `max_steps`
  budget to rule out simulation-horizon truncation as a confound; (3)
  investigate the one TraceLab window with exactly 0% hold rate, which
  coincided with slai_faithful's worst within-family result.

---

## I. Important Experiment Results (paths + conclusions only)

| Experiment | Root | Conclusion |
|---|---|---|
| Selector Dataset v2 Overnight Scale | `.../selector_v2_overnight_20260720T235405` | 1600 windows; RF beat WSP ID (`0.559` vs `0.527`), lost OOD (`0.248` vs `0.256`) |
| Selector v2 OOD Investigation | `.../selector_v2_ood_conclusive_20260721T133408Z` | `IMPROVE_DATA_OR_FEATURES`; BurstGPT WSP-vs-SCORPIO routing dominated error |
| Selector v3 Multi-Domain | `.../selector_v3_multidomain_causal_20260721T151341Z` | `DATA_LIMITED`; richer features didn't fix held-out WSP loss |
| Policy Library v2 Expanded Frontier | `.../policy_library_v2_expanded_20260721T171933Z` | Synthetic/frontier V2 expansion real but modest |
| **V2 Real-OOD 27-Policy Library Audit** | `.../v2_real_ood_library_20260721T222521Z` | `STRONG_EXPANSION`; V1→V2 oracle ANWG gain `+0.008904` (`3.54%`), CI `[0.008191,0.009646]` |
| Module Intervention/Structural Credit | `.../module_intervention_credit_20260721T224322Z` | Sparse single-module positive transfer; pairwise did NOT expand envelope |
| Native Composition Pilot | `.../native_composition_pilot_20260721T194929Z` | `NO_GO` — composition did not beat discrete selector |
| **27-Policy Selector/Regret Benchmark** | `.../v2_selector_regret_benchmark_20260722T134925Z` | `STRONG` per-report, but top-1 selector did NOT capture V1→V2 oracle gain OOD |
| SwissAI V2 27-Policy Sweep | `.../swissai_v2_policy_sweep_20260722T184451Z` | 512×27 complete matrix (reporting step failed, data intact); zero strict marginal gain, strong saturation |
| TraceLab V2 27-Policy Sweep | `.../tracelab_v2_policy_sweep_20260722T214129Z` | `REDUNDANT`; saturated, zero strict marginal gain |
| SLO/Deadline Augmented V2 Sweep | `.../slo_deadline_augmented_v2_sweep_20260722T194529Z` | `USEFUL_INCREMENTAL`; synthetic/regime-probing only |
| **Simulator Discriminative-Power Audit** | `.../simulator_discriminative_audit_20260722T223236Z` | `COMBINER_TRAINING_SIGNAL=WEAK`, `NEEDS_SIMULATOR_FIX` — the standing #1 bottleneck |
| **SLAI Bounded Pilot (this session)** | `.../slai_faithful_bounded_pilot_20260723T033609Z` | Oracle gain `0.0` everywhere; TraceLab loss; `NO_GO` for full sweep |

Full index with job IDs: `docs/current/EXPERIMENT_INDEX.md` (not modified
this session; the SLAI pilot row above is NOT yet added there — see §M).

### I.1 Experiment Reproducibility Table (added during the 2026-07-23 pause pass)

| Experiment | Durable result root | Script/entry point | Git SHA | Reproducible now? | Missing prerequisite |
|---|---|---|---|---|---|
| Selector Dataset v2 Overnight Scale | `.../selector_v2_overnight_20260720T235405` | not individually re-verified this pass | not individually verified this pass | Likely, via durable root | none identified this pass |
| V2 Real-OOD 27-Policy Library Audit | `.../v2_real_ood_library_20260721T222521Z` | scripts under root's `tools/`/`configs/` | not individually verified this pass (no dedicated `provenance.json` found under `manifests/`) | Likely, via durable root | git SHA not recorded in this root's own manifests |
| 27-Policy Selector/Regret Benchmark | `.../v2_selector_regret_benchmark_20260722T134925Z` | `scripts/selector_regret_benchmark.py` | not individually verified this pass (no dedicated `provenance.json`; per-task `manifest.json`s exist under `models/`) | Likely, via durable root + per-model manifests | git SHA not centrally recorded |
| SwissAI V2 27-Policy Sweep | `.../swissai_v2_policy_sweep_20260722T184451Z` | `scripts/swissai_v2_policy_sweep.py` | **`e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302`** (confirmed via `manifests/provenance.json`, branch `wulver-final-integration-20260721`) | Yes — commit confirmed to be an ancestor of current HEAD | Report step itself failed (`kv_proxy_p95` bug per `EXPERIMENT_INDEX.md`) — data matrix (`combined/policy_vectors.csv`) is intact |
| TraceLab V2 27-Policy Sweep | `.../tracelab_v2_policy_sweep_20260722T214129Z` | `scripts/tracelab_v2_policy_sweep.py` | **`e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302`** (confirmed via `manifests/provenance.json`) | Yes | none identified |
| SLO/Deadline Augmented V2 Sweep | `.../slo_deadline_augmented_v2_sweep_20260722T194529Z` | `scripts/slo_deadline_augmented_v2_sweep.py` | not individually verified this pass | Likely, via durable root | git SHA not individually confirmed this pass |
| Simulator Discriminative-Power Audit | `.../simulator_discriminative_audit_20260722T223236Z` | `run_discriminative_audit.py` | **`e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302`** (confirmed via `manifests/provenance.json`) | Yes | none identified |
| **SLAI Bounded Pilot** | `.../slai_faithful_bounded_pilot_20260723T033609Z` | `scripts/bounded_pilot.py` + `sbatch/bounded_pilot.sbatch` | Runs against whatever `src/` state is on disk at execution time — **not itself git-pinned** (the script does `sys.path.insert(0, ".../src")` against the live checkout, so re-running after any future `src/` change reproduces THAT state, not 2026-07-23's) | Yes, for the pilot mechanics; **not** for reproducing the exact 2026-07-23 numbers if `src/llmserveopt` changes before re-running | Should record `git rev-parse HEAD` into a manifest before future re-runs — not done for this first pilot, a gap to fix if the pilot is re-run |

**Verification method used for the three rows above with a confirmed SHA:**
read each root's `manifests/provenance.json` directly (`{"branch":
"wulver-final-integration-20260721", "commit":
"e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302"}`), then confirmed with
`git merge-base --is-ancestor e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302 HEAD`
(exit 0 — confirmed ancestor of the current, 2026-07-23 HEAD). Rows without
a confirmed SHA were not individually checked in this pass — their roots'
`manifests/` directories either lack a top-level `provenance.json` or it
was not opened; this is a **known gap**, not a claim that they're
unreproducible, just that the git-pin was not independently re-verified for
every row given this pass's time budget.

---

## J. Current Bottlenecks (ranked)

1. **Simulator/objective discriminative power** (standing #1 bottleneck,
   confirmed independently by this session's SLAI pilot). Everything else is
   downstream of this.
2. **No frozen selector artifact** — any future selector evaluation starts
   from retraining, not loading.
3. **SLAI's realistic-data value is currently unproven** (`NO_GO`) —
   needs load-calibrated re-testing, not more implementation.
4. **Composition/synthesis is infrastructure-ready but scientifically
   unvalidated** (`NO_GO` from the native pilot) — do not resume without new
   evidence.
5. **No real-hardware cross-check for `slai_faithful`** against the actual
   upstream repo (plan exists, not executed — `docs/slai_faithful_scheduler_reference.md` §Phase 8).
6. **10 stale, permanently-blocked Slurm jobs** remain queued from
   already-completed experiments (§N/Slurm pause state below) — a cleanup
   task, not a science risk. (The other linked worktree, previously flagged
   as a risk, was fully audited during the 2026-07-23 pause pass and found
   to contain **no unique or at-risk work** — see §WORKTREE_WARNING below.)

---

## K. Recommended Future Roadmap (when resumed)

1. ~~Finalize the current SLAI changes~~ **DONE as of 2026-07-23**: committed
   as `8c9cedbca171d44030a16cf630f81f99d15d729f` (parent
   `d1d5f12a0752a061e563f87dcf3e3289bee2e4bb`) and pushed to
   `origin/wulver-final-integration-20260721` — 0 ahead / 0 behind at pause
   time. A future agent should re-run `git status`/`git log` to confirm
   this is still true (nothing should have changed while the project was
   paused, but verify rather than assume).
2. **Realistic load calibration** — recalibrate window/arrival-rate
   construction so Azure/BurstGPT/SwissAI windows are not trivially
   underloaded (matches Stage 2 of `docs/current/RESEARCH_ROADMAP.md`,
   the project's own standing highest-priority stage — **do this before any
   new baseline work, not just for SLAI**).
3. **External-baseline expansion** — PARS-LTR, WAIT/Nested-WAIT are the
   most concretely-scoped candidates per `docs/external_baseline_decision.md`
   (fix its two citation-year errors first).
4. **One final selector benchmark** — only after (2), per the project's own
   stop/go position; do not retrain before that.
5. **Decide when to freeze selector development** — criterion already
   defined: "learned selectors should capture a meaningful fraction of the
   V2 oracle gain on held-out OOD" (`RESEARCH_ROADMAP.md` Stage 4).
6. **Composition/synthesis** — blocked until Stage 2-4 pass; do not resume
   from evidence available today.
7. **Real-hardware validation** — for both the general project (Stage 8) and
   specifically `slai_faithful` (its own §Phase 8 plan) — lowest priority,
   listed last in both this roadmap and the pre-existing one.

This order is **consistent with, not overriding,** the pre-existing
`docs/current/RESEARCH_ROADMAP.md` Stage 1-8 plan — treat that document as
the authoritative staged plan; this list is how the new SLAI work slots into
it.

---

## L. "Do Not Repeat These Mistakes"

- **Stale documentation vs. actual git/code**: multiple docs said "6 faithful
  baselines"/"33 total policies" — always verify counts by importing
  `registry.py`/`external_baselines_registry.py` directly, never trust a
  doc's stated count without cross-checking code.
- **Confusing style/inspired with faithful baselines**: `scorpio_style_slo_guard`,
  `sarathi_style`, `slai_style_phase_aware`, etc. are NOT pinned-commit
  reproductions — they're original heuristics with suggestive names. Only
  the 7 in `EXTERNAL_BASELINE_REGISTRY` are faithful.
- **Underloaded workloads create ANWG=1.0 ties**: a small/default-rate
  window will almost always saturate to "everyone completes everything on
  time" — this is a *known, recurring* trap (this session's pilot hit it on
  3/4 datasets), not a simulator bug each time it happens. Always check
  `num_dropped`/completion fraction before trusting a "tie" result.
- **SwissAI token reconstruction is fundamentally limited**: the raw
  bucket-reuse data has **zero** rows with real `prompt_tokens`/`output_tokens`
  — any per-request token count for SwissAI is a proxy, never a real
  observation. Don't present SwissAI results as if token counts were real.
- **TraceLab horizon/drop-rate concerns**: high drop rates for chunked-
  prefill-family policies on TraceLab in this session's pilot may be
  partly a `max_steps` truncation artifact — not confirmed either way. Don't
  over-interpret a single small pilot's drop-rate numbers as final.
- **Missing frozen selector artifact**: don't assume one exists — it doesn't,
  checked twice this session (repo + full shared data root).
- **Old dirty checkout vs. authoritative integration checkout**: this repo
  has **two linked git worktrees** sharing one `.git` — see the dedicated
  **WORKTREE_WARNING** section below for the full, verified audit. Short
  version: `.../llm-serving-heuristic-evolution-final-integration` (this
  checkout) is authoritative; the other
  (`.../llm-serving-heuristic-evolution`, branch
  `wulver-policy-composition-readiness`) was fully audited this pause pass
  and confirmed to hold **no unique or at-risk work** — but still don't
  confuse them, and don't run `git` commands assuming they share
  working-tree state (they don't — only history/objects).
- **Background/interactive jobs instead of Slurm for long tasks**: the login
  node (`login02`) has exactly 1 visible CPU and no active allocation —
  anything beyond a trivial smoke test must go through `sbatch`. The bounded
  SLAI pilot followed this rule (job 1129769); earlier full-repo test-suite
  runs in this session were borderline (~11-15 min each) and were run via
  the harness's own background-Bash mechanism rather than Slurm — acceptable
  for CPU-light pytest runs on a shared login node in practice, but the
  *actual experiment/data-generation* work should always go through `sbatch`
  going forward, per this session's explicit instruction.

---

## M. Resume Checklist

Before doing ANY new work, a future agent/researcher should:

1. `cd /mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration && git status` — confirm whether the SLAI changes described here are still uncommitted, already committed, or were reverted. Reconcile with §"Current Uncommitted Work" in the handoff-query final report before assuming anything.
2. Read `docs/current/README.md`, then this document, then `docs/current/PROJECT_STATUS.md`.
3. Re-verify policy counts live: `PYTHONPATH=src python -c "from llmserveopt.policies.registry import *; from llmserveopt.policies.external_baselines_registry import *; ..."` (§C) — do not trust any cached number, including the ones in this document, without a fresh check if significant time has passed.
4. Check `squeue -u $USER` and `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` before touching anything under `/mmfs1/project/ikoutis/sv96/llmserveopt-data/` — some roots there are protected/active.
5. Confirm which of the two worktrees you are in (§L) before running any git command.
6. Do not retrain the selector, launch composition/synthesis, or run a full SLAI dataset sweep without re-reading §J/§K — all three are explicitly gated on simulator-calibration work that has not yet happened.

---

## WORKTREE_WARNING

Full audit performed 2026-07-23 (read-only — the other worktree was
inspected but never modified).

> **Update (still 2026-07-23, after this audit was written):** the "13
> modified + 6 new" SLAI files described below for the authoritative
> checkout were subsequently committed as
> `8c9cedbca171d44030a16cf630f81f99d15d729f` and pushed to
> `origin/wulver-final-integration-20260721` (0 ahead / 0 behind). The
> authoritative checkout's working tree is clean as of the final pause
> state — the table below is a snapshot from earlier in the same pause
> sequence, not the final state. The other worktree's state (right column)
> is unaffected by this and remains as described.

| | Authoritative (this checkout) | Other worktree |
|---|---|---|
| Path | `.../llm-serving-heuristic-evolution-final-integration` | `.../llm-serving-heuristic-evolution` |
| Branch | `wulver-final-integration-20260721` | `wulver-policy-composition-readiness` |
| HEAD at time of this audit | `d1d5f12a0752a061e563f87dcf3e3289bee2e4bb` (now the parent of checkpoint commit `8c9cedb...`, see update note above) | `c8aee129f553f8dc3ede99eac60d5b14484beb41` |
| HEAD relationship | — | `c8aee12` **is an ancestor of** `d1d5f12` (`git merge-base --is-ancestor` confirmed) — no unique committed history |
| Dirty/uncommitted files at time of this audit | 13 modified + 6 new (SLAI work, this pause sequence) — **since committed, see update note above** | 3 modified, 34 untracked (31 files + 3 `.pyc` cache files after expanding the one untracked directory) |

**Per-file verification of the other worktree's uncommitted state (all 37
non-cache entries checked, not sampled):**
- **33 of 34 untracked non-cache files** are `git ls-files`-tracked in the
  authoritative checkout **and byte-identical** to the authoritative
  checkout's committed copy (verified via `diff`) — these are simply loose,
  never-`git add`-ed copies of work that is already safely committed on the
  authoritative branch. Includes all of Policy Library v2's actual policy
  source files (`slai_style_phase_aware.py`, `sola_style_state_aware.py`,
  `genome.py`, `structural_synthesis.py`, `composition.py`,
  `composition_experiment.py`, `parent_selection.py`,
  `policy_library_v2_helpers.py`, `flow_control_stability.py`,
  `kv_constrained_online.py`, `aging_priority.py`,
  `adaptive_chunked_prefill.py`, `weighted_fair_share.py`, and their tests).
- **7 files differ** — all 7 are `docs/current/*.md` files where the
  authoritative checkout's version has a **strictly newer mtime and more
  lines**, with the extra content being explanatory caveats/status updates
  layered on top of the other worktree's older draft (verified by reading
  the actual diffs, not just mtimes — e.g. `COMPOSITION_IMPLEMENTATION_STATUS.md`:
  other worktree's copy is the pre-"Current Caveat" draft; authoritative
  copy adds the caveat explaining the composition gate stayed closed).
  `WULVER_BRANCH_LINEAGE_AUDIT.md`'s own text confirms this was a
  **deliberate, documented split**: the final-integration branch/worktree
  was created specifically so the original worktree's active SLURM
  workflows could keep running undisturbed while integration/cleanup work
  happened separately.
- **3 files** exist only in the other worktree's untracked set and are not
  tracked in the authoritative checkout — all 3 are `tools/__pycache__/*.pyc`
  compiled bytecode caches, not source, not work product.
- The 3 tracked-but-modified files in the other worktree
  (`docs/current/README.md`, `src/llmserveopt/policies/__init__.py`,
  `src/llmserveopt/policies/registry.py`) were also diffed directly against
  the authoritative checkout's committed versions:
  `__init__.py`/`registry.py` are byte-identical; `README.md` is an older,
  pre-reorganization draft (matches the same "superseded" pattern above).

**Conclusion: no unique or at-risk work exists in the other worktree.**
Future agents do not need to reconcile, merge, or preserve anything from it
before proceeding — but should still avoid deleting or `git reset`-ing it
without cause, since it hosts (or may host) other active SLURM workflows
tied to its own working-tree state (see `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`,
tracked in the authoritative checkout, which documents protected experiment
roots regardless of which worktree originally launched them).

---

## Slurm Pause State (as of 2026-07-23)

Checked via `squeue -u sv96`; **zero jobs in state `RUNNING`.**

- **Active project jobs:** none.
- **Stale, permanently-blocked pending jobs (10 total):** `1127958_[0-39%6]`,
  `1127959`, `1127960`, `1127961`, `1127962`, `1127963`, `1127964`
  (`slo_aug_*` family, state `Dependency`), `1127943`, `1127944_[0-39%6]`,
  `1127950` (additional `slo_aug_*`, `Dependency`/`DependencyNeverSatisfied`),
  and `1127600` (`swissai_v2_report`, `DependencyNeverSatisfied`). All belong
  to experiments already marked COMPLETE (SLO/Deadline Augmented V2 Sweep)
  or PARTIAL-with-complete-data (SwissAI V2 Sweep) in
  `docs/current/EXPERIMENT_INDEX.md` — their upstream dependency jobs have
  already finished or failed, so several are marked
  `DependencyNeverSatisfied` by Slurm itself, meaning **these will never
  run and cannot modify any output directory.**
- **Jobs safe to ignore:** all 10 above, for exactly that reason.
- **Jobs requiring future cleanup:** the same 10 — `scancel` them in a
  future session (not done in this pause pass per explicit instruction not
  to cancel anything here). No other cleanup identified.
- **Unrelated jobs:** none observed in this account's queue at pause time.

Full durable record with per-job-family detail:
`docs/current/PAUSE_PROVENANCE_2026-07-23.md`.

---

*End of handoff document. See `docs/current/project_handoff_state.json` for
the machine-readable summary, and `docs/current/PAUSE_PROVENANCE_2026-07-23.md`
for durable evidence backing every test/pilot result cited above.*

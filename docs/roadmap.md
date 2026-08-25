# Roadmap

> **Canonical research roadmap note (2026-08-07):** the whole-project
> research roadmap — north star, workstream map, status dashboard,
> supported/hypothesis/not-established findings, negative results, and
> dependency-aware future plan — now lives at
> [`docs/PROJECT_MAP.md`](PROJECT_MAP.md). This document remains a
> historical numbered-phase roadmap and Selector v2/external-baseline
> status bridge; treat `docs/PROJECT_MAP.md` as authoritative if the two
> disagree.

> **Contextual-composition branch note (2026-08-04, active -- not paused):**
> The active roadmap for branch `contextual-compositional-heuristics-20260731`
> is [contextual_composition_roadmap.md](contextual_composition_roadmap.md).
> CC1-CC5 are complete -- CC5 closed `COMPLETE_REGIME_SPECIFIC` on
> 2026-08-03 (frozen operating-envelope system statistically beats best
> fixed and the hard selector; edge over `best_global_composition` not
> statistically distinguishable from zero) -- current status:
> [CC5 final operating envelope report](audits/contextual_composition_cc5_final_operating_envelope_20260803.md);
> CC6 is queued but **restricted** to the CC5 trusted envelope
> (`burst_transition`, `kv_pressure`, `long_output`, `prediction_noise`,
> `saturated`, `selective_admission_trap`, `underloaded`), not yet
> started; active issue:
> [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6).
> A separate, parallel baseline-integration effort is also underway on
> this branch (vLLM-LTR complete/evaluation-only; PARS-Serve-2026 complete,
> independently verified, EVALUATION_ONLY; VTC fairness-validated
> comparative sweep complete 2026-08-05 -- EVALUATION_ONLY (deployment),
> scientific classification FOUNDATIONAL_CANDIDATE (not registered) -- see
> [vtc_fairness_comparative_evaluation_20260805.md](audits/vtc_fairness_comparative_evaluation_20260805.md) --
> see [BASELINE_STATUS.md](BASELINE_STATUS.md)); it does not affect CC5/CC6
> status.
> Resume that branch from
> [RESUME_CONTEXTUAL_COMPOSITION.md](RESUME_CONTEXTUAL_COMPOSITION.md).
> The document below remains a historical numbered-phase roadmap and Selector
> v2/external-baseline status bridge; do not treat it as the current roadmap for
> the contextual-compositional heuristic path.

> **Note (2026-06-27):** The original phase structure below was written early in the
> project. The numbered-phase table reflects completed phases only through the
> 2026-06-27 pause checkpoint -- see the current-track section immediately below for
> everything since. For up-to-date status, see
> [docs/current/PROJECT_STATUS.md](current/PROJECT_STATUS.md), not `research_status.md`
> (retained only as a historical redirect).

## Current track: Selector v2 / external-baseline validation (unnumbered, active)

Development did not stop at the Phase 2C pause below -- it continued on an
**unnumbered track** not reflected anywhere in the phase table. In order:

1. External faithful baseline implementation -- 6 pinned-commit
   reimplementations (vLLM, vLLM-chunked-prefill, Sarathi-Serve, DistServe,
   TetriInfer, Llumnix). `docs/external_baseline_integration.md`.
2. Real-hardware runtime validation (local RTX 5060 Ti + Wulver A100,
   including N=5-repeated-trial Sarathi-vs-vLLM comparisons and a committed
   runtime-validation benchmark pack). `docs/wulver_sarathi_vllm_repeated_validation.md`.
3. Objective correction: `weighted_goodput`'s completed-only-denominator bias
   fixed via `arrival_normalized_weighted_goodput` (ANWG).
   `docs/selector_objective_audit.md`.
4. Selector Dataset v2 infrastructure, scenario redesign, and a
   policy-independent SLO calibration fix. `docs/selector_dataset_v2.md`,
   `docs/selector_v2_slo_calibrated_frontier_search.md`.
5. Faithful-baseline scope audit -> **Option B** decision: the Selector v2
   trainable action space is 8 of the 20 internal policies; faithful
   baselines are evaluation-only. `docs/selector_v2_faithful_baseline_scope_audit.md`.
6. Calibrated targeted pilot (250 windows, Option B scope) -- all pipeline
   quality gates passed, but an independent audit confirmed a real
   cross-transform leakage bug in the non-OOD splits; the one confirmed-clean
   split (OOD_TEST) loses to best-fixed. Not yet a finished result.
7. Baseline-integration phase begun and completed (2026-08-04): official
   vLLM-LTR (hao-ai-lab/vllm-ltr, pinned commit
   `13bbf6ff3dab661791d41362551b089e5f77c91c`; paper is NeurIPS 2024 main
   conference) integrated under `baselines/vllm_ltr/` as an isolated,
   evaluation-ready, offline-scored external baseline -- official
   checkpoint downloaded/hash-verified/architecturally verified, offline
   scoring pipeline built, semantic equivalence confirmed bit-exact via
   independent recomputation. Not yet a selector candidate.
   `docs/audits/vllm_ltr_baseline_audit_20260804.md`.
8. First comparison sweep run and independently re-verified (2026-08-04,
   after recovering from an initial run that never finished -- a selector
   performance bug, fixed): real WildChat-1M text, 300 requests, 3 seeds, 10
   policies. Result: `vllm_ltr_semantic_reference` tied FIFO/EDF/EST/SOF/
   WSP/`oracle_srtf` exactly in this regime (the oracle itself ties FIFO --
   no reorderable headroom for any policy here); moderate-not-near-1.0
   ranking agreement with EST/SOF confirms it is not behaviorally redundant.
   Classification: EVALUATION_ONLY; foundational-library eligibility not
   established by this run (needs a higher-contention regime).
   `docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`,
   `docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md`.
9. Canonical discriminative benchmark suite designed, generated, and
   characterized (2026-08-04): the WildChat-only comparison above gave
   ordering policies essentially zero headroom (oracle ties fifo exactly)
   -- diagnosed as a benchmark limitation, not a scheduler limitation
   (`docs/audits/ordering_workload_headroom_audit_20260804.md`). 9
   synthetic workload families were designed and headroom-validated; 7
   accepted (real, substantial ordering headroom restored), 2 rejected
   with documented reasons. A foundational-heuristic characterization
   pass across the accepted suite + WildChat control found
   `scorpio_style_slo_guard` -- the *worst* policy on the WildChat control
   -- is the *best* policy in 4 of 7 accepted synthetic families,
   confirming the WildChat-only comparison was hiding real
   admission-control value. `docs/audits/canonical_benchmark_suite_design_20260804.md`,
   `benchmarks/canonical_suite/`.
10. PARS-Serve-2026 baseline integration begun 2026-08-04, **complete and
    independently verified as of 2026-08-05** (see
    `docs/BASELINE_STATUS.md` for current status): official code
    (`SPEAR-UIC/PARS`, pinned commit `fd4e125b65bb73aef5eccafa79c2509434be61ec`)
    integrated unmodified; no pretrained checkpoint is released, so a
    real `bert-base-uncased` pairwise ranker was trained locally with the
    official, unmodified training script (`best_val_accuracy=0.9141`,
    checkpoint hash-verified). Named "PARS-Serve-2026" in this project's
    prose to disambiguate from an unrelated, earlier "PARS" already
    referenced elsewhere in this repo's docs (now "PARS-2023"). Known
    license gap (no upstream LICENSE file) disclosed, not hidden -- see
    `baselines/pars/PROVENANCE.md`. Comparative evaluation across WildChat
    control + all 7 accepted canonical-suite families completed via a
    combination of the original run (3 families) and per-family timeout
    recovery (5 families, after the original run's 10800s timeout killed
    it mid-flight); classified **EVALUATION_ONLY** -- zero unique wins
    across 8 families, dominated by simpler existing policies in every
    discriminative regime. `docs/audits/pars_baseline_implementation_20260804.md`,
    `docs/audits/pars_first_comparative_evaluation_20260804.md`.
11. VTC baseline integration begun 2026-08-05 (initial integration +
    smoke evaluation), then repaired and completed with a
    fairness-validated comparative sweep the same day (see
    `docs/BASELINE_STATUS.md` for current status): official artifact
    (`Ying1123/VTC-artifact`, pinned commit
    `192c2e2014c69c8c6c699d7113c3822e4db632e6`, Apache-2.0) is a full
    S-LoRA-based GPU serving engine this machine's GPU (RTX 5060 Ti,
    Blackwell) cannot build the CUDA kernels for (a compiler-generation
    gap, not a version-pin fix) -- but VTC's fairness-scheduling
    **algorithm** is pure Python/NumPy and was dynamically imported and
    executed completely unmodified via `baselines/vtc/adapter/`.
    Classified "official policy reused with simulator adapter." The
    initial smoke pass found a methodological confound (5/6 families
    showed no policy divergence at all -- insufficient backlog
    contention; the 1 that diverged was admission-gate-driven, traced
    precisely to a `max_batch_tokens` units mismatch between this
    simulator's native request-count interpretation and the official
    code's real token-budget interpretation, not a fairness-ordering
    effect). The repair pass built three labeled comparison variants
    (official VTC / matched-admission FIFO via the official FCFS base
    class / fairness-isolation VTC), retuned all six fairness-extension
    workloads for verified genuine contention, gated them with
    `scripts/check_vtc_fairness_headroom.py` (all 6 pass), and ran a
    108-run comparative sweep independently re-verified with **zero
    unexplained mismatches**. Result: VTC wins/ties the fairness
    comparison in 17/18 family x seed combinations, isolated to be an
    ordering effect, with a real bounded ANWG trade-off (0.680 vs.
    SCORPIO's 0.984) in the one family designed to expose its
    SLO-blindness. 45/45 tests pass across three test files. Deployment
    classification remains **EVALUATION_ONLY**; scientific
    classification **FOUNDATIONAL_CANDIDATE** (not registered).
    `docs/audits/vtc_official_artifact_audit_20260805.md`,
    `docs/audits/vtc_initial_integration_20260805.md`,
    `docs/audits/vtc_fairness_benchmark_repair_20260805.md`,
    `docs/audits/vtc_fairness_comparative_evaluation_20260805.md`.

**Full synthesis, in narrative order, with the current (confirmed leakage
bug, split-fix pending) result:** [docs/current/SELECTOR_V2.md](current/SELECTOR_V2.md).
**Next step:** [docs/current/NEXT_STEPS.md](current/NEXT_STEPS.md).

---

## Historical: numbered phase status (through 2026-06-27 pause)

| Phase | Description | Status |
|---|---|---|
| 1 | Simulator + classical baselines | ✅ Complete |
| 1.5 | Serving-style baselines (Orca/vLLM/Sarathi/SplitFuse) | ✅ Complete |
| 1.7A | BurstGPT + ShareGPT trace ingestion | ✅ Complete |
| 1.7B | GPU calibration (RTX 5060 Ti, Qwen2.5-0.5B) | ✅ Complete |
| 1.7C | Calibrated real-trace replay (7 experiments) | ✅ Complete |
| 2A.1 | Metric finalization + oracle wiring | ✅ Complete |
| 2A.2–2A.3 | Selector dataset + training + evaluation | ✅ Complete |
| 2A.3B | Hardened baselines (LLF, ESTF) + priority_weighted alias | ✅ Complete |
| 2B.1 | LLM heuristic DSL + verifier + policy wrapper | ✅ Complete |
| 2B.2 | LLM offline heuristic generation loop | ✅ Complete |
| 2B.3 | Controlled LLM heuristic search (multi-regime eval) | ✅ Complete |
| 2A.4/2B.4 | Final evaluation hardening (shortlist freeze, held-out test, bootstrap CIs) | ✅ Complete |
| 2B.5–2B.9 | External baselines, admission control, rule selector repair, robustness audit | ✅ Complete |
| 2B.10–2B.13 | SCORPIO policy, selector training on diversified suite (256 windows) | ✅ Complete |
| 2B.14–2B.16 | Metric audit (ANWG), corrected-objective retraining, fresh validation | ✅ Complete |
| 2C.1 | Real-trace ingestion: Azure 2023 + BurstGPT validation | ✅ Complete |
| 2C.2 | Causal selector retraining on real traces (ANWG = 0.8021; envelope = 0.8297) | ✅ Complete |
| 2C.3 | External-aware pool analysis (negative finding: no orca recovery) | ✅ Complete |
| 2C.4 | Pairwise/regret-weighted selector training from labeled dataset | 🔲 Not started |
| — | LLM evolution loop (full) | 🔲 Not started |
| — | Shifted-workload evaluation + paper write-up | 🔲 Not started |

---

## Original Phase Plan (historical reference)

## Phase 1 (current) — Baseline Infrastructure

- [x] Deterministic iteration-level simulator
- [x] Synthetic workload generators (Poisson, bursty, heavy-tail)
- [x] 8 classical baseline policies
- [x] Oracle policy (hindsight SRTF, non-deployable)
- [x] Evaluation metrics (latency, P95/P99, SLO violation, throughput, utilization)
- [x] Experiment runner with YAML configs
- [x] Result tables and figures
- [x] Unit tests
- [x] Documentation

## Phase 2 — LLM-Generated Restricted Policy Code

- [ ] Define a restricted policy DSL (whitelisted Python subset)
- [ ] Integrate Cohere / CloudRift API calls for code generation
- [ ] Build a sandboxed executor for LLM-generated policies
- [ ] Implement automated feasibility checking before execution
- [ ] Initial LLM-generated policy experiments on Phase 1 traces
- [ ] Compare LLM-generated policies against all Phase 1 baselines

## Phase 3 — Verifier and Sandbox

- [ ] Formal constraint verifier: check that generated policies respect all capacity
  constraints by static analysis or exhaustive small-trace testing
- [ ] Sandboxed execution environment (resource limits, import restrictions)
- [ ] Automated test suite for generated policies
- [ ] Rejection / repair mechanism for invalid generated policies

## Phase 4 — LLM-Evolution Loop

- [ ] Iterative prompt engineering: use metric feedback to guide re-generation
- [ ] Evolutionary search over policy code (mutation, crossover via LLM)
- [ ] Population-based optimization across multiple seeds
- [ ] Integration with Cohere Command R+ and CloudRift inference endpoints
- [ ] Track Pareto front: latency vs. SLO violation rate vs. throughput

## Phase 5 — Shifted Workload Evaluation and Paper Write-up

- [ ] Evaluate evolved policies on out-of-distribution workloads (shifted arrival rates,
  different length distributions)
- [ ] Compare against production vLLM baselines (requires additional validation)
- [ ] Ablation studies on policy complexity vs. performance
- [ ] Write-up: problem formulation, method, simulator, results, limitations
- [ ] Reproducibility package: frozen traces, policy code, result logs
- [ ] Submission target: MLSys, OSDI, or similar venue

## Known gaps before Phase 2

1. **Prefill cost**: Phase 1 treats prefill as free; realistic serving includes a
   non-trivial prefill step proportional to prompt length.
2. **KV cache paging**: No eviction/preemption means long-running requests can hold
   KV capacity indefinitely; real systems use block-level KV managers.
3. **Throughput saturation**: In Phase 1, decode throughput is constant regardless
   of batch size; real GPUs slow down under memory pressure.
4. **Real workload traces**: Phase 1 uses synthetic traces only.
5. **LLM policy interface**: The generated policy API needs to be frozen and
   documented before Phase 2 begins.

# Roadmap

> **Contextual-composition branch note (2026-08-03, active -- not paused):**
> The active roadmap for branch `contextual-compositional-heuristics-20260731`
> is [contextual_composition_roadmap.md](contextual_composition_roadmap.md).
> CC1-CC4 are complete; CC5 is `IN PROGRESS` (targeted dataset expansion +
> rerun underway) -- current status:
> [CC4b/CC5 retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md);
> active issue:
> [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
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

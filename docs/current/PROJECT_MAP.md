# Project Map

A stable navigation map of this repository — not a dated narrative. For
current status, see `docs/current/RESUME_HERE.md` and
`docs/current/WORK_STATUS.md`; this file only answers "where do I look."

> **Not to be confused with `docs/PROJECT_MAP.md`** (one directory up),
> which is the canonical *research-program roadmap* — north star, math
> objects, workstream/status dashboard, and dependency-aware future
> roadmap for the whole project. This file only answers "where is the
> code for X"; that file answers "why does the code exist and what's
> next."

---

## Core simulator

- **Purpose:** GPU-calibrated discrete-event simulator for LLM inference serving (admission, batching, KV-cache, preemption/migration).
- **Path:** `src/llmserveopt/simulator/` (`gpu.py`, `kv_block_manager.py`, `calibrated_service_model.py`, `service_model_factory.py`, `constraints.py`, `contention_diagnostics.py`, `request.py`)
- **Maturity:** Mature, heavily tested, foundation for everything else.
- **Look here first for:** simulator semantics, `Action` verbs (including `Action.migrate` for Llumnix), GPU/KV state model.
- **Common confusion to avoid:** this is a discrete-event abstraction, not a real inference engine — no real GPU execution happens here. See `docs/simulator_design.md`.

## Policy implementations

- **Purpose:** the scheduling policies (internal + faithful external reimplementations) the simulator can run.
- **Path:** `src/llmserveopt/policies/` — registries in `registry.py` (internal/historical/Policy Library v2) and `external_baselines_registry.py` (faithful external baselines).
- **Maturity:** Mature; live counts must be verified against the registry files, never trusted from a cached number.
- **Look here first for:** any specific policy's implementation.
- **Common confusion to avoid:** "style"/"inspired" policies (`sarathi_style`, `scorpio_style_slo_guard`, etc.) are original heuristics with suggestive names, **not** faithful reproductions — only names in `EXTERNAL_BASELINE_NAMES` are faithful.

## DSL / composition

- **Purpose:** restricted, verifiable JSON DSL for composing scheduling primitives into new policies.
- **Path:** `src/llmserveopt/heuristics/` (DSL + verifier), `src/llmserveopt/policies/primitives.py` (28 registered primitives), `src/llmserveopt/policies/primitive_reconstructions.py`, `src/llmserveopt/policies/composition.py`.
- **Maturity:** COMPLETE (CC2/CC3) — 6/7 representative-policy reconstructions EXACT, 1/7 documented APPROXIMATE.
- **Look here first for:** how a composed/weighted-mixture policy is specified and verified.
- **Docs:** `docs/architecture/contextual_composition_primitives.md`, `docs/architecture/contextual_composition_dsl.md`, `docs/llm_heuristic_dsl.md`.

## Policy Separation Dataset (WS-P)

- **Purpose:** theory-grounded / space-filling scenarios that separate classical
  scheduling mechanisms (prediction, deadlines/admission, fairness/aging, …)
  before QD/selector expansion.
- **Code:** `src/llmserveopt/policy_separation/`
- **Design:** `docs/design/POLICY_SEPARATION_DATASET_V1.md` (status banner is
  authoritative; original Phase-1 “fully implemented” wording is superseded),
  `docs/design/POLICY_SEPARATION_SOBOL_PILOT_V1.md`
- **Recent provenance:**
  - Sobol pilot: `experiments/policy_separation_sobol_pilot_20260816T183600Z_1182183/`
  - Family A pilot: `experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/`
    (analysis under `…/analysis/`; audit
    `docs/audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`)
- **Audits:** `docs/audits/policy_separation_*`
- **Common confusion to avoid:** Job 1182306’s CSV column `anwg` is unweighted
  SLO-success, not canonical `RunMetrics.arrival_normalized_weighted_goodput`;
  that job used synthetic token lengths, not BurstGPT anchoring. Family A v1 is
  diagnostic-only; next is Family A v2 redesign.

## Datasets

- **Purpose:** real-trace and synthetic workload sources.
- **Path:** `data/` (gitignored, generated — not committed), ingestion code in `src/llmserveopt/workloads/`, provenance in `docs/data_field_provenance.md`.
- **Real traces (Tier 1):** BurstGPT, ShareGPT, Azure, Mooncake, TraceLab, SwissAI — see `docs/current/REAL_DATASET_EXPANSION_STATUS.md`.
- **Maturity:** Real-trace ingestion COMPLETE (Phase 1.7A/1.7C lineage); load-calibration is a standing, tracked bottleneck (see `docs/current/RESUME_HERE.md`'s historical section and `docs/current/RESEARCH_ROADMAP.md`).
- **Common confusion to avoid:** Mooncake's license is `NOT_EXPLICITLY_SPECIFIED` — internal OOD-only use, not redistributable.

## Benchmark suites

- **Canonical synthetic suite:** `benchmarks/canonical_suite/`, generator `scripts/generate_canonical_benchmark_suite.py`, design doc `docs/audits/canonical_benchmark_suite_design_20260804.md`. Used by vLLM-LTR/PARS-Serve-2026 evaluations.
- **VTC fairness suite:** `baselines/vtc/fairness_workloads.py`, results in `baselines/vtc/sweep_results/`. 6 repaired workloads × 3 seeds × 6 policies. See `docs/audits/vtc_fairness_benchmark_repair_20260805.md`.
- **WildChat control:** used by vLLM-LTR/PARS as a fixed control regime — see `docs/BASELINE_STATUS.md` for exact usage per baseline.

## External baselines

- **Purpose:** faithful, pinned reimplementations (or adapter-wrapped official code) of published serving systems, used as evaluation-only comparison points — never selector actions.
- **Path:** `baselines/{pars,vllm_ltr,vtc}/` (dedicated adapter directories), plus faithful policy files directly in `src/llmserveopt/policies/` (`sarathi_faithful.py`, `distserve_faithful.py`, `llumnix_faithful.py`, `vllm_faithful.py`, `vllm_chunked_prefill_faithful.py`, `tetriinfer_paper_reimplementation.py`, `slai_faithful.py`).
- **Status index (authoritative):** `docs/BASELINE_STATUS.md` — always check this before trusting any other doc's baseline-status claim.
- **Per-baseline narrative detail:** `docs/baselines.md`.
- **Look here first for:** VTC adapter (`baselines/vtc/adapter/`, wraps the real unmodified `VTCReqQueue`), PARS/vLLM-LTR checkpoint loaders (`baselines/{pars,vllm_ltr}/adapter/checkpoint_loader.py`), Llumnix (`src/llmserveopt/policies/llumnix_faithful.py` + `docs/llumnix_faithful_scheduler_reference.md`), Apt-Serve (`src/llmserveopt/policies/apt_serve_faithful.py`, `scripts/run_apt_serve_phase_g.py`, `scripts/analyze_apt_serve_phase_g.py`, and the Phase G audit).
- **Common confusion to avoid:** this project has four distinct "vLLM" things, two "Sarathi" things — see `docs/baselines.md`'s disambiguation section before citing any of them.

## Stress-test library

- **Purpose:** target/counter-regime catalog testing whether a baseline's claimed mechanism holds under adversarial, not just average-case, conditions.
- **Catalog:** `configs/stress_tests/algorithm_stress_test_catalog.yaml`, narrative `docs/research/algorithm_stress_tests/STRESS_TEST_CATALOG.md`, candidate inventory `docs/research/algorithm_stress_tests/ALGORITHM_INVENTORY_20260805.md`.
- **Coverage today:** Sarathi-Serve, Llumnix, DistServe, VTC, and Apt-Serve all have point-in-time coverage/evaluation evidence. Current baseline status is centralized in `docs/BASELINE_STATUS.md`; Phase G Apt-Serve status is summarized in `docs/audits/apt_serve_phase_g_analysis_20260809.md`.
- **Common confusion to avoid:** a row in `ALGORITHM_INVENTORY_20260805.md` is candidate-identification, not catalog coverage — check the actual catalog YAML for whether a test genuinely exists.

## Audits

- **Purpose:** point-in-time, dated investigative reports (official-artifact audits, comparative-evaluation writeups, reconciliation reports). Treat as historical provenance, accurate as of their own date, not living documents.
- **Path:** `docs/audits/` (~70 files).
- **Most relevant currently:** `apt_serve_official_artifact_audit_20260805.md`, `apt_serve_strategy_c_wulver_probe_20260806.md`, `llumnix_official_artifact_audit_20260806.md`, `sarathi_official_artifact_audit_20260805.md`, `vtc_fairness_comparative_evaluation_20260805.md`, `pars_first_comparative_evaluation_20260804.md`, the CC5 report set (`contextual_composition_cc5_*_20260803.md`), and the three `project_pause_*_query{1,2,3}_20260806.md` reports (this pause sequence).

## Result directories

- **Purpose:** generated experiment outputs (metrics, plots, raw logs).
- **Path:** `results/` — **gitignored, ~109 GB, not committed by design.** Regenerable from configs + code, or durably archived on Wulver (`/mmfs1/...`) — see individual audit docs for exact Wulver paths per experiment.
- **Common confusion to avoid:** an empty/missing `results/<name>/` locally does not mean an experiment never ran — check the relevant audit doc's evidence citations, not just the local filesystem.

## Status / roadmap documents

- **Canonical entry point:** `docs/current/RESUME_HERE.md`.
- **CC-specific technical roadmap:** `docs/START_HERE_CONTEXTUAL_COMPOSITION.md` → `docs/contextual_composition_roadmap.md` → `docs/contextual_composition_decisions.md` (decision log).
- **Cross-baseline status index:** `docs/BASELINE_STATUS.md`.
- **This directory's other current docs:** `docs/current/WORK_STATUS.md`, `docs/current/NEXT_ACTIONS.md`, `docs/current/SCIENTIFIC_DECISIONS.md`, `docs/current/PROJECT_SNAPSHOT_20260806.md`, `docs/current/PROJECT_PAUSE_HANDOFF_20260806.md`.
- **Everything else in `docs/current/`:** historical, pre-CC-branch provenance — see `docs/current/README.md`'s notice.

## Scripts

- **Purpose:** experiment runners, dataset generators, status/consistency checkers, Wulver probe/SLURM scripts.
- **Path:** `scripts/` (top-level runners), `scripts/wulver_probes/` (Apt-Serve import/micro-trace probes, historical after execution), `scripts/slurm/` (`.sbatch` job scripts), `scripts/stress_tests/`, `scripts/workloads/`, `scripts/experiments/`, `scripts/data/`.
- **Status checkers:** `scripts/check_contextual_composition_status.py` (run before every commit on this branch; `--resume-readiness` flag for a stricter post-commit check).

## Tests

- **Path:** `tests/` (3488 tests collected as of this writing — always re-verify with `pytest --collect-only -q`, never trust a cached count).
- **Structure:** mirrors `src/llmserveopt/` roughly; `tests/stress_tests/` for the stress-test catalog; baseline-specific fidelity tests are named `test_<baseline>_faithful*.py` or similarly.

## HPC / SLURM assets

- **Path:** `scripts/slurm/*.sbatch` — includes Sarathi/vLLM real-hardware validation scripts and Apt-Serve Strategy C probe scripts. Check the relevant audit before submitting anything; the 2026-08-06 Apt-Serve Strategy C probe already executed.
- **Convention:** `account=ikoutis`, `qos=standard`, `partition=general` for CPU-only, `partition=gpu` + `--gres=gpu:a100:N` for GPU. Durable Wulver storage root: `/mmfs1/project/ikoutis/sv96/`.
- **Common confusion to avoid:** this workstation cannot execute or directly observe Wulver jobs without a working SSH/Kerberos session — see `docs/current/RESUME_HERE.md` §E `WULVER_DEFERRED` before assuming any job state.

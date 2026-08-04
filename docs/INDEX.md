# Documentation Index

Repository navigation hub, created 2026-08-04 during a repository
organization pass. This is a map, not a replacement for the documents it
links to — start here to find the right doc, then read that doc for full
detail.

## Start here

- **Resuming work on the active branch:** [`RESUME_CONTEXTUAL_COMPOSITION.md`](RESUME_CONTEXTUAL_COMPOSITION.md)
- **Current roadmap for this branch:** [`contextual_composition_roadmap.md`](contextual_composition_roadmap.md)
- **Historical numbered-phase roadmap** (Selector v2 / external-baseline
  status bridge — not the current CC roadmap):
  [`roadmap.md`](roadmap.md)
- **Cross-baseline status at a glance:** [`BASELINE_STATUS.md`](BASELINE_STATUS.md)
- **Where to run what:** [`COMPUTE_POLICY.md`](COMPUTE_POLICY.md)

## Roadmaps and decisions

- [`contextual_composition_roadmap.md`](contextual_composition_roadmap.md) — authoritative CC1-CC7 status
- [`roadmap.md`](roadmap.md) — historical numbered-phase roadmap (Phase 1-2C, baseline-integration items)
- [`external_baseline_decision.md`](external_baseline_decision.md) — original baseline-selection rationale (B.1-B.5 categories), updated with vLLM-LTR/PARS-Serve-2026 outcomes
- [`research_status.md`](research_status.md) — broader research status
- [`result_claims.md`](result_claims.md) — what claims are safe to make from which results

## Baselines and external integrations

- [`BASELINE_STATUS.md`](BASELINE_STATUS.md) — single cross-baseline status table (**start here**)
- [`baselines.md`](baselines.md) — per-baseline narrative detail (vLLM-LTR, PARS-Serve-2026, internal style/inspired policies)
- [`external_baseline_decision.md`](external_baseline_decision.md) — selection rationale
- [`external_baseline_coverage_report.md`](external_baseline_coverage_report.md) — historical coverage snapshot (superseded, retained for context; PARS-2023 naming lives here)
- [`external_baseline_correctness_audit.md`](external_baseline_correctness_audit.md) — fidelity/correctness checks on internal style/inspired policies
- `baselines/vllm_ltr/PROVENANCE.md`, `baselines/vllm_ltr/CHECKPOINT_PROVENANCE.md` — vLLM-LTR provenance
- `baselines/pars/PROVENANCE.md` — PARS-Serve-2026 provenance

## Benchmark suite

- [`audits/canonical_benchmark_suite_design_20260804.md`](audits/canonical_benchmark_suite_design_20260804.md) — the canonical discriminative benchmark suite: design, acceptance criteria, results
- [`audits/ordering_workload_headroom_audit_20260804.md`](audits/ordering_workload_headroom_audit_20260804.md) — the prerequisite diagnosis (why WildChat alone gave zero ordering headroom)
- `benchmarks/canonical_suite/` — the actual generated/accepted workload datasets and manifests

## Baseline evaluation audits (chronological, 2026-08-04)

- [`audits/vllm_ltr_baseline_audit_20260804.md`](audits/vllm_ltr_baseline_audit_20260804.md) — vLLM-LTR checkpoint verification
- [`audits/vllm_ltr_comparative_evaluation_recovery_20260804.md`](audits/vllm_ltr_comparative_evaluation_recovery_20260804.md) — recovery from a selector-performance bug that stalled the first comparison attempt
- [`audits/vllm_ltr_first_comparative_evaluation_20260804.md`](audits/vllm_ltr_first_comparative_evaluation_20260804.md) — vLLM-LTR's completed, independently-verified WildChat comparison
- [`audits/pars_baseline_implementation_20260804.md`](audits/pars_baseline_implementation_20260804.md) — PARS-Serve-2026 implementation record (training in progress as of this writing)
- [`audits/branch_and_pars_readiness_audit_20260804.md`](audits/branch_and_pars_readiness_audit_20260804.md) — read-only audit of branch/PARS-training state mid-training

## Compute and reproducibility

- [`COMPUTE_POLICY.md`](COMPUTE_POLICY.md) — local workstation vs. NJIT Wolverine, logging/manifest/checkpointing conventions
- [`dataset_workload_decision.md`](dataset_workload_decision.md), [`dataset_workload_plan.md`](dataset_workload_plan.md) — dataset/workload selection process
- `external/datasets/wildchat.md`, `external/datasets/sharegpt.md` — real-dataset provenance

## Status / resume / handoff documents

- [`RESUME_CONTEXTUAL_COMPOSITION.md`](RESUME_CONTEXTUAL_COMPOSITION.md) — how to resume the active CC branch
- `current/RESUME_HERE.md`, `current/PROJECT_STATUS.md`, `current/NEXT_STEPS.md` — an older, parallel status-tracking set (Selector v2 era — check dates before trusting over the CC roadmap above)
- `current/SELECTOR_V2.md` — full narrative synthesis of the Selector v2 line of work

## Simulator and architecture reference

- [`simulator_design.md`](simulator_design.md) — core simulator design
- [`policy_library_v2.md`](policy_library_v2.md) — the deployable policy library
- [`selector.md`](selector.md), [`selector_objective_audit.md`](selector_objective_audit.md) — selector design/objective history
- `docs/*_faithful_scheduler_reference.md` (Sarathi, DistServe, Llumnix, SLAI, vLLM chunked-prefill) — literature-fidelity reference notes for internal style/inspired policies

## Full audit archive

`docs/audits/` contains the complete chronological record of every phase
(Phase 2A-2C, CC1-CC5, the baseline-integration and benchmark-suite work
above) — 50+ documents. This index links only the most load-bearing ones;
browse the directory directly for anything not listed above.

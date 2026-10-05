# Documentation Index

Documentation is ordered from current to historical. If two documents
disagree, the one higher in this list wins.

## 1. Current study

1. [`../README.md`](../README.md): public overview.
2. [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md): reproducibility entry point.
3. [`current/README.md`](current/README.md): current-study documentation index
   (evidence reports, artifacts, scope).
4. Study-specific documents linked from that index:
   - [`current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md`](current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md): detailed reproducibility guide.
   - [`../paper/performance_evaluation/README.md`](../paper/performance_evaluation/README.md): manuscript source, build, and checks.
   - [`FRESH_CAUSAL_ROBUSTNESS_REPORT.md`](FRESH_CAUSAL_ROBUSTNESS_REPORT.md) and [`FRESH_CAUSAL_ARTIFACT_CORRECTION.md`](FRESH_CAUSAL_ARTIFACT_CORRECTION.md).
   - [`current/REAL_VLLM_PRESSURE_ACTION_VALIDATION_REPORT.md`](current/REAL_VLLM_PRESSURE_ACTION_VALIDATION_REPORT.md): bounded vLLM probe.
   - [`DATA_RELEASE_POLICY.md`](DATA_RELEASE_POLICY.md): what data is and is not redistributed.

## 2. Repository navigation

- [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md): top-level directories and their roles.
- [`../scripts/README.md`](../scripts/README.md), [`../configs/README.md`](../configs/README.md), [`../data/README.md`](../data/README.md), [`../tools/README.md`](../tools/README.md).
- Design references that still describe the code:
  [`simulator_design.md`](simulator_design.md),
  [`external_baseline_integration.md`](external_baseline_integration.md),
  [`llm_heuristic_dsl.md`](llm_heuristic_dsl.md),
  [`architecture/`](architecture/).
- [`COMPUTE_POLICY.md`](COMPUTE_POLICY.md), [`current/LOCAL_ARTIFACT_RETENTION.md`](current/LOCAL_ARTIFACT_RETENTION.md).

## 3. Historical research and provenance

These documents describe earlier research lines and the state of the project
on their own dates. Many use present tense ("current", "next", "not
started"). The major entry points carry a *Historical document* banner.

- Long-term research-program roadmap (as of 2026-08-19): [`PROJECT_MAP.md`](PROJECT_MAP.md).
- SBS-override selector line (ended 2026-09-19): [`current/RESUME_HERE.md`](current/RESUME_HERE.md),
  [`current/WORK_STATUS.md`](current/WORK_STATUS.md), [`current/NEXT_ACTIONS.md`](current/NEXT_ACTIONS.md).
- Submission-time roadmap of the current study (2026-09-20):
  [`current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md`](current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md).
- External baselines: [`BASELINE_STATUS.md`](BASELINE_STATUS.md), [`baselines.md`](baselines.md),
  [`external_baseline_decision.md`](external_baseline_decision.md).
- Contextual composition and synthesis: [`contextual_composition_roadmap.md`](contextual_composition_roadmap.md),
  [`START_HERE_CONTEXTUAL_COMPOSITION.md`](START_HERE_CONTEXTUAL_COMPOSITION.md),
  [`experiments/cc1_composition_opportunity_spec.md`](experiments/cc1_composition_opportunity_spec.md).
- Withdrawn LLM 2026 manuscript: [`RESULTS_INDEX.md`](RESULTS_INDEX.md), `current/llm2026_*.md`,
  [`../paper/history/llm2026/`](../paper/history/llm2026/).
- Superseded FGCS manuscript: [`current/FGCS_CURRENT_STATUS.md`](current/FGCS_CURRENT_STATUS.md),
  [`current/FGCS_REPRODUCIBILITY_MATRIX_V1.md`](current/FGCS_REPRODUCIBILITY_MATRIX_V1.md),
  [`../paper/history/fgcs/`](../paper/history/fgcs/).
- Older roadmaps and status snapshots: [`INDEX.md`](INDEX.md), [`roadmap.md`](roadmap.md),
  [`research_status.md`](research_status.md), [`current/PROJECT_STATUS.md`](current/PROJECT_STATUS.md),
  [`current/NEXT_STEPS.md`](current/NEXT_STEPS.md), [`current/RESEARCH_ROADMAP.md`](current/RESEARCH_ROADMAP.md),
  [`current/pause_2026_07_25/`](current/pause_2026_07_25/).
- Public-release preparation before the repository was made public (2026-08-24): `PUBLIC_RELEASE_*.md`.
- Dated technical and scientific audits: [`audits/`](audits/). Each one records
  point-in-time evidence.

Generated results under `results/` are local and gitignored. Curated artifacts
are under [`../experiments/`](../experiments/).

# Start Here

This directory is the current source-of-truth navigation layer for
`wulver-selector-v2-and-composition-integrated`, which combines
`origin/wulver-final-integration-20260721`'s Policy Library v2/composition/
structural-synthesis work with the Phase 2C selector-improvement and
leakage-fix work from `phase2c-final-selector-improvement`. Read these
documents in order when picking up the project.

## Project Paused 2026-07-23 — Read This First

The project is paused for potentially several months as of 2026-07-23.
**[RESUME_HERE.md](RESUME_HERE.md)** is the single, concise (5–10 min)
resume entry point — read it first. For full detail, see
**[PROJECT_HANDOFF_2026-07-23.md](PROJECT_HANDOFF_2026-07-23.md)**,
its machine-readable companion
**[project_handoff_state.json](project_handoff_state.json)**, and durable
test/experiment evidence in
**[PAUSE_PROVENANCE_2026-07-23.md](PAUSE_PROVENANCE_2026-07-23.md)**. Read
all four before `PROJECT_STATUS.md` below.

## Current Bottom Line

As of 2026-07-22, the primary bottleneck is **simulator/objective
discriminative power**, not generic dataset volume, selector model choice, or
unrestricted structural synthesis. Recent SwissAI and TraceLab sweeps added
raw workload novelty but collapsed to near-ceiling ANWG and weak policy
separation. The next major development step is bounded simulator calibration
and pressure validation before more selector, module-credit, or combiner
training.

## Core Current Docs

1. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - canonical current scientific state and bottleneck.
2. **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** - staged next plan, starting with simulator calibration.
3. **[ROADMAP_GAP_ANALYSIS.md](ROADMAP_GAP_ANALYSIS.md)** - evidence-ranked bottlenecks, including Phase 2C selector-improvement gaps.
4. **[REPO_ARCHITECTURE_MAP.md](REPO_ARCHITECTURE_MAP.md)** - current code structure, including policy/composition/synthesis modules and the reconciled selector-v2 split-leakage architecture.
5. **[SELECTOR_STATUS.md](SELECTOR_STATUS.md)** - selector/suitability status and caveats.
6. **[POLICY_LIBRARY.md](POLICY_LIBRARY.md)** - 27-policy library inventory, V2 results, and unsupported policy families.
7. **[COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md](COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md)** - implemented composition and structural synthesis architecture.
8. **[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)** - durable index of major experiment roots, jobs, reports, and conclusions.
9. **[POLICY_FRONTIER_STATUS.md](POLICY_FRONTIER_STATUS.md)** - frontier/library workflow summaries.
10. **[ACTIVE_EXPERIMENT_PROTECTED_PATHS.md](ACTIVE_EXPERIMENT_PROTECTED_PATHS.md)** - protected SLURM roots that must not be modified.

## Selector-v2 / Local-Lineage Integration Docs

- **[LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md)** -- how
  `phase2c-final-selector-improvement`'s selector-improvement and
  split-leakage-fix work was reconciled onto this integration branch: what
  was transferred intact, what was manually reconciled (and why), and the
  resulting single source of truth for split grouping. Read this for the
  history of *this specific merge*.
- **[WULVER_HANDOFF.md](WULVER_HANDOFF.md)** -- what is ready to scale on
  Wulver, what remains local-only, expected CPU/GPU job classes, and
  commands to wrap in SLURM later (written from the local-branch side prior
  to this integration; still applicable to the selector-v2 pilot path).

## Operational Audit Docs

- **[WULVER_UNPUSHED_WORK_AUDIT.md](WULVER_UNPUSHED_WORK_AUDIT.md)** - Query 1 audit of local-only work.
- **[WULVER_BRANCH_LINEAGE_AUDIT.md](WULVER_BRANCH_LINEAGE_AUDIT.md)** - Query 1 branch/source-of-truth map.
- **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)** - older operational handoff retained for continuity.
- **[LOCAL_ARTIFACT_CLEANUP.md](LOCAL_ARTIFACT_CLEANUP.md)** - older local artifact classification.

## Detailed Research Docs

- **[POLICY_COMPOSITION_READINESS.md](POLICY_COMPOSITION_READINESS.md)**
- **[COMPOSITION_EXPERIMENT_DESIGN.md](COMPOSITION_EXPERIMENT_DESIGN.md)**
- **[COMPOSITION_IMPLEMENTATION_STATUS.md](COMPOSITION_IMPLEMENTATION_STATUS.md)**
- **[WOLVERINE_ORACLE_MIXTURE_HANDOFF.md](WOLVERINE_ORACLE_MIXTURE_HANDOFF.md)** -- handoff spec for the not-yet-submitted large-scale composition sweep; currently BLOCKED pending read-only artifact recovery on Wulver, not launched.
- **[STRUCTURAL_SYNTHESIS_READINESS.md](STRUCTURAL_SYNTHESIS_READINESS.md)**
- **[structural_synthesis_experiment_design.md](structural_synthesis_experiment_design.md)**

## Machine-Readable Artifacts

- [policy_component_matrix.json](policy_component_matrix.json)
- [composable_primitives.json](composable_primitives.json)
- [composition_operators.json](composition_operators.json)
- [policy_complementarity.json](policy_complementarity.json)
- [composition_experiment_schema.json](composition_experiment_schema.json)
- [composition_hypotheses.json](composition_hypotheses.json)
- [scheduler_genome_v1.schema.json](scheduler_genome_v1.schema.json)
- [wolverine_oracle_mixture_spec.json](wolverine_oracle_mixture_spec.json) -- machine-readable spec for the handoff doc above.

## Historical Docs

`docs/README.md` remains the exhaustive historical documentation index. If older current docs such as `ARCHITECTURE.md`, `BASELINES.md`, `SELECTOR_V2.md`, `EXPERIMENTS_AND_RESULTS.md`, `REPRODUCIBILITY.md`, or `NEXT_STEPS.md` disagree with the files listed above, treat this directory's current docs ([PROJECT_STATUS.md](PROJECT_STATUS.md), [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md), and the cited experiment final reports) as authoritative.

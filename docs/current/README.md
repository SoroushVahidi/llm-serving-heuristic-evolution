# Start Here

This directory is the current source-of-truth navigation layer for
`wulver-selector-v2-and-composition-integrated`, which combines
`origin/wulver-final-integration-20260721`'s Policy Library v2/composition/
structural-synthesis work with the Phase 2C selector-improvement and
leakage-fix work from `phase2c-final-selector-improvement`. Read these
documents in order when picking up the project.

## Core Current Docs

1. **[REPO_ARCHITECTURE_MAP.md](REPO_ARCHITECTURE_MAP.md)** - current code structure, including where the policy/composition/synthesis modules and the reconciled selector-v2 split-leakage architecture live.
2. **[SELECTOR_STATUS.md](SELECTOR_STATUS.md)** - concise Selector v2/v3 status and stop/go interpretation.
3. **[POLICY_LIBRARY.md](POLICY_LIBRARY.md)** - 27-policy library inventory, v2 additions, and unsupported policy families.
4. **[POLICY_FRONTIER_STATUS.md](POLICY_FRONTIER_STATUS.md)** - status of active frontier and expanded-library workflows.
5. **[COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md](COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md)** - implemented composition and structural synthesis architecture.
6. **[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)** - durable index of major experiment roots, jobs, reports, and conclusions.
7. **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** - completed/running/pending work and decision gates.
8. **[ROADMAP_GAP_ANALYSIS.md](ROADMAP_GAP_ANALYSIS.md)** - current bottleneck diagnosis, now including the Phase 2C selector-improvement gap.
9. **[ACTIVE_EXPERIMENT_PROTECTED_PATHS.md](ACTIVE_EXPERIMENT_PROTECTED_PATHS.md)** - active SLURM roots that must not be modified.

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

`docs/README.md` remains the exhaustive historical documentation index. If older current docs such as `PROJECT_STATUS.md`, `ARCHITECTURE.md`, `BASELINES.md`, `SELECTOR_V2.md`, `EXPERIMENTS_AND_RESULTS.md`, `REPRODUCIBILITY.md`, or `NEXT_STEPS.md` disagree with the files listed above, treat the newer Query 2 current docs, [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md), and the cited experiment final reports as authoritative.

# Contextual Compositional Heuristics Development Branch

Branch name: `contextual-compositional-heuristics-20260731`

Base branch: `reality-grounded-dataset-expansion-20260724`

Base commit SHA: `775147beec997b14039bbaa088d17630a32156cf`

Creation date: 2026-07-31

## Objective

This branch is the authoritative development branch for the
contextual-compositional heuristic research path.

The central research objective is to represent scheduling knowledge as reusable,
verifiable heuristic components; estimate component usefulness from workload
context; compose those components into scenario-specific heuristics; compile the
composition into the verified scheduling DSL or genome path; and execute it with
safety checks and robust fallback.

## Continuity

Start from the synchronization-aware audit:

- [Start Here: Contextual Composition](START_HERE_CONTEXTUAL_COMPOSITION.md)
- [Contextual Compositional Heuristics Roadmap](contextual_composition_roadmap.md)
- [Contextual Composition Decision Log](contextual_composition_decisions.md)
- [Local Branch Compositional Path Audit](audits/local_branch_compositional_path_audit_20260731.md)

Current high-level status: Query 3 makes CC1 implementation-ready by adding the
approved [CC1 composition opportunity specification](experiments/cc1_composition_opportunity_spec.md)
and auditing existing composition prototypes. Selector, DSL, generation, and
native composition prototypes exist, but the verified contextual-composition
method and CC1 experiment have not been implemented.

## Six-Query Sequence

1. Query 1: synchronize, preserve the audit, establish this branch, validate,
   commit, and push. COMPLETE.
2. Query 2: establish the persistent roadmap, repository navigation path,
   milestones, and decision gates. COMPLETE.
3. Query 3: specify the CC1 composition opportunity experiment and polish
   continuity. COMPLETE.
4. Query 4: implement the approved CC1 specification without broad refactors.
   NEXT.
5. Query 5: prepare the first implementation slice without launching large
   experiments.
6. Query 6: finalize readiness for implementation and experimental execution.

## Guardrail

Do not implement major architectural changes, primitive rewrites, DSL
extensions, selector redesigns, simulator changes, or new large experiments
before the roadmap and invariant documents are established.

## Next Action

Query 4 should implement the approved CC1 true simulator-executed
composition-opportunity specification and keep CC1 as `NEXT` until full
simulator-executed results are documented.

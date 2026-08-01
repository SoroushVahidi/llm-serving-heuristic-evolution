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

Current high-level status: Query 5 completed the CC1b discriminativeness
review. The original CC1 suite was nondiscriminative, but the strengthened
CC1b suite found a true simulator-executed weighted Borda composition
opportunity and cleared the `PROCEED` gate. CC2 is now the next phase.
The approved CC1 experiment remains documented in
[CC1 composition opportunity specification](experiments/cc1_composition_opportunity_spec.md).

## Six-Query Sequence

1. Query 1: synchronize, preserve the audit, establish this branch, validate,
   commit, and push. COMPLETE.
2. Query 2: establish the persistent roadmap, repository navigation path,
   milestones, and decision gates. COMPLETE.
3. Query 3: specify the CC1 composition opportunity experiment and polish
   continuity. COMPLETE.
4. Query 4: implement the approved CC1 specification without broad refactors.
   COMPLETE.
5. Query 5: diagnose CC1 discriminativeness and run the bounded CC1b follow-up.
   COMPLETE.
6. Query 6: begin CC2 by defining the canonical primitive interface. NEXT.

## Guardrail

Do not implement CC3 DSL extensions, selector redesigns, real-vLLM jobs, hosted
API experiments, or large ungated sweeps before the roadmap allows them. CC2 is
limited to the canonical primitive interface and representative-policy
equivalence evidence.

## Next Action

Query 6 should define the CC2 canonical primitive interface and equivalence
tests needed to reproduce representative policies from reusable components. Do
not begin CC3 DSL work in Query 6.

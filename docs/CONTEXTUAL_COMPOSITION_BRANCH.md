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
- [Pause Checkpoint](audits/contextual_composition_pause_checkpoint_20260731.md)
- [Resume Guide](RESUME_CONTEXTUAL_COMPOSITION.md)
- [Final Pause-Readiness Report](audits/contextual_composition_query7_final_pause_readiness_20260731.md)
- [Architecture: CC2 Canonical Scheduling Primitive Interface](architecture/contextual_composition_primitives.md)
- [CC2 Primitive Interface Report](audits/contextual_composition_cc2_primitive_interface_report_20260802.md)

Current high-level status: Query 8 implemented CC2 (the canonical scheduling
primitive interface) and its equivalence gate passed: six of seven
representative-policy reconstructions are EXACT and one
(`scorpio_style_slo_guard`) is documented APPROXIMATE. CC2 is COMPLETE; CC3
(compositional DSL and verifier) is the single `NEXT` phase. Query 5
completed the CC1b discriminativeness review. The original CC1 suite was
nondiscriminative, but the strengthened CC1b suite found a true
simulator-executed weighted Borda composition opportunity and cleared the
`PROCEED` gate. The approved CC1 experiment remains documented in
[CC1 composition opportunity specification](experiments/cc1_composition_opportunity_spec.md).

## Query Sequence

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
6. Query 6: create the pause checkpoint and operational resume guide. COMPLETE.
7. Query 7: perform final polish and resume-readiness verification without
   implementing CC2. COMPLETE.
8. Query 8: implement the CC2 canonical scheduling primitive interface and
   representative-policy equivalence tests. COMPLETE.

## Guardrail

Do not implement CC3 DSL extensions, CC4 dataset generation, CC5 predictor
training, selector redesigns, real-vLLM jobs, hosted API experiments, or
large ungated sweeps before the roadmap allows them. CC3 is limited to
extending the JSON DSL/verifier over the CC2 primitive registry.

## Next Action

Begin CC3 by extending the JSON DSL/verifier to expose named references to
the CC2 primitive registry (`src/llmserveopt/policies/primitives.py`),
weighted sums, sparse top-k mixtures, conditional branches, admission gates,
and placement scores, per the roadmap's CC3 required constructs.

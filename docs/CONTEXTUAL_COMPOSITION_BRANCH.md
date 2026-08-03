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
- [Architecture: CC3 Compositional DSL](architecture/contextual_composition_dsl.md)
- [CC3 DSL/Verifier Report](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md)
- [CC4 Oracle Dataset Report](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md)
- [CC5 Predictor Report](audits/contextual_composition_cc5_predictor_report_20260803.md)

Current high-level status: Query 11 implemented and attempted CC5 (the
deployable contextual composition predictor) against CC4's oracle dataset.
The pipeline itself is complete and tested (22 new tests), but the exit
gate did **not** pass: verdict `INCONCLUSIVE`. The trained predictor (KNN
regret regression + OOD-gated fallback) ties the best fixed policy on CC4's
6 held-out evaluation windows (mean ANWG 0.2306 vs 0.2310) and is beaten by
the single best global composition (0.2633) -- judged a data-scarcity
finding (n=6 evaluation windows cannot statistically distinguish these
methods at any interesting effect size), not a methodology failure. CC6 is
**not** queued as a result; CC5 remains the roadmap's `NEXT` phase, with an
exact remaining task (expand the CC4 dataset, then retry) recorded in the
CC5 report. Query 10 implemented CC4 (the true
simulator-executed oracle composition dataset over the CC2/CC3
primitive-composition surface) and its exit gate passed: 12 workload
windows across all required regime categories, 34 verified candidates (0
rejected), 408 simulator executions, reproducible (byte-identical verdict
across an independent from-scratch re-run) and resumable (verified via an
interrupt-and-resume cycle plus an automated integration test). A
composition-family candidate is the oracle winner on 66.7% of held-out
evaluation windows; completion-fraction constraints hold on every window.
Query 9
implemented CC3 (the compositional DSL and verifier extension over the CC2
primitive registry) and its exit gate passed: all 8 required constructs
implemented, 447 focused+regression tests pass, and every pre-CC3 example
and genome-derived heuristic remains backward compatible. Query 8
implemented CC2 (the canonical scheduling primitive
interface) and its equivalence gate passed: six of seven
representative-policy reconstructions are EXACT and one
(`scorpio_style_slo_guard`) is documented APPROXIMATE. Query 5
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
9. Query 9: implement the CC3 compositional DSL/verifier extension over the
   CC2 primitive registry. COMPLETE.
10. Query 10: build the CC4 true simulator-executed oracle composition
    dataset. COMPLETE.
11. Query 11: implement and attempt CC5 (contextual composition predictor).
    Pipeline COMPLETE; decision gate NOT PASSED (verdict `INCONCLUSIVE`).

## Guardrail

Do not retry CC5 predictor training, implement CC6 adaptation, selector
redesigns, real-vLLM jobs, hosted API experiments, or large ungated sweeps
before the roadmap allows them. CC5's exit gate did not pass (see the CC5
predictor report); it must not be retried without a separate, explicitly
authorized query, and CC6 must not begin until CC5 actually passes.

## Next Action

CC5 (contextual composition predictor) remains `NEXT` -- attempted, not
passed. A future, explicitly authorized query should first expand the CC4
dataset (more windows, more per regime -- see
`docs/audits/contextual_composition_cc5_predictor_report_20260803.md`
section 10 for the exact rationale), then retrain against the larger
dataset using the existing, tested CC5 pipeline
(`src/llmserveopt/experiments/cc5_contextual_predictor.py`) -- no code
changes are anticipated to be required.

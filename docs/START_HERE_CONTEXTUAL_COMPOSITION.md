# Start Here: Contextual Composition

Authoritative branch: `contextual-compositional-heuristics-20260731`

Current phase: `CC3 - Compositional DSL and verifier`

Pause state: none. CC2 is COMPLETE; CC3 is `NEXT`.

CC2 implemented the canonical primitive interface
(`src/llmserveopt/policies/primitives.py`, 28 registered primitives across
RANKING/ADMISSION/PLACEMENT/BATCHING/RESOURCE_GUARD) and reconstructed
seven representative policies from it
(`src/llmserveopt/policies/primitive_reconstructions.py`), with 107 new
equivalence/registry tests. Six of seven reconstructions are EXACT; one
(`scorpio_style_slo_guard`) is documented APPROXIMATE. See the
[architecture doc](architecture/contextual_composition_primitives.md) and the
[CC2 primitive interface report](audits/contextual_composition_cc2_primitive_interface_report_20260802.md).

Exact next task: begin CC3 by extending the JSON DSL/verifier
(`src/llmserveopt/heuristics/`) to expose named primitive references,
weighted sums, sparse top-k mixtures, conditional branches, admission
gates, and placement scores over the CC2 primitive registry, per the
roadmap's CC3 required-constructs list. Do not begin CC4 dataset work.

## Read In This Order

1. `docs/START_HERE_CONTEXTUAL_COMPOSITION.md`
2. `docs/contextual_composition_roadmap.md`
3. `docs/contextual_composition_decisions.md`
4. `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`
5. `docs/experiments/cc1_composition_opportunity_spec.md`
6. `docs/audits/local_branch_compositional_path_audit_20260731.md`
7. `docs/audits/contextual_composition_query2_roadmap_report_20260731.md`
8. `docs/audits/contextual_composition_query3_cc1_spec_report_20260731.md`
9. `docs/audits/contextual_composition_query4_cc1_results_20260731.md`
10. `docs/audits/contextual_composition_query5_discriminativeness_review_20260731.md`
11. `docs/audits/contextual_composition_pause_checkpoint_20260731.md`
12. `docs/RESUME_CONTEXTUAL_COMPOSITION.md`
13. `docs/audits/contextual_composition_query6_pause_report_20260731.md`
14. `docs/audits/contextual_composition_query7_final_pause_readiness_20260731.md`
15. `docs/architecture/contextual_composition_primitives.md`
16. `docs/audits/contextual_composition_cc2_primitive_interface_report_20260802.md`
    or the latest later contextual-composition audit report

## What Not To Do Yet

CC2's primitive interface and representative-policy reconstructions are
complete; CC3 (JSON DSL/verifier extension over that primitive registry) is
now in scope. Do not begin CC4 offline oracle dataset generation,
contextual predictor training (CC5), dynamic adaptation (CC6),
counterexample hardening (CC7), real-vLLM jobs, hosted API experiments, or
large simulator sweeps before the roadmap phase gate allows them.

## How To Update Status

Every future implementation query must:

- verify branch and upstream;
- read the roadmap;
- identify the current `NEXT` phase;
- update roadmap status, evidence links, decisions, risks, and next action;
- add or update an audit report;
- run relevant tests;
- commit and push;
- leave the branch clean and synchronized.

Run the consistency checker before committing:

```bash
python scripts/check_contextual_composition_status.py
python scripts/check_contextual_composition_status.py --resume-readiness
```

## Historical Status And Safe Claims

Historical project status lives in `docs/current/` and the full historical index
is `docs/README.md`. Those documents remain provenance but may describe older
branches or phases.

Safe claim guidance lives in `docs/result_claims.md`. Contextual-composition
claims must distinguish historical simulator claims, corrected-objective claims,
real-trace-derived claims, hosted-API claims, and real-vLLM claims.

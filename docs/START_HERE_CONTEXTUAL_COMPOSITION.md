# Start Here: Contextual Composition

Authoritative branch: `contextual-compositional-heuristics-20260731`

Current phase: `CC5 - Contextual composition predictor`

Pause state: none. CC4 is COMPLETE. CC5 remains `IN PROGRESS` after three
evidence stages: first attempt `INCONCLUSIVE` (n=6); CC4b retry
`REGIME_SPECIFIC_ONLY` (n=76); uncertainty/regime refinement also
`REGIME_SPECIFIC_ONLY` (hybrid ANWG 0.4019 vs global 0.4025). CC6 is
**BLOCKED**. Read the
[CC5 uncertainty/regime report](audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md)
for current status (also: [retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md),
[first CC5 report](audits/contextual_composition_cc5_predictor_report_20260803.md)).

## Active Task Right Now (read this section first)

The CC5 uncertainty/regime refinement is **complete** with verdict
`REGIME_SPECIFIC_ONLY`. Model-agnostic `normalized_split_conformal`
uncertainty now works for gradient boosting; best completion-safe hybrid
system reaches ANWG **0.4019** (beats fixed 0.3895 and hard selector 0.3938;
0 completion violations) but remains **0.0006 short** of
`best_global_composition` (0.4025). Pure best-global fallback fails
completion (7 violations) and is rejected. Restricted envelope: trust the
predictor on `kv_pressure` / `saturated`; use validation-tuned hybrid
fallback elsewhere. Reference artifacts:

```bash
tmux attach -t cc5_uncertainty_regime
# refinement: results/cc5_uncertainty_regime_refinement/20260803T202108Z/
# CC4b dataset: results/cc4b_oracle_composition_expansion/20260803T182426Z/
# CC5 retry:    results/cc5_contextual_composition_predictor_retry/20260803T192246Z/
# session log:  logs/cc5_uncertainty_regime_20260803_195020.log
```

**Do not begin CC6.** Exact next decision: either freeze the restricted
operating envelope above as accepted CC5 scope, or run a narrow
regime-specialist follow-up on the six global-win regimes only. Active
issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5)
(remains open). Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6)
stays blocked/not ready.

CC5 trains a regret-regression predictor over CC4/CC4b's already-verified
candidate pool with leave-one-window-out model selection and calibrated
uncertainty-aware abstention/fallback
(`src/llmserveopt/experiments/cc5_contextual_predictor.py`,
`cc5_uncertainty_regime_refinement.py`).

CC4 built the first reproducible, resumable, simulator-derived oracle
composition dataset (`src/llmserveopt/experiments/cc4_oracle_composition_dataset.py`,
`configs/cc4_oracle_composition_dataset.yaml`): 12 workload windows across
all required regime categories, 34 verified DSL/fixed candidates (0
rejected), 408 true simulator executions, 0 GPU/live-API/real-vLLM. A
composition-family candidate is the oracle winner on 66.7% of held-out
evaluation windows; completion-fraction constraints hold on every window.
20 new focused tests pass, including a real (non-mocked) resume/reproducibility
integration test. See
[CC4 oracle dataset report](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md).

CC3 extended the JSON DSL/verifier (`src/llmserveopt/heuristics/`) to expose
named references to the CC2 primitive registry: weighted sums, sparse top-k
mixtures, conditional branches, admission gates with declared fallback,
placement-score composition, and externally supplied bounded parameters,
via a new read-only adapter module (`heuristics/primitive_bridge.py`). All
447 focused+regression tests pass and every pre-CC3 example and
genome-derived heuristic remains backward compatible. See the
[architecture doc](architecture/contextual_composition_dsl.md) and the
[CC3 DSL/verifier report](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md).

CC2 implemented the canonical primitive interface
(`src/llmserveopt/policies/primitives.py`, 28 registered primitives across
RANKING/ADMISSION/PLACEMENT/BATCHING/RESOURCE_GUARD) and reconstructed
seven representative policies from it
(`src/llmserveopt/policies/primitive_reconstructions.py`), with 107 new
equivalence/registry tests. Six of seven reconstructions are EXACT; one
(`scorpio_style_slo_guard`) is documented APPROXIMATE. See the
[architecture doc](architecture/contextual_composition_primitives.md) and the
[CC2 primitive interface report](audits/contextual_composition_cc2_primitive_interface_report_20260802.md).

Exact next task: see "Active Task Right Now" above. In summary: freeze the
restricted envelope or run a narrow global-win-regime follow-up. Do not
begin CC6 until CC5's exit gate fully passes.

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
17. `docs/architecture/contextual_composition_dsl.md`
18. `docs/audits/contextual_composition_cc3_dsl_verifier_report_20260803.md`
19. `docs/audits/contextual_composition_cc4_oracle_dataset_report_20260803.md`
20. `docs/audits/contextual_composition_cc5_predictor_report_20260803.md`
21. `docs/audits/contextual_composition_cc4b_cc5_retry_report_20260803.md`
    or the latest later contextual-composition audit report

## What Not To Do Yet

CC2's primitive interface, representative-policy reconstructions, CC3's
compositional DSL/verifier extension, and CC4's oracle composition dataset
are all complete. CC5's first attempt (`INCONCLUSIVE`) and its retry
(`REGIME_SPECIFIC_ONLY`) have both completed without fully passing the
exit gate; do not start a third retry before addressing the exact next
research step in the retry report, and do not begin dynamic adaptation
(CC6) until CC5 actually passes. Do not begin counterexample hardening
(CC7), real-vLLM jobs, hosted API experiments, evolutionary/QD
library-expansion work, or LLM-guided synthesis work (all future
directions, none implemented -- see the roadmap's "Future Research
Directions" section) before the roadmap phase gate allows them.

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

# Start Here: Contextual Composition

Authoritative branch: `contextual-compositional-heuristics-20260731`

Current phase: `CC5 - Contextual composition predictor`

Pause state: none. CC4 is COMPLETE. CC5's first attempt returned verdict
`INCONCLUSIVE` (exit gate NOT passed, not a negative result -- see below);
CC5 is `IN PROGRESS` via an active retry. CC6 is **BLOCKED** until the retry
verdict lands. See the
[CC5 predictor report](audits/contextual_composition_cc5_predictor_report_20260803.md)
(first attempt) and the
[CC4b/CC5 retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)
(current status -- read this one first if it exists).

## Active Task Right Now (read this section first)

A targeted oracle-dataset expansion (CC4b) is growing CC4's held-out set
from 6 to 50-100+ windows, after which CC5's existing, unchanged pipeline
reruns against it -- because the first CC5 attempt was statistically
inconclusive at n=6 held-out windows (`best_global_composition` beat the
trained predictor, but the gap was well within noise). Exact commands used:

```bash
tmux attach -t cc4b_cc5_retry
# build config:      configs/cc4b_oracle_composition_expansion.yaml
# build script:       scripts/run_cc4_oracle_composition_dataset.py --config configs/cc4b_oracle_composition_expansion.yaml --full-run --allow-dirty --timestamp <TS>
# checkpoints/log:     results/cc4b_oracle_composition_expansion/<TS>/checkpoints/{heartbeat.json,trial_results.jsonl}
# session log:         logs/cc4b_cc5_retry_YYYYMMDD_HHMMSS.log
# quality gates:       scripts/check_cc4b_quality_gates.py results/cc4b_oracle_composition_expansion/<TS>
# CC5 rerun:            scripts/run_cc5_contextual_predictor.py --dataset-dir results/cc4b_oracle_composition_expansion/<TS> --full-run --timestamp <TS>
```

**Do not start a second CC4b build, a second CC5 training run, or any CC6
work in parallel with this.** Check `results/cc4b_oracle_composition_expansion/*/checkpoints/heartbeat.json`
for the latest progress before starting anything else. Exact next decision
once the run completes: check the quality gates, then interpret the CC5
retry verdict (`PROCEED` / `REGIME_SPECIFIC_ONLY` / `STOP_OR_REDESIGN` /
`INCONCLUSIVE`) before any model redesign or CC6 work -- see the
[CC4b/CC5 retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)
for the actual outcome. Active issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).

CC5 trained a KNN regret-regression predictor over CC4's already-verified,
already-executed candidate pool (`src/llmserveopt/experiments/cc5_contextual_predictor.py`),
with leave-one-window-out model selection, OOD-gated abstention/fallback,
and a full evaluation against CC4's 6 held-out windows. Result: the
predictor ties the best fixed policy (mean ANWG 0.2306 vs 0.2310) and is
beaten by the single best global composition (0.2633) -- judged a
data-scarcity finding (n=6 held-out windows cannot statistically
distinguish these methods), not a methodology failure. 22 new focused tests
pass. Completion-fraction constraints hold (0 violations).

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

Exact next task: see "Active Task Right Now" above. In summary: interpret
the CC4b/CC5 retry verdict once the current run completes, before any model
redesign or CC6 work. Do not begin CC6 dynamic adaptation until CC5 passes.

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
are all complete. CC5's first attempt did not pass its exit gate
(`INCONCLUSIVE`) and a retry (CC4b expansion + unchanged CC5 rerun) is
underway; do not start a second, parallel retry, and do not begin dynamic
adaptation (CC6) until CC5 actually passes. Do not begin counterexample
hardening (CC7), real-vLLM jobs, hosted API experiments, evolutionary/QD
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

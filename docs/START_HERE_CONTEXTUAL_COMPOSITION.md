# Start Here: Contextual Composition

> This is the detailed CC-roadmap technical status document. The overall
> repository entry point (Wulver/local task split, baseline status,
> guardrails, exact next actions) is
> **[docs/current/RESUME_HERE.md](current/RESUME_HERE.md)** — read that
> first if you haven't already.

Authoritative branch: `contextual-compositional-heuristics-20260731`

Current phase: `CC6 - Dynamic adaptation and stability`

Pause state: none. CC5 is **COMPLETE** (`COMPLETE_REGIME_SPECIFIC`) after
four evidence stages: first attempt `INCONCLUSIVE` (n=6); CC4b retry
`REGIME_SPECIFIC_ONLY` (n=76); uncertainty/regime refinement also
`REGIME_SPECIFIC_ONLY` (hybrid ANWG 0.4019 vs global 0.4025); final
operating-envelope finalization `COMPLETE_REGIME_SPECIFIC` (frozen system
ANWG 0.4044, statistically beats fixed and hard selector, ties global).
CC6 is **NEXT, but restricted** to the CC5 trusted envelope. Read the
[CC5 final operating envelope report](audits/contextual_composition_cc5_final_operating_envelope_20260803.md)
for current status (also: [uncertainty/regime report](audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md),
[retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md),
[first CC5 report](audits/contextual_composition_cc5_predictor_report_20260803.md)).

## Active Task Right Now (read this section first)

CC5 is **finalized and CLOSED** with verdict `COMPLETE_REGIME_SPECIFIC`.
A frozen, deterministic, versioned operating-envelope gate (derived
entirely from development-split LOWO evidence, never touching held-out
data) trusts the contextual predictor in 7 of 12 regimes --
`burst_transition`, `kv_pressure`, `long_output`, `prediction_noise`,
`saturated`, `selective_admission_trap`, `underloaded` -- and falls back
to a validation-tuned completion-safe choice (best-global or best-fixed)
elsewhere. Evaluated once on 76 held-out windows: frozen system ANWG
**0.4044**, statistically beating best fixed (paired 95% CI
[+0.0074, +0.0235], p<0.0001) and the hard selector (paired 95% CI
[+0.0020, +0.0199], p=0.021), 0 completion violations. Its point-estimate
edge over `best_global_composition` (+0.0019 ANWG) is **not**
statistically distinguishable from zero (paired 95% CI [-0.0044, +0.0083],
p=0.5654) -- full-context superiority over global composition was not
established; this is documented honestly, not hidden. Reference artifacts:

```bash
tmux attach -t cc5_uncertainty_regime
# final envelope: results/cc5_final_operating_envelope/20260804T024524Z/
# refinement:     results/cc5_uncertainty_regime_refinement/20260803T202108Z/
# CC4b dataset:   results/cc4b_oracle_composition_expansion/20260803T182426Z/
# CC5 retry:      results/cc5_contextual_composition_predictor_retry/20260803T192246Z/
# session log:    logs/cc5_uncertainty_regime_finalization_20260804_024058.log
```

**CC6 is queued but NOT implemented in this query, and is restricted in
scope.** Exact next task: evaluate controlled temporal adaptation only
inside the CC5 trusted envelope above, with hysteresis and fallback; do
not enable contextual switching in unsupported regimes
(`azure_conversation_like`, `burstgpt_derived`, `long_prompt`, `mixed_slo`,
`priority_conflict`). Do not begin CC6 implementation until a future query
explicitly authorizes it. Issue
[#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5)
is **closed**. Issue
[#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6)
is ready, restricted scope, **not started**.

CC5 trains a regret-regression predictor over CC4/CC4b's already-verified
candidate pool with leave-one-window-out model selection and calibrated
uncertainty-aware abstention/fallback
(`src/llmserveopt/experiments/cc5_contextual_predictor.py`,
`cc5_uncertainty_regime_refinement.py`), finalized by a frozen,
development-evidence-only operating envelope plus paired statistical
analysis (`cc5_final_operating_envelope.py`).

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

Exact next task: see "Active Task Right Now" above. In summary: CC5 is
closed `COMPLETE_REGIME_SPECIFIC`; CC6 is queued but restricted to the
frozen envelope. Do not begin CC6 implementation in this query.

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
22. `docs/audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md`
23. `docs/audits/contextual_composition_cc5_final_operating_envelope_20260803.md`
    or the latest later contextual-composition audit report

## What Not To Do Yet

CC2's primitive interface, representative-policy reconstructions, CC3's
compositional DSL/verifier extension, CC4's oracle composition dataset, and
CC5's contextual composition predictor (finalized `COMPLETE_REGIME_SPECIFIC`)
are all complete. CC6 (dynamic adaptation) is now queued, but **restricted**
to the CC5 trusted envelope (`burst_transition`, `kv_pressure`,
`long_output`, `prediction_noise`, `saturated`, `selective_admission_trap`,
`underloaded`) -- do not enable contextual switching in unsupported
regimes, and do not begin CC6 implementation itself until a future query
explicitly authorizes it. Do not begin counterexample hardening
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

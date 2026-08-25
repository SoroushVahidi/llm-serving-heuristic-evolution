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
- [CC4b/CC5 Retry Report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)
- [CC5 Uncertainty/Regime Report](audits/contextual_composition_cc5_uncertainty_regime_report_20260803.md)
- [CC5 Final Operating Envelope Report](audits/contextual_composition_cc5_final_operating_envelope_20260803.md) -- **current status**

Current high-level status: Query 14 finalized CC5 with a frozen,
development-evidence-only operating envelope and a paired statistical
analysis. Verdict `COMPLETE_REGIME_SPECIFIC`: the frozen system (ANWG
0.4044) statistically beats best fixed (paired 95% CI [+0.0074,+0.0235],
p<0.0001) and the hard selector (paired 95% CI [+0.0020,+0.0199],
p=0.021), 0 completion violations, but its edge over best global
composition (+0.0019 ANWG) is **not** statistically distinguishable from
zero (paired 95% CI [-0.0044,+0.0083], p=0.5654). Trusted envelope (7 of
12 regimes, dev-LOWO-derived, no held-out tuning): `burst_transition`,
`kv_pressure`, `long_output`, `prediction_noise`, `saturated`,
`selective_admission_trap`, `underloaded`. **CC5 is now COMPLETE**; CC6 is
queued but **restricted** to this envelope, not yet implemented. Query 13
completed the CC5 uncertainty/regime
refinement (`normalized_split_conformal` + completion-safe hybrid fallback).
Verdict was `REGIME_SPECIFIC_ONLY`: hybrid ANWG 0.4019 beats fixed
0.3895 and hard selector 0.3938 with 0 completion violations, but stayed
0.0006 short of best global composition 0.4025 under independent-CI
comparison. Query 12 completed the CC4b expansion + unchanged CC5 retry
(`REGIME_SPECIFIC_ONLY` at n=76). Query 11 implemented and attempted CC5 (the
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
12. Query 12: targeted CC4b oracle-dataset expansion and unchanged CC5
    rerun. COMPLETE -- verdict `REGIME_SPECIFIC_ONLY`, exit gate not fully
    passed. See the CC4b/CC5 retry report for full evidence.
13. Query 13: CC5 uncertainty/regime refinement. COMPLETE -- verdict
    `REGIME_SPECIFIC_ONLY`; exit gate still not fully passed. See the
    uncertainty/regime report.
14. Query 14: CC5 finalization -- paired statistical analysis and frozen
    regime-specific operating envelope. COMPLETE -- verdict
    `COMPLETE_REGIME_SPECIFIC`. CC5 exit gate now PASSED (regime-specific
    scope). See the final operating envelope report.

## Guardrail

Do not begin CC6 implementation until a future query explicitly
authorizes it, and when it does, keep CC6 restricted to the CC5 trusted
envelope (`burst_transition`, `kv_pressure`, `long_output`,
`prediction_noise`, `saturated`, `selective_admission_trap`,
`underloaded`) -- do not enable contextual switching in unsupported
regimes. Do not implement selector redesigns, real-vLLM jobs, hosted API
experiments, evolutionary/QD library-expansion work, LLM-guided synthesis
work, or large ungated sweeps before the roadmap allows them -- see the
roadmap's "Future Research Directions -- Not Yet Implemented" section for
what remains future work, not current capability.

## Next Action

Per `docs/audits/contextual_composition_cc5_final_operating_envelope_20260803.md`:
CC5 is finalized `COMPLETE_REGIME_SPECIFIC`. CC6 is now the single `NEXT`
phase, restricted to the CC5 trusted envelope above, with hysteresis and
fallback; do not enable contextual switching in unsupported regimes
(`azure_conversation_like`, `burstgpt_derived`, `long_prompt`, `mixed_slo`,
`priority_conflict`). Do not begin CC6 implementation in this query.

Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) (ready, restricted scope, not started). Issue [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5) is now closed.

## Family B v2 — PrefillControl Composition Falsification

**Preregistered audit (FAMILY_B_COMPOSITION_READY):** `docs/audits/policy_separation_prefill_decode_pilot_v2_20260817.md`
**Parents:** `full_prefill` (unlimited chunk) vs `chunked_prefill_small` (chunk=64)
**Design doc:** `configs/prefill_control_composition_v2.yaml` + this section

### Implementation (current session)

| File | Purpose |
|---|---|
| `p3_chunk_control.py` | Composition logic: Family B v2 template imports, feature extraction (13 online-observable, 0 forbidden), composition operator, endpoint identity |
| `p7_runner.py` | End-to-end runner with `ScenarioBatch` / `ChildCompositionConfig` abstractions, deterministic eval-ID, split-aware parent + child evaluation |
| `p5_analysis_chunk_comp.py` | Preregistered analysis: integrity, split validation, envelope gain, bootstrap CI, paired deltas, coverage diagnostics, verdict |
| `p8_test_runner.py` | 51 focused tests: scenario-batch / config, eval-ID determinism, leakage protection, split integrity, verdict logic, all three verdict paths |
| `p2_config.yaml` | Configuration: parents, children, split spec, feature forbid/allow list, statistical settings, provenance fields |

**Bug fix in this session:** `prefill_control_splits.py` — line 107 tried to unpack plain strings as tuples in the degenerate split case; fixed with type guard.

### Bug fixes + pipeline completion (continuation session, 2026-08-17)

- **OOD=0 launch blocker (root cause, category C — implementation bug):**
  `p7_runner.py`'s `build_scenarios_from_config` was passing
  `runner.max_active_sequences` (a simulator capacity setting) as BOTH
  `n_hog` and `n_late` to `case_prefill_decode_ttft_contention`, overriding
  the intended `HOG_COUNT[hog_count]` / `LATE_PRESSURE[late_pressure]`
  derivation. This collapsed the `hog_count`/`late_pressure` sweep factors
  to a single fixed value (`hog512.late512` regardless of grid label) and
  broke the OOD split predicate (`"late40" in scenario_id`), which never
  matched. Fixed by removing the override; `n_hog`/`n_late` now derive from
  the grid labels as designed. Full 32-scenario grid now produces the
  preregistered train=16/val=8/test=4/ood=4 split. Two test-helper files
  (`p8_test_runner.py`, `tests/test_prefill_control_composition_v2.py`) had
  the same masking bug in their `_make_scenario` fixtures (hardcoded
  `n_hog=6, n_late=6`) and were fixed the same way, with new regression
  tests proving OOD is non-empty, disjoint, and exactly matches the
  preregistered split sizes.
- **Smoke-check `p3` NameError:** an ad hoc validation snippet referenced
  `p3.PRIMARY` without importing `p3_chunk_control as p3` in that scope.
  Replaced with `scripts/smoke_prefill_control_composition_v2.py`, a
  reusable launch-gate script that imports `PRIMARY` directly.
- **Pipeline completion — dynamic composition + real selector wiring:**
  `p7_runner.py` previously only evaluated the two parents and the three
  fixed-intermediate chunks; `prefill_control_child` (the actual
  falsification target) was never simulated, and `p5_analysis_chunk_comp.py`
  compared against a hindsight-oracle placeholder instead of a genuinely
  fitted selector. Closed both gaps:
  - `Action.prefill_chunk_override` (new, sixth narrowly-scoped Action verb,
    same opt-in pattern as `hold_decode`/`swap`/`migrate`): lets a policy
    set `ServiceModel.max_prefill_chunk_tokens` per-GPU, per-step, without
    mutating the frozen `ServiceModel`. Threaded through
    `GPUState.step`/`_step_phase15`/`_advance_decode_protected`/
    `_advance_shared_contention` and `Simulator._advance_decode`. Defaults
    to empty — zero behavior change for every pre-existing policy (see 456
    passing simulator/gpu/action/contention regression tests).
  - `PrefillControlChildPolicy.select_action` now makes a genuine per-step
    decision via `default_step_level_chunk_rule` (pre-specified, not
    data-fit — same H3/decode-protection mechanism theory as
    `hard_conditional_rule`) and attaches it as `prefill_chunk_override`.
    Wired into `p7_runner.py` Step 4b, evaluated on test+ood like the fixed
    intermediates.
  - `p7_runner.py` Step 4c fits a real top-1 selector + alpha model on
    TRAIN, picks model type on VAL (`select_prefill_model_on_val`), and
    writes `contextual_top1`/`hard_conditional`/`contextual_alpha` rows for
    test+ood computed analytically from already-simulated parent scores
    (per p2_config.yaml: "composite baselines computed at analysis time,
    not re-run") — TEST/OOD never enter fitting or model selection.
  - `p5_analysis_chunk_comp.py`'s `analyse()` now uses real `contextual_top1`
    rows when present (falls back to the hindsight oracle only for older
    result sets that predate this wiring). Also fixed a latent bug in the
    oracle-fallback path itself (`get_oracle_scores` was looking up
    `full_scores` by policy-name string instead of `scenario_id`).

### Preregistered verdict criteria

| Verdict | Conditions |
|---|---|
| `COMPOSITION_GO` | TEST envelope gain > ε=0.01, bootstrap CI lo > 0, adequate samples (test≥4, ood≥2), composition beats selector |
| `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | Contextual top-1 selector already matches parent envelope (gap < 0.005), no composition gain beyond selection |
| `INCONCLUSIVE` | Insufficient evidence (too few scenarios), ambiguous results, or envelope gain ≤ 0 |

### Current status

- **Composition falsification COMPLETE.** Run: `experiments/prefill_control_composition_v2_20260817T154633Z/` (32 scenarios, train=16/val=8/test=4/ood=4, 120/120 success, 0 failed). Analysis: `docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`.
- **Verdict: `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.** The real TRAIN/VAL-fitted contextual top-1 selector reaches the two-parent oracle envelope exactly (0 regret) on both TEST and OOD. `prefill_control_child` — independently verified to make genuine, scenario-varying, multi-valued per-step chunk decisions and never collapse to a fixed baseline — never beats the fitted selector and never expands the oracle envelope on held-out data. See the audit's mechanism section for why (the tested rule only ever selects 3 of its 6 configured chunk options on this workload).
- Broad synthesis/QD work (MAP-Elites, GP, LLM-guided synthesis, symbolic distillation) remains **gated** — not justified by this result, same as ESTF/WFS.
- All tests pass (120 total in `p8_test_runner.py` + `tests/test_prefill_control_composition_v2.py`; 456 simulator/gpu/action/contention regression tests unaffected; full project suite 3869 passed / 62 skipped).

### Launch gate

- ✅ `p3_chunk_control.py` uses `templates_prefill_decode_v2` (Family B v2), not Family A v2
- ✅ No generator-label leakage into features (35-column forbidlist verified)
- ✅ Family B v2 classes: `tenant_prefill` / `tenant_late` (no `.hog` suffix assumptions)
- ✅ Composition endpoints exactly reproduce parents (chunk=64 → `chunked_prefill_small`, chunk=65536 → `full_prefill`), including the dynamic `prefill_chunk_override` path (forced single-chunk grid reproduces the fixed-chunk endpoint exactly)
- ✅ Split integrity: disjoint, covered, held-out seed=20260823, OOD non-empty (train=16/val=8/test=4/ood=4 on the full grid)
- ✅ Deterministic eval IDs: sha256(scenario_id|method|config_hash)[:16]
- ✅ Canonical metric: `arrival_normalized_weighted_goodput`
- ✅ TEST/OOD never enter selector fitting or model selection (guarded by a dedicated test asserting the fitting functions' signatures carry no OOD parameter)
- ✅ `prefill_control_child` (the falsification target) and the real fitted top-1/hard-conditional/alpha selector are now genuinely wired into the runner and analysis, not a hindsight-oracle placeholder
- ✅ All 120 focused composition tests + 456 simulator/gpu/action regression tests pass

> **Verdict: `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.** See `docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md` for the full scientific analysis.

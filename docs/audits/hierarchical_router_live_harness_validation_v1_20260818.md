# Hierarchical Regime Router v1 — Live Closed-Loop Evaluation Harness: Design, Implementation, Validation

Date: 2026-08-18

**Scope: HARNESS DESIGN + IMPLEMENTATION + VALIDATION ONLY. No new scientific
TEST verdict is computed or implied by anything in this document.**

**Harness readiness verdict: `LIVE_HIERARCHICAL_HARNESS_READY`**

This task follows the first held-out TEST evaluation
([`hierarchical_regime_router_v1_20260818.md`](hierarchical_regime_router_v1_20260818.md),
verdict `HIERARCHICAL_ROUTER_NO_GO`), which traced its own negative result
to a specific, previously-documented limitation of its own evaluation
methodology, not to Stage-1 or Stage-2 competence. That prior verdict is
**preserved exactly as written** — nothing here modifies it, the frozen
router implementation, the frozen Stage-1/Stage-2 models, the frozen
feature definitions, `dwell=20`, the frozen fallback logic, the frozen gate
thresholds, or the TEST split. This task builds and validates a genuine
per-step live-simulation evaluator, as that prior audit's own §13 named as
the exact next step — and stops there.

## 1. Initial git state

| | |
|---|---|
| HEAD at start of this task | `21bfff165d5d24ed85683d05f90e33b1f36a7785` (analysis: evaluate hierarchical regime router v1) |
| Working tree | clean |
| Branch | `contextual-compositional-heuristics-20260731` |

## 2. Current majority-vote path — diagnosis

**A. Current dataflow.** `hierarchical_router_evaluation_v1.py`'s own
module docstring states it plainly: "a scenario's end-to-end outcome is
approximated by the MAJORITY effective regime over its per-step online
telemetry... an offline scenario-level approximation." Concretely
(`scenario_regime_from_telemetry`, lines 82–95):

```
telemetry_by_scenario: Dict[scenario_id -> ordered list of raw per-step regimes]
  -> for each scenario:
       effective, _ = dwell_fn(raw_regimes)          # frozen apply_dwell_and_fallback, correct
       vals, counts = np.unique(effective, return_counts=True)
       dispatched_regime = vals[np.argmax(counts)]    # <-- THE APPROXIMATION POINT
```

That single `dispatched_regime` is then used to index into ONE
already-precomputed, whole-scenario ANWG column
(`unified_utility_matrix_wide_v2.csv`, one static number per
(scenario, policy) pair) via `baseline_d_anwg`/`baseline_e_anwg`. No
simulator is ever re-run per router decision; the "hierarchy's" reported
ANWG for a scenario is literally the pre-existing ANWG of whichever ONE
fixed policy the majority vote happened to pick, applied as if that policy
had run for the entire scenario.

**B. Exact approximation point.** The line
`vals[np.argmax(counts)]` — collapsing an entire per-step trajectory of
potentially-changing effective regimes into one modal label — is where
information is destroyed. Everything upstream of it (Stage-1 prediction,
the dwell/fallback FSM) is computed correctly and per-step; everything
downstream of it (dispatch, ANWG lookup) treats the scenario as if that one
majority regime had been active for 100% of it.

**C. Why it loses minority-but-critical regime activity.** The prior TEST
audit already quantified this precisely: `KV_MEMORY_PRESSURE` is active on
only 8%–25% of a scenario's steps even in scenarios that ARE genuinely
KV-pressured (§10 of that audit). A regime that is *never* the plurality
of a scenario's steps can *never* win `np.argmax(counts)`, regardless of
how well Stage-1 identifies it exactly when it's active, and regardless of
how good Stage-2's native-pair selection is once routed there (that audit's
§4 showed Stage-2 in isolation, given ground truth, achieves 0 regret on
both evaluable regimes — the loss is 100% attributable to dispatch, not to
either stage's own competence).

**D. What can be reused safely (and is, unmodified, by this task).**
- `Stage1Router` (fit/predict), `apply_dwell_and_fallback`,
  `count_dwell_violations`, `route_action`, `STAGE1_INPUT_COLUMNS`,
  `STAGE2_CANDIDATES`, `FALLBACK_POLICY`, `DWELL_MINIMUM_STEPS`, the frozen
  blended-microcase builders (`hierarchical_regime_router_v1.py`).
- `Stage2Selector` (fit/predict, native-pair-only, hard-assertion-guarded)
  (`hierarchical_stage2_selectors_v1.py`).
- `compute_regime_signals`/`compute_activity_labels`
  (`online_regime_signals_v1.py`).
- All six native policies' real, unmodified `select_action` implementations.
- The unmodified `Simulator`/`SimulatorConfig`/`ServiceModel` — **no
  Simulator code change was needed or made** (§3 below explains why).
- `unified_utility_matrix.py`'s own `_build_policy` convention for how
  `full_prefill`/`chunked_prefill_small` are paired with a
  `ServiceModel`-level chunk override (reused as the reference baseline
  for forced-equivalence, §6).

**E. What must be bypassed/replaced.** Only
`scenario_regime_from_telemetry` and the `baseline_d_anwg`/
`baseline_e_anwg` dispatch-by-precomputed-column pattern that depends on
it. Neither is modified — they remain exactly as frozen, still valid for
their own documented purpose (an offline TRAIN/VAL/smoke-scoped
approximation) — this task adds an entirely separate module that never
calls them from within its own routing path (§5).

## 3. Live harness architecture

**Key structural fact that makes this simple:** `Simulator.run()`
(`simulator.py` §3, unmodified) already does, every step, exactly what a
closed loop requires: `action = policy.select_action(state)` immediately
followed by `self._apply_action(action)` and `self._advance_decode(action)`
— i.e. whatever the policy just decided causally determines the very next
`ObservableState`. **No Simulator change was needed.** The entire closed
loop falls out of wrapping the six frozen native policies inside one
`BasePolicy` and handing that ONE wrapper to the existing, completely
unmodified `Simulator.run()`.

New module (additive only):
`src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py`

```
ObservableState (this step, real)
  -> compute_regime_signals / compute_activity_labels   (frozen, reused)
  -> Stage1Router.predict                                (frozen, reused)
  -> IncrementalDwellFallbackFSM.step                     (new; proven == frozen apply_dwell_and_fallback, §8)
  -> Stage2Selector.predict (if an active regime)         (frozen, reused)
  -> resolve exactly ONE native policy id
  -> that policy's REAL select_action(state)               (frozen, reused, unmodified)
  -> Action (+ prefill_chunk_override if Regime B, §4)
  -> Simulator._apply_action / _advance_decode              (frozen, unmodified)
  -> ObservableState (next step, real, causally different)
```

`LiveHierarchicalRouterPolicy.select_action` is the single entry point;
`run_live_scenario(...)` builds one `Simulator`, hands it this policy, and
returns the resulting `RunMetrics` (real, simulator-computed
`arrival_normalized_weighted_goodput`) plus a full step-by-step trajectory
log (§9).

## 4. Policy-state preservation rule

**Audit finding: none of the six frozen native policies carry meaningful
cross-call state.** Direct source inspection of all six
(`estimated_service_time_first.py`, `weighted_fair_share.py`,
`least_laxity_first.py`, `kv_constrained_online.py`,
`prefill_control_variants.py::GreedyArrivalPrefillControlPolicy`):

| Policy | Constructor state | Mutated inside `select_action`? | Verdict |
|---|---|---|---|
| `estimated_service_time_first` | `alpha`, `beta` (fixed) | No | stateless |
| `weighted_fair_share` | `alpha`, `beta` (fixed) | No (`admitted_counts` is a local `Counter`, not `self.*`) | stateless |
| `least_laxity_first` | `alpha`, `beta` (fixed) | No | stateless |
| `kv_constrained_online` | `step_size`, `alpha`, `beta`, `target_kv_utilization`, `urgent_laxity_seconds` (fixed) | No | stateless |
| `full_prefill` / `chunked_prefill_small` (both `GreedyArrivalPrefillControlPolicy`) | none | No (`deterministic_place` is a pure function of its arguments) | stateless |

Every one is a pure function of its fixed constructor hyperparameters and
the `ObservableState` argument. `tests/test_hierarchical_router_live_harness_v1.py::test_all_six_native_policies_are_provably_stateless`
enforces this structurally (source-level: no `self.*` assignment inside
`select_action`).

**Consequence:** the "reset semantics" and "resume vs. reset on
reactivation" questions the task raises are moot for this policy set —
there is no meaningful internal state to preserve or reset. **Frozen rule
adopted (per the task's own preferred default, satisfied trivially here):**
`build_native_policy_instances()` instantiates all six exactly once per
`LiveHierarchicalRouterPolicy` (i.e. once per scenario run), and the same
six instances are reused for every step regardless of how many times the
router switches into and out of each one. This is implemented, not merely
argued: `LiveHierarchicalRouterPolicy.__init__` calls
`build_native_policy_instances()` exactly once.

## 5. Action-level compatibility audit

Five of the six native policies (`estimated_service_time_first`,
`weighted_fair_share`, `least_laxity_first`, `kv_constrained_online`, and
both prefill-control variants for their **admission** decision) return only
`Action.admit` — directly compatible with the same per-step contract the
Simulator already applies uniformly, no translation needed.

**The one real compatibility gap: Regime B's mechanism is NOT an admission
difference.** `full_prefill` and `chunked_prefill_small` share the
identical `GreedyArrivalPrefillControlPolicy` admission logic
(`prefill_control_variants.py`) — their entire distinguishing mechanism is
a `ServiceModel`-level execution-budget field
(`max_prefill_chunk_tokens`), fixed at `Simulator`/`ServiceModel`
construction time, not something either policy's own `select_action`
Action carries. This is confirmed directly in the frozen evaluation code:
`unified_utility_matrix.py::_build_policy` builds a *different merged
`ServiceModel`* per policy for exactly this reason
(`sm_override = {"max_prefill_chunk_tokens": ..., "decode_first": False}`).
A single live `Simulator` run has exactly one `ServiceModel`, constructed
once — it cannot be swapped mid-run just because the router switched which
Stage-2 candidate is active this step.

**Resolution: reuse an existing, already-authorized canonical adapter, not
invent one.** `Action.prefill_chunk_override` (`core/action.py`) is an
existing per-step verb — added for
`composition.prefill_control_policy.PrefillControlChildPolicy`, already
honored by `GPUState.step()` (`gpu.py` §`step`: "used in place of
`service_model.max_prefill_chunk_tokens` for THIS step's prefill budget on
this GPU only — the frozen ServiceModel itself is never mutated") — that
was built for exactly this situation. This harness attaches it whenever
the resolved policy id is `full_prefill` or `chunked_prefill_small`, using
the identical constants (`UNLIMITED_PREFILL_CHUNK`, `DEFAULT_CHUNK_SMALL`)
`unified_utility_matrix.py::_build_policy`'s own `sm_override` already
uses. Nothing new is invented; the harness only decides *when* to attach
it (Regime-B routing steps) and *with which* value (the frozen native-pair
constants).

**Empirical proof this is a faithful substitute, not an approximation**:
§6's forced-parent equivalence tests run each of `full_prefill`/
`chunked_prefill_small` two ways — (a) baked into `ServiceModel` for the
whole run (the existing evaluation convention) and (b) via
`prefill_chunk_override` every step through the live harness — and the
resulting ANWG is bit-identical (§6 table). `decode_first` never needs a
per-step override: it is `False` for both frozen Regime-B candidates.

No other action-verb translation is needed anywhere else: no hidden family
metadata, no unsupported verbs, no GPU-mutation assumption differs across
the six policies (all mutate `ObservableGPUState.active_request_ids`/
`current_kv_tokens` in place via `deterministic_place` or an equivalent
manual admit loop — a pre-existing, shared convention across all six, not
something this harness introduces).

## 6. Forced-expert equivalence results

Six small, fast, synthetic-token fixture scenarios (one per native family,
built directly from the frozen family template functions with
`allow_synthetic_tokens=True`) were run two ways: (a) the reference path —
`Simulator` + that one policy directly, with the same `ServiceModel`
merge convention `unified_utility_matrix.py::_build_policy` already uses;
(b) the live harness, `forced_expert=<policy_id>` (bypasses Stage-1/Stage-2/
dwell entirely, always delegates to that one policy). Both trajectories
are driven by the same deterministic `Simulator`, so equality is exact
(`abs(Δ) < 1e-9`), not merely "close."

| Regime | Policy | Reference ANWG | Live-forced ANWG | Match |
|---|---|---|---|---|
| A (F) | `estimated_service_time_first` | 0.75 | 0.75 | **PASS** |
| A (G) | `weighted_fair_share` | 0.75 | 0.75 | **PASS** |
| B (H) | `full_prefill` | 0.6875 | 0.6875 | **PASS** |
| B (I) | `chunked_prefill_small` | 0.625 | 0.625 | **PASS** |
| C (J) | `kv_constrained_online` | 0.7647058823529411 | 0.7647058823529411 | **PASS** |
| C (K) | `least_laxity_first` | 0.4411764705882353 | 0.4411764705882353 | **PASS** |

**All 6/6 bit-exact.** In particular, B's two cases prove the
`prefill_chunk_override` adapter (§5) is a faithful causal substitute for
the `ServiceModel`-level field it stands in for — this is the load-bearing
result the whole harness depends on, and it passes.

## 7. Causal-switch microcase (L)

Scenario: the frozen `build_blended_microcase_b_plus_c()` (unmodified,
reused as-is), run 201 steps (capped for speed; not a scientific claim
about the full scenario). Two runs from the identical initial scenario:
(1) real routing (Stage-1 fit on real TRAIN telemetry, Stage-2 fit on real
TRAIN scenario rows — same fitting code as the existing
`run_hierarchical_regime_router_v1_smoke.py`); (2) `forced_expert=
weighted_fair_share` for every step (what the router does whenever it is
in `NONE`/`OVERLAP`, i.e. "router permanently disabled").

| | |
|---|---|
| First real switch | step **21** (`NONE -> KV_MEMORY_PRESSURE`) |
| Total switches within the 201-step window | 3 |
| Trajectories identical for steps 0–20 | **Yes** (both select `weighted_fair_share`, identical `admitted_request_ids` every step) |
| Trajectories diverge at/after step 21 | **Yes** |
| Dwell violations | 0 |

This is the causality proof the task asked for: **same initial state,
same code path, up to step 21 — the only difference is the router's own
decision at step 21 (to activate `KV_MEMORY_PRESSURE` and delegate to
`kv_constrained_online` instead of `weighted_fair_share`) — and that one
decision provably changes the observable trajectory from that point on**
(different `admitted_request_ids`/`mean_kv_utilization_after_admission` at
or after step 21). This is not an observational label — the selected
policy's own `select_action` call is what produced the different `Action`,
which the Simulator then actually applied.

## 8. Dwell / fallback validation

The frozen `apply_dwell_and_fallback` is a **batch** function: it computes
the entire per-step transition history in one call, given the full raw
sequence. Calling it fresh every live step by replaying the growing
history (`apply_dwell_and_fallback(raw_regimes_seen_so_far)`) would be
byte-for-byte reuse with zero reimplementation risk — and was the first
implementation tried — but is O(steps) per call / **O(steps²) per
scenario**, which measured as prohibitively slow on real MF-PSD
trajectories (one scenario in this project's own prior audit has >10,000
steps; a live-harness prototype run on it did not finish in 120 seconds
and was killed).

**Resolution:** `IncrementalDwellFallbackFSM`, a genuine O(1)-per-step
streaming state machine carrying the exact same two pieces of state
(`effective`, `steps_since_change`) the batch function's own loop body
computes. This is **proven**, not merely argued, bit-identical to the
frozen batch function:
`test_incremental_fsm_matches_frozen_batch_fsm_on_random_sequences` runs
200 random raw-regime sequences (length 1–60, uniform over all 5 classes)
through both implementations and asserts identical effective-regime
sequences and identical diagnostics (`total_transitions`,
`switches_per_regime`, `switching_rate_per_1000_steps`, `fallback_rate`)
on every one — **200/200 pass**.

`dwell_violation_count` (design doc: "should be exactly 0 by construction
— a correctness check, not a tunable outcome") is verified once per live
run with a single O(steps) pass over the realized trajectory using the
frozen `count_dwell_violations` check directly (not per-step, and not
reimplemented) — 0 in every run performed for this task (forced-mode runs,
the causal-switch microcase, and every TRAIN/VAL smoke scenario, §12).

NONE and OVERLAP fallback semantics were tested directly with a scripted
Stage-1 stub that always returns `NONE` (resp. `OVERLAP`):
`selected_policy == "weighted_fair_share"` and `stage2_regime is None` on
every single step, confirming Stage-2 is never consulted for either
fallback outcome — matching §F of the frozen design doc exactly.

## 9. Temporal-leakage validation

- **Stage-1 inputs**: `test_stage1_input_row_contains_only_the_frozen_four_columns`
  intercepts the exact DataFrame passed to `Stage1Router.predict` and
  asserts its columns are exactly `STAGE1_INPUT_COLUMNS`, in that order —
  nothing else is ever passed.
- **Stage-2 inputs**: Stage-2 was frozen fitting on
  `multifamily_contextual_selector_v1.FEATURE_COLUMNS` — 33 columns that
  are scenario-**generation** parameters (`feat_A__target_utilization`,
  `feat_B__hog_count`, `feat_C__bulk_pressure`, ...), fixed at t=0 and
  constant for a scenario's entire trajectory. Reading them at every
  Stage-2-active step (rather than once per scenario, as the frozen
  offline evaluation already does) introduces **no new information** —
  the values are identical every time. Two sourcing paths, both read-only
  and both audited to never touch `scenario.requests` or
  `actual_output_tokens` (`test_feature_row_best_effort_never_reads_forbidden_scenario_fields`):
  1. **Exact**: for a real MF-PSD `canonical_scenario_id`, the row is
     looked up directly from the frozen `mf_psd_scenarios_v1.csv`.
  2. **Best-effort** (for scenarios not in that table, e.g. a fresh
     fixture or a blended microcase): derived from
     `scenario.params` — which `PolicySeparationScenario`'s own frozen
     docstring defines as "the exact keyword arguments the template
     function was called with" — plus `scenario.stress_control_relationship`,
     matched by exact column-name suffix. Every other `FEATURE_COLUMNS`
     entry is left `NaN`, using `build_X`'s own frozen, pre-existing
     missing-value contract (numeric NaN → 0.0 + `__missing` indicator;
     categorical NaN → `"__NONE__"`) — the identical contract every real
     cross-family training row already relies on (a real Family-A row
     already has NaN for every `feat_B__`/`feat_C__` column). This is a
     **named, bounded limitation**: fields this cannot honestly derive
     (request-trace aggregates like `feat_B__hog_prompt_median`) are left
     NaN, never fabricated.
- **No future-state access**: `compute_regime_signals(state)` is called
  exactly once per real step, on that step's own just-built
  `ObservableState` (`test_trajectory_features_depend_only_on_current_and_past_state`
  — structural: the call site appears exactly once in the module).
- **No hidden identity leakage**: neither Stage-1 nor Stage-2 nor the
  harness itself ever reads `mechanism_family`, `canonical_scenario_id`,
  `seed`, or any `CompletedRequest`/SLO/ANWG field as a routing input —
  the only two data paths into a routing decision are (a) the frozen
  4-column Stage-1 allowlist and (b) the frozen 33-column Stage-2 feature
  schema, both already leakage-audited by prior frozen work and reused
  unchanged here.

## 10. Majority-vote exclusion (regression guard)

`test_live_harness_module_never_imports_the_majority_vote_evaluation_module`
and `test_live_harness_module_never_computes_a_majority_over_effective_regimes`
parse the live harness module's own AST (not a text grep — the module's
own docstrings and comments legitimately *discuss* the forbidden pattern
for documentation purposes, so a plain substring search would false-fail)
and assert: (a) no `import`/`from ... import` anywhere in the module names
`hierarchical_router_evaluation_v1`; (b) no `Name`/`Attribute` node
anywhere references `scenario_regime_from_telemetry`; (c) no `Call` node
invokes `np.unique(..., return_counts=True)` or the paired
`np.argmax(counts)` pattern. **Both tests pass.** The smoke script (a
separate file, `scripts/run_hierarchical_router_live_harness_v1_smoke.py`)
DOES import `scenario_regime_from_telemetry` — deliberately, and only for
§13's side-by-side methodology comparison, never inside the harness
module's own routing path.

## 11. Family-B TEST representation — not touched

The prior audit's finding that Family B (8 groups) has zero TEST
scenarios on the current split (§3 of that audit) is unchanged and
untouched. This task does not read, recompute, or redesign the TRAIN/VAL/
TEST split (`build_splits`, `hierarchical_regime_router_v1.py`, unmodified
— confirmed by `test_frozen_router_module_symbols_unchanged_shape`).
Fixing it is explicitly named, per the task's own §11, as a separate
future replication-design issue.

## 12. TRAIN/VAL live smoke

Ran the live harness (real Stage-1 + real Stage-2, no forcing) on:
two Family-A-mechanism fixtures, two Family-B-mechanism fixtures (fresh,
synthetic-token, same generator functions/parameters real TRAIN/VAL
scenarios use), and the frozen B+C blended microcase (for Regime-C
reachability, per the Family-C reconstruction limitation below).

| Fixture | Steps | ANWG | Regimes visited | Dwell violations | All finite |
|---|---|---|---|---|---|
| A_fixture_1 (`skew=5.0`, stress) | 4,599 | 0.75 | NONE 666, RANKING_FAIRNESS 3,933 | 0 | yes |
| A_fixture_2 (`skew=1.0`, control) | 4,276 | 1.0 | NONE 4,276 | 0 | yes |
| B_fixture_1 (`hog=high`) | 614 | 0.65625 | NONE 608, PREFILL_DECODE_CONTENTION 6 | 0 | yes |
| B_fixture_2 (`hog=low`) | 458 | 1.0 | NONE 197, PREFILL_DECODE_CONTENTION 261 | 0 | yes |
| C via B+C blended microcase (capped 2,000 steps) | 2,001 | 0.640625 | NONE 1,461, KV_MEMORY_PRESSURE 540 | 0 | yes |

**All three regime paths reached. No NaN/Inf anywhere. Dwell violations:
0/0/0/0/0.** A_fixture_2 (the `skew=1.0` control scenario) correctly never
activates `RANKING_FAIRNESS`, exactly matching the frozen `a_active`
formula's own definition of a non-stressed control case (`priority_skew`
never exceeds 1.05 when the true generator ratio is 1.0) — this is
expected, correct behavior, not a harness defect.

**Family-C note (documented limitation, not fixed here):**
`unified_utility_matrix.py` already documents that Family C / KV v2
scenarios cannot be regenerated byte-exact from the current template code
(`kv_v2_reproducibility_forensic_20260817.md`). This task therefore never
attempts to re-simulate a *specific historical* Family-C MF-PSD scenario;
Regime-C reachability is instead demonstrated via the frozen B+C blended
microcase, which the prior TEST audit already confirmed empirically
activates genuine KV pressure. This is a pre-existing limitation this task
inherits and reports, not one it works around by touching Family-C
reconstruction.

## 13. Approximate-vs-live methodology comparison (diagnostic only)

On the same 6 real TRAIN/VAL scenarios (3 Family-A + 3 Family-B, first-by-id),
the OLD scenario-level majority-vote dispatch (imported only in the
comparison script, never in the harness itself, §10) was run against the
NEW live harness's behavior on matched-mechanism fixtures (§12):

| | OLD (majority-vote, real scenarios) | NEW (live, matched-mechanism fixtures) |
|---|---|---|
| Predicted/effective regime distribution | `NONE`: 6/6 (100%) | `RANKING_FAIRNESS`, `PREFILL_DECODE_CONTENTION`, `KV_MEMORY_PRESSURE` all genuinely visited |
| Mean ΔANWG vs. best-global-fixed | **0.0000** (identical to baseline A on all 6) | not directly comparable (different scenario instances) — but the live ANWGs (0.75, 1.0, 0.65625, 1.0, 0.640625) are computed while the router genuinely spent real steps in every active regime |

This is exactly the mechanism the failure analysis in
[`hierarchical_regime_router_v1_20260818.md`](hierarchical_regime_router_v1_20260818.md)
predicted: on this small real sample, the majority-vote dispatch **never
elected an active regime for even one of the 6 scenarios** — every one
fell back to `NONE`/baseline A, producing the flat "0 measurable delta"
signature that audit diagnosed. The live harness, on matched-mechanism
scenarios, **does** route into and act from all three active regimes.
This is reported strictly as a methodology diagnostic, per the task's own
instruction — **it is not evidence for or against `HIERARCHICAL_ROUTER_GO`**,
since the scenario instances differ and no TEST data was touched.

## 14. Implementation artifacts

| File | Status |
|---|---|
| `src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py` | new |
| `scripts/run_hierarchical_router_live_harness_v1_smoke.py` | new |
| `tests/test_hierarchical_router_live_harness_v1.py` | new (26 tests) |
| `docs/audits/hierarchical_router_live_harness_validation_v1_20260818.md` | new (this document) |
| `experiments/hierarchical_router_live_harness_v1_smoke/live_harness_smoke_summary.json` | new (smoke output) |

No existing file listed in the prior audit's frozen-artifact set was
modified (verified in §17 below).

## 15. Harness readiness verdict

**`LIVE_HIERARCHICAL_HARNESS_READY`**

All required conditions hold:
- Forced-parent equivalence: **6/6 PASS**, bit-exact (§6).
- Closed-loop causality: demonstrated, same-state/different-decision/
  different-outcome, with the divergence traced to a specific step (§7).
- Dwell/fallback semantics: 0 violations across every run performed;
  NONE/OVERLAP correctly never reach Stage-2 (§8).
- No majority vote in the live path: both structural AST-level guards pass (§10).
- Temporal leakage checks: pass (§9).
- TRAIN/VAL smoke: all three regime paths reached, 0 dwell violations,
  no NaN/Inf (§12).
- Frozen router/policies/gates: unchanged (§17).

## 16. Tests

26 new tests in `tests/test_hierarchical_router_live_harness_v1.py`:
forced ESTF/WFS/full_prefill/chunked_prefill_small/kv_constrained_online/
least_laxity_first equivalence (6, parametrized across 2 test functions
covering 2 policies each for A/C and both B policies individually);
forced-mode bypasses Stage-1/Stage-2; policy-state statelessness audit;
single-instantiate-per-run check; incremental-FSM-vs-frozen-batch
equivalence (200 random sequences) plus a switch-flag unit test;
dwell-minimum-respected; NONE fallback; OVERLAP fallback; causal-switch
microcase; 2 majority-vote-exclusion structural guards; 3 temporal-leakage
checks; trajectory-log completeness; deterministic replay; canonical ANWG
range check; no-NaN/Inf smoke check; 2 frozen-artifact-immutability checks.
**26/26 pass.** The full pre-existing suite (`test_hierarchical_regime_router_v1.py`,
`test_hierarchical_router_evaluation_v1.py`, `test_hierarchical_router_gates_v1.py`,
73 tests) was re-run unmodified and **all 73 still pass**; the complete
repository test suite was also re-run in full as part of this task's
pre-commit check (§17/§18).

## 17. Files changed / frozen-artifact immutability

```
$ git diff --stat 21bfff1 -- \
    src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py \
    src/llmserveopt/selector/hierarchical_stage2_selectors_v1.py \
    src/llmserveopt/policy_separation/online_regime_signals_v1.py \
    configs/hierarchical_regime_router_v1_gates.json \
    docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md \
    docs/audits/hierarchical_regime_router_v1_20260818.md \
    experiments/hierarchical_regime_router_v1_test_evaluation/
(empty — no changes)
```

Only new, additive files were created (§14). No Simulator/Action/GPUState
code was modified — `Action.prefill_chunk_override` already existed
(§5); the harness only calls it.

## 18. Confirmation: original TEST result/audit untouched

`docs/audits/hierarchical_regime_router_v1_20260818.md` and
`experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json`
are byte-identical to their state at task start (§17). The mechanical
verdict `HIERARCHICAL_ROUTER_NO_GO` stands exactly as computed. This task
does not assign, imply, or gesture toward `HIERARCHICAL_ROUTER_GO`/`NO_GO`
for anything — only the harness-readiness verdict in §15.

## 19. Exact single next scientific action

**Not started, not authorized by this task.** With
`LIVE_HIERARCHICAL_HARNESS_READY`, the next well-defined step is a genuine
re-evaluation of the hierarchical router against the real, pre-registered
TEST split using this live harness instead of the majority-vote
approximation — which would also require first resolving the separately-
named Family-B zero-TEST-representation issue (§11) if a Regime-B TEST
signal is wanted. That re-evaluation, any TEST-split change, and any new
`HIERARCHICAL_ROUTER_GO`/`NO_GO` computation all require separate,
explicit authorization, per this task's own stop condition (§19 of the
task instructions). **This document does not begin that action.**

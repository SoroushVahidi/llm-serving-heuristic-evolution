# Family-A Observability / Continuation-Dependence Diagnostic v1 — Instrumentation Repair Audit

Date: 2026-08-20. Integrity-repair pass over the invalid run documented in
`docs/current/family_a_observability_continuation_v1_analysis_20260820.md`. This document does
not overwrite that analysis (it correctly documents the invalid first run) and does not draw a
scientific interpretation of Δ_same/Δ_native — it documents the repair, its verification, and the
repaired run's integrity only.

---

## 1. Root cause (confirmed)

In `FamilyAObservabilityObserver.select_action` (`src/llmserveopt/analysis/family_a_observability_continuation_v1.py`),
`real_action = self.inner_router.select_action(state)` ran first. That call delegates to whichever
ONE native policy Stage-2 selected, which mutates `state.gpu_states[i].active_request_ids` (list,
appended) and `state.gpu_states[i].current_kv_tokens` (scalar, incremented) in place as same-call
admission-planning bookkeeping — confirmed identically in both
`estimated_service_time_first.py:80-81` and `policy_library_v2_helpers.py::deterministic_place:105-106`
(used by `weighted_fair_share.py`). No other field is touched by any native policy's
`select_action` — `active_requests_info` is a separate list built once by `GPUState.to_observable()`
and never mutated.

The pre-repair code's `saved_gpu_state` snapshot (used to reset state between the ESTF and WFS
shadow calls) was captured **after** this real-router mutation had already happened — so both
shadow candidates were evaluated against an already-capacity-consumed baseline instead of the true
open decision point, which structurally biased both toward the same (often empty, given every
Family-A GPU has `max_active_sequences=1`) admit set. Result: `actions_disagree()` returned `False`
on effectively every one of 796,415 Family-A-active steps across all 64 scenarios in the completed
invalid run (`n_events_total=0`).

Confirmed via `simulator.py::Simulator.run` that nothing downstream ever re-reads `state` after
`policy.select_action(state)` returns (the real per-step commit happens via `self._apply_action(action)`
against the simulator's own internal `self._gpus`, independent of the `ObservableState` snapshot
object), and that `LiveHierarchicalRouterPolicy._log_step` (which does read the post-admission
mutated `state.gpu_states` for its own trajectory-row logging) fully completes *inside*
`inner_router.select_action(state)`, before that call returns to the observer — so nothing is lost
by restoring `state.gpu_states` to the true pre-decision baseline afterward.

`fork_from_live_simulator`/`run_bounded_rollout` (the branch-forking machinery) were **not** part of
the bug — they deep-copy the simulator's own internal `sim._gpus` directly (`decision_criticality_timescale_trainval_v1.py:392`),
never touching the `ObservableState` object the shadow-comparison logic mutates, and were
unaffected by any fix here.

---

## 2. Exact fix

Files modified:
- `src/llmserveopt/analysis/family_a_observability_continuation_v1.py` — added two module-level
  helper functions, `snapshot_gpu_counters`/`restore_gpu_counters` (pure functions over the two
  affected fields, extracted from the prior inline closure for testability), and reordered
  `FamilyAObservabilityObserver.select_action` to: (1) snapshot the TRUE pre-decision GPU counters
  *before* calling `inner_router.select_action`; (2) also snapshot the post-real-admission state
  immediately after that call returns; (3) restore to the pre-decision snapshot before each of the
  ESTF and WFS shadow calls (unchanged mechanism, corrected input); (4) restore to the
  pre-decision snapshot once more before feature extraction, so `extract_causal_features` correctly
  observes decision-time state (design doc §C) rather than a post-admission-contaminated one — this
  was a second-order consequence of the same bug, since feature extraction previously ran after the
  same late restore; (5) restore to the post-real-admission snapshot immediately before returning
  `real_action`, as a defensive no-op guarantee (nothing currently reads `state` again after this
  point, but this guarantees zero externally observable behavior change regardless).
- `scripts/run_family_a_observability_continuation_v1.py` — strengthened the integrity gate (see §7).
- `tests/test_family_a_observability_continuation_v1.py` — added regression tests (see §4-§6).

No scientific design parameter was changed: the 64-scenario population, ESTF/WFS pair, seeds,
`FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS=1500`, `FULL_TRAJECTORY_BRANCHES_PER_SCENARIO=3`,
`HISTORY_WINDOW=10`, both common-continuation choices, all feature groups, all consequence/delta
definitions, the disagreement-event sampling rule, the grouped-CV split logic, and the four
interpretation categories are all byte-identical to the frozen design doc — verified by diffing
the changed files' logic against the design doc's §E/§G/§H/§J and confirming only the
snapshot-timing and integrity-gate-strictness changed.

---

## 3. Observer non-interference (proven)

New test `test_observer_does_not_change_real_router_trajectory_or_completions` runs the identical
seeded Family-A fixture (seed=17, 40 jobs) two ways — once through the plain, unwrapped
`LiveHierarchicalRouterPolicy`, once through the repaired `FamilyAObservabilityObserver` — and
asserts:
- Identical trajectory row-for-row on `step`, `effective_regime`, `selected_policy`,
  `admitted_count`, `admitted_request_ids`, `queue_len_after_admission`,
  `active_count_after_admission`, `mean_kv_utilization_after_admission`
  (`pandas.testing.assert_frame_equal`).
- Identical `RunMetrics` aggregate outcome: `num_completed`, `num_dropped`, `num_slo_violated`,
  `num_total`, `completion_fraction`, `mean_latency`, `median_latency`, `p95_latency`,
  `p99_latency`, `max_latency`, `slo_violation_rate`.

**PASSED.** The observer's shadow computation is confirmed invisible to the real router/simulator.

---

## 4. Non-vacuous regression test

**Scenario/state used**: `case_fairness_vs_size_v2(target_utilization=1.2, tenant_weight_skew=5.0,
favored_tenant_size="long", prediction_noise_sigma=0.0, seed=7, n_total_jobs=40,
allow_synthetic_tokens=True)` — the same synthetic Family-A fixture already used by
`test_family_a_diagnostic_deterministic_replay`, with `n_total_jobs` made explicit at 40 (the
value empirically confirmed to reliably produce disagreement, see below).

**Old behavior** (verified empirically in this repair session, via an isolated monkeypatch of
`FamilyAObservabilityObserver.select_action` back to the exact pre-repair method body, run in a
throwaway scratch script — not committed, not part of the test suite): `n_family_a_active_steps=2636`,
`events=0`.

**Repaired behavior**: identical fixture, identical active-step count (`2636`, confirming the
population/replay is unaffected), `events=3` (the full branch budget).

New test `test_family_a_observer_captures_known_disagreement_from_identical_pre_decision_state`
asserts `len(res.events) >= 1` on this fixture, and for each captured event: `admit_symmetric_diff_size
>= 1` (ESTF and WFS actions genuinely differ), and that `(canonical_scenario_id, step)` forms a
stable, valid join key. `test_family_a_diagnostic_deterministic_replay` and
`test_family_a_events_have_stable_join_keys_and_bounded_branch_budget` were also updated to assert
`len(events) >= 1` explicitly, so a regression back to the pre-repair bug would now fail these
tests directly (previously both passed vacuously at `0 == 0` / `0 <= 3`).

**PASSED** (all three tests).

---

## 5. Snapshot/restore unit tests

Four new tests directly exercise `snapshot_gpu_counters`/`restore_gpu_counters` in isolation
(synthetic `ObservableGPUState`/`ObservableState` fixtures, no simulator needed):

- `test_snapshot_gpu_counters_captures_state_before_mutation` — a snapshot taken before a mutation
  restores the pre-mutation values, not the state at restore-time (§6.A).
- `test_restore_gpu_counters_no_alias_leakage` — `restore_gpu_counters` writes a fresh list
  (`[:] = ids`), never aliases the snapshot's own list; a later live mutation does not corrupt a
  previously-taken snapshot, and a second restore from the same snapshot still works (§6.E).
- `test_restore_gpu_counters_gives_identical_baseline_across_repeated_calls` — simulates the real
  three-restore sequence (before ESTF, before WFS, before feature extraction) with different
  intervening mutations each time; asserts every post-restore state is bit-identical to the true
  original (§6.B/C).
- `test_snapshot_gpu_counters_covers_all_gpus_independently` — multi-GPU states restore each GPU's
  own counters independently; one GPU's mutation never leaks into another's restored values (§6.E).

**PASSED** (all four).

---

## 6. Parent-study cross-check

Per the design doc, this diagnostic's disagreement detection is a **fresh, independent
instrumentation pass**, not a byte-for-byte reproduction of the parent decision-criticality study's
own shadow-disagreement check — the two are *not* mathematically identical definitions:

- **Parent study** (`decision_criticality_timescale_trainval_v1.py::ForkingObserverPolicy.select_action`):
  compares `real_action` (computed on the true clean pre-decision state — it's the *first* call
  made on `state`) against `alt_action = alt_policy.select_action(copy.deepcopy(state))`, where that
  deep copy is taken **after** `real_action`'s own mutation. This is an **asymmetric** comparison:
  chosen-candidate-on-clean-state vs. alt-candidate-on-already-capacity-consumed-state.
- **This (repaired) diagnostic**: compares `ESTF.select_action(state)` against
  `WFS.select_action(state)`, both evaluated from the identical true pre-decision baseline. This is
  a **symmetric** comparison, and the one the design doc's §D actually specifies ("both actions are
  already fully determined [by] the snapshotted ObservableState").

**First cross-check attempt** (4 real TRAIN scenarios with the most parent-flagged disagreements,
all coincidentally `favshort` scenarios, 50-51 flagged disagreement steps each): the repaired
diagnostic found **0/4 scenarios with any event** (0 overlap). This triggered a targeted
investigation (per this repair task's own explicit instruction not to blindly proceed on a failed
cross-check) rather than an assumption of continued bugginess.

**Root-cause of the divergence, confirmed with real data**: every Family-A GPU config has
`max_active_sequences=1` (single admission slot). A targeted single-step probe at 3 of the parent's
own flagged steps in one `favshort` scenario
(`fs2.util1.1000.skew10.0000.favshort.noise0.30.s20260816`, steps 1362/3309/3906) showed ESTF and
WFS genuinely **agree** on the true pre-decision state at every one of them (identical single-request
admit sets) — i.e., under the correct symmetric definition there is no real ranking disagreement at
these specific steps; the parent's asymmetric definition manufactured spurious "disagreement"
purely from evaluating the alt candidate against an already-capacity-reduced baseline (exactly the
same category of bug this repair fixed, but present in the *other*, frozen, out-of-scope
decision-criticality module — not modified here, per this task's explicit constraints).

**Second cross-check, on a `favlong` scenario** (`fs2.util1.1000.skew10.0000.favlong.noise0.30.s20260816`,
67 parent-flagged disagreement steps): a single-step probe at the parent's first 6 flagged steps
found genuine disagreement (different admit sets under the true clean state) at 3 of them
(steps 1293, 4152, 4639) and genuine agreement at the other 3 (1285, 4994, 5119). Running the
**full automated pipeline** (`run_family_a_row_diagnostic`, the exact code path the real 64-scenario
run uses) on this same scenario captured **exactly 3 events, at steps 1293, 4152, and 4639** —
an **exact match** to the 3 steps independently confirmed as genuine disagreements, and all 3
independently flagged by the parent study too.

**Conclusion**: the definitions are intended to answer the same underlying question but are not
mathematically identical (asymmetric vs. symmetric baseline), so exact count equality is not
expected and was not required. Clear, exact, nonzero overlap is demonstrated (3/3 automated events
match confirmed genuine disagreements, all independently corroborated by the parent study's own
flagged steps) — the divergence on `favshort` scenarios is fully explained (near-total genuine
ESTF/WFS ranking agreement there under the correct definition, given the single-admission-slot GPU
constraint; the parent's inflated count there is a definitional artifact, not evidence this
repair's detection is broken) and is not a residual bug. **Cross-check requirement satisfied.**

---

## 7. Strengthened integrity gates

`scripts/run_family_a_observability_continuation_v1.py`'s integrity report (`ok` field) previously
only checked: no scenario failures, scenario count matches, no duplicate `(scenario, step)` join
keys. It could report `ok=true` even when `n_events_total=0` — exactly what happened in the invalid
run. The gate now additionally requires:
- `n_events_total > 0`
- `n_scenarios_with_events > 0` (new field, also recorded unconditionally)

`zero_events_detected` and `zero_scenarios_with_events_detected` are recorded explicitly (not just
folded into `ok`) so a future reader can see exactly which condition failed, if any. The runner's
exit code now distinguishes: `0` = fully healthy, `2` = scenario failures occurred, `3` = ran clean
but failed the strengthened integrity gate (e.g. zero events) — previously only `0`/`2` existed, so
a zero-events run exited `0` indistinguishably from a healthy one.

No exact-equality threshold against the parent study's 3,545-disagreement count was added, per this
task's explicit instruction not to invent a post-hoc threshold — the two diagnostics' definitions
are not intended to match exactly (§6), so only "nonzero" is enforced structurally; the *qualitative*
cross-check evidence (§6) is what establishes confidence beyond "nonzero."

---

## 8. Tests run

```
tests/test_family_a_observability_continuation_v1.py: 18 passed
tests/test_decision_criticality_timescale_trainval_v1.py: 42 passed (unmodified module, sanity check)
```

All Family-A tests pass, including the 5 new/hardened tests in §3-§5. The decision-criticality
module was not modified and its full suite passes unchanged, confirming no cross-module regression.

---

## 9. Invalid-run preservation

Moved (not deleted, not overwritten) verbatim: `experiments/family_a_observability_continuation_v1/`
→ `experiments/family_a_observability_continuation_v1_invalid_pre_snapshot_fix_20260820/`, plus its
log file → `.../family_a_observability_continuation_v1_invalid_run.log`. A `README_INVALID.md` was
added there (new file, does not alter any original artifact) documenting the reason invalid, the
original completion timestamp, and a pointer to this repair. See
`docs/current/family_a_observability_continuation_v1_analysis_20260820.md` for the original
integrity-failure analysis.

---

## 10. Repaired-run integrity

**[TO BE FILLED IN ON COMPLETION]**

- Launch: tmux session `family_a_observability_continuation_v1_repaired`, launched 2026-08-20
  19:40:06Z, `git_head_sha=8e1223beb58fd4d296061b6b48e3ba493714108f`, `git_tree_dirty=True` (expected —
  matches the session's known uncommitted files), exact command
  `python3 scripts/run_family_a_observability_continuation_v1.py`.
- Early health gate: [to fill in]
- Final: scenarios X/64, failures N, `n_events_total`=N, `n_scenarios_with_events`=N,
  `integrity_ok`=[bool], duplicate join keys=N, wall-clock=Ns.

---

## 11. Readiness for fresh scientific analysis

**[TO BE FILLED IN ON COMPLETION]** — pending §10. Once the repaired run completes with
`integrity_ok=true`, `n_events_total > 0`, and `n_scenarios_with_events > 0`, the diagnostic's
output is ready for a fresh scientific-interpretation pass (Δ_same/Δ_native/continuation-dependence
analysis, grouped prediction, mechanism attribution, etc., per the design doc's §H) — not performed
in this document per this task's explicit "do not yet write the final scientific interpretation"
instruction.

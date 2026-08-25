# Family-A Observability / Continuation-Dependence Diagnostic v1 — Analysis

Date: 2026-08-20
Analysis-only pass over the completed run at
`experiments/family_a_observability_continuation_v1/`. No code modified, no experiment rerun,
nothing committed/pushed. This document is the only repository file created or modified by this
analysis.

---

## 1. Executive verdict

**The run is structurally invalid for scientific interpretation: `n_events_total = 0`.** Across
all 64/64 Family-A TRAIN/VAL scenarios (0 failures, `integrity_ok=True` per the run's own mechanical
integrity gate), the diagnostic captured **zero disagreement events** between ESTF and WFS at any
of the 796,415 Family-A-active steps it evaluated. This directly contradicts the population this
run itself reproduced: total `n_family_a_active_steps` summed across scenarios is exactly
796,415 — identical to the parent decision-criticality study's own `RANKING_FAIRNESS` active-step
count (`docs/current/decision_criticality_timescale_trainval_v1_analysis_20260820.md` §3), which
found disagreement at 3,545/796,415 steps (0.445%) across 44/64 scenarios using the same
ESTF/WFS native pair over the same scenario population. The design doc itself predicted "132
actual branches attempted" as the expected realized count (§G). Zero is not a plausible
same-population replication of that result.

This is **not** a "no local signal" scientific finding — it is a **data-generation defect**. None
of §§3–14's requested statistics (Δ_same, Δ_native, continuation dependence, grouped prediction,
feature ablations, mechanism interpretation, concentration, sign flips) can be computed: the events
table is empty by construction (`family_a_observability_continuation_events.csv` was never even
written — the runner's `if len(events_df):` guard at `scripts/run_family_a_observability_continuation_v1.py:159`
short-circuits when there are no rows, and the results JSON correspondingly has none of the
`mean_delta_same`/`mean_delta_native`/`mean_continuation_dependence`/etc. keys that would exist had
even one event been captured — confirmed directly against the JSON's key set).

Per the authorizing task's own §1 instruction — "If structurally invalid, stop and classify:
`NEED_RESULT_INTEGRITY_REPAIR`. Do not repair anything in this task." — that is the classification
returned here, in place of forcing this run's (non-existent) evidence into one of §13's four
interpretation categories, all of which presuppose a nonempty event population.

**Classification: `NEED_RESULT_INTEGRITY_REPAIR`. No next-scientific-step from §14's four options
applies** (none of `DESIGN_FAMILY_A_INTERPRETABLE_CHILD` / `DESIGN_STATEFUL_FAMILY_A_CONTROLLER` /
`INVESTIGATE_MISSING_STATE` / `STOP_FAMILY_A_SYNTHESIS` is a correct characterization of "the
instrumentation has a data-generation bug and must be fixed and rerun before any interpretation is
possible"). The concrete engineering next step, outside this diagnostic's own four-option
taxonomy, is: fix the state-mutation-order defect identified in §12 below, add a test that would
have caught it, and rerun.

---

## 2. Structural integrity

| Check | Expected | Observed | Status |
|---|---|---|---|
| Scenario count | 64 | 64 | OK |
| Split composition | 54 train / 10 val | 54 train / 10 val | OK |
| Failures | 0 | 0 | OK |
| TEST leakage | none | none (`assert_trainval_only` guard held; `split` column is train/val only) | OK |
| Duplicate scenario IDs | 0 | 0 (64 unique `canonical_scenario_id` rows in `family_a_scenario_summaries.csv`) | OK |
| Null/NaN critical fields | none | none in scenario summary (`n_steps`, `n_family_a_active_steps`, `n_events`, `elapsed_s` all populated) | OK |
| Native Family-A policy pair | `estimated_service_time_first` / `weighted_fair_share` | matches (`provenance.estf_id`/`provenance.wfs_id`, and `assert STAGE2_CANDIDATES[REGIME_A] == (ESTF_ID, WFS_ID)` in the module) | OK |
| Disagreement-state count | ≤192 (≈132 expected, per parent-study reproduction) | **0** | **FAIL** |
| Expected branch counts | ≈132 events × 4 branches ≈ 528 rollouts | 0 branches run | **FAIL** |
| Observed branch counts | — | 0 (`family_a_observability_continuation_events.csv` does not exist — never written, since `len(all_events_rows) == 0`) | **FAIL** |
| Counterfactual outputs structurally complete | one row per event with all 4 branch outcomes + deltas | N/A — no rows exist | **FAIL (vacuous)** |
| `duplicate_join_keys` | 0 | 0 (vacuously — no keys to duplicate) | OK (uninformative) |
| `integrity_ok` (run's own gate) | — | `true` | **Misleading** — see below |

**Why the run's own `integrity_ok=true` does not certify scientific validity**: reading
`scripts/run_family_a_observability_continuation_v1.py:168-183`, the `ok` flag is computed as
`len(failures) == 0 and len(scenario_summaries) == EXPECTED_FAMILY_A_TOTAL and (not
len(events_df) or events_df.duplicated(...).sum() == 0)` — it checks that the run *completed
mechanically* (no crashes, right scenario count, no duplicate keys *if* any events exist) but
**never asserts `n_events_total > 0`**. A run that silently finds zero disagreement events on
every single step passes this gate exactly as cleanly as a run that found genuine, correctly
instrumented events. This is the same reason the pre-launch test suite (§12 below) did not catch
the defect before the 3h21m production run was launched.

**Population reproduction check (the one part of this run that *is* independently verified
correct)**: `n_family_a_active_steps` summed over all 64 scenarios = 796,415, exactly matching the
parent study's `RANKING_FAIRNESS` "Steps evaluated" figure (796,415, per the frozen
`decision_criticality_timescale_trainval_v1_analysis_20260820.md` §3 table). This confirms the
scenario rebuild, `LiveHierarchicalRouterPolicy` replay, and `effective_regime == REGIME_A`
detection are all correctly reproducing the identical population — the defect is isolated to the
**shadow disagreement-detection step**, not the scenario/population machinery.

**Verdict: structurally invalid for scientific interpretation. Stopping per §1's instruction.
`NEED_RESULT_INTEGRITY_REPAIR`.**

---

## 3–11. Δ_same / Δ_native / continuation dependence / grouped prediction / feature ablations /
history / mechanism interpretation / concentration

**Not computed. Not applicable.** All of these sections require a nonempty disagreement-event
table; the run produced zero rows. Reporting any of these statistics — even as "0/0 undefined" —
would risk being read as a null scientific finding ("no material same-continuation effect") when
the actual cause is that no comparison was ever attempted. Per the task's own explicit
instruction, partial/degenerate "results" of this kind are not interpreted here.

---

## 12. Root-cause hypothesis (diagnostic only — no code touched)

Reading `src/llmserveopt/analysis/family_a_observability_continuation_v1.py`'s
`FamilyAObservabilityObserver.select_action` (lines 415–495) against the parent diagnostic's
analogous `ForkingObserverPolicy.select_action` (`decision_criticality_timescale_trainval_v1.py`
lines 692–786) surfaces a plausible, code-grounded explanation — offered as a hypothesis for the
eventual repair task, not as a fix applied here:

1. `real_action = self.inner_router.select_action(state)` runs **first** (line 416), exactly as in
   the parent diagnostic. This call delegates to whichever native policy Stage-2 selected, and —
   per the code's own comment (lines 424–441) — that policy mutates `state.gpu_states` in place as
   an admission-planning side effect (`gpu.active_request_ids.append(...)`,
   `gpu.current_kv_tokens += ...`; confirmed directly in
   `src/llmserveopt/policies/estimated_service_time_first.py:80-81`).
2. `saved_gpu_state` (line 442) is snapshotted **after** this mutation has already happened, not
   before it. `_restore()` (lines 446–451) therefore does not restore the true pre-decision GPU
   state — it repeatedly resets both shadow calls back to the **post-real-admission** baseline,
   i.e. GPUs whose free KV capacity and active-slot budget have already been at least partially
   consumed by the real router's own committed admission.
3. Both `self.shadow_policies[ESTF_ID].select_action(state)` and
   `self.shadow_policies[WFS_ID].select_action(state)` (lines 452, 454) then run against this
   *identical*, already-capacity-reduced baseline — not the original open decision point. Since
   both candidates greedily admit up to remaining GPU capacity in `_sort_key`/score order and stop
   when full, and both now see the same reduced (frequently near-zero) headroom, they are
   structurally biased toward producing the same (often empty) canonical admit set, so
   `actions_disagree()` (`canonical_action(a) != canonical_action(b)`) essentially never returns
   `True` — consistent with observing exactly 0 disagreements across all 796,415 active steps.
4. The parent diagnostic's own alt-candidate computation (`shadow_state = copy.deepcopy(state)` at
   `decision_criticality_timescale_trainval_v1.py:731`) has a related but *not identical*
   ordering issue (it also deep-copies `state` after the real router's mutation) — but it only ever
   computes **one** alt candidate against **one** already-committed baseline (the real router's
   own choice), which is a fundamentally different comparison than this diagnostic's two-candidate,
   both-against-the-same-contaminated-baseline design, and evidently still permitted disagreement
   to be detected in that study (3,545 events). This diagnostic's *additional* restore-between-two-
   shadow-calls step is the most direct candidate for the regression, but a definitive root cause
   would require running the fixed code, which is out of scope here.

**This is offered strictly as a hypothesis for the repair task, not verified by execution, and no
file was modified to test it.**

### Test-coverage gap (why this was not caught before the 3h21m run)

`tests/test_family_a_observability_continuation_v1.py` contains two tests that exercise this exact
code path — `test_family_a_diagnostic_deterministic_replay` (asserts
`len(res1.events) == len(res2.events)`) and
`test_family_a_events_have_stable_join_keys_and_bounded_branch_budget` (asserts
`len(res.events) <= fac.FULL_TRAJECTORY_BRANCHES_PER_SCENARIO`) — **neither asserts
`len(res.events) > 0`**. Both pass vacuously when `events` is always empty (`0 == 0`, `0 <= 3`).
The design doc's own pre-launch verification plan (§ "Pre-launch verification plan", item 7:
"Focused test suite... passes before the real 64-scenario run is launched") was satisfied by a
green test run that could not have distinguished "instrumentation works" from "instrumentation
silently finds nothing, always." This is the concrete reason the defect reached a completed
12,099-second production run undetected.

---

## 13. Exact classification

`NEED_RESULT_INTEGRITY_REPAIR` (per authorizing-task §1 — this pre-empts and overrides §13's
four-option enumeration, all of which presuppose a nonempty, structurally valid event population).

## 14. Exact next scientific step

**None of the four §14 options is a correct characterization of this outcome** — they all presume
Family-A evidence exists to act on. Stated plainly instead of force-fitting: the run's own
instrumentation must be repaired (see §12 hypothesis) and rerun before any of
`DESIGN_FAMILY_A_INTERPRETABLE_CHILD` / `DESIGN_STATEFUL_FAMILY_A_CONTROLLER` /
`INVESTIGATE_MISSING_STATE` / `STOP_FAMILY_A_SYNTHESIS` can be responsibly chosen. This report
recommends, as the concrete engineering step (outside this task's scope to execute): (a) fix the
snapshot/restore ordering so both shadow candidates are evaluated against the true pre-real-
admission `ObservableState`, not the post-admission one; (b) add a test asserting
`len(res.events) > 0` on a realistic Family-A fixture before any future long run is launched; (c)
rerun this diagnostic only after both land.

---

## 15. Novelty-aware interpretation

Not evaluable. The chain under investigation — *policy disagreement → downstream consequence →
local-action/continuation separation → mechanism attribution → standalone interpretable scheduler
→ positive marginal portfolio contribution* — had its first two links already established by the
prior decision-criticality study (disagreement occurs for Family A; downstream consequence is real
and TEST-corroborated, +0.886 mean completions/branch, +0.0302 ANWG). This run was intended to
extend the third link (mechanism attribution / continuation separation) but produced no usable
evidence toward it due to the defect in §12. **Number of chain links now empirically supported by
this run: 0 (unchanged from before this run; the two links already established stand only on the
prior study's own evidence, not on anything new from this run).**

## 16. Relation to prior Family-A result

None of the prior findings are contradicted, confirmed, or reinterpreted by this run — this run
produced no comparable evidence either way:

| Prior finding | Status after this run |
|---|---|
| Disagreement present in 68.75% of Family-A scenarios (44/64) | Neither confirmed nor refuted — this run's own independent re-derivation found 0/64, but per §12 that is attributed to an instrumentation defect in *this* run's shadow-comparison step, not a re-measurement of the same quantity under the same method. The two are not comparable as stated. |
| Consequential episodes median ~223 steps | Untouched — this run did not reach the point of measuring episode-level consequence. |
| Native-pair ceiling mean +0.886 completions/branch | Untouched — no branches were run in this diagnostic to compare against. |
| Prior standalone TEST-side audit +0.0302 ANWG | Untouched (correctly — TEST was never read here) and remains external corroboration only, unaffected by this run. |
| Prior advantage split ~61.4%/38.6% (native, chosen-policy direction) | Untouched — this run's `router_chosen_policy_id` field exists in the event schema but no events were captured to populate it. |

**None of the five prior figures survive or are reinterpreted here — they are simply not addressed
by this run's (empty) output.** Any future rerun after repair should be evaluated fresh against
these figures, not treated as having already superseded them.

---

## 17. Limitations

- This is a **TRAIN/VAL-only** diagnostic by design; nothing here says anything about TEST-split
  performance, and no TEST data was read in this run or this analysis.
- The Family-A scenario population (64 scenarios) is finite and fixed; even a repaired rerun would
  inherit the same population-size limitations the parent study already documented.
- The `FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS = 1500` bounded-rollout horizon and the "first 3
  disagreement events per scenario" sampling rule were never exercised by this run (0 events), so
  no branch-horizon-adequacy question arises from *this* run specifically.
- **This run's central limitation is not a modeling/observability limitation — it is a data-
  generation defect** (§12). No conclusion about Family-A's observability, continuation-
  dependence, or synthesis-readiness should be drawn from this run in either direction.
- Whether the next experiment (a repaired rerun) is justified: **yes** — the population-
  reproduction check (§2) shows the run's scenario/population/replay machinery is correct, and the
  defect appears narrowly isolated to the two-candidate shadow-comparison snapshot/restore
  ordering (§12), which is a scoped, well-understood-looking fix, not a redesign. This is a
  strong candidate for a fast repair-and-rerun rather than a redesign.

---

## 18. Artifact paths and reproducible commands

Source artifacts read (unmodified):
- `docs/design/FAMILY_A_OBSERVABILITY_CONTINUATION_DIAGNOSTIC_V1.md`
- `docs/current/decision_criticality_timescale_trainval_v1_analysis_20260820.md`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_integrity_report.json`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_v1_results.json`
- `experiments/family_a_observability_continuation_v1/family_a_scenario_summaries.csv`
- `src/llmserveopt/analysis/family_a_observability_continuation_v1.py`
- `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py` (comparison reference)
- `src/llmserveopt/policies/estimated_service_time_first.py` (mutation-side-effect confirmation)
- `scripts/run_family_a_observability_continuation_v1.py`
- `tests/test_family_a_observability_continuation_v1.py`

Reproducible commands (read-only; all figures in this report were derived this way):

```bash
# Top-level results (verbatim JSON fields) -- note the absence of any
# mean_delta_same / mean_delta_native / mean_continuation_dependence key
python3 -c "import json; print(json.dumps(json.load(open('experiments/family_a_observability_continuation_v1/family_a_observability_continuation_v1_results.json')), indent=2))"

# Integrity report
python3 -c "import json; print(json.dumps(json.load(open('experiments/family_a_observability_continuation_v1/family_a_observability_continuation_integrity_report.json')), indent=2))"

# Scenario-level zero-events confirmation + population-reproduction cross-check
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('experiments/family_a_observability_continuation_v1/family_a_scenario_summaries.csv')
print('total n_events:', df['n_events'].sum())
print('scenarios with n_events>0:', (df['n_events']>0).sum())
print('total n_family_a_active_steps:', df['n_family_a_active_steps'].sum())
print('scenarios with active_steps>0:', (df['n_family_a_active_steps']>0).sum())
PY
```

---

## Confirmation

No experiment was rerun. No TEST-split data was read. No simulator/scientific/analysis code was
modified (only read, for root-cause hypothesis purposes in §12). No files were staged, committed,
or pushed. This document is the only file created by this analysis.

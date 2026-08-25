# Family-A Receding-Horizon Oracle Feasibility V1

Date: 2026-08-20

Status: frozen before controller evaluation. This document is not modified after
scientific results are known, except to append a clearly marked execution record
at the end.

## 0. Scope And Non-Goals

This is an **oracle feasibility test**, not a deployability claim. The controller
uses the simulator itself as a perfect short-horizon transition model, evaluated
via counterfactual forks of the live simulator. Nothing here claims this is
cheaply reproducible in a real serving stack. This is NOT a classifier, NOT a
dwell/threshold tuning experiment, and NOT a retry of
`family_a_stateful_controller_v1` (that controller predicted the *sign of native
continuation advantage* from current state via a fitted tree; this controller
never fits anything — it *simulates* each candidate's own induced future
directly, every decision, and compares realized objective).

## 1. Scientific Hypothesis

`family_a_stateful_controller_v1` (frozen 2026-08-20,
`docs/current/family_a_stateful_controller_v1_analysis_20260820.md`) exercised
both ESTF/WFS modes deterministically but lost to fixed WFS on held-out
TRAIN/VAL ANWG (`STATEFUL_CONTROLLER_NO_GO`). It was not a mode-collapse or
thrashing failure. The repaired observability/continuation diagnostic
(`docs/current/family_a_observability_continuation_v1_repaired_analysis_20260820.md`)
established that Family-A's native ESTF/WFS advantage is `CONTINUATION_DOMINATED`:
the same-continuation local-action effect (`Delta_same`) is small and mostly
zero, while the native-continuation effect (`Delta_native`) is large and
common. The stateful controller's offline scorer was trained to predict
`sign(Delta_native)` from *current* observable state alone — but
`Delta_native` is a property of the trajectory induced by *which policy
continues afterward*, not a property of the pre-decision state in isolation.
Deployment changes which future states get visited (the switching policy's own
actions alter the state distribution it is scored on), so a state-only
predictor of a continuation-dependent quantity is exactly the kind of target
expected to generalize poorly online, matching what was observed.

**Hypothesis under test**: if a controller explicitly evaluates the future
trajectory *induced by each candidate's own actions* (a short-horizon
counterfactual rollout / receding-horizon / MPC-style evaluation), rather than
predicting the native-continuation sign from static state, does the Family-A
adaptive opportunity become exploitable in closed loop?

This is an oracle test: the "transition model" is the real simulator itself
(via `fork_from_live_simulator`/`LiveFork`, already proven non-interfering by
the repaired observability diagnostic and the decision-criticality diagnostic).
If oracle-style planning has no value, learned approximations are not
justified. If it has value, the interesting scientific claim is narrow: the
prior failure was specifically caused by neglecting policy-induced future-state
evolution, not that MPC/receding-horizon control is itself novel (it is not).

## 2. Reused Infrastructure (no reimplementation)

All imported unmodified from
`src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py` (`dcm`)
and `src/llmserveopt/analysis/family_a_observability_continuation_v1.py` (`fac`):

- `fac.load_family_a_trainval_scenario_table()` — the exact 64 Family-A
  TRAIN/VAL scenarios (54 train / 10 val) used by both prior Family-A studies.
- `fac.rebuild_scenario_from_row(row)` — scenario reconstruction.
- `dcm.fork_from_live_simulator(sim, policy=..., policy_id=..., first_action=...)`
  and `dcm.LiveFork.advance_one_step(...)` — isolated, deep-copied simulator
  forks driven only via the simulator's own unmodified
  `_apply_action`/`_advance_decode`/`_build_observable_state`. `sim` itself is
  never mutated. Chaining is supported: a `LiveFork.shell` is itself a
  `Simulator` instance, so `fork_from_live_simulator(fork.shell, ...)` forks
  *from* an existing fork to implement the H-step-then-common-continuation
  chain (never previously exercised this way, but uses only the existing,
  already-verified fork constructor with no modification).
- `dcm.run_bounded_rollout(sim, policy=..., policy_id=..., first_action=..., max_extra_steps=...)`
  — bounded single-segment rollout (used as-is for the H=1 arm and as the
  template for the chained rollout below).
- Candidate-region / eligibility gate: `snapshot_gpu_counters`,
  `restore_gpu_counters`, `canonical_action`, `actions_disagree` from
  `src/llmserveopt/policies/family_a_stateful_controller_v1.py` — the
  identical mechanism `FamilyAStatefulControllerV1._candidate_region` already
  uses (snapshot true pre-decision GPU counters, call ESTF, restore, call WFS,
  restore, compare canonical admit sets).
- `EstimatedServiceTimeFirstPolicy`, `WeightedFairSharePolicy` — unmodified
  parent policies.
- `RunMetrics`/`CompletedRequest` semantics
  (`src/llmserveopt/core/metrics.py`) for the frozen window-objective
  definition (SS5 below) and for full-trajectory ANWG.

**Determinism**: verified by source inspection that neither
`simulator.py`/`service_model.py` nor
`estimated_service_time_first.py`/`weighted_fair_share.py`/
`policy_library_v2_helpers.py` use any random-number source. There is no RNG
state to snapshot/restore for rollout isolation — confirmed by grep, not
assumed.

## 3. Snapshot/Restore Integrity (P0)

The 2026-08-20 repair
(`docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md`)
found that `select_action` on one native policy mutates
`state.gpu_states[i].active_request_ids` and `.current_kv_tokens` in place as
same-call admission bookkeeping, and that snapshotting *after* the real
router's own decision silently biased both shadow candidates toward an
already-capacity-consumed baseline. This controller avoids that class of bug
structurally:

- The eligibility/candidate-gate probe (SS4) snapshots the true pre-decision
  `ObservableState` GPU counters **before** calling either shadow policy, and
  restores after each of the two probe calls — identical, reused mechanism to
  `FamilyAStatefulControllerV1._candidate_region` (already regression-tested).
- The rollout branches never touch the live `ObservableState` object at all —
  they operate on `dcm.fork_from_live_simulator`'s deep-copied `Simulator`
  shell (`fork.shell`), which is a structurally separate object graph from
  `sim` (`_gpus`, `_waiting`, `_migrating`, `_relocating`,
  `_pending_arrivals` suffix are all `copy.deepcopy`'d — see `dcm`'s own
  module docstring and `fork_from_live_simulator` implementation). The real
  `sim` is provably never mutated by rollout evaluation: this is the exact
  same isolation guarantee already exercised at scale (768 four-branch
  rollouts) by the repaired observability diagnostic and by the
  decision-criticality diagnostic's own full-trajectory branches.
- Candidate A (ESTF rollout) and Candidate B (WFS rollout) are each built from
  an independent `fork_from_live_simulator(sim, ...)` call against the same,
  unmutated `sim` — so candidate A cannot affect candidate B, and neither can
  affect the real trajectory.
- After planning, the controller returns exactly one of the two
  already-computed first actions (`action_estf` or `action_wfs`) to the real
  `Simulator.run()` loop, which applies it through the simulator's own
  unmodified `_apply_action` — the real trajectory is therefore driven by the
  *actual* simulator, not a copy, at every real step.
- No feature of a rollout branch (completion counts, objective values, which
  candidate won) is ever read by the eligibility gate for a *later* step,
  and no branch touches a request object shared with the real trajectory
  (deep-copied per fork). No future information leaks into the controller
  beyond the explicit, declared rollout.
- Policy-specific persistent state: `EstimatedServiceTimeFirstPolicy` and
  `WeightedFairSharePolicy` are stateless across calls (verified by reading
  both classes — no instance attributes mutated by `select_action` beyond the
  passed-in `state`), so no persistent-state copy/reset is required for the
  shadow/rollout policy instances; fresh instances are still constructed once
  per scenario for defensiveness and are `reset()` between scenarios exactly
  like every other harness in this repo.

Regression tests proving all of the above (SS9) are required to pass before
any TRAIN/VAL scenario is run (integrity gate, SS12).

## 4. Eligibility / Candidate Region And Fallback

Identical definition to `FamilyAStatefulControllerV1._candidate_region`:
at a decision step, snapshot the true pre-decision GPU counters, call
`EstimatedServiceTimeFirstPolicy.select_action(state)`, restore, call
`WeightedFairSharePolicy.select_action(state)`, restore. If
`actions_disagree(action_estf, action_wfs)` is `True`, the step is **eligible**
for rollout planning. Both already-computed actions (`action_estf`,
`action_wfs`) are reused as the two candidates' first actions — no
recomputation.

**Fallback** (outside the candidate region, or if rollout evaluation raises):
execute fixed **WFS**'s action directly (no rollout). Justification, per the
task's own instruction: WFS was the strongest fixed Family-A parent in the
prior closed-loop TRAIN/VAL evaluation (`family_a_stateful_controller_v1`
full-simulation mean ANWG: WFS `0.7478` > ESTF `0.7296`). WFS is also the
fairness-preserving default used as the stateful controller's `initial_mode`.
An empty queue (`not state.waiting_queue and not state.migrating_queue`) is
also treated as non-eligible (nothing to plan over) and falls back to WFS
directly, matching the prior controller's convention.

## 5. Rollout Objective (frozen before any run)

`arrival_normalized_weighted_goodput` (ANWG) has denominator
`arrival_weight` = the weighted sum over **all** arrivals in the full scenario
(`_arrival_weight_denominator`, `src/llmserveopt/core/metrics.py`). This
denominator is a whole-scenario constant, not decomposable to a bounded
rollout window that ends before scenario completion — using it inside a
partial-horizon rollout would silently change the "arrival population" the
score is normalized against between calls at different points of the
trajectory, which is not mathematically meaningful. Per SS7 of the task
instructions, this is not invented; instead the window objective reuses only
already-established quantities from `RunMetrics`/`CompletedRequest`:

**Window Weighted SLO Goodput**: for a bounded rollout window (a set of newly
`CompletedRequest` objects realized strictly during that fork's own advance
calls, exactly the set `dcm.LiveFork` already tracks via
`completed_in_window`/`shell._completed` growth), the objective is

```
W(branch) = sum_i  weight_i * 1[not slo_violated_i]      for i completed in branch's window
weight_i = request.priority if request.priority > 0 else 1.0
slo_violated_i = completion_time_i > slo_deadline_i
```

This is exactly the *numerator* of `weighted_goodput`/ANWG
(`compute_metrics`'s `success_weight` computation), restricted to one bounded
window instead of the whole scenario, and is well-defined because: (a) both
candidate branches at a decision step start from the *identical* snapshotted
pre-decision state and therefore face the identical residual population and
identical future-arrival stream (arrivals are policy-independent), so the
comparison is apples-to-apples without needing a common denominator; and (b)
it only sums quantities (`priority`, `slo_deadline`, `completion_time`)
`RunMetrics` already treats as primary. Raw (unweighted) `completed_count`
(what `family_a_observability_continuation_v1`'s `Delta_same`/`Delta_native`
used) is retained as a secondary/diagnostic quantity only, not the selection
criterion, since the primary project utility is ANWG and Family-A scenarios
carry per-class weights (`tenant_weight_skew`) this window objective must be
sensitive to.

**Full-trajectory reporting** (SS8) uses the standard, unmodified
`RunMetrics.arrival_normalized_weighted_goodput` computed once per full
real-scenario run — never a rollout-window quantity.

## 6. Horizons And Terminal/Common-Continuation Semantics

`H in {1, 5, 20}`. `H=1` is the local-action control (one candidate-controlled
step). `H=5` tests short future dependence. `H=20` matches the prior
controller's frozen dwell/reaction timescale
(`DWELL_MINIMUM_STEPS`/`min_dwell_steps=20`).

To avoid the horizon-truncation bias flagged in SS8 of the task (a short
rollout can favor a candidate that grabs immediate completions while leaving a
worse queue behind), **every** horizon uses the same terminal-handling
structure, so that `H` differences are isolated to exactly "how many steps the
candidate itself controls before a shared, fixed continuation takes over":

```
candidate branch(H) =
    fork_from_live_simulator(sim, policy=candidate, first_action=candidate's
        precomputed first action)          # step 1, candidate-controlled
    -> advance_one_step() under `candidate` policy, up to H steps total
       (fewer if the fork naturally drains)
    -> from the resulting fork.shell, fork again
       (fork_from_live_simulator(fork.shell, policy=COMMON_CONTINUATION, ...))
    -> advance_one_step() under COMMON_CONTINUATION for up to
       CONTINUATION_BUDGET further steps (fewer if it naturally drains)

objective(branch) = W(candidate segment) + W(continuation segment)
```

`COMMON_CONTINUATION = WeightedFairSharePolicy` — the same frozen fallback
default (SS4), applied identically to both the ESTF-first and WFS-first
branches at every `H`, so it cannot itself create an ESTF/WFS preference.
Per the task's explicit instruction not to build an enormous branch tree, this
is the **single** common-continuation formulation (not the further
`ESTF-common-ESTF`/`WFS-common-ESTF` symmetrization the task offers as
optional) — the smallest formulation that still controls for
terminal/truncation bias identically across `H`.

`CONTINUATION_BUDGET = 200` steps, frozen a priori (not tuned on any
observed result): large enough relative to `H<=20` to expose near-term
consequence beyond the candidate segment (10x the largest `H`), small
relative to Family-A's median active episode length (223 steps, p90 662 —
`decision_criticality_timescale_trainval_v1` §4) so it stays a *bounded*
proxy rather than a near-full-scenario oracle, and small enough to keep total
rollout cost tractable across up to 3 horizons x 64 scenarios x many
eligible decision points.

`H=1` therefore is not "no future information" — it is "future information
only through the shared WFS default," isolating the marginal value of letting
the *candidate itself* (rather than WFS) control 1 vs. 5 vs. 20 of the next
steps. This is the direct, controlled test of SS21's causal question ("does
H>1 outperform H=1?").

## 7. Replanning Semantics

Because `Simulator.run()` already calls `policy.select_action(state)` with the
**true** current state at every real scheduling step, receding-horizon
replanning falls out for free from the existing harness: at each eligible
step the controller plans (SS4-6), executes **only the first action of the
winning candidate** in the real trajectory (never the full H-step rollout),
and returns control to the simulator. The next call to `select_action`
necessarily reflects the true post-action state (including anything a
different, unplanned-for arrival or completion changed), so "observe actual
state -> replan" happens automatically, once per real step, with no extra
bookkeeping. Ties (`W(ESTF branch) == W(WFS branch)`, exact float equality is
not expected but handled) resolve to the WFS action, consistent with the
frozen fallback.

## 8. Candidate Selection Rule

At each eligible step: run both branches (ESTF-first, WFS-first) from the
identical snapshot; execute `action_estf` if
`objective(ESTF branch) > objective(WFS branch)`, else execute `action_wfs`
(covers both the WFS-favored and tie cases, matching the WFS-default
fallback).

## 9. Non-Interference Regression Tests (required before any TRAIN/VAL run)

Synthetic, deterministic tests in
`tests/test_family_a_receding_horizon_oracle_v1.py`:

A. `H=1` reproduces the immediate-effect behavior expected from a
   single-step-lookahead comparison (a constructed fixture where the two
   candidates' single-step window objectives are known analytically).
B. A constructed case where the immediate/local choice looks better under
   `H=1` (or under raw first-action completions) but a longer horizon
   (`H=5` or `H=20`) correctly reverses the choice because of a worse
   induced future (e.g., one candidate's first action starves a
   high-priority request that later misses its deadline).
C. A constructed case where the rollout-aware controller chooses ESTF.
D. A constructed case where the rollout-aware controller chooses WFS.
E. Replanning reverses a prior choice after the observed real state changes
   between two consecutive real steps.
F. Counterfactual planning leaves the real simulator state and the real
   `ObservableState` object byte-identical to what an unwrapped WFS/ESTF
   call would have left (fingerprint-style comparison, mirroring
   `dcm._state_fingerprint` and the repair audit's own non-interference
   test methodology).
G. `fork_from_live_simulator` chaining (candidate segment -> common
   continuation segment) is itself non-interfering: forking from
   `fork.shell` never mutates `fork.shell` in a way that would be visible if
   `fork.shell` were used again (defensive; the design does not reuse
   `fork.shell` after chaining, but the invariant is still asserted).
H. The eligibility gate is exactly `actions_disagree(action_estf, action_wfs)`
   evaluated from the identical pre-decision snapshot (byte-for-byte reuse
   of the already-tested `family_a_stateful_controller_v1` mechanism) —
   asserted directly rather than re-derived.

## 10. Safety / Fairness Metrics (tracked, not optimized against)

Per full real-scenario run, using unmodified `RunMetrics`: `completion_fraction`,
`weighted_completion_fraction`, `p95_latency`, `p95_queuing_delay`,
`slo_violation_rate`, `mean_latency`/`median_latency` for context. A rollout
controller is not counted as successful if any of these collapse relative to
**both** fixed parents (tolerance: no more than 0.02 absolute degradation in
`completion_fraction`/`weighted_completion_fraction` below
`min(mean_ESTF, mean_WFS)`, and no more than 0.02 absolute increase in
`slo_violation_rate` above `max(mean_ESTF, mean_WFS)` — frozen numeric
tolerances, chosen a priori as small round numbers consistent with the
existing repo convention of reporting these metrics at 2-4 decimal places,
not tuned on any observed result).

## 11. TRAIN/VAL Only

Uses exactly `fac.load_family_a_trainval_scenario_table()` — the same 64
Family-A TRAIN/VAL scenarios (54 train / 10 val) as
`family_a_observability_continuation_v1` and `family_a_stateful_controller_v1`.
No TEST scenario is loaded, generated, or read. No TEST metric from any prior
study is used to pick `H`, the continuation budget, or any other design
parameter (`CONTINUATION_BUDGET=200`, `H in {1,5,20}`, and all thresholds in
SS10/SS13 are frozen in this document before the run). `assert_trainval_only`
(reused from `dcm`) guards every scenario-row entry point.

## 12. Pre-Run Integrity Gate

Before the full TRAIN/VAL run, require ALL of:

- All tests in `tests/test_family_a_receding_horizon_oracle_v1.py` pass.
- SS9.F (non-interference) passes on a real Family-A TRAIN scenario, not just
  a synthetic fixture.
- No TEST-split reference anywhere in the new module/script (grep check).
- A small pilot (2-3 TRAIN scenarios, all three horizons) completes with 0
  failures and produces a nonzero number of eligible/candidate decisions, so
  the eligibility gate is confirmed non-vacuous before committing to the full
  64-scenario x 3-horizon run.
- Deterministic reproducibility: running the same scenario+horizon twice
  yields bit-identical per-scenario ANWG and identical controller decision
  logs.
- The controller genuinely replans (decision log shows more than one planning
  call per scenario in the pilot, with plans keyed to distinct real states).

If any of these fail: **`RECEDING_HORIZON_INTEGRITY_NO_GO`**, and the full run
is not launched.

## 13. Baselines (six comparison arms + envelope context)

1. Fixed ESTF.
2. Fixed WFS.
3. `family_a_stateful_controller_v1` (refit identically: `DecisionTreeClassifier`,
   `max_depth=3`, `class_weight="balanced"`, `random_state=20260820`, on the
   same 91 repaired events, `min_dwell_steps=20`, thresholds 0.65/0.35 —
   byte-identical recipe to the frozen prior script).
4. `H=1` rollout controller (this design).
5. `H=5` rollout controller.
6. `H=20` rollout controller.
7. (context only, not a comparison arm) native ESTF/WFS envelope per scenario:
   `max(ANWG_ESTF, ANWG_WFS)`.

No VTC/FSP/vLLM-LTR/PARS/SCORPIO integration in this internal oracle
feasibility run, consistent with both prior Family-A studies.

## 14. GO/NO_GO Criteria (frozen)

Let `best_fixed_mean = max(mean ANWG_ESTF, mean ANWG_WFS)` and
`oracle_gap = mean(native_pair_envelope) - best_fixed_mean` (only defined,
and `recovered_fraction` only reported, when `oracle_gap > 0`).

**`RECEDING_HORIZON_POSITIVE_SIGNAL`** requires all of:
- best short-horizon arm (`max` over `H=5,H=20` mean ANWG) exceeds the `H=1`
  arm's mean ANWG, AND has more per-scenario wins than losses vs `H=1`.
- The best short-horizon arm's mean ANWG exceeds `best_fixed_mean`, AND has
  more per-scenario wins than losses vs the per-scenario best-fixed parent.
- Gain is not pathologically concentrated: no single TRAIN/VAL scenario
  accounts for more than 50% of the total positive
  `(controller - best_fixed)` mass across scenarios with a positive diff.
- Safety/fairness tolerances (SS10) hold.
- `oracle_gap > 0` and `recovered_fraction > 0` (some native envelope gap is
  recovered).

**`RECEDING_HORIZON_MIXED_SIGNAL`**: the short-horizon arm improves over
`H=1` (first bullet above holds) but either does not clearly beat
`best_fixed_mean`, or does but with a small/unstable/concentrated margin
(fails the concentration or win-margin bullets), while safety holds.

**`RECEDING_HORIZON_NO_GO`**: `H>1` does not improve over `H=1`
(mean ANWG and win/loss both fail to show improvement), or the best arm still
loses to `best_fixed_mean`, or safety/fairness tolerances are violated.

**`RECEDING_HORIZON_INTEGRITY_NO_GO`**: SS12 gate fails.

## 15. Computational-Cost Reporting

Per scenario and per horizon, record: number of eligible/candidate decisions,
number of rollout branches evaluated (2 per eligible decision), total
simulator steps spent on rollout planning (candidate segment + continuation
segment, both branches), wall-clock seconds for planning vs. total scenario
wall-clock, and a hard per-scenario planning-call safety cap
`MAX_PLANNING_CALLS_PER_SCENARIO` (set from the pilot in SS12, documented
in the execution record below, not tuned on TRAIN/VAL scientific outcomes —
purely a runtime-tractability bound; if hit, remaining real steps in that
scenario fall back to WFS directly and this is logged and reported as a
limitation, never silently).

## 16. Deliverables

- `src/llmserveopt/policies/family_a_receding_horizon_oracle_v1.py`
- `scripts/run_family_a_receding_horizon_oracle_v1.py`
- `tests/test_family_a_receding_horizon_oracle_v1.py`
- `experiments/family_a_receding_horizon_oracle_v1/`
- `docs/current/family_a_receding_horizon_oracle_v1_analysis_20260820.md`

---

## Execution Record (appended after results are known)

See `docs/current/family_a_receding_horizon_oracle_v1_analysis_20260820.md` for
the full scientific record. This section intentionally left for a short
pointer only, appended post-hoc.

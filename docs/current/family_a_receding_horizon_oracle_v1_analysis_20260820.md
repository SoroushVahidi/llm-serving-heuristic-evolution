# Family-A Receding-Horizon Oracle Feasibility V1 -- Analysis

Date: 2026-08-20

## 1. Executive Verdict

Classification: **`RECEDING_HORIZON_MIXED_SIGNAL`**.

Next step: **`DIAGNOSE_ROLLOUT_VALUE_LIMIT`**.

Explicit action-aware short-horizon planning (H=5, H=20) modestly outperforms the
purely-local H=1 rollout controller and clearly outperforms both fixed ESTF and
the previously-failed supervised stateful controller. But **no rollout horizon
beats fixed WFS** on TRAIN/VAL mean ANWG, and the gains that do exist over H=1
are small, non-monotonic in H, and concentrated in a handful of scenarios. The
oracle-gap "recovered fraction" is negative for every horizon: none of the
rollout controllers recover any of the native ESTF/WFS envelope gap over the
best fixed parent -- they all sit *below* the best fixed parent, not between it
and the envelope.

This is a real, measured, non-zero signal that future-aware planning changes
the ESTF/WFS choice in a horizon-dependent way (WFS's share of chosen
decisions drops from 73.5% at H=1 to 54.8% at H=20 -- the rollout genuinely
starts trusting ESTF more as it can see further), but that shift does not
translate into a closed-loop win over the already-strong fixed WFS baseline on
this scenario population.

## 2. Relation To The Prior Failure

`family_a_stateful_controller_v1` predicted `sign(Delta_native)` -- a
continuation-dependent quantity -- from *static* pre-decision state via a
fitted tree, and lost to fixed WFS in closed loop
(`docs/current/family_a_stateful_controller_v1_analysis_20260820.md`: mean
ANWG 0.7281 vs WFS 0.7478). This experiment replaces that static predictor
with a direct oracle simulation of each candidate's own induced future. The
result: even with a perfect (oracle) short-horizon transition model, all
three horizons (0.7360 / 0.7380 / 0.7361) beat the *failed* stateful
controller (0.7281) and beat fixed ESTF (0.7296) but still fall short of
fixed WFS (0.7478). The hypothesis that "explicit future-state modeling
recovers the opportunity" is **partially supported** (oracle planning beats
both the naive local baseline and the prior failed approach) but **not fully
supported** (it does not close the gap to the strongest fixed parent). This
narrows the failure explanation: neglecting policy-induced future-state
evolution was A real problem (fixing it recovers value over ESTF and over the
prior controller), but it was not the ONLY problem preventing a Family-A
adaptive controller from beating WFS -- see SS9 (interpretation) below.

## 3. Integrity

- All 15 new regression tests pass (`tests/test_family_a_receding_horizon_oracle_v1.py`),
  covering: eligibility gate byte-identical to the reused
  `family_a_stateful_controller_v1` mechanism; fallback to WFS outside the
  candidate region and on an empty queue; H=1 with zero continuation
  reproduces a purely local window-objective comparison; a constructed case
  where H=1 myopically favors ESTF but H=5 correctly reverses to WFS;
  rollout can choose either ESTF or WFS; replanning reverses a prior choice
  after the real state changes; chained-fork rollouts never mutate the
  source simulator (fingerprint-identical before/after, order-independent,
  deterministic); the planning-call safety cap falls back to WFS and is
  reported; the window-objective arithmetic is correct.
- The full existing Family-A test suite (44 tests: 11
  `test_family_a_stateful_controller_v1.py` + 18
  `test_family_a_observability_continuation_v1.py` + 15 new) passes
  unchanged -- no regression to frozen prior modules.
- No RNG exists anywhere in the simulator, service model, or ESTF/WFS policy
  code (confirmed by source inspection), so there is no random state to
  snapshot/restore for rollout isolation.
- Rollout branches are built exclusively via
  `dcm.fork_from_live_simulator`/`LiveFork.advance_one_step`
  (`decision_criticality_timescale_trainval_v1.py`), reused unmodified,
  including a new (previously unexercised, but structurally supported)
  second-level fork: `fork_from_live_simulator(candidate_fork.shell, ...)`
  to implement the H-step-candidate -> common-continuation chain. This is
  proven non-interfering by direct `dcm._state_fingerprint` comparison
  before/after chained-branch evaluation, and by order-independence
  (ESTF-branch-then-WFS-branch gives identical WFS-branch results to
  WFS-branch-first).
- The full 64-scenario x 3-horizon run completed with **0 failures**,
  0 planning-call-cap hits (max observed eligible decisions per scenario:
  107, cap: 150), and reproducible per-scenario outcomes.
- No TEST scenario or TEST metric was read, generated, or used at any point
  (`assert_trainval_only` guards every scenario-row entry point; the
  Family-A TRAIN/VAL table load asserts no TEST row and exact count 64).

## 4. Exact Receding-Horizon / MPC Formulation

At each real scheduling step: snapshot the true pre-decision GPU counters,
compute `action_estf`/`action_wfs`, restore between calls (byte-identical
mechanism to `family_a_stateful_controller_v1._candidate_region`). If they
disagree (eligible), fork the LIVE simulator twice from the identical
snapshot: "ESTF drives up to H steps, then WFS (common continuation) drives
up to 200 further steps" vs. "WFS drives up to H steps, then WFS drives up
to 200 further steps." Execute only the already-computed first action of
whichever branch realizes the higher windowed objective (SS5); return
control to the real simulator, which calls `select_action` again next real
step with the true new state -- replanning is therefore free, exactly once
per real step, with no extra bookkeeping. Outside the candidate region or on
an empty queue: execute fixed WFS directly (no rollout).

## 5. Horizons

`H in {1, 5, 20}`, as frozen. `COMMON_CONTINUATION_BUDGET = 200` steps,
applied identically after every candidate segment at every `H` (never
symmetrized with an ESTF common continuation, per the frozen "smallest
defensible formulation" decision).

## 6. Rollout Objective

`W(branch) = sum(weight_i * 1[not slo_violated_i])` over requests newly
completed within the branch's own bounded window (candidate segment +
common continuation), where `weight_i = request.priority` (or 1.0 fallback)
-- the numerator of `weighted_goodput`/ANWG restricted to one apples-to-apples
window (both candidates share an identical pre-decision snapshot and future
arrival stream). Full-trajectory reporting below always uses the standard,
unmodified `RunMetrics.arrival_normalized_weighted_goodput`, never a
rollout-window quantity.

## 7. Baseline Results (TRAIN/VAL, 64 scenarios, 54 train / 10 val)

| Policy | Mean ANWG | Median ANWG |
|---|---:|---:|
| Fixed WFS | **0.74777** | 0.74583 |
| Fixed ESTF | 0.72962 | 0.77500 |
| `family_a_stateful_controller_v1` (V1, refit identically) | 0.72813 | 0.76250 |
| RHO H=1 | 0.73600 | 0.74167 |
| RHO H=5 | **0.73795** (best RHO) | 0.74167 |
| RHO H=20 | 0.73607 | 0.75417 |
| Native ESTF/WFS envelope (per-scenario max, context only) | 0.76255 | -- |

Fixed WFS remains the strongest single policy on mean ANWG, matching the
prior `family_a_stateful_controller_v1` study's finding exactly. All three
oracle rollout controllers beat fixed ESTF and beat the previously-failed
stateful controller on mean ANWG, but none beats fixed WFS. Interestingly,
fixed ESTF has the *highest median* ANWG (0.775, above WFS's 0.746) despite
the lowest mean -- ESTF's distribution is right-skewed with a worse tail that
pulls its mean down; H=20's median (0.754) sits closest to ESTF's typical
per-scenario behavior among the RHO variants.

## 8. H=1 vs H>1 (the causal test)

| Comparison | Mean diff | Wins | Ties | Losses |
|---|---:|---:|---:|---:|
| H=5 vs H=1 | **+0.00195** | 10 | 50 | 4 |
| H=20 vs H=1 | +0.00008 | 27 | 20 | 17 |

**H>1 does modestly outperform H=1**: H=5 has more than double the wins over
losses against H=1 (10 vs 4, mean diff positive and the largest of any
horizon-pair comparison), and H=20 also has more wins than losses (27 vs 17)
though its mean-diff edge over H=1 is nearly zero. This is evidence that
explicit multi-step self-control by the candidate (vs. handing off to the
common WFS continuation almost immediately) does add value beyond a purely
1-step lookahead -- but the effect is small in magnitude, not monotonic in H
(H=5 > H=20 on mean ANWG), and 50-78% of eligible decisions tie exactly
across horizons (the horizon only matters when the candidate's own multi-step
admission ordering diverges from what the common WFS continuation would have
done anyway). The horizon-dependent choice-fraction shift is larger and more
monotonic than the ANWG effect: WFS's share of chosen decisions falls from
73.5% (H=1) to 72.0% (H=5) to 54.8% (H=20) -- the rollout genuinely trusts
ESTF more as horizon grows, but that shifted trust does not translate into a
correspondingly large ANWG gain.

## 9. Best-Fixed-Parent Comparison

| Horizon | Mean diff vs best fixed | Wins | Ties | Losses |
|---|---:|---:|---:|---:|
| H=1 | -0.02655 | 3 | 19 | 42 |
| H=5 | -0.02460 | 4 | 22 | 38 |
| H=20 | -0.02648 | 5 | 25 | 34 |

Every horizon loses to the per-scenario best fixed parent on both mean ANWG
and win/loss count, by a wide margin (34-42 losses out of 64 scenarios).
H=20 has the best win/loss ratio (5/34) and the fewest losses, consistent
with its choice-fraction shift toward ESTF, but still clearly loses overall.

## 10. Envelope-Gap Recovery

`oracle_gap = native_pair_envelope_mean (0.76255) - best_fixed_mean (0.74777) = +0.01477`.

`recovered_fraction`: H=1 = **-0.797**, H=5 = **-0.665**, H=20 = **-0.792**.

All three are strongly negative: rather than recovering part of the gap
between the best fixed parent and the native envelope, every rollout
controller sits *below* the best fixed parent itself. There is no partial
recovery to report.

## 11. Safety / Fairness

`completion_fraction` and `weighted_completion_fraction` are exactly 1.0 for
every policy (no completion collapse anywhere). `p95_latency`
(15.47-15.90 for RHO variants) and `p95_queuing_delay` (14.96-15.39) sit
between ESTF's (14.71 / 14.18) and WFS's (16.03 / 15.57) values -- no
unpaired regression beyond either parent. `slo_violation_rate`
(0.247-0.256) is close to WFS's (0.254) and modestly above ESTF's (0.237),
well within the frozen 0.02-absolute safety tolerance relative to
`max(ESTF, WFS)`. **All safety/fairness tolerances hold**; the rollout
controllers do not trade fairness/completion behavior for their (modest)
ANWG gain over ESTF.

## 12. Computational Overhead

| Horizon | Total planning calls (64 scen.) | Mean eligible/scenario | Mean scenario wall-clock (s) | Total wall-clock (s) | Cap hits |
|---|---:|---:|---:|---:|---:|
| H=1 | 2,707 | 42.3 | 3.15 | 201.7 | 0 |
| H=5 | 2,736 | 42.8 | 3.17 | 203.1 | 0 |
| H=20 | 2,752 | 43.0 | 3.12 | 199.8 | 0 |

Full end-to-end run (6 policies x 64 scenarios, all provenance/IO included):
**838.7 s (~14 min)**, 0 failures. The frozen safety cap
(`MAX_PLANNING_CALLS_PER_SCENARIO=150`) was never hit (max observed eligible
decisions in one scenario: 107). Notably, planning cost is essentially flat
across H -- the fixed 200-step common continuation dominates rollout cost at
every horizon, so H itself is cheap to vary once the continuation budget is
paid.

## 13. Induced State-Distribution Comparison (descriptive)

Computed from `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_state_distribution.json`
(fixed ESTF, fixed WFS, `family_a_stateful_controller_v1`, and RHO H=20,
across all 64 scenarios, using the simulator's own already-tracked per-step
`_waiting_queue_history`/`_util_history`/`_batch_history`; mean-of-per-scenario-means).

| Policy | Mean queue length | Median queue length | p90 queue length |
|---|---:|---:|---:|
| Fixed ESTF | **9.93** (shortest) | 10.45 | 15.12 |
| `family_a_stateful_controller_v1` | 10.07 | 10.45 | 15.70 |
| RHO H=20 | 10.67 | 10.95 | 16.28 |
| Fixed WFS | **11.66** (longest) | 11.47 | 17.83 |

`active_batch_size` and `kv_utilization` are statistically identical across
all four policies (Family A's GPUs all have `max_active_sequences=1`, so
with exactly one admission slot per GPU, batch size/KV occupancy is
essentially saturated/deterministic regardless of *which* request holds the
slot -- this is a real structural property of the scenario family, not a
measurement artifact). Queue length is therefore the only descriptive
state-distribution signal available at this level, and it does differ
meaningfully: ESTF drains the queue fastest (shortest-service-first
throughput), WFS lets it build up the most (subordinating throughput to
priority/fairness), and the RHO-H=20 controller's induced queue-length
distribution sits **closer to WFS's fuller-queue regime than to ESTF's**,
even though it chooses ESTF's action on 45% of eligible decisions (SS8) --
i.e. the *state trajectory* the oracle controller navigates resembles WFS's
regime more than its own action-choice frequency alone would suggest. This
is consistent with (not proof of) the hypothesis that a controller's
realized state distribution is a distinct, load-bearing quantity from its
per-decision action mix -- exactly the mechanism this experiment was
designed to make visible, though the previously-failed
`family_a_stateful_controller_v1` induces a queue-length distribution
*closer to ESTF* than the oracle controller does, despite both being
Family-A ESTF/WFS switchers. Descriptive only; no causal claim.

## 14. Classification

**`RECEDING_HORIZON_MIXED_SIGNAL`**

Per the frozen SS14 criteria: the best short-horizon arm (H=5) improves over
H=1 (mean ANWG higher, wins 10 > losses 4) and safety holds, but it does not
beat `best_fixed_mean` (0.738 < 0.748) and its gain over best-fixed is
concentrated (top-1 scenario carries 57% of its positive mass over best
fixed, exceeding the 50% not-concentrated threshold) -- exactly the MIXED
definition ("future-aware planning helps relative to H=1 but gain vs best
fixed is small/unstable/concentrated").

## 15. Limitations

- Oracle only: this uses the real simulator as a perfect transition model at
  planning time, unavailable cheaply in a production server. No deployability
  claim is made.
- `COMMON_CONTINUATION_BUDGET=200` and the single WFS-only common
  continuation (not symmetrized with an ESTF common continuation) are one
  frozen, defensible choice; a different terminal-handling scheme could in
  principle shift the H=1-vs-H>1 or vs-best-fixed comparison.
- TRAIN/VAL only (54 train / 10 val); no TEST, public-trace, or real-serving
  validation.
- Wins over best-fixed are concentrated in a small number of scenarios
  (mostly `skew5.0000.favlong.noise0.00`-family scenarios for H=1/H=5;
  a more varied but still small set for H=20), consistent with (but not
  identical to) the repaired observability diagnostic's finding that native
  Family-A value concentrates in `favlong`/higher-skew scenarios.
- The window objective (SS6) is a considered, documented, non-outcome-tuned
  surrogate for ANWG, but it is still a surrogate, not ANWG itself; a
  different valid surrogate could shift branch preferences at the margin.
- No VTC/FSP/vLLM-LTR/PARS/SCORPIO external baselines are integrated.

## 16. Next Step

**`DIAGNOSE_ROLLOUT_VALUE_LIMIT`**

The oracle recovers value over ESTF and over the prior failed controller but
still loses to fixed WFS. Before considering any learned approximation
(which would only ever underperform this oracle ceiling), the open question
is *why* oracle planning itself caps out below WFS here -- e.g., whether the
single WFS-only common continuation structurally biases the comparison
toward WFS-like outcomes, whether the window objective under-values
long-tail fairness effects WFS's whole-trajectory behavior captures better
than any bounded window can, or whether Family-A's real opportunity is
simply smaller than the native ESTF/WFS envelope suggested once continuation
effects are evaluated exactly rather than sampled at 3 events/scenario.

## 17. Reproducible Commands / Artifacts

- Design: `docs/design/FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md`
- Policy: `src/llmserveopt/policies/family_a_receding_horizon_oracle_v1.py`
- Tests: `python3 -m pytest -q tests/test_family_a_receding_horizon_oracle_v1.py` (15 passed)
- Run: `python3 scripts/run_family_a_receding_horizon_oracle_v1.py`
- State-distribution supplement: `python3 scripts/analyze_family_a_receding_horizon_oracle_v1_state_distribution.py`
- Results: `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_results.json`,
  `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_per_scenario_results.csv`
- git SHA: `8e1223beb58fd4d296061b6b48e3ba493714108f` (dirty tree; matches the
  session's known pre-existing uncommitted files plus this task's new files)

# PHASE_D_CAUSAL_OBJECTIVE_IDENTIFIABILITY_AUDIT_V1

Date: 2026-09-20

Status: objective-identifiability audit after Phase-D V1 execution. Phase-D V1 result artifacts are preserved unchanged.

## Preservation

Phase-D V1 result commit: `3d59065f16d5379d67d295329a0a5c99b8dc9342`.

Frozen preregistration commit: `540aeedaff1fc53f79cb034899859e162dd9c242`.

Phase-D V1 completed exactly:

- SBS references: `11328 / 11328`
- unique non-SBS branches: `12169 / 12169`
- total continuations: `23497 / 23497`
- duplicate branch rows: `0`
- missing reference branches: `0`
- missing counterfactual branches: `0`
- fingerprint failures: `0`
- forced-action warning rows: `0`
- SBS hash mismatches: `0`
- state mutation failures: `0`

Phase-D V1 is classified as:

`PHASE_D_V1_RESULT = ZERO_ADVANTAGE_UNDER_FROZEN_TERMINAL_ANWG`

It is not classified as proving global absence of causal scheduling headroom.

## Exact ANWG Definition

Implementation source:

- `src/llmserveopt/core/metrics.py`
- `compute_metrics(...)`
- `RunMetrics.arrival_normalized_weighted_goodput`

Phase-D extraction source:

- `scripts/industry_realism_causal_headroom_phase_d_v1_execute.py`
- `metric_row(metrics)["q_sbs_anwg"] = metrics_to_dict(metrics)["arrival_normalized_weighted_goodput"]`

For arriving requests \(i\), let:

- \(w_i = priority_i\) if `priority_i > 0`, else \(1.0\);
- \(C_i\) be completion time;
- \(D_i\) be SLO deadline;
- \(completed_i\) indicate completed before terminal evaluation.

The denominator is:

\[
W = \sum_{i \in arrivals} w_i
\]

The numerator is:

\[
\sum_i w_i \cdot 1[completed_i] \cdot 1[C_i \le D_i]
\]

Therefore:

\[
ANWG = \frac{\sum_i w_i \cdot 1[completed_i] \cdot 1[C_i \le D_i]}{\sum_i w_i}
\]

Dropped, unfinished, and late requests receive zero numerator credit but remain in the arrival-weight denominator.

ANWG is bounded in `[0, 1]` for positive request weights. It equals `1.0` iff every positive-weight arriving request completes and every completion satisfies its SLO deadline. It is insensitive to latency, TTFT, TPOT, queueing delay, request order, transient utilization, and completion ordering once all requests satisfy the deadline indicator.

## Why Phase-D ANWG Equals One

Every Phase-D continuation row has:

- `q_sbs_anwg = 1.0`
- `completion_fraction = 1.0`
- `weighted_completion_fraction = 1.0`
- `slo_violation_rate = 0.0`
- `num_completed = 200`
- `num_dropped = 0`
- `num_total = 200`

Thus every continuation completed every request before its SLO deadline. Under the formula above, the numerator equals the denominator for every SBS reference and every forced one-step counterfactual branch.

## Sensitivity Analysis

ANWG can detect:

- fewer completed requests at terminal evaluation;
- dropped/cancelled/unfinished requests;
- completed requests whose `completion_time > slo_deadline`;
- priority-weighted changes in those binary success events.

ANWG cannot detect, after all requests complete before deadline:

- different TTFT;
- different TPOT;
- different mean request latency;
- different p95 latency;
- different queueing delay;
- different request completion order;
- different transient queue length;
- different transient active sequence or KV occupancy;
- different sim duration, provided all deadlines are still met.

Minimal example:

Two equal-priority requests with deadline `10`.

- Trajectory A completes at times `1` and `2`: ANWG = `1.0`.
- Trajectory B completes at times `8` and `9`: ANWG = `1.0`.

Mean latency changes from `1.5` to `8.5`, but ANWG is unchanged because both requests still satisfy `C_i <= D_i`.

## Joint240 Comparison

Earlier joint240 terminal causal experiments used the same ANWG implementation:

- `metrics.arrival_normalized_weighted_goodput`
- `Simulator.continue_run(...)`
- terminal drain semantics

The difference is workload/objective sensitivity, not a different ANWG formula.

Joint240 had many synthetic scenarios with reference ANWG below one and nonzero deadline-credit slack. The stored joint240 branch table has:

- `3541` branches;
- `206` branches with `|delta_anwg| > 1e-12`;
- reference ANWG range `0.0` to approximately `1.0`;
- counterfactual ANWG range `0.0` to approximately `1.0`;
- `delta_anwg` range approximately `[-0.07445, 0.24193]`;
- no dropped requests in the branch table.

Thus joint240 nonzero ANWG arose from changing which requests met binary SLO deadlines, not from completion-count differences.

The later joint240 utility-robustness study is directly relevant. It records that ANWG is a hard step-function objective and that continuous metrics such as weighted mean tardiness can expose effects not visible in ANWG. In joint240 utility robustness:

- ANWG meaningful prevalence: approximately `0.0582`;
- WCG meaningful prevalence: `0`;
- WMT meaningful prevalence: approximately `0.499`;
- ANWG-zero to WMT-meaningful fraction: approximately `0.468`.

This prior evidence supports the interpretation that Phase-D V1's universal ANWG zero advantage may reflect objective saturation rather than identical trajectories.

## Termination Semantics

Phase-D continuations call `Simulator.continue_run(...)` from the forked live state.

The continuation loop terminates when one of the following holds:

- `max_steps` is reached;
- all arrivals are consumed, the waiting/migration/relocation queues are empty, and all GPUs are inactive;
- all arrivals are consumed and `drain_steps` is reached.

For Phase D, all rows report full completion and no drops. Therefore terminal evaluation occurred after all branch work was completed or, equivalently for ANWG, after every request had already received successful deadline credit.

The reference and intervention branches may run for different simulated durations; Phase-D branch rows show nonzero `sim_duration`/`extra_steps` differences for some branches. Queue-drain evaluation can erase scheduling effects for binary-goodput objectives when all work still completes before deadlines.

## Trajectory Difference Audit

Using only existing Phase-D continuation artifacts:

- branches with nonzero mean-latency difference vs SBS reference: `11935 / 12169`;
- branches with nonzero p95-latency difference vs SBS reference: `2433 / 12169`;
- branches with nonzero sim-duration difference vs SBS reference: `418 / 12169`;
- branches with nonzero extra-step difference vs SBS reference: `418 / 12169`;
- states with any mean-latency difference: `11202 / 11328`;
- states with any p95-latency difference: `2286 / 11328`;
- states with any sim-duration difference: `418 / 11328`.

Therefore:

`TRAJECTORIES_DIFFER_BUT_TERMINAL_ANWG_COLLAPSES = YES`

This is diagnostic only. It does not authorize post hoc replacement of the Phase-D V1 primary outcome.

## Implementation Bug Audit

No implementation bug was found.

Evidence:

- `q_sbs_anwg` is read directly from `arrival_normalized_weighted_goodput`;
- `all_requests` and `num_total=len(all_requests)` are passed into every continuation;
- branch rows contain `num_completed`, `num_dropped`, `num_total`, completion fraction, weighted completion fraction, SLO violation rate, mean latency, p95 latency, and sim duration;
- every row has `num_completed=200`, `num_dropped=0`, `num_total=200`, `weighted_completion_fraction=1.0`, and `slo_violation_rate=0.0`, which algebraically implies ANWG `1.0`;
- counterfactual branches have distinct latency/duration fields, ruling out simple reference-row copying;
- causal correctness checks passed for all 11,328 states;
- exact-zero advantages follow from the metric formula, not from clipping or a default/fallback `1.0`.

Implementation-bug audit: `PASS`.

## Interpretability Classification

`PHASE_D_V1_INTERPRETABILITY = METRIC_SATURATION_LIMITS_IDENTIFIABILITY`

Rationale:

- Phase-D V1 correctly establishes zero one-step SBS-relative advantage for frozen terminal ANWG.
- However, ANWG is saturated at `1.0` in every branch.
- Existing branch artifacts show many transient latency differences despite identical ANWG.
- Therefore Phase-D V1 does not identify whether one-step scheduler choices affect operational latency, flow-time, TTFT, TPOT, or fixed-horizon queueing objectives.

## Industry Objective Audit

Operationally meaningful serving objectives from the manuscript and completed literature audit include:

| Objective | Represented in simulator? | Existing Phase-D artifact? | Scheduler-sensitive? | Matches industry claim? | New assumptions? |
| --- | --- | --- | --- | --- | --- |
| ANWG / weighted SLO goodput | yes | yes | only through completion/deadline success | yes, for deadline-goodput claims | no |
| completion fraction / WCG | yes | yes | only if unfinished/dropped differ | limited; all Phase-D rows complete | no |
| mean request latency / flow time | yes | yes (`mean_latency`) | yes | yes | no |
| p95 request latency | yes | yes (`p95_latency`) | yes | yes | no |
| sim-duration / drain time | yes | yes (`sim_duration`) | yes | limited as workload-level makespan | no |
| queueing delay | yes in metric code | not stored in Phase-D result rows | yes | yes | rerun or richer tracing |
| TTFT | represented when first-token times are recorded | not stored/usefully available in Phase-D rows | yes | highly relevant | rerun or richer tracing |
| TPOT / inter-token latency | represented when token timing is recorded | not stored/usefully available in Phase-D rows | partly | relevant | rerun or richer tracing |
| deadline tardiness / soft goodput | computable from per-request traces | not stored in Phase-D rows | yes | yes | rerun unless per-request C_i/D_i stored |
| fixed-horizon completed work | computable if horizon chosen prospectively | not Phase-D V1 primary | yes | yes under throughput-at-horizon question | requires D2 design choice |

## Phase-D2 Decision

`PHASE_D2_REQUIRED = YES`

Justification:

Phase-D V1 cannot identify operational latency/flow-time scheduling effects because its primary utility is saturated. A Phase-D2 is needed only to answer the narrower question:

When canonical action disagreement exists, do one-step alternate actions change operational latency under the same production-derived constrained regimes?

Phase-D2 must not alter the Phase-D V1 conclusion for ANWG.

## D2 Design Summary

Design artifact: `docs/design/INDUSTRY_REALISM_CAUSAL_HEADROOM_PHASE_D2_V1.md`

Draft preregistration: `experiments/industry_realism_causal_headroom_phase_d2_v1/PREREGISTRATION_DRAFT_V1.json`

Core anti-rescue constraints:

- same 11,328 Phase-D V1 disagreement states;
- same 12,169 unique non-SBS branches;
- same regimes and workload windows;
- same one-step forced-action then SBS continuation;
- no new scheduler;
- no state/regime selection based on latency sign or magnitude;
- no selector training;
- no changes to Phase-A/B/D V1 artifacts.

Primary D2 objective:

`mean_latency_improvement = mean_latency_SBS_reference - mean_latency_counterfactual`

Positive means the alternate one-step action reduces mean request latency. This objective is selected because request latency/flow time is a standard serving objective and is already produced by the simulator, not because of favorable D2 outcome selection.

Secondary D2 objectives:

- `p95_latency_improvement = p95_latency_ref - p95_latency_cf`;
- `sim_duration_improvement = sim_duration_ref - sim_duration_cf`;
- state-level oracle latency headroom;
- harmful/zero/mixed structure under latency objectives.

D2 can be analysis-only using existing Phase-D continuation outputs for mean/p95 latency and sim duration. It cannot answer TTFT/TPOT/tardiness without a separately preregistered richer-tracing rerun.

## Revised Gates

`CAUSAL_HEADROOM_GATE = UNRESOLVED_METRIC_SATURATION`

FGCS readiness is revised downward from the Phase-D report because the previous `CAUSAL_HEADROOM_GATE = FAIL` was too strong. A correct but saturated ANWG result is useful, but it does not settle the operational causal-headroom question.

Revised internal checkpoint:

- scientific novelty: `17/20`
- industry realism: `16/20`
- technical depth: `15/20`
- experimental rigor: `16/20`
- practitioner value: `6/10`
- reproducibility/community value: `8/10`

`FGCS_CONTRIBUTION_READINESS_SCORE = 78/100`

`FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 68%`

## Manuscript Implication

Current safe claim:

Phase D V1 found zero one-step SBS-relative causal headroom for frozen terminal ANWG because every reference and counterfactual continuation completed all requests by deadline.

Current unsafe claim:

Phase D V1 proves that scheduler disagreement has no causal value or no operational scheduling headroom.

Full manuscript rewrite around a zero-headroom conclusion is not authorized until D2 or an equivalent objective-validity resolution is completed.

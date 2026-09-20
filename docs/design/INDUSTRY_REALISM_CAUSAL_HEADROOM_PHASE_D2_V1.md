# INDUSTRY_REALISM_CAUSAL_HEADROOM_PHASE_D2_V1

Status: design draft only. Do not execute in the objective-identifiability audit task.

## Motivation

Phase-D V1 correctly measured one-step SBS-relative causal headroom for frozen terminal ANWG and found exact-zero advantage in every branch. However, ANWG was saturated at `1.0` for every SBS reference and counterfactual continuation, while existing branch artifacts show latency/duration differences. Phase-D2 asks whether the same forced one-step alternatives affect operational latency objectives.

Phase-D2 is not a rescue of Phase-D V1. Phase-D V1 remains the canonical result for terminal ANWG:

`PHASE_D_V1_RESULT = ZERO_ADVANTAGE_UNDER_FROZEN_TERMINAL_ANWG`

## Population

Use exactly the Phase-D V1 causal population:

- `11328` canonical SBS-vs-P6 disagreement states;
- `12169` unique non-SBS canonical branches;
- same workloads;
- same faithful windows;
- same Phase-B selected regimes;
- same state IDs and branch IDs.

Do not add, remove, or resample states based on latency sign/magnitude.

## Intervention Semantics

Exactly the Phase-D V1 intervention:

- force exactly one non-SBS canonical action once at state `s`;
- immediately continue with fixed SBS / `kv_constrained_online`;
- preserve future arrivals, simulator configuration, and random state;
- compare to the SBS reference branch for the same state.

No policy-forever switching. No learned continuation. No selector training.

## Primary Objective

Primary D2 branch advantage:

\[
A_{latency}(s,a) = mean\_latency_{ref}(s) - mean\_latency_{cf}(s,a)
\]

Positive means the forced alternative reduces mean request latency relative to SBS continuation.

State-level oracle latency headroom:

\[
H_{latency}(s) = max(0, max_a A_{latency}(s,a))
\]

Beneficial latency opportunity:

\[
B_{latency}(s)=1 \iff max_a A_{latency}(s,a) > 0
\]

The primary metric is selected because request latency/flow time is an operational serving objective represented by the simulator and already stored in Phase-D continuation rows. It is not selected by inspecting favorable cells.

## Secondary Objectives

Secondary diagnostics:

- `p95_latency_improvement = p95_latency_ref - p95_latency_cf`;
- `sim_duration_improvement = sim_duration_ref - sim_duration_cf`;
- action-level positive/negative/zero fractions for each secondary metric;
- relationship between Phase-B `P(D)` and D2 latency headroom.

Not in D2 analysis-only unless new tracing is separately preregistered:

- TTFT;
- TPOT;
- per-request completion order;
- weighted tardiness;
- soft deadline credit;
- queue-area/waiting-time integrals.

## Reuse Existing Trajectories

D2 should be analysis-only if the existing Phase-D V1 continuation table is sufficient:

- `PHASE_D_CONTINUATION_RESULTS_V1.csv` contains `mean_latency`, `p95_latency`, and `sim_duration` for every SBS reference and counterfactual branch.
- The completeness and causal-integrity gates already passed.

If richer metrics are required, create a separate D3 or D2b design with per-request tracing before rerunning simulations.

## Statistics

Use the Phase-D V1 statistical protocol unless explicitly superseded:

- cluster unit: faithful window;
- bootstrap replicates: `2000`;
- bootstrap seed: `20260920`;
- percentile `95%` CI;
- minimum contributing clusters for CI: `5`;
- sparse regimes: descriptive only.

Bootstrap states by faithful window and retain all state/branch rows for sampled windows with multiplicity.

## Reporting

For every workload/regime report:

- Phase-B `P(D)`;
- `P(B_latency | D)`;
- mean state-level oracle latency headroom;
- positive latency-headroom mean/median;
- harmful/zero/mixed state structure;
- secondary p95 and sim-duration diagnostics.

Report zeroes and harms. Do not select only positive regimes.

## Claim Boundaries

D2 can support claims about modeled one-step local latency headroom under the frozen Phase-D causal population. It cannot establish:

- learned-policy gains;
- closed-loop selector gains;
- production deployment improvements;
- real-vLLM latency improvements;
- TTFT/TPOT effects unless separately traced.

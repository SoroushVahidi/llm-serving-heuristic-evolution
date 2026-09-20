# JOINT240_VS_PHASE_D_OBJECTIVE_COMPARISON_V1

## Question

Earlier SBS-relative and terminal one-step experiments produced nonzero causal advantages, while Phase-D V1 produced universal exact-zero ANWG advantages. This document explains the difference.

## Same Utility Formula

Both pipelines used `metrics.arrival_normalized_weighted_goodput` from `src/llmserveopt/core/metrics.py`.

The formula is:

\[
ANWG = \frac{\sum_i w_i 1[completed_i] 1[C_i \le D_i]}{\sum_i w_i}
\]

The denominator is all arriving request weight. The numerator is deadline-met completed request weight.

## Same Broad Terminal Convention

Both pipelines use `Simulator.continue_run(...)` after a one-step forced action. Both continue until terminal/drain semantics, with unfinished/dropped work penalized by the denominator if present.

Joint240 used `max_steps=80000` and `drain_steps=20000`.

Phase-D uses the Phase-B transformed scenario semantics and continuation drain path. In the completed Phase-D V1 artifacts all continuations complete all 200 requests and report zero SLO violations.

## Why Joint240 Could Have Nonzero ANWG

Joint240 synthetic/generated scenarios were designed to stress SLO and scheduler mechanisms. The frozen branch table contains:

- `3541` branch rows;
- `206` nonzero ANWG deltas under `|delta_anwg| > 1e-12`;
- reference ANWG ranging from `0.0` to approximately `1.0`;
- counterfactual ANWG ranging from `0.0` to approximately `1.0`;
- `delta_anwg` ranging from approximately `-0.07445` to `0.24193`;
- no dropped requests in the branch table.

Thus nonzero ANWG was driven by changed deadline attainment among completed requests, not by selective dropping.

## Why Phase-D V1 Has Universal ANWG One

Phase-D V1 continuation rows all have:

- `q_sbs_anwg = 1.0`
- `completion_fraction = 1.0`
- `weighted_completion_fraction = 1.0`
- `slo_violation_rate = 0.0`
- `num_completed = 200`
- `num_dropped = 0`
- `num_total = 200`

Therefore every reference and every counterfactual branch receives full binary deadline credit.

## Prior Joint240 Utility-Robustness Warning

The joint240 terminal utility robustness study already showed ANWG's hard-step limitation:

- ANWG meaningful prevalence was about `0.0582`.
- WCG meaningful prevalence was `0.0`.
- WMT meaningful prevalence was about `0.499`.
- Among ANWG-zero branches, about `0.468` were meaningful under WMT.

This means ANWG zeroes do not necessarily mean identical trajectories or absence of continuous deadline/latency effects.

## Conclusion

Joint240 and Phase-D V1 differ because their workload/deadline regimes place ANWG in different parts of its response curve. Joint240 had deadline-attainment variation. Phase-D V1 saturated terminal ANWG at one in all branches. Phase-D V1 is valid for ANWG, but it does not settle operational latency headroom.

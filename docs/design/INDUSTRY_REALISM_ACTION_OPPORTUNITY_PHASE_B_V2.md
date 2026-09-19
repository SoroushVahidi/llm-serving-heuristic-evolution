# Industry Realism Action Opportunity Phase B V2

Status: design/source/config freeze target before execution.

Phase B V2 asks:

> As the same production-derived request traces are replayed under progressively tighter system constraints, where does canonical SBS-vs-P6 action disagreement begin to emerge?

This is a support/prevalence experiment only. It does not compute SBS-relative causal gains, terminal counterfactuals, selector predictions, learned models, or closed-loop learned scheduling.

## Provenance

Phase A established zero canonical disagreement under native replay for the same 60 faithful production-trace windows:

- Azure 2023 code: 0 / 99,992
- Azure 2023 conversation: 0 / 461,985
- BurstGPT: 0 / 440,461

Phase A also showed that the native configuration was extremely lightly constrained:

- max KV utilization below 0.004
- max active sequences: 9, 7, and 31 by workload
- zero KV, active-sequence, or token-budget binding states

The earlier Phase-B V1 grid is preserved with SHA-256:

`72eb3e7d6915e93c237cd721baec32e526d91c6ed4689da5fdc5b64e42965711`

Classification:

`PHASE_B_V1_GRID = SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY`

It is not invalid; it was designed before the Phase-A pressure telemetry made clear that the old KV and active-cap floors were too weak to cross the observed native operating regime.

## Population

Phase B V2 uses exactly the Phase-A faithful windows:

- Azure 2023 code: 20 windows
- Azure 2023 conversation: 20 windows
- BurstGPT: 20 windows

No augmented windows, synthetic windows, joint240 scenarios, request substitutions, prompt/output-length scaling, or post-result window selection are allowed.

## One-Axis Rule

Each condition varies exactly one serving-system axis.

Arrival sweep:

- vary only interarrival spacing through a fixed arrival multiplier
- keep KV capacity native
- keep active-sequence capacity native

KV sweep:

- vary only KV capacity
- keep native arrival timing
- keep active-sequence capacity native

Active-sequence sweep:

- vary only active-sequence capacity
- keep native arrival timing
- keep KV capacity native

Joint combinations belong to a future Phase C.

## Frozen V2 Grids

Arrival pressure multiplier:

`0.5, 1.0, 2.0, 4.0, 8.0`

Active-sequence capacity:

`512, 64, 32, 16, 8, 4`

This grid includes native 512 and crosses below, near, and above the observed native maxima of 7, 9, and 31.

KV capacity tokens:

`8,000,000, 240,000, 120,000, 60,000, 32,000, 16,000, 8,000`

This grid is absolute rather than a fixed fraction of native capacity. It is calibrated against the Phase-A observed peak KV occupancy of roughly 30k tokens and remains at or above the largest single prompt token count observed in the selected windows.

Expected matrix:

- 18 conditions per faithful window
- 60 faithful windows
- 1,080 window-level conditions

## Metrics

Primary metric:

`disagreement_rate = canonical_disagreement_states / total_SBS_decision_states`

Every condition reports exact numerator and denominator.

Pressure telemetry includes:

- max/mean active sequences
- max/mean queue length
- max/mean KV utilization
- active pressure: observed active sequences / max active sequences
- KV pressure: observed KV tokens / max KV tokens
- active-cap binding fraction
- KV-cap binding fraction
- near-KV-cap fraction
- token-budget binding fraction
- completion fraction
- unfinished request count
- validity class

## Transition Metrics

For each workload and axis:

- first-binding point: first preregistered setting with positive binding count for the manipulated or induced bottleneck
- first-disagreement point: first preregistered setting with at least one canonical SBS-vs-P6 disagreement
- sustained-disagreement point: first preregistered setting with disagreement in at least 2 faithful windows

The sustained criterion is fixed before outcomes.

## Validity Classes

Each window-level condition is classified as:

- `VALID_UNCONSTRAINED`
- `VALID_PARTIALLY_CONSTRAINED`
- `VALID_STRONGLY_CONSTRAINED`
- `INVALID_RESOURCE_INFEASIBLE`
- `INVALID_HORIZON_TRUNCATED`
- `INVALID_OTHER`

Invalid or truncated conditions are retained and reported. They are not interpreted as meaningful scheduler opportunity.

## Reporting Rules

All preregistered workload, window, axis, and pressure settings must be reported, including zero-support, non-monotonic, invalid, or extreme regimes.

Phase B V2 must not be extended around interesting transitions within the same experiment. Any refinement requires a separately versioned phase.

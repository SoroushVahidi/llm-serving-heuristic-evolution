# FRESH_LATENCY_HEADROOM_CONFIRMATORY_DESIGN_REPORT

## Methodological Correction

The existing Phase-D2 latency design is reclassified as
`POST_HOC_OBJECTIVE_REANALYSIS_DESIGN`. Mean latency was selected after the
Phase-D V1 outputs had already established that latency varies, so same-data D2
cannot be treated as fresh preregistered confirmation or independent
generalization evidence.

Allowed role: `POST_HOC_REANALYSIS`.

## Unused Trace Capacity

| workload | total requests | Phase-A/B/D requests | unused full windows | frozen fresh windows |
|---|---:|---:|---:|---:|
| azure_2023_code | 8819 | 4000 | 24 | 20 |
| azure_2023_conv | 19366 | 4000 | 76 | 20 |
| burstgpt | 1404294 | 4000 | 7001 | 20 |

Overlap is checked by `source_record_id`; global overlap count is `0`.

## Fresh Window Freeze

Selection rule: first 20 eligible full contiguous 200-request windows by
ascending window index after excluding all Phase-A/B/D windows.

- Azure 2023 code: `1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39`
- Azure 2023 conversation: `1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26`
- BurstGPT: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20`

Machine-readable proof:
`experiments/fresh_production_latency_headroom_confirmatory_v1/OVERLAP_AUDIT_V1.json`.

## Support Protocol

Fresh support mapping must run before causal labeling. It reuses the Phase-B V2
one-axis grid:

- arrival: `{0.5, 1, 2, 4, 8}`
- KV: `{8000000, 240000, 120000, 60000, 32000, 16000, 8000}`
- active sequences: `{512, 64, 32, 16, 8, 4}`

Expected fresh support matrix: `60` windows x `18` pressure settings = `1080`
window conditions.

## Regime Selection Rule

For each workload and axis, include where distinct:

1. first valid disagreement/onset regime;
2. first sustained-disagreement regime;
3. strongest valid pressure regime.

Arrival remains a structural control if it has no canonical disagreement.

## Causal Protocol

Causal population: all canonical disagreement states in mechanically selected
fresh regimes. Exhaustive labeling is preferred if reference plus unique
alternative continuations are `<=50000`; otherwise use deterministic stratified
sampling by workload, axis, regime stage, and source window with seed
`20260920`.

Intervention semantics: force one unique non-SBS canonical action once, then
immediately revert to SBS / `kv_constrained_online`.

## Latency Metric Semantics

For state `s`, the primary request population `R_s` contains requests not
completed before the intervention state: active, queued, and future requests.
Pre-intervention completed requests are excluded.

`A_LAT(s,a) = L_SBS(s) - L_CF(s,a)`, where `L` is mean request latency in
seconds over `R_s`. Positive means the alternative action reduces mean latency.

Primary state headroom: `H_LAT(s)=max(0,max_a A_LAT(s,a))`.

## Primary Confirmation Rule

Primary endpoint: state-weighted `mean_s H_LAT(s)`, zeros included.

Clustered bootstrap: faithful source window clusters, `2000` replicates, seed
`20260920`, percentile `95%` CI, minimum `5` contributing windows.

`LATENCY_HEADROOM_CONFIRMED` iff the primary endpoint is greater than `0` and
the lower endpoint of the clustered 95% CI is greater than `0`.

## Secondary Metrics

Secondary only: `P(B_LAT|D)`, positive-headroom magnitude, harm/zero/mixed state
structure, action-level sign fractions, relative latency headroom, p95 latency,
and ANWG continuity.

## Old Phase-D Latency Status

`POST_HOC_REANALYSIS_ONLY`.

## Data Sufficiency

Existing local traces are sufficient for a three-workload, 60-window fresh
confirmation with zero request-identity overlap.

## Additional Trace Value

Planning rank if a fourth trace is later needed:

1. Azure 2024 or comparable newer Azure-derived trace with timestamp and token
   fidelity.
2. Agentic/coding trace with replay-compatible arrivals and token counts.
3. Bailian/Qwen or Mooncake/Kimi style trace, contingent on license and replay
   compatibility.

No fourth trace is added in this protocol.

## FGCS Readiness Forecast

Readiness remains `78/100`; confidence remains `68%`. This protocol can resolve
the metric-saturation causal-headroom uncertainty only after fresh support and
fresh latency causal labeling are executed.

## Validation

- Protocol JSON parsed successfully.
- `git diff --check` passed for the touched artifacts.
- `PYTHONPATH=. pytest -q tests/test_industry_realism_action_opportunity_phase_b_v2.py tests/test_industry_realism_causal_headroom_phase_d_v1.py tests/test_industry_realism_causal_headroom_phase_d_v1_execute.py`
  passed: `24 passed`.
- Public-trace/Phase-A test collection was attempted with `PYTHONPATH=.` but
  the ambient shell lacked `pandas`; collection stopped before test execution.

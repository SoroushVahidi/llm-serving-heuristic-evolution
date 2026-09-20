# FRESH_LATENCY_HEADROOM_CONFIRMATORY_REPORT

Execution commit: `38e02bcdaeb8f9cec2289b05dea9a1406318d14f`

## Completeness

- SBS references: 720 / 720
- Non-SBS branches: 831 / 831
- Total continuations: 1551 / 1551
- Duplicate/missing/request-population/fingerprint failures: 0/0/0/0

## Primary Result

- Fresh disagreement states: 720
- Beneficial states: 590
- P(B_LAT|D): 0.819444444
- Mean oracle latency headroom (s): 0.00199579117
- Clustered 95% CI (s): [0.00018458898060051367, 0.003497418373764222]
- Primary verdict: **LATENCY_HEADROOM_CONFIRMED**

## Workload-Regime Map

| workload | axis | regime | P(D) | P(B_LAT|D) | mean H_LAT (s) | windows |
|---|---|---|---:|---:|---:|---:|
| azure_2023_code | active_sequence_capacity | active_8 | 0.000105924 | 0.4 | 3.34267e-05 | 2 |
| azure_2023_code | active_sequence_capacity | active_4 | 0.00128971 | 0.52459 | 7.6685e-05 | 15 |
| azure_2023_code | kv_capacity | kv_16000 | 0.00456109 | 0.923434 | 0.00322728 | 14 |
| azure_2023_conv | active_sequence_capacity | active_4 | 0.000107592 | 0.484375 | 0.000216798 | 17 |
| azure_2023_conv | kv_capacity | kv_16000 | 0.000157048 | 1 | 0.000241346 | 1 |

## Objective Sensitivity

The fresh primary endpoint is continuation-population mean latency. Phase-D V1 ANWG saturation remains a distinct secondary-objective result; fresh causal ANWG was not captured in this execution output and was not used to select, alter, or override the latency verdict.

## Causal Headroom Gate

PASS: the frozen primary mean-headroom endpoint is positive and its clustered 95% CI lower endpoint is positive across 26 independent fresh windows.

## Boundaries

This establishes one-step SBS-relative oracle latency headroom under the frozen production-derived replay scope. It does not establish a deployable selector, production deployment improvement, or general scheduler superiority.

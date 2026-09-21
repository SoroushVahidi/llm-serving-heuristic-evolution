# Erratum to FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT_V1.md

Date: 2026-09-21. The original report is preserved unchanged because it records the support-mapping stage of
`FRESH_PRODUCTION_LATENCY_HEADROOM_CONFIRMATORY_V1`. This erratum states what the canonical machine-readable
artifact shows where the report is wrong. Where a report and a CSV disagree, the CSV governs.

Canonical artifact: `experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv`
(1,080 rows). Recomputed by `paper/performance_evaluation/scripts/reference_policy_numbers.py` and registered in
`paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json` (claims `burstgpt.fresh_mechanism`, `fresh.support_transfer`).

## Incorrect statement

Report §4, table row `BurstGPT | KV | none | zero at every valid pressure point; kv_8000 is invalid/truncated`.

## What the CSV shows

| Quantity | Value |
|---|---|
| BurstGPT fresh conditions | 360 (20 windows x 18 settings) |
| BurstGPT conditions with `validity_class` starting `VALID` | **360 of 360** (no invalid BurstGPT condition, including `kv_8000`) |
| BurstGPT canonical disagreement states | 0 |
| BurstGPT active-sequence or KV binding states | 0 |
| BurstGPT maximum queue length | 3 |
| Invalid (`INVALID_HORIZON_TRUNCATED`) conditions, all workloads | 21 |
| ... of which Azure code | 20 (all `kv_capacity=8000`) |
| ... of which Azure conversation | 1 (`kv_capacity=8000`) |
| ... of which BurstGPT | 0 |

## Consequence

The report's conclusion that BurstGPT support did not transfer to the fresh windows is correct (see the report's own
§6), but the reason is not invalidity of any BurstGPT condition. The untouched BurstGPT windows never queued more than
three requests, so none of the tested caps (down to four active sequences and 8,000 KV tokens) was ever reached.
The manuscript states this mechanism (Section 6.1).

## Other statements in the report

§3 counts (1,080 completed; 1,059 valid; 21 invalid; 951 valid-unconstrained, 38 partially constrained, 70 strongly
constrained) and §6 (transition reproduces for Azure code and conversation; does not for BurstGPT; arrival scaling null)
agree with the CSV. §8 state counts (Azure code active 132 and KV 431; Azure conversation active 64 and KV 93) agree
with `FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv`.

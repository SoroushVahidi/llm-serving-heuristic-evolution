# Corrected Summary (reprocessed from existing requests.jsonl)

Source: `/home/soroush/llm-serving-heuristic-evolution/experiments/real_llm/cohere_pilot_20260703T040421Z/requests.jsonl`
Has native rate_limiter_wait_seconds field: False

## Caveat

This log predates the rate_limiter_wait_seconds/provider_request_latency_seconds split, so the exact amount of local RPM-limiter wait inside each request's recorded latency cannot be recovered. 'corrected_stats_excluding_flagged' drops requests where latency exceeded ttft by more than 5.0s (or exceeded 10.0s absolute for non-streaming requests with no ttft) as likely-RPM-wait-polluted, rather than correcting their value. ttft_seconds was always measured from inside the provider call and is unaffected by this artifact in either raw or corrected form. p50 latency is typically also reliable since the artifact affects a small number of outliers concentrated at the tail (compare raw vs. corrected p50 below to confirm for this specific log).

## Raw stats (all successful requests, unmodified)
- raw count: 180
- raw mean / p50 / p95 / p99 latency (s): 3.057322222222222 / 0.68705 / 1.1494349999999998 / 53.30867400000001
- raw mean / p50 / p95 / p99 TTFT (s): 0.27298944444444445 / 0.25825 / 0.43521 / 0.5302530000000002

## Corrected stats (flagged likely-RPM-wait outliers excluded)
- corrected count: 172
- corrected mean / p50 / p95 / p99 latency (s): 0.7212668604651163 / 0.6828 / 0.99798 / 1.1438879999999998
- corrected mean / p50 / p95 / p99 TTFT (s): 0.27298944444444445 / 0.25825 / 0.43521 / 0.5302530000000002

## Flagged requests (8 of 180 successful)
```
long__mt128__c1__i0
long__mt256__c1__i0
long__mt64__c1__i0
medium__mt128__c1__i0
medium__mt256__c1__i0
medium__mt64__c1__i0
short__mt128__c1__i0
short__mt256__c1__i0
```

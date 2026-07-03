# Cohere API Calibration — Summary

**Model:** `command-r7b-12-2024`
**Generated:** 2026-07-03 13:50:11 UTC

## Status counts
```json
{
  "success": 108
}
```

## Latency / TTFT (successful requests)
Latency here is `provider_request_latency_seconds` — timed from after
the local RPM limiter released the request, so it excludes local
rate-limiter wait. See 'Local rate-limiter wait' below for that.
- count: 108
- mean latency (s): 1.9390574074074076
- p50 / p95 / p99 latency (s): 1.85955 / 3.77436 / 4.069901999999999
- mean TTFT (s): 0.24620277777777777
- p50 / p95 / p99 TTFT (s): 0.23515 / 0.33625499999999997 / 0.38657299999999994

## Local rate-limiter wait (all dispatched requests)
- requests with nonzero wait: 32
- total wait (s): 276.4793
- max wait (s): 44.4258

## Throughput / cost
- mean output tokens: 140.37037037037038
- mean tokens/sec (provider latency basis): 69.75810694906207
- total billed input tokens: 93924.0
- total billed output tokens: 15160.0
- estimated cost (USD, approximate pricing): $0.005796

## Output length vs. target (v2 length-targeted workload)
- overall fraction reaching target range (>= 0.7 x target): 0.9444444444444444

| target_output_tokens | n_success | mean_output_tokens | p50_output_tokens | mean_output_token_ratio | frac_reached_target_range |
|---|---|---|---|---|---|
| 64 | 36 | 51.97222222222222 | 52.0 | 0.8120659722222222 | 0.8333333333333334 |
| 128 | 36 | 127.55555555555556 | 125.0 | 0.9965277777777778 | 1.0 |
| 256 | 36 | 241.58333333333334 | 232.0 | 0.9436848958333334 | 1.0 |

See `aggregate_by_cell.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, `aggregate_by_target_output_tokens.csv` for v2 achieved-vs-target output length, and `errors.jsonl` for failure detail.

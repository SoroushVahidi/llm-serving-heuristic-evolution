# Cohere API Calibration — Summary

**Model:** `command-r7b-12-2024`
**Generated:** 2026-07-03 13:44:06 UTC

## Status counts
```json
{}
```

## Latency / TTFT (successful requests)
Latency here is `provider_request_latency_seconds` — timed from after
the local RPM limiter released the request, so it excludes local
rate-limiter wait. See 'Local rate-limiter wait' below for that.
- count: 0
- mean latency (s): None
- p50 / p95 / p99 latency (s): None / None / None
- mean TTFT (s): None
- p50 / p95 / p99 TTFT (s): None / None / None

## Local rate-limiter wait (all dispatched requests)
- requests with nonzero wait: None
- total wait (s): None
- max wait (s): None

## Throughput / cost
- mean output tokens: None
- mean tokens/sec (provider latency basis): None
- total billed input tokens: None
- total billed output tokens: None
- estimated cost (USD, approximate pricing): $None

See `aggregate_by_cell.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, `aggregate_by_target_output_tokens.csv` for v2 achieved-vs-target output length, and `errors.jsonl` for failure detail.

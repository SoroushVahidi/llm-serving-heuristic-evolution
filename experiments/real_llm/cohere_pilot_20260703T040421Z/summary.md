# Cohere API Calibration — Summary

**Model:** `command-r7b-12-2024`
**Generated:** 2026-07-03 04:12:35 UTC

## Status counts
```json
{
  "success": 180
}
```

## Latency / TTFT (successful requests)
- count: 180
- mean latency (s): 3.057322222222222
- p50 / p95 / p99 latency (s): 0.68705 / 1.1494349999999998 / 53.30867400000001
- mean TTFT (s): 0.27298944444444445
- p50 / p95 / p99 TTFT (s): 0.25825 / 0.43521 / 0.5302530000000002

## Throughput / cost
- mean output tokens: 31.988888888888887
- mean tokens/sec: 43.781687125724225
- total billed input tokens: 150060.0
- total billed output tokens: 5758.0
- estimated cost (USD, approximate pricing): $0.006491

See `aggregate_by_cell.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.

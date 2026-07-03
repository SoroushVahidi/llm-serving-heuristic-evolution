# Gemini API Calibration — Summary

**Model:** `gemini-3.1-flash-lite`
**Generated:** 2026-07-03 04:57:21 UTC

## Status counts
```json
{
  "success": 180
}
```

## Latency / TTFT (successful requests)
- count: 180
- mean latency (s): 3.067105555555555
- p50 / p95 / p99 latency (s): 0.7419 / 1.4764849999999876 / 53.024069000000004
- mean TTFT (s): 0.5558561111111111
- p50 / p95 / p99 TTFT (s): 0.56345 / 0.7359099999999998 / 0.8007640000000009

## Throughput / cost
- mean output tokens: 23.266666666666666
- mean tokens/sec: 30.220820282725942
- total billed input tokens: 148320.0
- total billed output tokens: 4188.0
- estimated cost (USD, approximate pricing): $0.016507

See `aggregate_by_cell.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.

# Gemini API Calibration — Summary

**Model:** `gemini-3.1-flash-lite`
**Generated:** 2026-07-03 14:22:36 UTC

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
- mean latency (s): 1.267486111111111
- p50 / p95 / p99 latency (s): 1.1126999999999998 / 1.9757449999999996 / 2.217192
- mean TTFT (s): 0.6740833333333334
- p50 / p95 / p99 TTFT (s): 0.6006 / 1.267825 / 1.7924159999999987

## Local rate-limiter wait (all dispatched requests)
- requests with nonzero wait: 37
- total wait (s): 324.6874
- max wait (s): 46.911

## Throughput / cost
- mean output tokens: 139.33333333333334
- mean tokens/sec (provider latency basis): 107.22041310715265
- total billed input tokens: 92880.0
- total billed output tokens: 15048.0
- estimated cost (USD, approximate pricing): $0.015307

## Output length vs. target (v2 length-targeted workload)
- overall fraction reaching target range (>= 0.7 x target): 1.0

| target_output_tokens | n_success | mean_output_tokens | p50_output_tokens | mean_output_token_ratio | frac_reached_target_range |
|---|---|---|---|---|---|
| 64 | 36 | 66.66666666666667 | 65.0 | 1.0416666666666667 | 1.0 |
| 128 | 36 | 118.66666666666667 | 118.0 | 0.9270833333333334 | 1.0 |
| 256 | 36 | 232.66666666666666 | 233.0 | 0.9088541666666666 | 1.0 |

See `aggregate_by_cell.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, `aggregate_by_target_output_tokens.csv` for v2 achieved-vs-target output length, and `errors.jsonl` for failure detail.

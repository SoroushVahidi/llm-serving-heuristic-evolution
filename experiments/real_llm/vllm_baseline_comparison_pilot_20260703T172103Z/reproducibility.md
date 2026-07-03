# Reproducibility Metadata

- Generated: 2026-07-03T17:21:36.501410+00:00
- Git branch: `phase2c1-real-trace-ingestion-validation`
- Git commit: `f3044d4561d012ea9b8522e63ac4a6198517773f`
- Git dirty: True
- Run status: `completed`

## Config
```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "policies": [
    "vllm_direct",
    "fifo",
    "edf",
    "shortest_output_first",
    "least_laxity_first",
    "estimated_service_time_first"
  ],
  "prompt_buckets": [
    "short",
    "medium"
  ],
  "target_output_tokens_list": [
    64,
    128
  ],
  "concurrency_list": [
    1,
    2,
    4
  ],
  "requests_per_cell": 2,
  "timeout_seconds": 180.0,
  "max_total_requests": 1000,
  "fail_fast": true,
  "seed": 20260703,
  "mock": false,
  "server_url": "http://127.0.0.1:8001",
  "run_status": "completed",
  "not_wired_policies": {
    "generated_heuristic": "No current, methodologically-valid model artifact exists to load (see the module-level comment above): the only serialized *.joblib files on disk predate the Phase 2B.14 objective correction. Wiring this safely would require re-running the Phase 2B.15/16 corrected-objective training pipeline to produce a fresh artifact, then building a feature adapter for the ~17 of 18 features that ARE client-observable plus an honest placeholder (or /metrics scrape) for kv_utilization. Neither is done here.",
    "best_generated": "alias of generated_heuristic -- see that entry.",
    "selector": "Same gap as generated_heuristic: no current corrected-objective selector model was ever persisted to disk. Not wired."
  }
}
```

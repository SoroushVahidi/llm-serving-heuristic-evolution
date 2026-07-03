# Reproducibility Metadata

- Generated: 2026-07-03T04:04:27.563109+00:00
- Git branch: `phase2c1-real-trace-ingestion-validation`
- Git commit: `43602e5feb217d324974713e7b31568a0866114f`
- Git dirty: True
- Full diff saved to: `git_diff.patch`

```
docs/README.md | 2 ++
 1 file changed, 2 insertions(+)
```

- Python version: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Hostname: `al-khwarizmi`
- CWD: `/home/soroush/llm-serving-heuristic-evolution`
- cohere SDK version: `6.1.0`
- Command line: `scripts/run_cohere_api_calibration.py --allow-live-api --stream --model command-r7b-12-2024 --prompt-buckets short,medium,long --max-tokens-list 64,128,256 --concurrency-list 1,2,4,8 --requests-per-cell 5 --timeout-seconds 90 --rpm-limit 20 --max-total-requests 180 --max-total-input-tokens 250000 --max-total-output-tokens 50000 --max-estimated-cost-usd 5 --seed 20260703 --fail-fast --output-dir experiments/real_llm/cohere_pilot_20260703T040421Z`
- COHERE_API_KEY present: True

## Config
```json
{
  "experiment_id": "cohere_pilot_20260703T040421Z",
  "model": "command-r7b-12-2024",
  "seed": 20260703,
  "prompt_buckets": [
    "short",
    "medium",
    "long"
  ],
  "max_tokens_list": [
    64,
    128,
    256
  ],
  "concurrency_list": [
    1,
    2,
    4,
    8
  ],
  "requests_per_cell": 5,
  "timeout_seconds": 90,
  "rpm_limit": 20,
  "max_total_requests": 180,
  "max_total_input_tokens": 250000,
  "max_total_output_tokens": 50000,
  "max_estimated_cost_usd": 5.0,
  "stream": true,
  "resume": false,
  "fail_fast": true,
  "mock": false,
  "mode": "live"
}
```

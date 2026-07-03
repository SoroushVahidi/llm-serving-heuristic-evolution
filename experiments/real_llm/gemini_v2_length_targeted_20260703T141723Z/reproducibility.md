# Reproducibility Metadata

- Generated: 2026-07-03T14:17:23.384499+00:00
- Git branch: `phase2c1-real-trace-ingestion-validation`
- Git commit: `e2dc3edad7b128d28d0ff4f3054348236f2826ee`
- Git dirty: True
- Full diff saved to: `git_diff.patch`

```
scripts/run_gemini_real_llm_calibration.py |  45 ++++++++
 tests/test_gemini_real_llm_calibration.py  | 176 ++++++++++++++++++++++++++++-
 2 files changed, 220 insertions(+), 1 deletion(-)
```

- Python version: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Hostname: `al-khwarizmi`
- CWD: `/home/soroush/llm-serving-heuristic-evolution`
- SDK package: `google-genai` version `2.6.0`
- Command line: `scripts/run_gemini_real_llm_calibration.py --allow-live-api --stream --model gemini-3.1-flash-lite --workload-version v2 --prompt-buckets short,medium,long --target-output-tokens-list 64,128,256 --concurrency-list 1,2,4,8 --requests-per-cell 3 --timeout-seconds 120 --rpm-limit 20 --max-total-requests 108 --max-total-input-tokens 250000 --max-total-output-tokens 50000 --max-estimated-cost-usd 5 --seed 20260703 --fail-fast --output-dir experiments/real_llm/gemini_v2_length_targeted_20260703T141723Z`
- Env var presence: GOOGLE_CLOUD_PROJECT_present=True

## Config
```json
{
  "provider": "Gemini",
  "experiment_id": "gemini_v2_length_targeted_20260703T141723Z",
  "model": "gemini-3.1-flash-lite",
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
  "requests_per_cell": 3,
  "timeout_seconds": 120,
  "max_total_requests": 108,
  "max_total_input_tokens": 250000,
  "max_total_output_tokens": 50000,
  "max_estimated_cost_usd": 5.0,
  "resume": false,
  "mock": false,
  "mode": "live",
  "workload_version": "v2",
  "rpm_limit": 20,
  "fail_fast": true,
  "stream": true,
  "target_output_tokens_list": [
    64,
    128,
    256
  ],
  "min_output_token_ratio": 0.7,
  "record_output_text_preview_chars": 80
}
```

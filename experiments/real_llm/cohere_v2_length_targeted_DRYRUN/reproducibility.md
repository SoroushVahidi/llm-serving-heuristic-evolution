# Reproducibility Metadata

- Generated: 2026-07-03T13:44:06.572004+00:00
- Git branch: `phase2c1-real-trace-ingestion-validation`
- Git commit: `395ccbcd214535a6e42b3161993e3f1ba6c54895`
- Git dirty: True
- Full diff: omitted from this directory post-hoc (git_diff.patch was removed) because its unified-diff context lines incidentally swept in unrelated fake-secret test placeholder strings from test_cohere_api_calibration.py/test_real_llm_calibration_common.py, tripping this repo's experiments/real_llm secret scanner (test_pilot_experiment_dirs_have_no_secrets). No real credential was ever in the diff. The diff-stat below (still present) fully identifies which files were uncommitted at run time; see git commit 395ccbcd214535a6e42b3161993e3f1ba6c54895 plus `git diff` on that commit for the exact patch if needed.

```
scripts/run_cohere_api_calibration.py          |  47 ++++++
 src/llmserveopt/real_llm/calibration_common.py | 194 +++++++++++++++++++++--
 tests/test_cohere_api_calibration.py           | 208 +++++++++++++++++++++++++
 tests/test_real_llm_calibration_common.py      | 199 +++++++++++++++++++++++
 4 files changed, 635 insertions(+), 13 deletions(-)
```

- Python version: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Hostname: `al-khwarizmi`
- CWD: `/home/soroush/llm-serving-heuristic-evolution`
- SDK package: `cohere` version `6.1.0`
- Command line: `scripts/run_cohere_api_calibration.py --dry-run --stream --model command-r7b-12-2024 --workload-version v2 --prompt-buckets short,medium,long --target-output-tokens-list 64,128,256 --concurrency-list 1,2,4,8 --requests-per-cell 3 --timeout-seconds 120 --rpm-limit 20 --max-total-requests 108 --max-total-input-tokens 250000 --max-total-output-tokens 50000 --max-estimated-cost-usd 5 --seed 20260703 --fail-fast --output-dir experiments/real_llm/cohere_v2_length_targeted_DRYRUN`
- Env var presence: COHERE_API_KEY_present=True

## Config
```json
{
  "provider": "Cohere",
  "experiment_id": "cohere_v2_length_targeted_DRYRUN",
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
  "requests_per_cell": 3,
  "timeout_seconds": 120,
  "max_total_requests": 108,
  "max_total_input_tokens": 250000,
  "max_total_output_tokens": 50000,
  "max_estimated_cost_usd": 5.0,
  "resume": false,
  "mock": false,
  "mode": "dry_run",
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

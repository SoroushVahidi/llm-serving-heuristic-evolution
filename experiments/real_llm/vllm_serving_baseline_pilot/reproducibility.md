# Reproducibility Metadata

- Generated: 2026-07-03T16:01:51.266938+00:00
- Git branch: `phase2c1-real-trace-ingestion-validation`
- Git commit: `f9130ebe326db7a5de12f6c0ad742d77bb83fccb`
- Git dirty: True
- Run status: `planned_only_vllm_not_installed`

**vLLM is not installed in this environment** (CUDA 13.0 / PyTorch 2.12.0 is too new for vLLM's prebuilt wheels — see `configs/gpu_calibration/online_validation.yaml`, which documents the same constraint for GPU calibration). This directory contains a dry-run plan only: no vLLM server was launched or queried, no GPU inference occurred. Rerun with `--allow-live-server` (plus `--server-url` or `--launch-server`) once vLLM is installed on compatible hardware.

## Config
```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "seed": 20260703,
  "prompt_buckets": [
    "short",
    "medium",
    "long"
  ],
  "target_output_tokens_list": [
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
  "timeout_seconds": 120.0,
  "max_total_requests": 108,
  "min_output_token_ratio": 0.7,
  "record_output_text_preview_chars": 80,
  "mock": false,
  "server_url": null,
  "launch_server": false,
  "run_status": "planned_only_vllm_not_installed"
}
```

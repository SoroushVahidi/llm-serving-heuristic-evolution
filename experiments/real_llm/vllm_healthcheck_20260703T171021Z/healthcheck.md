# vLLM Real-Server Health Check — Summary

**Generated:** 2026-07-03T17:14:41Z
**Result: PASS.** A real vLLM server started, loaded the model, and served
both a non-streaming and a streaming completion request successfully. No
hosted API (Cohere/Gemini/OpenAI/Azure/Fireworks/CloudRift) was called.

## Server

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- vLLM version: 0.24.0
- Endpoint: `http://127.0.0.1:8001/v1`
- Flags: `--gpu-memory-utilization 0.5 --max-model-len 4096 --enforce-eager`
- Environment overrides needed (see `reproducibility.md` for why):
  `CUDA_HOME`/`PATH` pointed at the pip-installed `nvcc`/`ninja` inside the
  isolated venv, and `VLLM_USE_FLASHINFER_SAMPLER=0` to bypass a FlashInfer
  JIT-kernel CCCL/nvcc version mismatch (an officially documented vLLM env
  var, not a workaround outside vLLM's own support surface).

## Checks

| Check | Result |
|---|---|
| Server starts | ✅ Yes — "Application startup complete" |
| Model loads | ✅ Yes — 0.93 GiB, 0.72s load time |
| Non-streaming request succeeds | ✅ Yes |
| Streaming request succeeds | ✅ Yes |
| TTFT measurable | ✅ Yes — **0.0145s** |
| Token usage fields present | ✅ Yes — both requests report `prompt_tokens`/`completion_tokens`/`total_tokens` |
| Server crashed | ❌ No |

## Request results

**Non-streaming** (`stream: false`):
- Total latency: 0.3483s
- `usage`: `{"prompt_tokens": 14, "completion_tokens": 64, "total_tokens": 78}`
- `finish_reason`: `length` (hit `max_tokens=64`, as expected)

**Streaming** (`stream: true, stream_options.include_usage: true`):
- Chunks received: 65 (64 text deltas + 1 final usage-only chunk)
- TTFT: **0.0145s**
- Total latency: 0.3221s
- `usage` present in the final SSE chunk: `{"prompt_tokens": 14, "completion_tokens": 64, "total_tokens": 78}`
- `finish_reason`: `length`

Both requests generated identical text (temperature=0.0, deterministic),
confirming streaming and non-streaming paths are consistent.

## GPU memory

| | Used | Total |
|---|---|---|
| Before request | 8267 MiB | 16311 MiB |
| After request | 8267 MiB | 16311 MiB |

Unchanged — vLLM pre-allocates its full KV-cache pool at server startup
(`--gpu-memory-utilization 0.5` → ~8GiB reserved upfront), so a single tiny
64-token request does not measurably change reported usage.

## Warnings observed (non-fatal)

- `Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
  — expected one-time cost under `--enforce-eager` (no ahead-of-time
  CUDA-graph warmup); this is why the very first request took longer
  (client-side timeout of 60s was insufficient; 180s succeeded). Subsequent
  requests should not re-trigger this for the same input shape.
- `Default vLLM sampling parameters have been overridden by the model's generation_config.json` — informational, not an error.

## What this confirms

This is a genuine, real vLLM server — not mocked, not dry-run, not a fake
HTTP server — running Qwen2.5-0.5B-Instruct on the RTX 5060 Ti in an
isolated venv, serving real generated text with real, measured TTFT and
token usage. This is sufficient to proceed to the tiny 24-request external
baseline comparison pilot in a future step (not run in this task).

## See also

- `tmux_server_command.txt` — exact server launch command
- `server.log` — full server log (startup + one-time JIT warning)
- `client_healthcheck.log` — full client-side request/response transcript
- `healthcheck.json` — machine-readable version of this summary
- `reproducibility.md` — environment/git state at health-check time

# real_vllm_runtime_probe_v1

Label: `NON_SCIENTIFIC_RUNTIME_FEASIBILITY_PROBE`

Date: 2026-08-24

Scope: restore a controlled local vLLM runtime and run only tiny
instrumentation probes for `real_vllm_mechanism_validation_v1`. This is not the
scientific prefill/decode mechanism comparison.

## Preflight

| Field | Value |
| --- | --- |
| Repository | `/home/soroush/llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | `origin/contextual-compositional-heuristics-20260731` |
| Ahead/behind | local ahead by 2 |
| Git lock/merge/rebase/cherry-pick/bisect state | none observed |

Existing untracked scientific artifacts were preserved.

## Hardware

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Count | 1 |
| Compute capability | 12.0 / SM120 |
| VRAM | 16,311 MiB |
| Preflight VRAM | 15 MiB used, 15,834 MiB free |
| Driver | 580.173.02 |
| CUDA reported by driver | 13.0 |
| RAM | 62 GiB total, 58 GiB available |
| Disk | 654 GB available |

## Environment Created

New isolated environment:

`/home/soroush/.venvs/vllm_real_validation_v1`

Install path used after correcting environment targeting:

```bash
/home/soroush/.venvs/vllm_real_validation_v1/bin/uv pip install \
  --python /home/soroush/.venvs/vllm_real_validation_v1/bin/python \
  vllm --torch-backend=cu130
```

Official guidance consulted: vLLM GPU installation docs and the vLLM installer
page. The relevant constraint is that Blackwell requires CUDA 12.8 or newer;
the stable CUDA-13 backend was therefore tried before any nightly build.

Installed versions in the new environment:

| Package | Version |
| --- | --- |
| vLLM | 0.27.1 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| Transformers | 5.15.1 |
| FlashInfer Python | 0.6.16.post3 |
| uv | 0.12.5 |
| cuda-python | 13.3.1 |

Important isolation incident: the first `uv pip install` invocation selected
the already-active `/home/soroush/modal-venv` despite calling `uv` from the new
environment path. That unintentionally installed vLLM 0.27.1 and upgraded Torch
to 2.13.0+cu130 in `modal-venv`. The install was then repeated with explicit
`--python` targeting the new environment. No rollback was attempted because
reverting an existing research environment without explicit instruction would
be riskier than recording the exact incident.

## Blackwell SM120 Compatibility

Checks in the new environment:

| Check | Result |
| --- | --- |
| `torch.cuda.is_available()` | true |
| device count | 1 |
| device name | NVIDIA GeForce RTX 5060 Ti |
| `torch.cuda.get_device_capability()` | `(12, 0)` |
| CUDA tensor kernel | executed successfully |
| vLLM import | success |
| SM120 architecture mismatch | none observed |

## Model

Primary model:

`Qwen/Qwen2.5-0.5B-Instruct`

Local snapshot:

`/home/soroush/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`

Local snapshot size: 999,586,347 bytes. Tokenizer and config loaded
local-only. Config: Qwen2, 24 layers, hidden size 896, 14 attention heads,
2 KV heads, bf16, max position embeddings 32,768.

## Server Launches

Required environment:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
VLLM_USE_FLASHINFER_SAMPLER=0
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is required on this workstation unless
`CUDA_HOME`/`nvcc` is configured. Without it, FlashInfer sampler warmup fails
with `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist`.

### Chunked-Prefill Disabled

Successful command shape:

```bash
vllm serve <local-qwen05b-snapshot> \
  --served-model-name qwen05b-local \
  --host 127.0.0.1 --port 8051 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --block-size 16 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --no-enable-chunked-prefill
```

Current vLLM rejects disabled chunked-prefill with
`max_num_batched_tokens=512` and `max_model_len=4096`; for disabled mode,
`max_num_batched_tokens` must be at least the max model length.

Server log facts:

- model load: 0.93 GiB
- available KV cache memory: 4.17 GiB
- GPU KV cache size: 364,592 tokens
- maximum concurrency for 4,096 tokens/request: 89.01x
- mode served 5/5 tiny requests successfully

Probe note: first concurrent requests included large warmup/JIT latency, so
scientific runs must include explicit warmup and exclude it.

### Chunked-Prefill Enabled

Successful command shape:

```bash
vllm serve <local-qwen05b-snapshot> \
  --served-model-name qwen05b-local \
  --host 127.0.0.1 --port 8052 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --enable-chunked-prefill
```

Server log facts:

- vLLM logged: `Chunked prefill is enabled with max_num_batched_tokens=512`
- model load: 0.93 GiB
- available KV cache memory: 4.29 GiB
- GPU KV cache size: 375,200 tokens
- maximum concurrency for 4,096 tokens/request: 91.60x
- mode served 5/5 tiny requests successfully

Queueing probe:

- 5/5 requests successful
- max running: 4
- max waiting: 3
- max KV-cache usage: 0.0157
- preemptions: 0

## Metrics Available

The `/metrics` endpoint exists and exposes at least:

| Quantity | Metric |
| --- | --- |
| waiting requests | `vllm:num_requests_waiting` |
| waiting by reason | `vllm:num_requests_waiting_by_reason` |
| running requests | `vllm:num_requests_running` |
| KV cache usage | `vllm:kv_cache_usage_perc` |
| preemptions | `vllm:num_preemptions_total` |
| prompt tokens | `vllm:prompt_tokens_total` |
| prompt tokens by source | `vllm:prompt_tokens_by_source_total` |
| generation tokens | `vllm:generation_tokens_total` |
| request success | `vllm:request_success_total` |
| request prompt-token histogram | `vllm:request_prompt_tokens` |
| request generation-token histogram | `vllm:request_generation_tokens` |
| iteration token histogram | `vllm:iteration_tokens_total` |
| requested max-token histogram | `vllm:request_params_max_tokens` |

## Readiness

`local_probe_ready: true`

`scientific_local_run_ready: true`

Basis:

- stable vLLM install succeeded in the isolated environment
- PyTorch sees SM120 and runs CUDA kernels
- vLLM imports successfully
- cached Qwen2.5-0.5B loads
- both chunked-prefill modes launch
- streaming TTFT/E2E harness works
- modest concurrency and queueing are observable
- VRAM/KV headroom is sufficient for the small 4k-context validation

Required cautions for the next task:

- Always target the new venv explicitly.
- Set `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Warm up each server mode before measured requests.
- Use `max_num_batched_tokens >= 4096` for disabled chunked-prefill at
  `max_model_len=4096`.
- Use `max_num_batched_tokens=512` for enabled chunked-prefill.
- Do not interpret this probe as scientific mechanism evidence.

## Artifacts

Created/updated under `experiments/real_vllm_mechanism_validation_v1/`:

- `runtime_environment.json`
- `vllm_install.json`
- `probe_config.json`
- `probe_results.json`
- `gpu_memory_probe.json`
- `metrics_inventory.json`
- `readiness.json`
- `probe_disabled_results.json`
- `probe_enabled_results.json`
- `probe_enabled_queue_results.json`

## Safety

No scientific mechanism comparison, TEST, FINAL, Wulver/Vulver job, external
API call, large benchmark, large model download, git push, or destructive git
operation was performed. All vLLM server processes were stopped before this
report was written.

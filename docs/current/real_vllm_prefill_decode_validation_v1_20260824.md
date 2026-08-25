# real_vllm_prefill_decode_validation_v1

Date: 2026-08-24

Experiment: `real_vllm_mechanism_validation_v1`

Subrun: `prefill_decode_local_v1`

Verdict: `PREFILL_REAL_VALIDATION_NO_GO`

Scope: first real local vLLM mechanism-validation run for native full prefill
versus native chunked prefill only. No WFS, ESTF, LLF, KV-aware policy, Wulver,
TEST, FINAL, DEV redesign, external API, or new model download was used.

## Question

Does native vLLM chunked prefill produce the same qualitative prefill/decode
contention tradeoff and workload-dependent winner reversal observed in the
simulator?

## Runtime

Repository: `/home/soroush/llm-serving-heuristic-evolution`

Branch: `contextual-compositional-heuristics-20260731`

HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`

Environment: `/home/soroush/.venvs/vllm_real_validation_v1`

vLLM: 0.27.1

Model: cached local `Qwen/Qwen2.5-0.5B-Instruct`

GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM

## Server Treatments

Both treatments used:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `--served-model-name qwen05b-local`
- `--gpu-memory-utilization 0.35`
- `--max-model-len 4096`
- `--max-num-seqs 4`
- `--block-size 16`
- `--no-enable-prefix-caching`
- `--enforce-eager`

`FULL` used `--no-enable-chunked-prefill --max-num-batched-tokens 4096`.

`CHUNKED` used `--enable-chunked-prefill --max-num-batched-tokens 512`.

The different `max_num_batched_tokens` values are the intended intervention:
current vLLM 0.27.1 rejects disabled chunked prefill when
`max_num_batched_tokens < max_model_len`.

## Workloads

Manifest hash:
`194d8c5f2eb3f3dc8a6840c7a1a293e2afe895dae71e4e08748a6238eefe28c3`

Four deterministic regimes were run:

| Regime | Requests per run | Geometry |
| --- | ---: | --- |
| `hog_tight_low_late` | 3 hog + 3 late | hog convoy first, low late pressure |
| `late_tight_low_late` | 3 hog + 3 late | same geometry, late class tight |
| `hog_tight_high_late` | 3 hog + 6 late | hog convoy first, high late pressure |
| `late_tight_high_late` | 3 hog + 6 late | same geometry, late class tight |

Hog prompts were 3200, 3280, and 3360 tokens with `max_tokens=128`. Late
prompts were 96, 104, and 112 tokens with `max_tokens=32`, arriving at
0.05-0.10 s. All prompts were tokenized with the actual Qwen tokenizer before
running and fit inside `max_model_len=4096`.

## Run Integrity

Measured design: 2 treatments x 4 regimes x 5 repetitions = 40 measured
regime-runs.

Observed:

- 40/40 measured regime-runs
- 300/300 measured requests successful
- 0 request failures
- 0 prompt-generation mismatches
- 821 metrics samples
- 0 metrics scrape errors
- warmup performed for each server block and excluded from measured rows
- post-run audit found no active vLLM processes
- post-run GPU memory returned to 15 MiB used / 15834 MiB free

Telemetry confirmed real queueing:

- max `vllm:num_requests_running`: 4
- max `vllm:num_requests_waiting`: 7
- max `vllm:kv_cache_usage_perc`: 0.0284
- max `vllm:num_preemptions_total`: 0

No server-log matches were found for obvious CUDA/OOM/Traceback/failure errors.

## Results

The preregistered expected pattern did not appear.

Late-request TTFT comparison reports `CHUNKED - FULL`; negative favors
`CHUNKED`.

| Regime | Mean diff (s) | 95% bootstrap CI (s) | Reps chunked lower | Direction |
| --- | ---: | --- | ---: | --- |
| `hog_tight_low_late` | +0.0236 | [0.0094, 0.0371] | 1/5 | FULL lower late TTFT |
| `late_tight_low_late` | +0.0355 | [0.0315, 0.0399] | 0/5 | FULL lower late TTFT |
| `hog_tight_high_late` | +0.0161 | [-0.0053, 0.0336] | 1/5 | FULL lower/equal late TTFT |
| `late_tight_high_late` | +0.0185 | [0.0030, 0.0310] | 1/5 | FULL lower late TTFT |

Hog E2E comparison reports `CHUNKED - FULL`; positive favors `FULL`.

| Regime | Mean diff (s) | 95% bootstrap CI (s) | Reps full lower | Direction |
| --- | ---: | --- | ---: | --- |
| `hog_tight_low_late` | -0.0165 | [-0.0411, 0.0097] | 2/5 | CHUNKED lower/equal hog E2E |
| `late_tight_low_late` | -0.0056 | [-0.0215, 0.0127] | 2/5 | no full advantage |
| `hog_tight_high_late` | -0.0079 | [-0.0303, 0.0110] | 2/5 | no full advantage |
| `late_tight_high_late` | -0.0056 | [-0.0217, 0.0084] | 3/5 | no stable full advantage |

Throughput differences were small and inconsistent:

| Regime | Mean `CHUNKED - FULL` throughput (requests/s) |
| --- | ---: |
| `hog_tight_low_late` | -0.0918 |
| `late_tight_low_late` | -0.1616 |
| `hog_tight_high_late` | +0.0979 |
| `late_tight_high_late` | +0.0899 |

SLO/goodput was not a useful separator in this local run. Most rows saturated
the configured SLO thresholds; late-tight high-late was 0.50 for `CHUNKED`
late requests and 0.60 for `FULL` late requests, again not favoring chunking.

## Simulator Alignment

| Simulator mechanism | Real-vLLM analogue | Expected direction | Observed direction | Agreement |
| --- | --- | --- | --- | --- |
| hog-tight low-late | `hog_tight_low_late` | FULL lowers hog E2E/completion | CHUNKED lower/equal | no |
| late-tight low-late | `late_tight_low_late` | CHUNKED lowers late TTFT | FULL lower/equal | no |
| hog-tight high-late | `hog_tight_high_late` | FULL lowers hog E2E/completion | CHUNKED lower/equal | no |
| late-tight high-late | `late_tight_high_late` | CHUNKED lowers late TTFT | FULL lower/equal | no |

## Interpretation

This is a scientifically valid local real-vLLM no-go for the specific
Qwen2.5-0.5B, vLLM 0.27.1, `max_num_seqs=4`, 4k-context workload used here.
It does not prove that chunked prefill never helps in real systems. It does
show that the simulator's qualitative Family-B prefill/decode contention
reversal does not automatically transfer to this native vLLM setup.

The most plausible evidence-based explanation is that vLLM's native chunked
prefill implementation and scheduler behavior under this small-model local
workload do not match the simulator's simplified chunking mechanism. Queueing
was present, but the 0.5B model kept absolute prefill/decode service times low,
KV pressure was minimal, and native chunking overhead/scheduling details appear
to dominate the expected late-request TTFT protection.

## Decision

`PREFILL_REAL_VALIDATION_NO_GO`

The result should be used in the LLM 2026 paper as a cautionary real-system
validation finding: simulator mechanism claims should not be promoted as real
serving claims without direct system validation.

Next task: perform a targeted workload-fidelity diagnosis using the same local
setup and artifacts, without changing the scientific conclusion post hoc. The
question should be whether the local 0.5B workload failed to induce the same
mechanism, or whether vLLM's native chunked prefill semantics genuinely differ
from the simulator abstraction. Do not escalate to Wulver or a 7B run until
that diagnosis is preregistered.


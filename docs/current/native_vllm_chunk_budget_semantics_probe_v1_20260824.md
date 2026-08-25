# native_vllm_chunk_budget_semantics_probe_v1

Date: 2026-08-24

Verdict: `NATIVE_VLLM_BUDGET_EFFECT_STRONG`

Scope: native vLLM mechanism experiment. This is not a Family-B simulator
validation. Both treatments kept chunked prefill enabled; only
`max_num_batched_tokens` changed.

## Question

In native vLLM semantics, how does the scheduler token budget change
prefill/decode latency tradeoffs when chunked prefill is enabled in all
treatments?

## Treatments

Common controls:

- cached `Qwen/Qwen2.5-0.5B-Instruct`
- vLLM 0.27.1
- `/home/soroush/.venvs/vllm_real_validation_v1`
- `--enable-chunked-prefill`
- `--max-model-len 4096`
- `--max-num-seqs 4`
- `--gpu-memory-utilization 0.35`
- `--no-enable-prefix-caching`
- `--enforce-eager`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

Treatments:

- `T512`: `--max-num-batched-tokens 512`
- `T4096`: `--max-num-batched-tokens 4096`

## Workloads

Reused exact request manifests from
`prefill_decode_local_v1/workload_manifest.json`.

Source manifest SHA256:
`194d8c5f2eb3f3dc8a6840c7a1a293e2afe895dae71e4e08748a6238eefe28c3`

Selected before running:

- `late_tight_low_late`: 3 hog + 3 late
- `late_tight_high_late`: 3 hog + 6 late

Prompts, tokenized lengths, output caps, arrivals, classes, and inherited SLO
fields were reused unchanged.

## Run Integrity

Design: 2 treatments x 2 regimes x 5 repetitions = 20 measured regime-runs.

Observed:

- 20/20 measured regime-runs
- 150/150 measured requests successful
- 0 request failures
- matched warmup per server block, excluded from measurement
- balanced ABBA server block order
- scheduler trace captured
- no server-log error matches

Prometheus metric scraping produced 413 rows with no scrape errors, but this
runner's parser returned empty metric dictionaries. Therefore KV/preemption
Prometheus maxima are marked unavailable for this run. Queue/running evidence
comes from the scheduler trace.

## Engine-Step Mechanism

Scheduler trace summary:

| Treatment | Scheduled steps | Mixed prefill+decode steps | Mixed fraction | Partial prefill items | Mean prompt tokens / prefill step | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `T512` | 1680 | 165 | 0.0982 | 194 | 436.1 | 511 |
| `T4096` | 1536 | 55 | 0.0358 | 9 | 1535.9 | 112 |

Trace interpretation:

- `T512` spread long prompt work across many smaller partial-prefill chunks.
- `T4096` admitted much larger prompt chunks, including near-4096-token steps.
- Both had the same aggregate decode-token share in this run, but the schedule
  geometry differed strongly.

Max waiting/running from scheduler trace:

- `T512`: max waiting 7, max running 4
- `T4096`: max waiting 5, max running 4

## Latency Results

All differences below are `T4096 - T512`.

Late-request TTFT:

| Regime | Mean diff | 95% bootstrap CI | Reps T4096 lower |
| --- | ---: | --- | ---: |
| `late_tight_low_late` | -0.0306 s | [-0.0380, -0.0228] | 5/5 |
| `late_tight_high_late` | -0.0025 s | [-0.0077, 0.0018] | 3/5 |

Hog E2E:

| Regime | Mean diff | 95% bootstrap CI | Reps T4096 lower |
| --- | ---: | --- | ---: |
| `late_tight_low_late` | +0.0163 s | [0.0092, 0.0235] | 0/5 |
| `late_tight_high_late` | +0.0233 s | [0.0156, 0.0318] | 0/5 |

Throughput:

| Regime | Mean diff | 95% bootstrap CI | Reps T4096 higher |
| --- | ---: | --- | ---: |
| `late_tight_low_late` | +0.0970 rps | [0.0497, 0.1443] | 5/5 |
| `late_tight_high_late` | -0.1994 rps | [-0.2515, -0.1363] | 0/5 |

## Interpretation

The native vLLM token budget has a real, trace-explained effect.

In the low-late-pressure regime, `T4096` improved late TTFT by about 31 ms in
all five repetitions and improved throughput, while worsening hog E2E by about
16 ms in all five repetitions. That is a reproducible tradeoff.

In the high-late-pressure regime, `T4096` did not materially improve late TTFT,
worsened hog E2E in every repetition, and reduced throughput in every
repetition. Pressure changed the utility of the larger budget.

This supports the native-vLLM-specific claim: once chunked prefill is enabled,
`max_num_batched_tokens` itself is an important control dimension. The result
does not rescue the simulator Family-B validation, because this experiment asks
a different question under native vLLM semantics.

## Paper Implication

The paper can safely distinguish:

1. Simulator Family-B: fixed shared 512-token budget, chunk-size mechanism.
2. Direct native-vLLM analogue: failed because the mapping bundled chunking and
   budget changes.
3. Native vLLM budget semantics: under chunked prefill, budget changes produce
   real mechanism and latency tradeoffs.

Best placement: main validation/discussion section if the paper has a
real-system validation paragraph; otherwise appendix with a concise pointer in
limitations.

## Decision

`NATIVE_VLLM_BUDGET_EFFECT_STRONG`

Stop after this experiment. The next paper-driven task should be integration:
fold this into the LLM 2026 evidence/claim-safety plan and proceed toward
manuscript figures/tables. Do not automatically add 1024/2048 budgets, a 7B
replication, or Wulver runs.


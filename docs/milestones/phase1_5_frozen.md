# Phase 1.5 — Frozen State

**Date**: 2026-06-10
**Status**: FROZEN — do not modify simulator, policy, or evaluation code from this phase

## Test Status

102 tests passing at freeze time.

## Frozen Configurations

The following YAML configs are frozen at Phase 1.5 and must not be modified:

1. `configs/overloaded_prefill_comparison.yaml`
2. `configs/prefill_heavy_comparison.yaml`
3. `configs/decode_heavy_comparison.yaml`
4. `configs/mixed_slo_comparison.yaml`
5. `configs/burst_heavy_tail_comparison.yaml`

## Canonical Result Paths

Results from the frozen Phase 1.5 runs are stored at:

1. `results/baseline_comparison/`
2. `results/overloaded_prefill_comparison/`
3. `results/prefill_heavy_comparison/`
4. `results/decode_heavy_comparison/`
5. `results/burst_heavy_tail_comparison/`

## Simulator State at Freeze

- Phase 1.5 adds `first_token_time` to `CompletedRequest`
- Phase 1.5 adds TTFT and TPOT properties
- Phase 1.5 adds `prefilling_count` and `decoding_count` to `ObservableGPUState`
- Phase 1.5 adds prefill modeling: chunked prefill, prefill cost per token,
  step token budget, decode-first scheduling

## Known Limitations (as of Phase 1.5)

- **Single GPU**: No multi-GPU load balancing
- **No preemption**: Requests run to completion once admitted
- **No context switching**: Partial decode state not tracked
- **No KV cache eviction**: Once KV cache is full, new requests wait
- **Synthetic SLOs**: All deadline/priority annotations are generated
- **No network latency modeling**
- **Deterministic token generation**: Actual output length is fixed at
  request creation time

## Phase 1.7A Additions (not part of frozen Phase 1.5)

Phase 1.7A adds real trace support:
- BurstGPT loader (`src/llmserveopt/workloads/burstgpt.py`)
- ShareGPT loader (`src/llmserveopt/workloads/sharegpt.py`)
- Augmentation (`src/llmserveopt/workloads/augmentation.py`)
- Extended JSONL format (`src/llmserveopt/workloads/trace_io_extended.py`)

These additions do NOT modify Phase 1.5 simulator, policy, or evaluation code.

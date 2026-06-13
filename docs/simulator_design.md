# Simulator Design

## Overview

`llmserveopt` implements a **deterministic, iteration-level discrete-event simulator**
inspired by the continuous batching ideas from Orca, DeepSpeed-FastGen, and vLLM,
but retaining only the abstractions needed for research-level policy evaluation.

## Time model

- Time is measured in **seconds** (float).
- The simulator advances in discrete **decode steps** of `step_size` seconds each.
- Default `step_size = 0.001` s (1 ms per decode step).
- One decode step = one iteration of the continuous batching loop.

## Phase 1 simplifications

The following are deliberate simplifications for Phase 1 feasibility:

| Aspect | Phase 1 model | Realistic model |
|---|---|---|
| Prefill cost | Zero (instantaneous) | Proportional to prompt tokens, often amortized |
| Decode throughput | 1 token/request/step always | Decreases with large batch sizes (memory-bandwidth bound) |
| KV cache eviction | Not modeled | vLLM-style block-level paging and eviction |
| Preemption | Not modeled | Preempt-and-requeue long requests |
| GPU heterogeneity | Identical GPUs | Different FLOPS and memory per GPU |
| Network / dispatch latency | Zero | Routing overhead in multi-node setups |

## Phase 1.5 realism upgrades

Phase 1.5 adds **explicit prefill modeling** while preserving all Phase 1 behavior
when `ServiceModel(enable_prefill_modeling=False)` (the default).

### Enabling prefill modeling

```python
from llmserveopt.simulator.service_model import ServiceModel

sm = ServiceModel(
    enable_prefill_modeling=True,
    prefill_cost_per_token=1.0,       # budget tokens per prompt token
    max_prefill_chunk_tokens=512,     # max prefill tokens processed per step
    step_token_budget=4096,           # total token budget per GPU per step
    decode_first=False,               # if True: guarantee decode budget first (Sarathi-style)
)
```

### Prefill phase mechanics

1. When a request is admitted, `prefill_remaining = ceil(prompt_tokens × prefill_cost_per_token)` is set.
2. While `prefill_remaining > 0`, the request is in **prefill phase** — no output tokens are produced.
3. Each step the GPU allocates a chunk of `min(max_prefill_chunk_tokens, prefill_remaining, budget_left)` tokens to advance prefill.
4. Once `prefill_remaining == 0`, the request enters **decode phase** on the next step.
5. `first_token_time` is recorded the moment the first `advance_decode()` call succeeds.

### Token budget allocation (Phase 1.5 step)

```
budget = step_token_budget
each decoding request uses 1 token  → subtract len(decoding_reqs)
remaining budget → chunked prefill for each prefilling request
```

When `decode_first=True`, decode requests are guaranteed their tokens first; prefill
only gets what remains.  This matches the stall-free principle from Sarathi-Serve.

### TTFT and TPOT metrics

Phase 1.5 introduces two new request-level metrics:

| Metric | Formula | Notes |
|---|---|---|
| **TTFT** (Time to First Token) | `first_token_time − arrival_time` | Includes queuing + prefill |
| **TPOT** (Time Per Output Token) | `(completion − first_token) / max(1, output_tokens−1)` | Mean inter-token latency |
| **prefill_delay** | `first_token_time − admission_time` | Waiting inside GPU for prefill |

In Phase 1 mode (instant prefill), `first_token_time` is still recorded (set when
the first decode step runs), so TTFT ≈ queuing delay + first decode step.
In Phase 1.5, TTFT includes the explicit prefill duration.

### KV cache during prefill

During prefill, only the tokens already processed count toward KV usage:

$$\text{KV}_r(t) = (p_r - \text{prefill\_remaining}_r) + \text{tokens\_decoded}_r(t)$$

This allows the scheduler to admit more requests while a large prompt is being prefilled.

## Simulator loop

```
for each step t = 0, 1, 2, ...:
    time = t * step_size
    1. Enqueue all requests with arrival_time <= time
    2. Build ObservableState (no ground-truth leakage)
    3. Call policy.select_action(state) → Action
    4. Validate and apply action (admission with constraint checks)
    5. Advance each GPU:
       Phase 1: every active request → +1 output token
       Phase 1.5: prefilling requests → advance prefill chunk;
                  decoding requests → +1 output token
    6. Remove completed requests; record CompletedRequest (with first_token_time)
    7. Record utilization history
    8. Check termination (all done, or drain_steps exceeded)
```

## Constraint checking

When a policy admits request $r$ to GPU $g$, the simulator checks:

1. `request_id` is in the waiting queue
2. `arrival_time <= current_time`
3. `gpu_id` exists
4. `len(active) + 1 <= max_active_sequences`
5. `kv_used + r.prompt_tokens <= max_kv_tokens`
6. `len(active) + 1 <= max_batch_tokens`

Invalid admissions are silently dropped with a `warnings.warn()`.

## Information hiding

The `ObservableState` passed to policies contains `ObservableRequest` objects that
**omit** `actual_output_tokens`.  This enforces the online constraint.

Phase 1.5 adds `prefilling_count` and `decoding_count` to `ObservableGPUState` so
that serving-style policies (Sarathi, SplitFuse) can reason about GPU phase without
ground-truth KV or token access.

The oracle policy (`policies/oracle.py`) receives a pre-built map of
`request_id -> actual_output_tokens` at construction time and is explicitly
labeled as non-deployable.

## Determinism

Given the same trace and `seed`, every policy and the simulator will produce
identical results.  The `RandomFeasiblePolicy` carries its own seeded `np.random.Generator`
and resets it on `policy.reset()` between runs.

## Extension points (Phase 2+)

- Memory-bandwidth-limited decode slow-down at large batch sizes.
- Block-level KV paging and preemption / eviction.
- Realistic GPU FP16 FLOPS model for prefill.
- Heterogeneous GPU throughput multipliers.
- Speculative decoding.

# Real-vLLM Pressure/Action Validation V1

Status: frozen retrospective audit of the completed bounded local validation.

The canonical fresh latency campaign was complete at `b196c3e27d1e5a0d43fd0664e24e51090ec8e442`. Existing local vLLM runs were audited rather than repeated. This preserves the bounded design and avoids selecting another configuration after observing the result.

## Scope

The validation has two executable tiers:

- **R1 pressure validation:** compare low/transition/strong contention using native vLLM prefill-budget controls and record queueing, running sequences, request latency, TTFT, throughput, and scheduler traces.
- **R2 action-opportunity validation:** compare nearest observable scheduler signatures: scheduled token budget, mixed prefill/decode steps, partial-prefill events, waiting/running counts, and prompt tokens scheduled per step.
- **R3 controlled causal action validation:** excluded. The harness has no hook that forces one alternative action while preserving the same live state, arrivals, random state, and continuation population.

## Frozen matrix

The completed validation used the existing deterministic token-shape replay (`TOKEN_SHAPE_REPLAY`) with Qwen2.5-0.5B-Instruct and vLLM 0.27.1 on one RTX 5060 Ti:

- full-prefill versus native chunked-prefill: 2 treatments x 4 workload regimes x 5 repetitions = 40 measured regime runs;
- native chunked-prefill budget probe: T512 versus T4096 x 2 selected contention regimes x 5 repetitions = 20 measured regime runs;
- warmups were excluded; request-level timings and scheduler traces were retained.

The design uses qualitative mechanism support as the outcome. It does not claim numerical simulator-to-vLLM parameter equivalence or one-step causal latency transfer.

## Decision rule

`STRONG_SUPPORT` requires clear pressure and action-mechanism corroboration without a material fidelity failure. `PARTIAL_SUPPORT` is used when R1/R2 are supported but the direct simulator mechanism or R3 is unavailable. `NULL_RESULT` is reserved for a credible absence of the pressure/action transition. `INCONCLUSIVE` is reserved for incomplete or invalid execution.

The existing run is complete and mechanically valid, but the direct full-versus-chunked comparison is a no-go for the simulator's expected qualitative reversal. The native budget probe does show stable vLLM-native action differentiation and latency tradeoffs. The frozen validation verdict is therefore `PARTIAL_SUPPORT`.

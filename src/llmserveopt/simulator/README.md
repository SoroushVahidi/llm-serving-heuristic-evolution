# src/llmserveopt/simulator

The discrete-event simulation engine. Design rationale:
`docs/simulator_design.md`, `docs/decode_prefill_contention_execution_model.md`.

## Key files

- **`simulator.py`** -- `Simulator`, the top-level step loop.
- **`gpu.py`** -- `GPUState`: per-GPU admission/eviction/step state machine.
  Two execution paths live here (see "What not to confuse" below).
- **`service_model.py`** -- `ServiceModel`: the synthetic prefill/decode
  timing model.
- **`calibrated_service_model.py`** -- `CalibratedServiceModel`: the
  measured-GPU-curve alternative, same interface as `ServiceModel` so
  `gpu.py` needs no branching between them.
- **`service_model_factory.py`** -- builds either model from a YAML config
  (`type: synthetic|calibrated`).
- **`kv_block_manager.py`** -- `KVBlockAllocator` / `KVBlockSpaceManager`:
  a from-scratch reimplementation of vLLM v0.1.0's paged-KV block manager.
  **Opt-in** -- only the 6 faithful baseline policies use it; every other
  policy uses the simpler legacy scalar KV model
  (`GPUConfig.max_kv_tokens`). Two parallel, non-interoperating KV
  accounting systems coexist by design.
- **`constraints.py`** -- feasibility checks shared across policies.
- **`contention_diagnostics.py`** -- diagnostic-only per-step signal
  collection; never consulted by execution or objective code.

## What not to confuse

- **`decode_first` is a dead parameter by default.** Under the default
  execution path (`enable_decode_prefill_contention=False`, still the
  default), decode is unconditionally protected regardless of this flag's
  value -- preserved intentionally for backward compatibility with existing
  configs. It only takes effect on the opt-in shared-contention path
  (`enable_decode_prefill_contention=True`, vLLM-v0.4.2-chunked-prefill-style
  FCFS budget sharing). See `docs/decode_prefill_contention_execution_model.md`
  before assuming this flag does anything.
- **`ServiceModel` vs. `CalibratedServiceModel`**: pick via config, not by
  importing one or the other directly in new code -- use
  `service_model_factory.build_service_model_from_config()`.
- **`Action`'s `preempt`/`swap`/`migrate` verbs** (`core/action.py`) are
  opt-in extensions used only by the faithful baselines that need them
  (`vllm_faithful`, `distserve_faithful`, `llumnix_faithful`). Most policies
  only ever use `admit`.

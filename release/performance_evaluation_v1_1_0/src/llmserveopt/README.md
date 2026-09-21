# src/llmserveopt

Local navigation for this package. For the full architecture picture (module
map, execution flow, key abstractions), see
**[docs/current/ARCHITECTURE.md](../../docs/current/ARCHITECTURE.md)** --
this file is a shorter pointer, not a duplicate.

## Subpackages

| Package | Purpose | See also |
|---|---|---|
| `core/` | Foundational types: `Action`, `ObservableState`, `RunMetrics` | [ARCHITECTURE.md](../../docs/current/ARCHITECTURE.md) |
| `simulator/` | Discrete-event engine, `GPUState`, `ServiceModel` | [simulator/README.md](simulator/README.md) |
| `policies/` | Policy implementations + registries | [policies/README.md](policies/README.md) |
| `selector/` | Selector v1 + `selector/dataset_v2/` (current focus) | [selector/README.md](selector/README.md) |
| `evaluation/` | Run/compare/aggregate harness | [evaluation/README.md](evaluation/README.md) |
| `heuristics/` | LLM-generated heuristic DSL (secondary track) | `docs/llm_heuristic_dsl.md` |
| `llm_generation/` | Offline LLM heuristic-candidate generation loop | `docs/llm_generation_loop.md` |
| `calibration/` | GPU service-curve measurement | `docs/gpu_calibration.md` |
| `real_llm/` | Real-LLM-API calibration shared infra | `docs/api_provider_setup.md` |
| `workloads/` | Synthetic generators + trace loaders | `docs/workload_realism.md` |
| `plotting/`, `utils/` | Small, stable support code | -- |

## What not to confuse

- **`policies/registry.py` vs. `policies/external_baselines_registry.py`**:
  two separate registries. The first is the 20-policy historical/internal
  portfolio; the second is the 6 faithful external baselines. Neither
  imports the other.
- **`selector/` vs. `selector/dataset_v2/`**: v1 (older, still present, not
  removed) vs. v2 (the current active research track). See
  [selector/README.md](selector/README.md).
- **`ServiceModel` vs. `CalibratedServiceModel`** (`simulator/`): synthetic
  vs. measured-GPU-curve timing. Same interface, different backing data.
- **`decode_first`**: a config-visible flag that is a *dead parameter* under
  the default execution path, and only takes effect when
  `enable_decode_prefill_contention=True` is also set. See
  `docs/decode_prefill_contention_execution_model.md`.

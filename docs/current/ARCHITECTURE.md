# Architecture (Canonical)

High-level map of `src/llmserveopt/`. This synthesizes and links to detailed
design docs rather than duplicating implementation detail -- when this doc
and a linked detailed doc disagree, the detailed doc is authoritative for
specifics; this doc is authoritative for the overall shape.

No module-level `README.md` files exist under `src/llmserveopt/` yet (a
Query-4 candidate); this doc is the closest current substitute.

## Execution flow

```
workload (synthetic generator or real-trace loader)
  -> Request stream (arrival time, prompt/output tokens, SLO, priority)
  -> Simulator step loop
       -> ObservableState / ObservableGPUState  (policy's read surface;
          ground-truth output length deliberately excluded)
       -> Policy.select_action(state) -> Action (admit / preempt / swap / migrate)
       -> GPUState.admit/evict + ServiceModel or CalibratedServiceModel
          (prefill/decode timing, KV accounting)
  -> RunMetrics (per-run: weighted_goodput, arrival_normalized_weighted_goodput, ...)
  -> [selector path only] PolicyOutcomeVector per window -> Selector Dataset v2 row
  -> [selector path only] selector model training / evaluation
```

## Module map

| Module | Purpose | Key entry points | Status |
|---|---|---|---|
| `core/` | Foundational types | `Action`, `ObservableState`, `ObservableGPUState`, `RunMetrics`/`compute_metrics` | Stable, heavily depended upon |
| `simulator/` | Discrete-event engine | `Simulator`, `GPUState`, `ServiceModel`, `CalibratedServiceModel`, `KVBlockAllocator`/`KVBlockSpaceManager` | Active |
| `policies/` | Policy implementations + registries | `BASELINE_NAMES` (`policies/registry.py`), `EXTERNAL_BASELINE_REGISTRY` (`policies/external_baselines_registry.py`), `BasePolicy` (`policies/base.py`) | Active |
| `selector/` | Selector v1 (feature extraction, labeling, models) | `SELECTOR_CANDIDATES` (`selector/candidates.py`), `EXTERNAL_STYLE_BASELINES` (`selector/roles.py`) | Superseded in active-development focus by `selector/dataset_v2/`, not removed |
| `selector/dataset_v2/` | Selector v2 pipeline (this project's current focus) | `calibrated_targeted_pilot.py` (current, Option B scope), `slo_calibration.py`, `builder.py`, `candidates.py` (stale, see [BASELINES.md](BASELINES.md)) | Active, most recently modified subpackage |
| `heuristics/` | LLM-generated heuristic DSL | `HeuristicPolicy`, `verifier.py`, `compiler.py` | Present, comparatively dormant vs. the selector track |
| `llm_generation/` | Offline LLM heuristic-candidate generation loop | `generation_loop.py`, `providers.py` (CloudRift/Cohere/Mistral/Mock) | Present, comparatively dormant |
| `calibration/` | GPU service-curve measurement | `BenchmarkBackend`, feeds `CalibratedServiceModel` via `service_curves.json` | Active |
| `real_llm/` | Real-LLM-API calibration shared infra | `calibration_common.py` (output schema for provider pilots) | Active |
| `workloads/` | Synthetic generators + trace loaders | `synthetic.py`, `burstgpt.py`, `sharegpt.py`, `trace_io_extended.py` | Active |
| `evaluation/` | Run/compare/aggregate harness | `run_policy.py`, `compare.py`, `external_baseline_harness.py` | Active |
| `plotting/`, `utils/` | Small, stable support code | -- | Active |

## Key abstractions

- **`Action`** (`core/action.py`): `admit`/`preempt`/`swap`/`migrate`. Only
  `admit` is used by the 20 historical policies; the other three verbs are
  opt-in, used only by `vllm_faithful`/`distserve_faithful`/`llumnix_faithful`.
- **`ObservableState`/`ObservableGPUState`** (`core/types.py`): the sole read
  surface a policy gets. `actual_output_tokens` is deliberately excluded
  (ground-truth-leakage prevention by field omission, not convention).
- **`GPUState`** (`simulator/gpu.py`): per-GPU admission/eviction/step state
  machine. Two execution paths: the default **decode-protected** path (where
  the `decode_first` flag is a dead parameter -- see
  `docs/decode_prefill_contention_execution_model.md`) and the opt-in
  **shared-contention** path (`enable_decode_prefill_contention=True`,
  vLLM-v0.4.2-chunked-prefill-style FCFS budget sharing).
- **`ServiceModel`** / **`CalibratedServiceModel`** (`simulator/service_model*.py`):
  synthetic vs. measured-GPU-curve timing models, sharing one interface so
  `gpu.py` needs no branching.
- **`KVBlockSpaceManager`** (`simulator/kv_block_manager.py`): from-scratch
  reimplementation of vLLM v0.1.0's paged-KV block manager. Opt-in --
  consumed only by the 6 faithful baseline policies. Every other policy uses
  a simpler legacy scalar KV model (`GPUConfig.max_kv_tokens`). **Two
  parallel, non-interoperating KV accounting systems coexist by design.**
- **External baseline registry** (`policies/external_baselines_registry.py`):
  `EXTERNAL_BASELINE_REGISTRY`, 6 entries, each an `ExternalBaselineSpec`
  with `fidelity_class`, `topology_class`, `pinned_source`, `selector_eligible`
  (always `False`).
- **Historical policy registry** (`policies/registry.py`): `BASELINE_NAMES`,
  20 entries, `selector_eligible` implicitly `True` for the Selector v1 pool;
  Selector v2's actual trainable pool is the narrower Option B 8-policy set
  (see [BASELINES.md](BASELINES.md), not `BASELINE_NAMES` directly).
- **Selector v2 candidate resolvers** (`selector/dataset_v2/candidates.py`):
  defines three distinct policy sets with disambiguating names --
  `BASELINE_NAMES` (20, historical portfolio), `MONOLITHIC_DIAGNOSTIC_POLICY_POOL`
  / `monolithic_candidate_policies()` (14, broader diagnostic pool, historical
  pilots), and `SELECTOR_V2_OPTION_B_POLICIES` (8, the current canonical
  trainable action space). `calibrated_targeted_pilot.py::CANDIDATE_POLICIES`
  imports `SELECTOR_V2_OPTION_B_POLICIES` directly rather than duplicating
  it. See [BASELINES.md](BASELINES.md) §B.
- **SLO calibration** (`selector/dataset_v2/slo_calibration.py`):
  `calibrate_window_e2e()` derives a per-request deadline from a
  policy-independent reference-service-model estimate. This fix is what
  took the fraction of genuinely ANWG-discriminative windows from 0/900 to
  16.6%/910 (`docs/selector_v2_slo_calibrated_frontier_search.md`).

## Where to go deeper

- Simulator timing/step semantics: `docs/simulator_design.md`,
  `docs/decode_prefill_contention_execution_model.md`
- GPU calibration: `docs/calibrated_service_model.md`, `docs/gpu_calibration.md`
- Policies/baselines: [BASELINES.md](BASELINES.md)
- Selector v2 pipeline: [SELECTOR_V2.md](SELECTOR_V2.md)
- Scripts as entry points: `scripts/README.md` covers Phase-1.7C-and-earlier
  scripts only; it is not sufficient alone for the current Selector v2 /
  external-baseline scripts (see `docs/README.md` §16A for those).

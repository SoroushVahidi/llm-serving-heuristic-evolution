# Apt-Serve External Scheduler Adapter Specification

**Date:** 2026-08-06
**Status:** Design Complete — Implementation NOT STARTED.

This document describes the external-checkout adapter architecture for executing Apt-Serve's `vllm.core.scheduler.Scheduler` code without violating legal redistribution constraints or polluting the primary simulator environment.

## 1. Execution Boundary Decision

Apt-Serve requires Python 3.11, `torch==2.3.0`, and a patched, highly specific build of `vllm==0.5.0.post1`. Our simulator runs on Python 3.12 with entirely distinct dependencies.

**Decision: Dedicated Subprocess (Option B).**
The simulator will run the adapter in a separate subprocess running under the pinned `apt-serve` conda environment. 
- **Determinism:** Maintained by passing identical serialized JSON states per step.
- **Portability:** Perfect separation of Python environments. If the user doesn't have the `apt-serve` conda env, the baseline simply gracefully fails initialization.
- **Legal Compliance:** No code vendored. The subprocess executes from a dynamically checked-out path outside the simulator's repository scope.

## 2. Adapter Components

### `AptServeLoader`
- Identifies the required `APT_SERVE_CHECKOUT_PATH` environment variable.
- Uses `git rev-parse HEAD` to verify the checkout matches the pinned artifact hash exactly (`c953217988`).
- Verifies the Python environment (`sys.version_info` == 3.11).

### `AptServeCompatibilityAdapter` (Subprocess)
Runs in the isolated 3.11 environment. 
- Receives JSON-serialized `ObservableState` via `stdin`.
- Maps fields to mock `SequenceGroup`, `SchedulerConfig`, and `CacheConfig` objects.
- Invokes `scheduler.schedule()`.
- Maps the returned `SchedulerOutputs` back to a JSON-serialized `Action` containing `admit`, `preempt`, and custom cache-assignment instructions.
- Writes to `stdout`.

### `AptServeSchedulerPolicy` (Simulator Side)
Implements `BasePolicy`.
- Dispatches state to the subprocess.
- Parses the JSON output.
- Emits the parsed `Action`.
- Charges execution overhead (policy time) matching the subprocess clock.

## 3. Compatibility Object Mapping

| Simulator Object | Apt-Serve Object | Classification |
| --- | --- | --- |
| `GPUConfig` | `SchedulerConfig` / `CacheConfig` | Derived mapping |
| `ObservableRequest` | `SequenceGroup` / `Sequence` | Exact mapping |
| `request.arrival_time` | `SequenceGroup.arrival_time` | Exact mapping |
| `request.slo_deadline` | Custom `SequenceGroup` fields (`TTFT_SLO`) | Exact mapping |
| `request.priority` | Used in the value-score heuristic | Derived mapping |
| `_active` | `scheduler.running` | Exact mapping |
| `_waiting` | `scheduler.waiting` | Exact mapping |
| `KVBlockSpaceManager` | `scheduler.block_manager` | Exact tracking |
| `request.actual_output_tokens` | N/A | **Forbidden** |
| `HybridCacheManager` state | `SequenceGroup.use_hidden` | Exact mapping |

## 4. Failure Handling
The simulator will **FAIL HARD** (raise exceptions and crash) rather than falling back to default behavior if:
- The Apt-Serve checkout is missing.
- The `git` SHA does not match.
- The subprocess crashes, times out, or returns invalid JSON.
- An over-capacity decision is made.
This prevents corrupted scientific evaluations.

## 5. Fidelity Levels
- **LEVEL 0:** Scheduler import only. *(Currently achieved via Wulver probe)*.
- **LEVEL 1:** Official value/order decisions on static snapshots.
- **LEVEL 2:** Official batch selection with synthetic cache state.
- **LEVEL 3:** Official cache assignment with simulator-executed tier transitions.
- **LEVEL 4:** End-to-end simulator scheduling with dual-tier accounting. *(Initial implementation target)*.
- **LEVEL 5:** Official-system differential on real hardware. *(Target before final publication)*.

## 6. Implementation Estimation
- **Files Added:** 3 (`apt_serve_faithful.py`, `apt_serve_subprocess.py`, `hybrid_cache_manager.py`).
- **LOC:** ~500.
- **Effort:** LARGE (Subprocess IPC, dual-tier mechanics).
- **Highest-Risk Component:** Subprocess IPC overhead bottlenecking large-scale evaluation sweeps.

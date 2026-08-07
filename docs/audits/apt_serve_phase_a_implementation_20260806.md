# Apt-Serve Phase A Implementation Report

**Date:** 2026-08-06
**Auditor:** Gemini CLI
**Status:** Scaffolding & Contracts Complete — Approved for Phase B.

---

## 1. Summary of Work

We have fully completed Phase A of the Apt-Serve implementation, successfully scaffolding all required structures, validation schemas, IPC protocols, and example configurations.

### Files Added/Modified:
- **Modified `src/llmserveopt/core/types.py`:** Added all 9 proposed hybrid cache configuration fields to `GPUConfig`, along with robust `__post_init__` validation rules. Added a strict **rejected** policy for non-default fields when `hybrid_cache_enabled=False`.
- **Modified `src/llmserveopt/policies/external_baselines_registry.py`:** Registered the new placeholder policy as `apt_serve_faithful` with its specification and factory mapping.
- **Created `src/llmserveopt/policies/apt_serve_faithful.py`:** Holds all typed structures, enums, dataclasses, adapter protocol contracts, versioned IPC schemas, and the placeholder `AptServeSchedulerPolicy` class.
- **Created `tests/test_apt_serve_phase_a.py`:** Added 24 highly comprehensive unit, config example, interface, IPC, and regression tests.
- **Created example configurations under `configs/examples/apt_serve/`:**
  - `legacy_disabled.yaml` (legacy default)
  - `valid_hybrid.yaml` (valid hybrid config)
  - `invalid_negative_capacity.yaml` (negative capacity rejected)
  - `invalid_ratio.yaml` (zero ratio rejected)
  - `invalid_recomputation_model.yaml` (unknown model rejected)

---

## 2. Configuration Schema & Validation

The new configuration fields have been added to `GPUConfig`:
- `hybrid_cache_enabled: bool = False`
- `hidden_cache_capacity_blocks: int = 0`
- `hidden_to_kv_memory_ratio: float = 0.1`
- `cache_switch_latency: float = 0.0`
- `hidden_restore_latency: float = 0.0`
- `recomputation_cost_model: str = "full"`
- `apt_serve_rho: float = 0.5`
- `apt_serve_ttft_slo: float = 2.0`
- `apt_serve_tbt_slo: float = 0.05`

### Validation Policy (Rejected):
If `hybrid_cache_enabled=False`, any non-default values specified for the hybrid fields are strictly rejected. This prevents mixed or ambiguous legacy/hybrid states.

---

## 3. Typed Cache Interfaces & IPC Envelopes

The scaffolding file `apt_serve_faithful.py` implements all needed structs:
- **Enums:** `CacheTier` (KV/HIDDEN/NONE), `CacheRepresentation`, `CacheTransitionKind` (KV_TO_HIDDEN/HIDDEN_TO_KV/EVICT_FULL).
- **Dataclasses:** `CacheAssignment`, `CacheTransitionRequest`, `CacheTransitionResult`, `CacheCapacitySnapshot`, `HybridCacheSnapshot`, `AptServeRequestView`, `AptServeSchedulerDecision`.
- **Adapter Contracts:** `AptServeAdapterConfig`, `AptServeEnvironmentSpec`, `AptServeSourceProvenance`. Defines the expected git commit pin `c953217988274a761da35cf06c01033b18dadf68`.
- **IPC Envelopes:** `AptServeSchedulerInput` and `AptServeSchedulerOutput` with versioned (schema_version=1) JSON-safe serialization and deserialization.

---

## 4. Test Verification Results

All 24 Phase A tests passed flawlessly under `modal-venv`:
- **Config Tests:** Verified defaults, valid config parsing, negative capacity rejection, negative latency rejection, ratio out of bounds, invalid SLO parameters, invalid recomputation cost model, and strict disabled-mode rejection of hybrid fields.
- **Interface Tests:** Construct enums, validate cache assignments, enforce deterministic sorted resident request IDs, and construct decisions.
- **IPC Tests:** Verified deterministic JSON byte serialization, version matching, and bad payload rejection.
- **Config Example Tests:** Verified parsing of the 5 newly added non-production yaml configurations under `configs/examples/apt_serve/`.

---

## 5. Phase B Handoff

With configuration schemas, contracts, and tests fully implemented, the workspace is **ready for Phase B**.
- **Phase B Target:** Implement the dual-tier `HybridCacheManager` ( Option C wrapper around `KVBlockSpaceManager` ) mapping capacity limits and enforcing atomicity of transitions.

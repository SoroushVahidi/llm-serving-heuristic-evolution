# Apt-Serve Simulator Architecture & Configuration Design

**Date:** 2026-08-06
**Status:** Design Complete — Implementation NOT STARTED.

This document scopes the precise backward-compatible simulator configuration additions and execution flow required to implement Apt-Serve's dual-tier (KV and hidden-state) cache semantics.

## 1. Simulator Memory Architecture Audit (Current State)

The current simulator memory model operates as follows:
- **Arrival/Admission:** Requests arrive in `Simulator._pending_arrivals` and move to `_waiting`. They are admitted to a `GPUState` via `can_admit`/`admit`, constrained by `GPUConfig.max_kv_tokens`.
- **Memory Allocation:** External baselines (like `vLLM` or `Llumnix`) rely on `KVBlockSpaceManager` to map request tokens to discrete blocks of KV capacity.
- **Execution:** During `step()`, requests consume execution budget (`step_token_budget`), governed by `ServiceModel` rules for prefill and decoding phases.
- **Preemption/Release:** Under memory pressure, requests are evicted (their blocks freed via `free()`). The default cost model assumes full recomputation on preemption unless otherwise modeled (e.g. swap to CPU).
- **Core Limitation:** This is a **single-tier** model. All tokens occupy KV blocks of equal cost and size. There is no concept of a dense, highly compressed hidden-state cache tier with a lower footprint and cheaper restoration cost.

## 2. Backward-Compatible Configuration Design

To support dual-tier semantics without breaking existing configurations (e.g., CC5 evaluation runs), we extend `GPUConfig` with explicit opt-in fields. Existing YAML/JSON files and programmatic instantiations will default to legacy behavior.

### Proposed Additions to `GPUConfig` (in `src/llmserveopt/core/types.py`)

```python
    # Dual-tier cache support (Apt-Serve style)
    hybrid_cache_enabled: bool = False
    hidden_cache_capacity_blocks: int = 0
    # Ratio representing how much smaller hidden-state blocks are compared to KV blocks
    # e.g. 0.1 means hidden state takes 10% the memory of full KV.
    hidden_to_kv_memory_ratio: float = 0.1  
    
    # Apt-Serve latency/cost models
    cache_switch_latency: float = 0.005      # Fixed latency to switch formats
    hidden_restore_latency: float = 0.01     # Time per block to decode/restore from hidden to KV
    recomputation_cost_model: str = "full"   # "full" vs "hidden_restore"
    
    # Apt-Serve specific SLO and priority parameters
    apt_serve_rho: float = 0.5               # Value weighting coefficient
    apt_serve_ttft_slo: float = 2.0
    apt_serve_tbt_slo: float = 0.05
```

### Validation Rules (Invariants)
1. **Backward Compatibility:** If `hybrid_cache_enabled=False`, all other hidden-cache fields must be ignored.
2. **Capacity Validation:** If `hybrid_cache_enabled=True`, `hidden_cache_capacity_blocks` must be non-negative, and `hidden_to_kv_memory_ratio` must be > 0.
3. **Legacy Tests:** No existing policy (e.g. `vllm_faithful`, `llumnix_faithful`) will observe or utilize `hidden_cache_capacity_blocks` or transition costs.

## 3. Simulator Execution Flow

To cleanly integrate Apt-Serve, the simulator loop must adopt a strict execution boundary that safely delegates logic to the external scheduler (detailed in the Adapter Spec).

The full proposed scheduling step:
1. **Snapshot Simulator State:** Gather `waiting_queue`, `active_requests`, and dual-tier memory utilization.
2. **Construct Compatibility Objects:** Map to the required Apt-Serve `Scheduler` format.
3. **Update Cache-Tier Occupancy:** Synchronize the shadow `HybridCacheManager` (if maintaining internal state).
4. **Invoke Official Scheduler Subprocess:** Execute `schedule()` inside the pinned 3.11 environment.
5. **Receive Decisions:** Parse the returned set of selected requests and their cache assignments (`use_hidden` = True/False).
6. **Validate Capacity:** Ensure the decisions do not violate `GPUConfig` constraints.
7. **Execute Cache Transitions:** Fire events for KV→Hidden (compaction) or Hidden→KV (restoration), charging `cache_switch_latency` and `hidden_restore_latency`.
8. **Execute Batch:** Run the standard `step()` token decoding.
9. **Update State:** Record diagnostics and validate invariants.

## 4. Failure Behavior
If the external official scheduler subprocess times out, crashes, or returns an over-capacity decision, the simulator will **fail hard** (raise an Exception). Scientific evaluation runs must fail loudly rather than silently falling back to a safe baseline, ensuring data integrity.

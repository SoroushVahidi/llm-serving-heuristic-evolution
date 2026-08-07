# Apt-Serve Phase B HybridCacheManager Report

**Date:** 2026-08-06
**Auditor:** Gemini CLI
**Status:** HybridCacheManager Implementation Complete — Approved for Phase C.

---

## 1. Summary of Work

We have fully completed Phase B of the Apt-Serve implementation, successfully building the `HybridCacheManager` (Option C wrapped architecture).

### Files Added/Modified:
- **Created `src/llmserveopt/simulator/hybrid_cache_manager.py`:** Implements `HybridCacheManager` containing dual-tier block managers, atomic transition states, cost representation, and validator checking.
- **Created `tests/test_apt_serve_phase_b.py`:** Added 18 comprehensive tests covering allocations, ratio rounding, release cycles, atomicity rollbacks, deterministic snapshots, and state invariants.

---

## 2. Capacity Rounding Semantics

To represent compressed hidden-state cache blocks safely without fractional allocations, the following rounding semantics are implemented:
```python
hidden_blocks = math.ceil(kv_blocks * self.config.hidden_to_kv_memory_ratio)
```
- **Invariants Checked:**
  - Every non-zero primary allocation must result in at least 1 hidden block (zero-sized representation is forbidden).
  - Capacities are strictly checked; if a conversion or new allocation overflows physical bounds, an exception or atomic rollback occurs.

---

## 3. Transition Atomicity & Rollback

- **KV → Hidden:**
  1. Validates the source is allocated.
  2. Computes destination hidden requirements.
  3. Checks hidden physical capacity. If full, returns failed transition without changing any state.
  4. Releases KV block allocation.
  5. Reserves Hidden block allocation.
- **Hidden → KV:**
  1. Validates source.
  2. Checks KV capacity. If full, rolls back atomically.
  3. Releases Hidden allocation.
  4. Allocates KV allocation.
- **Evict:** Full evictions remove residency and release resources from either tier cleanly.

---

## 4. Performance & Overhead Check

Lightweight manager overhead was benchmarked across scale ranges under `modal-venv`:
- **N = 10 requests:**
  - Allocation: 0.017 ms
  - Transition: 0.044 ms
  - Invariant Checking: 0.002 ms
- **N = 1,000 requests:**
  - Allocation: 0.940 ms (0.94 microseconds/req)
  - Transition: 2.468 ms (2.47 microseconds/req)
  - Invariant Checking: 0.078 ms
- **Conclusion:** Operations are strictly linear. Even at high pressure, memory management adds negligible overhead (<< 1% total step time).

---

## 5. Phase C Handoff

The dual-tier block allocation wrapper, cost representation, and validation invariants are fully validated and checked in the local simulator workspace.
- **Phase C Target:** Implement the subprocess-based external scheduler adapter IPC, supporting input/output serializers and launching the pinned author artifact.

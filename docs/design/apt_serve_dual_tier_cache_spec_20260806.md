# Apt-Serve Dual-Tier Cache Specification

**Date:** 2026-08-06
**Status:** Design Complete — Implementation NOT STARTED.

This document specifies the internal dual-tier memory manager (`HybridCacheManager`) required to implement Apt-Serve's KV and hidden-state cache tiering.

## 1. Dual-Tier Selection (Architecture)

**Option C (Wrap existing `KVBlockSpaceManager` with `HybridCacheManager`)** is selected as the lowest-risk architecture.
- **Fidelity:** It preserves bit-for-bit KV allocation semantics by relying on the exact existing `KVBlockSpaceManager` code.
- **Backward Compatibility:** Existing policies interact with the simulator normally. Only `apt_serve_faithful` directly interfaces with `HybridCacheManager`.
- **Complexity:** Minimal footprint addition.
- **Estimated LOC:** ~150 lines.

## 2. Memory Semantics

### A. KV Cache Tier
- **Stored:** Full key-value vectors for all decoded tokens.
- **Size Formula:** `blocks = ceil(tokens / block_size)`.
- **Behavior:** Execution can only advance (decode next token) if the request resides in the KV cache tier.
- **Release:** Explicitly freed upon completion, preemption, or compression to hidden-state.

### B. Hidden-State Cache Tier
- **Stored:** Highly compressed intermediate token representations (e.g. last-layer hidden states).
- **Size Formula:** `hidden_blocks = ceil((tokens * hidden_to_kv_memory_ratio) / block_size)`.
- **Why it is smaller:** Drops all intermediate layer KV matrices, retaining only the final representation necessary to reconstruct the KV state.
- **Recomputation:** Restoring from hidden state requires a partial re-decode (a forward pass without full recalculation of previous layers) governed by `hidden_restore_latency`.

### C. Cache Switching & Transitions
- **KV → Hidden (Compaction):** Valid for requests that are preempted or pushed out by high-value requests. Poses a one-time fixed penalty (`cache_switch_latency`).
- **Hidden → KV (Restoration):** Valid for requests moving back to decoding. Costs `hidden_restore_latency` per token/block. Execution pauses until restoration completes.
- **Atomicity:** Transitions are logically atomic. If capacity constraints fail during transition processing, a hard error is thrown.

### D. Eviction
- **Selection:** Handled externally by the Apt-Serve scheduler. The `HybridCacheManager` merely enforces capacity.
- **Total Loss:** Evicting a request from the hidden-state tier results in a total loss of progress. Restoring it requires full recomputation from scratch (as per `vllm_faithful`).

## 3. Interface Design (`HybridCacheManager`)

```python
class CacheTier(Enum):
    KV = "kv"
    HIDDEN = "hidden"
    NONE = "none"

class CacheTransition(Enum):
    KV_TO_HIDDEN = auto()
    HIDDEN_TO_KV = auto()
    EVICT_FULL = auto()

class HybridCacheManager:
    def __init__(self, config: GPUConfig, block_size: int = 16):
        self.kv_manager = KVBlockSpaceManager(block_size, config.max_kv_tokens // block_size)
        self.hidden_manager = KVBlockSpaceManager(block_size, config.hidden_cache_capacity_blocks)
        self.ratio = config.hidden_to_kv_memory_ratio
        
        # State tracking
        self.assignments: Dict[int, CacheTier] = {}

    def allocate(self, request_id: int, tokens: int, target_tier: CacheTier) -> bool:
        pass

    def switch_tier(self, request_id: int, target_tier: CacheTier) -> CacheTransition:
        pass

    def release(self, request_id: int) -> None:
        pass

    def validate_invariants(self) -> None:
        # Asserts no request occupies both tiers simultaneously.
        pass
```

## 4. Invariants
- `assert kv_manager.num_free_blocks >= 0`
- `assert hidden_manager.num_free_blocks >= 0`
- A single `request_id` can never have allocated blocks in both `kv_manager` and `hidden_manager` simultaneously.
- Total memory tracking strictly honors `GPUConfig` hard caps.

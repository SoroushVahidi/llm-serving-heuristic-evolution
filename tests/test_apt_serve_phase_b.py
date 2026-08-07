"""Comprehensive focused unit, schema, invariant, and micro-scenario tests for Apt-Serve Phase B (HybridCacheManager)."""
from __future__ import annotations

import json
import math
import pytest
from dataclasses import asdict

from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.apt_serve_faithful import CacheTier, CacheTransitionKind
from llmserveopt.simulator.hybrid_cache_manager import HybridCacheManager, HybridCacheInvariantError


# ======================================================================
# 1. ALLOCATION, RELEASE, AND RATIO TESTS (Step 12)
# ======================================================================

def test_kv_allocation_under_hybrid():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32)
    mgr = HybridCacheManager(gpu)
    assert mgr.can_allocate(100, CacheTier.KV)
    mgr.allocate(request_id=1, prompt_tokens=100, target_tier=CacheTier.KV)
    
    assert mgr.get_request_tier(1) == CacheTier.KV
    assert mgr.get_request_capacity(1) == mgr.blocks_needed(100)
    assert mgr.kv_manager.num_used_blocks == mgr.blocks_needed(100)


def test_hidden_allocation_under_hybrid():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32, hidden_to_kv_memory_ratio=0.2)
    mgr = HybridCacheManager(gpu)
    assert mgr.can_allocate(100, CacheTier.HIDDEN)
    mgr.allocate(request_id=1, prompt_tokens=100, target_tier=CacheTier.HIDDEN)
    
    # kv_blocks needed for 100 tokens is ceil(100/16) = 7 blocks
    # hidden_blocks needed is ceil(7 * 0.2) = ceil(1.4) = 2 blocks
    assert mgr.get_request_tier(1) == CacheTier.HIDDEN
    assert mgr.get_request_capacity(1) == 2
    assert mgr.hidden_manager.num_used_blocks == 2


def test_allocation_insufficient_capacity():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=64, # 4 blocks!
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=2, hidden_to_kv_memory_ratio=0.5)
    mgr = HybridCacheManager(gpu)
    
    # KV capacity exceeded (needs 5 blocks, we only have 4)
    assert not mgr.can_allocate(80, CacheTier.KV)
    with pytest.raises(Exception):
        mgr.allocate(1, 80, CacheTier.KV)
        
    # Hidden capacity exceeded
    assert not mgr.can_allocate(80, CacheTier.HIDDEN)
    with pytest.raises(Exception):
        mgr.allocate(2, 80, CacheTier.HIDDEN)


def test_duplicate_allocation_raises():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32)
    mgr = HybridCacheManager(gpu)
    mgr.allocate(1, 32, CacheTier.KV)
    with pytest.raises(ValueError, match="already has an allocation"):
        mgr.allocate(1, 32, CacheTier.KV)


def test_release_is_idempotent():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32)
    mgr = HybridCacheManager(gpu)
    mgr.allocate(1, 32, CacheTier.KV)
    mgr.release(1)
    # double release is noop, no raises
    mgr.release(1)
    mgr.release(999)


# ======================================================================
# 2. RATIO ROUNDING TESTS (Step 12)
# ======================================================================

@pytest.mark.parametrize("kv_blocks, ratio, expected_hidden", [
    (1, 0.1, 1), # ceil(0.1) = 1 (at least 1 block)
    (5, 0.1, 1), # ceil(0.5) = 1
    (10, 0.1, 1), # ceil(1.0) = 1
    (11, 0.1, 2), # ceil(1.1) = 2
    (7, 0.25, 2), # ceil(1.75) = 2
    (8, 0.25, 2), # ceil(2.0) = 2
])
def test_ratio_rounding_cases(kv_blocks, ratio, expected_hidden):
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=64, hidden_to_kv_memory_ratio=ratio)
    mgr = HybridCacheManager(gpu)
    assert mgr.hidden_blocks_needed(kv_blocks) == expected_hidden


# ======================================================================
# 3. TRANSITION & EVICTION TESTS (Step 12)
# ======================================================================

def test_transition_kv_to_hidden_success():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32, hidden_to_kv_memory_ratio=0.5,
                    cache_switch_latency=0.005)
    mgr = HybridCacheManager(gpu)
    mgr.allocate(1, 32, CacheTier.KV) # 2 blocks
    
    res = mgr.switch_tier(1, CacheTier.HIDDEN)
    assert res.success
    assert res.source_tier == CacheTier.KV
    assert res.destination_tier == CacheTier.HIDDEN
    assert math.isclose(res.expected_delay, 0.005)
    assert not res.recomputation_required
    
    assert mgr.get_request_tier(1) == CacheTier.HIDDEN
    assert mgr.kv_manager.num_used_blocks == 0
    assert mgr.hidden_manager.num_used_blocks == 1 # ceil(2*0.5)=1


def test_transition_hidden_to_kv_success():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32, hidden_to_kv_memory_ratio=0.5,
                    hidden_restore_latency=0.01, recomputation_cost_model="hidden_restore")
    mgr = HybridCacheManager(gpu)
    mgr.allocate(1, 32, CacheTier.HIDDEN) # tokens=32, hidden_blocks=1
    
    res = mgr.switch_tier(1, CacheTier.KV)
    assert res.success
    assert res.source_tier == CacheTier.HIDDEN
    assert res.destination_tier == CacheTier.KV
    assert math.isclose(res.expected_delay, 0.01 * 32)
    assert not res.recomputation_required # cost model says "hidden_restore" -> no recompute!
    
    assert mgr.get_request_tier(1) == CacheTier.KV
    assert mgr.kv_manager.num_used_blocks == 2
    assert mgr.hidden_manager.num_used_blocks == 0


def test_transition_insufficient_capacity_rollback():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=32, # 2 blocks
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=1, hidden_to_kv_memory_ratio=0.5)
    mgr = HybridCacheManager(gpu)
    
    # Allocate req 1 to hidden
    mgr.allocate(1, 16, CacheTier.HIDDEN) # 1 block (ceil(1*0.5)=1)
    assert mgr.hidden_manager.num_free_blocks == 0
    
    # Allocate req 2 to KV
    mgr.allocate(2, 32, CacheTier.KV) # 2 blocks
    assert mgr.kv_manager.num_free_blocks == 0
    
    # Transition KV -> hidden fails due to hidden being full!
    res = mgr.switch_tier(2, CacheTier.HIDDEN)
    assert not res.success
    assert res.error_message == "Insufficient destination hidden capacity"
    
    # Verify rollback: state and residency must remain exactly unchanged!
    assert mgr.get_request_tier(2) == CacheTier.KV
    assert mgr.kv_manager.num_used_blocks == 2
    assert mgr.hidden_manager.num_used_blocks == 1


def test_eviction_removes_residency():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32)
    mgr = HybridCacheManager(gpu)
    mgr.allocate(1, 32, CacheTier.KV)
    
    res = mgr.evict(1)
    assert res.success
    assert res.transition_kind == CacheTransitionKind.EVICT_FULL
    assert res.recomputation_required
    
    assert mgr.get_request_tier(1) == CacheTier.NONE
    assert mgr.kv_manager.num_used_blocks == 0


# ======================================================================
# 4. SNAPSHOT AND INVARIANT TESTS (Step 10)
# ======================================================================

def test_snapshots_deterministic():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32)
    mgr = HybridCacheManager(gpu)
    mgr.allocate(2, 32, CacheTier.KV)
    mgr.allocate(1, 16, CacheTier.HIDDEN)
    
    snap1 = mgr.snapshot(step=42, timestamp=1.5)
    snap2 = mgr.snapshot(step=42, timestamp=1.5)
    
    # Sorted request list must be deterministic and identical
    assert snap1.resident_request_ids == [1, 2]
    assert snap1 == snap2


def test_invariant_dual_residency_raises():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=32)
    mgr = HybridCacheManager(gpu)
    mgr.allocate(1, 32, CacheTier.KV)
    
    # Deliberately hack/corrupt state to bypass direct checks and test validator
    mgr.hidden_manager.allocate(1, 16)
    mgr.assignments[1] = CacheTier.KV # assignment still says KV but also allocated in hidden!
    
    with pytest.raises(HybridCacheInvariantError, match="Duplicate residency detected"):
        mgr.validate_invariants()


# ======================================================================
# 5. HAND-VERIFIABLE MICRO-SCENARIOS (Step 13)
# ======================================================================

def test_scenario_kv_to_hidden_freeing_space():
    """Scenario 1: Two KV requests fill KV capacity; one converts to hidden, freeing KV space."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=64, # 4 blocks total
                    hybrid_cache_enabled=True, hidden_cache_capacity_blocks=4, hidden_to_kv_memory_ratio=0.5)
    mgr = HybridCacheManager(gpu)
    
    # 1. Allocate 2 requests to KV, each needing 2 blocks (32 tokens)
    mgr.allocate(1, 32, CacheTier.KV)
    mgr.allocate(2, 32, CacheTier.KV)
    assert mgr.kv_manager.num_free_blocks == 0
    assert not mgr.kv_manager.can_allocate(16) # full!
    
    # 2. Convert request 1 to hidden, freeing up 2 blocks of KV capacity
    res = mgr.switch_tier(1, CacheTier.HIDDEN)
    assert res.success
    assert mgr.kv_manager.num_free_blocks == 2
    assert mgr.hidden_manager.num_used_blocks == 1 # ceil(2*0.5)=1
    
    # 3. Now we can allocate a new request to KV
    mgr.allocate(3, 16, CacheTier.KV) # needs 1 block
    assert mgr.kv_manager.num_free_blocks == 1

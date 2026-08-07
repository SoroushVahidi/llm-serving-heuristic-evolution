"""Comprehensive focused unit, transaction, rollback, timing, and multi-step tests for Apt-Serve Phase E."""
from __future__ import annotations

import json
import pytest
import math
from dataclasses import asdict

from llmserveopt.policies.apt_serve_faithful import (
    AptServeAdapterConfig,
    AptServeSchedulerPolicy,
    AptServeSchedulerInput,
    AptServeSchedulerDecision,
    CacheTier,
    AptServeAdapterError,
    AptServeProtocolMismatch,
    AptServeCapacityViolation
)
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableRequest, ObservableGPUState, ObservableState


# ======================================================================
# 1. RUNTIME POLICY & CLIENT LIFECYCLE TESTS (Step 16)
# ======================================================================

def test_policy_construction_and_reset():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5
    )
    assert policy.name == "apt_serve_faithful"
    assert policy.hybrid_cache_enabled
    assert policy.hidden_cache_capacity_blocks == 32
    
    # State reset cleans up client
    policy.reset()
    assert policy._client is None


def test_select_action_no_gpu_states_is_noop():
    policy = AptServeSchedulerPolicy()
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[], completed_count=0, step=0)
    action = policy.select_action(state)
    assert action.is_empty()


# ======================================================================
# 2. DECISION APPLICATION & TRANSACTION/ROLLBACK TESTS (Step 16)
# ======================================================================

def test_decision_transaction_success():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5
    )
    
    # Simulate a single step with 1 waiting request
    req = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=32, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={}
    )
    state = ObservableState(time=0.0, waiting_queue=[req], gpu_states=[gpu_state], completed_count=0, step=1)
    
    action = policy.select_action(state)
    assert 1 in action.all_admitted_ids()
    
    # Verify HybridCacheManager state updated
    mgr = policy._get_cache_manager(gpu_state)
    assert mgr.get_request_tier(1) == CacheTier.KV
    assert mgr.kv_manager.num_used_blocks == 2 # ceil(32/16) = 2


def test_decision_transaction_rollback_on_infeasible_allocation():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # Low memory budget (needs 2 blocks, we only have 1 block cap!)
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5
    )
    
    req = ObservableRequest(request_id=444444, arrival_time=0.0, prompt_tokens=32, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=16, # only 1 block!
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={}
    )
    state = ObservableState(time=0.0, waiting_queue=[req], gpu_states=[gpu_state], completed_count=0, step=1)
    
    with pytest.raises(AptServeAdapterError, match="rolled back"):
        policy.select_action(state)
        
    # Verify rollback: memory manager state remains clean with zero allocations
    mgr = policy._get_cache_manager(gpu_state)
    assert len(mgr.assignments) == 0


def test_policy_failed_transaction_leaves_state_unharmed():
    """Verify that any failure in the transaction (e.g. invalid decision) rolls back the cache state entirely."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=1, # 1 block hidden limit
        hidden_to_kv_memory_ratio=0.5
    )
    
    # Pre-populate with request 1 on Hidden (needs 1 hidden block)
    req1 = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=32, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[1], active_requests_info=[req1], current_kv_tokens=32, tokens_decoded_per_request={}
    )
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 32, CacheTier.HIDDEN)
    assert mgr.hidden_manager.num_free_blocks == 0
    
    # Mock client output to try to allocate request 2 on Hidden too (should trigger over-capacity failure!)
    class MockClient:
        def schedule_step(self, inp):
            return AptServeSchedulerDecision(
                selected_request_ids=[2],
                cache_assignments={2: CacheTier.HIDDEN},
                evictions=[],
                deprioritized_requests=[],
                value_scores={}
            )
        def terminate(self):
            pass
            
    policy._client = MockClient()
    
    req2 = ObservableRequest(request_id=2, arrival_time=0.1, prompt_tokens=32, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    state = ObservableState(time=0.1, waiting_queue=[req2], gpu_states=[gpu_state], completed_count=0, step=2)
    
    with pytest.raises(AptServeAdapterError, match="rolled back"):
        policy.select_action(state)
        
    # Verify rollback: request 1 remains safely in HIDDEN, and request 2 was NOT allocated
    assert mgr.get_request_tier(1) == CacheTier.HIDDEN
    assert mgr.get_request_tier(2) == CacheTier.NONE
    assert mgr.hidden_manager.num_free_blocks == 0


# ======================================================================
# 3. TIMING & COST INJECTION TESTS (Step 16)
# ======================================================================

def test_transition_latency_hold_decode():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5,
        cache_switch_latency=0.005 # non-zero switch latency
    )
    
    # Mock pre-decision client output directly by replacing schedule_step
    class MockClient:
        def schedule_step(self, inp):
            # simulate transition 1 to HIDDEN
            return AptServeSchedulerDecision(
                selected_request_ids=[],
                cache_assignments={1: CacheTier.HIDDEN},
                evictions=[],
                deprioritized_requests=[],
                value_scores={}
            )
        def terminate(self):
            pass

    policy._client = MockClient()
    
    # Active request 1 currently on KV
    req = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=32, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[1], active_requests_info=[req], current_kv_tokens=32, tokens_decoded_per_request={}
    )
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)
    
    # Populate cache manager with request 1 on KV prior to decision
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 32, CacheTier.KV)
    
    action = policy.select_action(state)
    
    # Decodes must be suspended during transition delay
    assert 1 in action.all_held_decode_ids()
    assert mgr.get_request_tier(1) == CacheTier.HIDDEN


# ======================================================================
# 4. MULTI-STEP REPEATED SIMULATION TESTS (Step 16)
# ======================================================================

def test_multi_step_state_persistence():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5
    )
    
    # Step 1: Admit Request 1
    req1 = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=16, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={}
    )
    state1 = ObservableState(time=0.0, waiting_queue=[req1], gpu_states=[gpu_state], completed_count=0, step=1)
    
    action1 = policy.select_action(state1)
    assert 1 in action1.all_admitted_ids()
    
    # Step 2: Request 1 is active, we admit Request 2
    req2 = ObservableRequest(request_id=2, arrival_time=0.1, prompt_tokens=16, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state2 = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[1], active_requests_info=[req1], current_kv_tokens=16, tokens_decoded_per_request={}
    )
    state2 = ObservableState(time=0.1, waiting_queue=[req2], gpu_states=[gpu_state2], completed_count=0, step=2)
    
    action2 = policy.select_action(state2)
    assert 2 in action2.all_admitted_ids()
    
    # Verify memory manager has both requests active
    mgr = policy._get_cache_manager(gpu_state2)
    assert mgr.get_request_tier(1) == CacheTier.KV
    assert mgr.get_request_tier(2) == CacheTier.KV


def test_completion_cleanup():
    """Verify that when a request finishes, its cache is fully cleaned up from HybridCacheManager."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5
    )
    
    req1 = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=16, predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[1], active_requests_info=[req1], current_kv_tokens=16, tokens_decoded_per_request={}
    )
    
    # Set request 1 as admitted
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 16, CacheTier.KV)
    assert len(mgr.assignments) == 1
    
    # In step 2, request 1 has completed (is no longer in active_request_ids)
    gpu_state2 = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={}
    )
    state2 = ObservableState(time=0.1, waiting_queue=[], gpu_states=[gpu_state2], completed_count=1, step=2)
    
    policy.select_action(state2)
    
    # Cache manager must be completely empty!
    assert len(mgr.assignments) == 0
    assert mgr.kv_manager.num_used_blocks == 0

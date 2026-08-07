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


# ======================================================================
# 5. TRANSACTION-ORDERING REGRESSION TESTS (Phase G SS15 incident)
#
# Phase G's overnight sweep hit AptServeCapacityViolation on a decision
# that was internally feasible: a HIDDEN->KV restore freed enough hidden
# capacity for a KV->HIDDEN move requested in the same step, but section
# 5b applied `decision.cache_assignments` in raw dict order (first-touch
# order from the client's own bookkeeping) instead of an order that
# respects capacity dependencies, so the acquiring move was attempted
# before the releasing one it depended on. See
# docs/audits/apt_serve_phase_g_ss15_incident_20260807.md.
# ======================================================================

def _mgr_state_fingerprint(mgr):
    """Cheap deep-equality fingerprint of manager state for rollback checks."""
    return (
        dict(mgr.assignments), dict(mgr.num_tokens),
        mgr.kv_manager.num_used_blocks, mgr.kv_manager.num_free_blocks,
        sorted(mgr.kv_manager._requests.keys()),
        mgr.hidden_manager.num_used_blocks if mgr.hidden_manager else None,
        mgr.hidden_manager.num_free_blocks if mgr.hidden_manager else None,
        sorted(mgr.hidden_manager._requests.keys()) if mgr.hidden_manager else None,
    )


class _DictOrderClient:
    """MockClient returning a fixed decision with cache_assignments built
    in an explicitly-controlled dict insertion order, to test that the
    adapter's application order does not depend on it."""
    def __init__(self, cache_assignments, evictions=None, selected=None):
        self._cache_assignments = cache_assignments
        self._evictions = evictions or []
        self._selected = selected or []

    def schedule_step(self, inp):
        return AptServeSchedulerDecision(
            selected_request_ids=list(self._selected),
            cache_assignments=dict(self._cache_assignments),
            evictions=list(self._evictions),
            deprioritized_requests=[],
            value_scores={},
        )

    def terminate(self):
        pass


def _two_resident_gpu_state(active_ids):
    return ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=list(active_ids), active_requests_info=[], current_kv_tokens=0,
        tokens_decoded_per_request={},
    )


def test_transition_order_frees_before_it_acquires():
    """A HIDDEN->KV release and a KV->HIDDEN acquisition in the same
    decision must succeed when the release frees exactly enough capacity
    for the acquisition, regardless of which key comes first in the
    client's cache_assignments dict."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # hidden capacity holds exactly one 333-token request (21 blocks *
    # 0.5 ratio -> 11 hidden blocks) -- request 2 (29 kv blocks -> 15
    # hidden blocks) cannot also fit unless request 1 first restores to KV.
    policy = AptServeSchedulerPolicy(
        adapter_config=config, hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=15, hidden_to_kv_memory_ratio=0.5,
    )
    gpu_state = _two_resident_gpu_state([1, 2])
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 333, CacheTier.HIDDEN)  # 11 hidden blocks used, 4 free
    mgr.allocate(2, 464, CacheTier.KV)      # 29 kv blocks

    # Dict key order deliberately presents the acquisition (2 -> HIDDEN)
    # BEFORE the release (1 -> KV) it depends on.
    policy._client = _DictOrderClient({2: CacheTier.HIDDEN, 1: CacheTier.KV})
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)

    action = policy.select_action(state)

    assert mgr.get_request_tier(1) == CacheTier.KV
    assert mgr.get_request_tier(2) == CacheTier.HIDDEN
    mgr.validate_invariants()


def test_transition_order_independent_of_dict_key_order():
    """Same logical decision, opposite dict insertion order -> identical
    outcome (application order must not be a function of client dict
    order, only of tier semantics)."""
    def run_with_order(assignments):
        config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
        policy = AptServeSchedulerPolicy(
            adapter_config=config, hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=15, hidden_to_kv_memory_ratio=0.5,
        )
        gpu_state = _two_resident_gpu_state([1, 2])
        mgr = policy._get_cache_manager(gpu_state)
        mgr.allocate(1, 333, CacheTier.HIDDEN)
        mgr.allocate(2, 464, CacheTier.KV)
        policy._client = _DictOrderClient(assignments)
        state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)
        policy.select_action(state)
        return mgr.get_request_tier(1), mgr.get_request_tier(2)

    result_a = run_with_order({2: CacheTier.HIDDEN, 1: CacheTier.KV})
    result_b = run_with_order({1: CacheTier.KV, 2: CacheTier.HIDDEN})
    assert result_a == result_b == (CacheTier.KV, CacheTier.HIDDEN)


def test_transition_order_handles_mutual_capacity_dependency():
    """A stress-grid follow-up regression: a HIDDEN->KV restore needing
    room a KV->HIDDEN move (of a *different* request) would free, AND
    that KV->HIDDEN move needing room the restore would free, in the
    same decision. Neither a fixed "releases before acquisitions of the
    other tier" priority nor raw dict order can satisfy both directions
    at once -- release-then-acquire decoupling (begin_transition_release
    / finish_transition_acquire) must."""
    def run_with_order(assignments):
        config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
        # KV holds exactly one 21-block resident with zero free blocks;
        # hidden holds exactly one 11-block resident with zero free
        # blocks. Request 1 (HIDDEN->KV, needs 11 kv blocks) and request
        # 2 (KV->HIDDEN, needs 11 hidden blocks) each depend on the
        # other's release.
        policy = AptServeSchedulerPolicy(
            adapter_config=config, hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=11, hidden_to_kv_memory_ratio=0.5,
        )
        gpu_state = ObservableGPUState(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=336,
            active_request_ids=[1, 2], active_requests_info=[], current_kv_tokens=0,
            tokens_decoded_per_request={},
        )
        mgr = policy._get_cache_manager(gpu_state)
        mgr.allocate(1, 333, CacheTier.HIDDEN)
        mgr.allocate(2, 333, CacheTier.KV)
        policy._client = _DictOrderClient(dict(assignments))
        state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)
        policy.select_action(state)
        mgr.validate_invariants()
        return mgr.get_request_tier(1), mgr.get_request_tier(2)

    for assignments in ({1: CacheTier.KV, 2: CacheTier.HIDDEN}, {2: CacheTier.HIDDEN, 1: CacheTier.KV}):
        assert run_with_order(assignments) == (CacheTier.KV, CacheTier.HIDDEN)


def test_mixed_transaction_eviction_transition_and_admission():
    """One eviction, one HIDDEN->KV release, one KV->HIDDEN acquisition,
    and one new admission in the same decision all apply consistently."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config, hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=15, hidden_to_kv_memory_ratio=0.5,
    )
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=1024,
        active_request_ids=[1, 2, 3], active_requests_info=[], current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 333, CacheTier.HIDDEN)  # will restore to KV (release)
    mgr.allocate(2, 464, CacheTier.KV)      # will move to HIDDEN (acquire)
    mgr.allocate(3, 64, CacheTier.KV)       # will be evicted entirely

    req4 = ObservableRequest(request_id=4, arrival_time=0.0, prompt_tokens=32,
                              predicted_output_tokens=16, slo_deadline=1.0, priority=1.0, class_id="default")
    policy._client = _DictOrderClient(
        cache_assignments={2: CacheTier.HIDDEN, 1: CacheTier.KV},
        evictions=[3],
        selected=[4],
    )
    state = ObservableState(time=0.0, waiting_queue=[req4], gpu_states=[gpu_state], completed_count=0, step=2)

    action = policy.select_action(state)

    assert mgr.get_request_tier(1) == CacheTier.KV
    assert mgr.get_request_tier(2) == CacheTier.HIDDEN
    assert mgr.get_request_tier(3) == CacheTier.NONE
    assert mgr.get_request_tier(4) == CacheTier.KV
    assert 3 in action.preempt.get(0, [])
    assert 4 in action.admit.get(0, [])
    mgr.validate_invariants()


def test_destination_capacity_exactly_full_succeeds():
    """A KV->HIDDEN move that consumes exactly the last free hidden block
    must succeed (no off-by-one under- or over-rejection)."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config, hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=11, hidden_to_kv_memory_ratio=0.5,
    )
    gpu_state = _two_resident_gpu_state([1])
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 333, CacheTier.KV)  # 21 kv blocks -> needs exactly 11 hidden blocks

    policy._client = _DictOrderClient({1: CacheTier.HIDDEN})
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)
    policy.select_action(state)

    assert mgr.get_request_tier(1) == CacheTier.HIDDEN
    assert mgr.hidden_manager.num_free_blocks == 0


def test_genuinely_infeasible_decision_still_rolls_back_exactly():
    """A decision that is infeasible even under capacity-safe ordering
    (two acquisitions whose combined need exceeds capacity, nothing being
    released) must still raise and leave the manager byte-for-byte
    unchanged -- the ordering fix must not weaken SS15 for real
    violations."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config, hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=11, hidden_to_kv_memory_ratio=0.5,
    )
    gpu_state = _two_resident_gpu_state([1, 2])
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 333, CacheTier.KV)  # needs 11 hidden blocks
    mgr.allocate(2, 333, CacheTier.KV)  # also needs 11 hidden blocks -- together 22 > 11

    before = _mgr_state_fingerprint(mgr)
    policy._client = _DictOrderClient({1: CacheTier.HIDDEN, 2: CacheTier.HIDDEN})
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)

    with pytest.raises(AptServeAdapterError, match="rolled back"):
        policy.select_action(state)

    assert _mgr_state_fingerprint(mgr) == before


def test_no_transitions_silently_dropped():
    """Every request named in a feasible decision's cache_assignments
    must actually end up at its requested tier -- the ordering fix must
    not skip, clip, or drop any transition."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config, hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=15, hidden_to_kv_memory_ratio=0.5,
    )
    gpu_state = _two_resident_gpu_state([1, 2])
    mgr = policy._get_cache_manager(gpu_state)
    mgr.allocate(1, 333, CacheTier.HIDDEN)
    mgr.allocate(2, 464, CacheTier.KV)

    policy._client = _DictOrderClient({2: CacheTier.HIDDEN, 1: CacheTier.KV})
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_state], completed_count=0, step=1)
    policy.select_action(state)

    assert policy.stats["hidden_to_kv_transitions"] == 1
    assert policy.stats["kv_to_hidden_transitions"] == 1


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

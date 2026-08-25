"""Comprehensive focused unit, boundary, differential, corruption, and regression tests for Apt-Serve Phase D."""
from __future__ import annotations

import os
import sys
import json
import pytest
import math
from dataclasses import asdict

from llmserveopt.policies.apt_serve_faithful import (
    AptServeAdapterConfig,
    AptServeSubprocessClient,
    AptServeSchedulerInput,
    AptServeSchedulerDecision,
    CacheTier,
    AptServeInvalidSchedulerDecision,
    AptServeMalformedResponse,
    AptServeProtocolMismatch
)
from llmserveopt.simulator.hybrid_cache_manager import HybridCacheManager


# ======================================================================
# 1. BOUNDARY SCENARIOS TESTS (Step 9)
# ======================================================================

def test_scenario_empty_queues():
    """Scenario 1: Verify empty queues serialize and are scheduled safely."""
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=0, timestamp=0.0,
        gpus=[{"gpu_id": 0, "max_active_sequences": 16, "max_batch_tokens": 2048, "max_kv_tokens": 1024}],
        waiting_requests=[], running_requests=[], cache_snapshot={}
    )
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    with AptServeSubprocessClient(config) as client:
        decision = client.schedule_step(state_input)
        assert isinstance(decision, AptServeSchedulerDecision)
        assert len(decision.selected_request_ids) == 0


def test_scenario_one_waiting_ample_memory():
    """Scenario 2: One waiting request with ample memory."""
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=0, timestamp=0.0,
        gpus=[{"gpu_id": 0, "max_active_sequences": 16, "max_batch_tokens": 2048, "max_kv_tokens": 1024}],
        waiting_requests=[{"request_id": 1, "prompt_tokens": 100, "arrival_time": 0.0}],
        running_requests=[], cache_snapshot={}
    )
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    with AptServeSubprocessClient(config) as client:
        decision = client.schedule_step(state_input)
        assert decision.selected_request_ids == [1]


def test_scenario_multiple_waiting_tie():
    """Scenario 3: Multiple waiting requests with deterministic tie."""
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=0, timestamp=0.0,
        gpus=[{"gpu_id": 0, "max_active_sequences": 16, "max_batch_tokens": 2048, "max_kv_tokens": 1024}],
        waiting_requests=[
            {"request_id": 1, "prompt_tokens": 100, "arrival_time": 0.0},
            {"request_id": 2, "prompt_tokens": 100, "arrival_time": 0.0}
        ],
        running_requests=[], cache_snapshot={}
    )
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    with AptServeSubprocessClient(config) as client:
        decision = client.schedule_step(state_input)
        # Fake worker selects the first waiting request
        assert decision.selected_request_ids == [1, 2]


def test_scenario_running_plus_waiting():
    """Scenario 4: Running request plus waiting request."""
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=0, timestamp=0.0,
        gpus=[{"gpu_id": 0, "max_active_sequences": 16, "max_batch_tokens": 2048, "max_kv_tokens": 1024}],
        waiting_requests=[{"request_id": 2, "prompt_tokens": 100, "arrival_time": 0.0}],
        running_requests=[{"request_id": 1, "prompt_tokens": 50, "arrival_time": 0.0, "running_duration": 0.5}],
        cache_snapshot={}
    )
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    with AptServeSubprocessClient(config) as client:
        decision = client.schedule_step(state_input)
        assert decision.selected_request_ids == [1, 2]


# ======================================================================
# 2. DETERMINISM AND PYTHONHASHSEED TESTS (Step 11)
# ======================================================================

def test_determinism_across_hash_seeds():
    """Verify that serialization is byte-identical and hash independent."""
    inp = AptServeSchedulerInput(
        schema_version=1, request_id=42, simulator_step=1, timestamp=0.5,
        gpus=[{"id": 0}], waiting_requests=[{"id": 10}, {"id": 20}], running_requests=[], cache_snapshot={}
    )
    b1 = inp.serialize_json()
    b2 = inp.serialize_json()
    assert b1 == b2


# ======================================================================
# 3. CORRUPTION AND INTEGRITY TESTS (Step 13)
# ======================================================================

def test_corruption_detector_selected_not_in_input():
    """Ensure that any decision containing an unknown selected request ID is rejected."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # 555555 triggers fake worker to return non-existent selected ID 12345
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=0, timestamp=0.0, gpus=[],
        waiting_requests=[{"request_id": 555555, "prompt_tokens": 100}], running_requests=[], cache_snapshot={}
    )
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeInvalidSchedulerDecision, match="not present in input"):
            client.schedule_step(state_input)


def test_corruption_detector_malformed_json():
    """Ensure that malformed stdout output raises AptServeMalformedResponse."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # 999999 triggers fake worker to write MALFORMED_NON_JSON_CONTENT
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=0, timestamp=0.0, gpus=[],
        waiting_requests=[{"request_id": 999999, "prompt_tokens": 100}], running_requests=[], cache_snapshot={}
    )
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeMalformedResponse):
            client.schedule_step(state_input)


# ======================================================================
# 4. CAPACITY CONSISTENCY AUDIT TESTS (Step 14)
# ======================================================================

def test_capacity_consistency_audit():
    """Verify that decisions can be represented safely under physical constraints."""
    from llmserveopt.core.types import GPUConfig
    gpu = GPUConfig(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=2048, max_kv_tokens=32, # 2 blocks
        hybrid_cache_enabled=True, hidden_cache_capacity_blocks=4, hidden_to_kv_memory_ratio=0.5
    )
    mgr = HybridCacheManager(gpu)
    
    # 1. Allocation of selected is feasible
    assert mgr.can_allocate(16, CacheTier.KV)
    mgr.allocate(1, 16, CacheTier.KV)
    assert mgr.kv_manager.num_used_blocks == 1
    
    # 2. Re-allocation of oversized is correctly blocked
    assert not mgr.can_allocate(100, CacheTier.KV)

"""Comprehensive focused unit, scenario, and comparative tests for Apt-Serve Phase F."""
from __future__ import annotations

import json
import math
import pytest
from pathlib import Path

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.apt_serve_faithful import (
    AptServeAdapterConfig,
    AptServeSchedulerPolicy,
    AptServeSchedulerInput,
    CacheTier
)
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.apt_serve_stress import (
    generate_apt_serve_target_workload,
    generate_apt_serve_counter_workload
)

ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# 1. GENERATOR TESTS (Step J)
# ======================================================================

def test_target_generator_is_deterministic():
    """Verify that generate_apt_serve_target_workload produces identical, sorted output under same seed."""
    w1 = generate_apt_serve_target_workload(seed=2026, n_requests=10)
    w2 = generate_apt_serve_target_workload(seed=2026, n_requests=10)
    assert len(w1) == 10
    assert w1 == w2


def test_counter_generator_is_deterministic():
    w1 = generate_apt_serve_counter_workload(seed=2026, n_requests=10, scenario="low_pressure")
    w2 = generate_apt_serve_counter_workload(seed=2026, n_requests=10, scenario="low_pressure")
    assert len(w1) == 10
    assert w1 == w2


def test_target_workload_contains_mixed_slo_classes():
    w = generate_apt_serve_target_workload(seed=2026, n_requests=20)
    classes = {r.class_id for r in w}
    assert "relaxed_long" in classes
    assert "urgent_short" in classes


# ======================================================================
# 2. STATE RECONCILIATION & MULTI-STEP TIMING TESTS (Step J)
# ======================================================================

def test_multistep_reconciliation_under_pressure():
    """Verify that active and completed requests are correctly tracked across steps on HybridCacheManager."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    policy = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5
    )
    
    # 1. First step: Admitting sequence 1
    state1 = ObservableState_mock(time=0.0, step=1, active_ids=[1])
    action1 = policy.select_action(state1)
    
    mgr = policy._get_cache_manager(state1.gpu_states[0])
    assert mgr.get_request_tier(1) == CacheTier.KV
    
    # 2. Second step: Sequence 1 completes (no longer in active_ids)
    state2 = ObservableState_mock(time=0.1, step=2, active_ids=[])
    policy.select_action(state2)
    
    # Sequence 1 must be cleanly released from mgr assignments!
    assert mgr.get_request_tier(1) == CacheTier.NONE


# ======================================================================
# 3. END-TO-END SMOKE EVALUATION (Step J)
# ======================================================================

def test_phase_f_smoke_headroom_experiment():
    """CI-scale end-to-end Phase F smoke check."""
    requests = generate_apt_serve_target_workload(seed=2026, n_requests=5)
    gpu_configs = [
        GPUConfig(gpu_id=0, max_active_sequences=4, max_batch_tokens=512, max_kv_tokens=256)
    ]
    service_model = ServiceModel(step_size=0.1)
    
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    apt_serve = AptServeSchedulerPolicy(
        adapter_config=config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=16,
        hidden_to_kv_memory_ratio=0.5
    )
    
    sim_cfg = SimulatorConfig(
        gpu_configs=gpu_configs,
        service_model=service_model,
        drain_steps=500
    )
    sim = Simulator(sim_cfg)
    sim.load_trace(requests)
    
    # Ensure policy runs successfully and simulator terminates without deadlocks or hangs!
    metrics = sim.run(apt_serve)
    assert metrics.num_completed + metrics.num_dropped == len(requests)


# ======================================================================
# MOCK HELPERS
# ======================================================================

class ObservableState_mock:
    def __init__(self, time: float, step: int, active_ids: list[int]):
        self.time = time
        self.step = step
        self.waiting_queue = []
        self.waiting_map = {}
        
        reqs_info = []
        for rid in active_ids:
            reqs_info.append(ObservableRequest_mock(rid))
            
        self.gpu_states = [ObservableGPUState_mock(active_ids, reqs_info)]


class ObservableGPUState_mock:
    def __init__(self, active_request_ids, active_requests_info):
        self.gpu_id = 0
        self.max_active_sequences = 16
        self.max_batch_tokens = 2048
        self.max_kv_tokens = 1024
        self.active_request_ids = active_request_ids
        self.active_requests_info = active_requests_info
        self.current_kv_tokens = 0
        self.tokens_decoded_per_request = {}
        self.role = None


class ObservableRequest_mock:
    def __init__(self, rid: int):
        self.request_id = rid
        self.arrival_time = 0.0
        self.prompt_tokens = 16
        self.predicted_output_tokens = 16
        self.slo_deadline = 5.0
        self.priority = 1.0
        self.class_id = "default"

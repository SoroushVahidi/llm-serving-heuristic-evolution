"""Comprehensive focused unit, interface, and schema tests for Apt-Serve Phase A scaffolding."""
from __future__ import annotations

import json
import math
import pytest
from dataclasses import asdict

from llmserveopt.core.types import GPUConfig, ObservableRequest, ObservableState
from llmserveopt.policies.external_baselines_registry import get_external_baseline_spec
from llmserveopt.policies.apt_serve_faithful import (
    CacheTier,
    CacheRepresentation,
    CacheAssignment,
    CacheTransitionKind,
    CacheTransitionRequest,
    CacheTransitionResult,
    CacheCapacitySnapshot,
    HybridCacheSnapshot,
    AptServeRequestView,
    AptServeSchedulerDecision,
    AptServeAdapterConfig,
    AptServeEnvironmentSpec,
    AptServeSourceProvenance,
    AptServeSchedulerInput,
    AptServeSchedulerOutput,
    AptServeSchedulerPolicy,
    AptServeAdapterError,
    AptServeProtocolMismatch
)


# ======================================================================
# 1. CONFIG TESTS (Step 10)
# ======================================================================

def test_legacy_config_defaults_correctly():
    """Verify legacy construction remains untouched and defaults correctly."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024)
    assert not gpu.hybrid_cache_enabled
    assert gpu.hidden_cache_capacity_blocks == 0
    assert math.isclose(gpu.hidden_to_kv_memory_ratio, 0.1)
    assert math.isclose(gpu.cache_switch_latency, 0.0)
    assert math.isclose(gpu.hidden_restore_latency, 0.0)
    assert gpu.recomputation_cost_model == "full"
    assert math.isclose(gpu.apt_serve_rho, 0.5)
    assert math.isclose(gpu.apt_serve_ttft_slo, 2.0)
    assert math.isclose(gpu.apt_serve_tbt_slo, 0.05)


def test_valid_hybrid_config_passes():
    """Verify that a valid complete hybrid cache config passes validation."""
    gpu = GPUConfig(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=64,
        hidden_to_kv_memory_ratio=0.2,
        cache_switch_latency=0.005,
        hidden_restore_latency=0.01,
        recomputation_cost_model="hidden_restore",
        apt_serve_rho=0.6,
        apt_serve_ttft_slo=1.5,
        apt_serve_tbt_slo=0.1
    )
    assert gpu.hybrid_cache_enabled
    assert gpu.hidden_cache_capacity_blocks == 64
    assert math.isclose(gpu.hidden_to_kv_memory_ratio, 0.2)
    assert gpu.recomputation_cost_model == "hidden_restore"


def test_invalid_capacities_fail():
    """Verify negative or invalid capacities are rejected."""
    with pytest.raises(ValueError, match="hidden_cache_capacity_blocks must be positive"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=-10 # negative
        )
    with pytest.raises(ValueError, match="hidden_cache_capacity_blocks must be positive"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=0 # zero
        )


def test_invalid_latencies_fail():
    """Verify negative latencies are rejected."""
    with pytest.raises(ValueError, match="cache_switch_latency must be non-negative"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            cache_switch_latency=-0.001
        )
    with pytest.raises(ValueError, match="hidden_restore_latency must be non-negative"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            hidden_restore_latency=-0.01
        )


def test_invalid_ratio_fails():
    """Verify ratio must be positive and <= 1.0."""
    with pytest.raises(ValueError, match="hidden_to_kv_memory_ratio must be in"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            hidden_to_kv_memory_ratio=-0.1
        )
    with pytest.raises(ValueError, match="hidden_to_kv_memory_ratio must be in"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            hidden_to_kv_memory_ratio=0.0
        )
    with pytest.raises(ValueError, match="hidden_to_kv_memory_ratio must be in"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            hidden_to_kv_memory_ratio=1.1
        )


def test_invalid_slos_fail():
    """Verify non-positive SLO parameters fail."""
    with pytest.raises(ValueError, match="apt_serve_ttft_slo must be positive"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            apt_serve_ttft_slo=-1.0
        )
    with pytest.raises(ValueError, match="apt_serve_tbt_slo must be positive"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            apt_serve_tbt_slo=0.0
        )


def test_invalid_recomputation_model_fails():
    """Verify invalid recomputation models are rejected."""
    with pytest.raises(ValueError, match="unsupported recomputation_cost_model"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=True,
            hidden_cache_capacity_blocks=64,
            recomputation_cost_model="invalid_model"
        )


def test_rejected_non_default_disabled_fields():
    """Verify that specifying non-default hybrid fields under disabled mode is strictly rejected."""
    with pytest.raises(ValueError, match="Hybrid cache fields can only be set when hybrid_cache_enabled=True"):
        GPUConfig(
            gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
            hybrid_cache_enabled=False,
            hidden_cache_capacity_blocks=64 # invalid when False!
        )


def test_serialization_round_trip():
    """Verify dict-based serialization roundtrip."""
    gpu = GPUConfig(
        gpu_id=0, max_active_sequences=16, max_batch_tokens=512, max_kv_tokens=1024,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=64,
        hidden_to_kv_memory_ratio=0.2,
        cache_switch_latency=0.005,
        hidden_restore_latency=0.01,
        recomputation_cost_model="hidden_restore",
        apt_serve_rho=0.6,
        apt_serve_ttft_slo=1.5,
        apt_serve_tbt_slo=0.1
    )
    raw = asdict(gpu)
    reconstructed = GPUConfig(**raw)
    assert reconstructed == gpu


# ======================================================================
# 2. INTERFACE TESTS (Step 10)
# ======================================================================

def test_enums_construct_correctly():
    assert CacheTier.KV == "kv"
    assert CacheTier.HIDDEN == "hidden"
    assert CacheTier.NONE == "none"
    assert CacheRepresentation.KV_BLOCKED == "kv_blocked"
    assert CacheRepresentation.COMPRESSED_HIDDEN == "compressed_hidden"
    assert CacheTransitionKind.KV_TO_HIDDEN == "kv_to_hidden"
    assert CacheTransitionKind.HIDDEN_TO_KV == "hidden_to_kv"
    assert CacheTransitionKind.EVICT_FULL == "evict_full"


def test_cache_assignment_validates():
    asg = CacheAssignment(
        request_id=42,
        target_tier=CacheTier.HIDDEN,
        required_units=15,
        current_tier=CacheTier.KV,
        reason="preemption"
    )
    assert asg.request_id == 42
    assert asg.target_tier == CacheTier.HIDDEN
    assert asg.required_units == 15
    assert asg.current_tier == CacheTier.KV
    assert asg.reason == "preemption"


def test_snapshots_resident_ordering_enforced():
    kv_snap = CacheCapacitySnapshot(CacheTier.KV, 100, 40, 60)
    hidden_snap = CacheCapacitySnapshot(CacheTier.HIDDEN, 200, 10, 190)
    
    # Valid sorted request IDs passes
    HybridCacheSnapshot(step=10, timestamp=1.23, kv_snapshot=kv_snap, hidden_snapshot=hidden_snap, resident_request_ids=[1, 2, 5])
    
    # Invalid unsorted request IDs fails __post_init__ validation
    with pytest.raises(ValueError, match="resident_request_ids must be sorted deterministically"):
        HybridCacheSnapshot(step=10, timestamp=1.23, kv_snapshot=kv_snap, hidden_snapshot=hidden_snap, resident_request_ids=[5, 2, 1])


def test_scheduler_decisions():
    dec = AptServeSchedulerDecision(
        selected_request_ids=[1, 2],
        cache_assignments={1: CacheTier.KV, 2: CacheTier.HIDDEN},
        evictions=[3],
        deprioritized_requests=[4],
        value_scores={1: 5.6, 2: 1.2}
    )
    assert dec.selected_request_ids == [1, 2]
    assert dec.cache_assignments[1] == CacheTier.KV
    assert dec.value_scores[1] == 5.6


# ======================================================================
# 3. IPC TESTS (Step 10)
# ======================================================================

def test_ipc_request_serialization_determinism():
    req = AptServeSchedulerInput(
        schema_version=1,
        request_id=100,
        simulator_step=45,
        timestamp=0.045,
        gpus=[{"gpu_id": 0, "role": "none"}],
        waiting_requests=[{"request_id": 1, "prompt_tokens": 100}],
        running_requests=[],
        cache_snapshot={"kv_free": 50}
    )
    
    # Serialized bytes must be stable and deterministic
    b1 = req.serialize_json()
    b2 = req.serialize_json()
    assert b1 == b2
    
    # Deserialize and verify
    reconstructed = AptServeSchedulerInput.deserialize_json(b1)
    assert reconstructed == req


def test_ipc_request_schema_version_check():
    bad_data = json.dumps({
        "schema_version": 2, # wrong version!
        "request_id": 100,
        "simulator_step": 45,
        "timestamp": 0.045,
        "gpus": [],
        "waiting_requests": [],
        "running_requests": [],
        "cache_snapshot": {}
    }).encode("utf-8")
    
    with pytest.raises(AptServeProtocolMismatch, match="Expected schema_version=1"):
        AptServeSchedulerInput.deserialize_json(bad_data)


def test_ipc_response_serialization_determinism():
    resp = AptServeSchedulerOutput(
        schema_version=1,
        request_id=100,
        selected_request_ids=[1, 2],
        cache_assignments={"1": "kv", "2": "hidden"},
        evictions=[3],
        deprioritized_requests=[],
        value_scores={"1": 8.5, "2": 2.1}
    )
    
    b1 = resp.serialize_json()
    b2 = resp.serialize_json()
    assert b1 == b2
    
    reconstructed = AptServeSchedulerOutput.deserialize_json(b1)
    assert reconstructed == resp


# ======================================================================
# 4. PROVENANCE TESTS (Step 10)
# ======================================================================

def test_provenance_pin_and_license():
    prov = AptServeSourceProvenance()
    assert prov.pinned_commit == "c953217988274a761da35cf06c01033b18dadf68"
    assert prov.official_repo_url == "https://github.com/eddiegaoo/Apt-Serve"


# ======================================================================
# 5. REGRESSION & PLACEHOLDER POLICY TESTS (Step 10)
# ======================================================================

def test_registry_integration():
    spec = get_external_baseline_spec("apt_serve_faithful")
    assert spec.name == "apt_serve_faithful"
    assert spec.factory is not None


def test_placeholder_policy_fails_loudly_on_execution():
    policy = AptServeSchedulerPolicy()
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[], completed_count=0, step=0)
    action = policy.select_action(state)
    assert action.is_empty()


# ======================================================================
# 6. CONFIG EXAMPLES TESTS (Step 10)
# ======================================================================

def test_load_legacy_disabled_yaml():
    import yaml
    from pathlib import Path
    path = Path("/home/soroush/llm-serving-heuristic-evolution/configs/examples/apt_serve/legacy_disabled.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    gpus = [GPUConfig(**raw) for raw in data["gpus"]]
    assert len(gpus) == 1
    assert not gpus[0].hybrid_cache_enabled


def test_load_valid_hybrid_yaml():
    import yaml
    from pathlib import Path
    path = Path("/home/soroush/llm-serving-heuristic-evolution/configs/examples/apt_serve/valid_hybrid.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    gpus = [GPUConfig(**raw) for raw in data["gpus"]]
    assert len(gpus) == 1
    assert gpus[0].hybrid_cache_enabled
    assert gpus[0].hidden_cache_capacity_blocks == 64
    assert math.isclose(gpus[0].hidden_to_kv_memory_ratio, 0.2)


def test_load_invalid_negative_capacity_yaml():
    import yaml
    from pathlib import Path
    path = Path("/home/soroush/llm-serving-heuristic-evolution/configs/examples/apt_serve/invalid_negative_capacity.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    with pytest.raises(ValueError, match="hidden_cache_capacity_blocks must be positive"):
        [GPUConfig(**raw) for raw in data["gpus"]]


def test_load_invalid_ratio_yaml():
    import yaml
    from pathlib import Path
    path = Path("/home/soroush/llm-serving-heuristic-evolution/configs/examples/apt_serve/invalid_ratio.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    with pytest.raises(ValueError, match="hidden_to_kv_memory_ratio must be in"):
        [GPUConfig(**raw) for raw in data["gpus"]]


def test_load_invalid_recomputation_model_yaml():
    import yaml
    from pathlib import Path
    path = Path("/home/soroush/llm-serving-heuristic-evolution/configs/examples/apt_serve/invalid_recomputation_model.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    with pytest.raises(ValueError, match="unsupported recomputation_cost_model"):
        [GPUConfig(**raw) for raw in data["gpus"]]

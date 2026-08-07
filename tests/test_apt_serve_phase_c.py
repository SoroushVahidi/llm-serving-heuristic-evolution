"""Comprehensive focused unit, verification, protocol, and recorded-trace tests for Apt-Serve Phase C."""
from __future__ import annotations

import json
import os
import sys
import pytest
import subprocess
import hashlib
from pathlib import Path

from llmserveopt.policies.apt_serve_faithful import (
    AptServeAdapterConfig,
    AptServeSubprocessClient,
    AptServeSchedulerInput,
    AptServeSchedulerOutput,
    AptServeSchedulerDecision,
    CacheTier,
    AptServeSourceCheckoutMissing,
    AptServeWrongCommit,
    AptServeSourceHashMismatch,
    AptServeEnvironmentMissing,
    AptServeProtocolMismatch,
    AptServeSubprocessTimeout,
    AptServeMalformedResponse,
    AptServeInvalidSchedulerDecision,
    AptServeUnsupportedConfiguration,
    AptServeAdapterError,
    AptServeSourceProvenance
)

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "results" / "provenance" / "apt_serve_strategy_probe"


# ======================================================================
# 1. CLIENT & LIFE CYCLE TESTS (Step 14)
# ======================================================================

def test_client_valid_one_shot_test_mode():
    """Verify that AptServeSubprocessClient initializes and runs cleanly under 'test' execution mode."""
    config = AptServeAdapterConfig(
        checkout_path="",
        conda_env_name="test-env",
        execution_mode="test"
    )
    
    state_input = AptServeSchedulerInput(
        schema_version=1,
        request_id=1,
        simulator_step=10,
        timestamp=0.1,
        gpus=[],
        waiting_requests=[{"request_id": 42, "prompt_tokens": 100}],
        running_requests=[],
        cache_snapshot={}
    )
    
    with AptServeSubprocessClient(config) as client:
        decision = client.schedule_step(state_input)
        assert isinstance(decision, AptServeSchedulerDecision)
        assert decision.selected_request_ids == [42]
        assert decision.cache_assignments[42] == CacheTier.KV


def test_client_worker_crash_translated_correctly():
    """Verify that a worker crash (nonzero exit) raises AptServeAdapterError."""
    config = AptServeAdapterConfig(
        checkout_path="",
        execution_mode="test"
    )
    # 888888 request ID triggers fake worker crash
    state_input = AptServeSchedulerInput(
        schema_version=1,
        request_id=1,
        simulator_step=10,
        timestamp=0.1,
        gpus=[],
        waiting_requests=[{"request_id": 888888, "prompt_tokens": 100}],
        running_requests=[],
        cache_snapshot={}
    )
    
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(Exception, match="exited with non-zero code"):
            client.schedule_step(state_input)


def test_client_timeout_handled_and_cleaned():
    """Verify that a slow worker trigger a SubprocessTimeout exception."""
    config = AptServeAdapterConfig(
        checkout_path="",
        execution_mode="test",
        subprocess_timeout_seconds=0.5 # tight timeout
    )
    # 777777 request ID triggers 20 seconds sleep
    state_input = AptServeSchedulerInput(
        schema_version=1,
        request_id=1,
        simulator_step=10,
        timestamp=0.1,
        gpus=[],
        waiting_requests=[{"request_id": 777777, "prompt_tokens": 100}],
        running_requests=[],
        cache_snapshot={}
    )
    
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeSubprocessTimeout):
            client.schedule_step(state_input)
        assert client.proc is None # closed and cleaned up


def test_client_malformed_response_fails():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # 999999 request ID triggers malformed non-JSON response
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=10, timestamp=0.1, gpus=[],
        waiting_requests=[{"request_id": 999999, "prompt_tokens": 100}], running_requests=[], cache_snapshot={}
    )
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeMalformedResponse):
            client.schedule_step(state_input)


def test_client_wrong_schema_version_fails():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # 666666 request ID triggers schema version 2 response
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=10, timestamp=0.1, gpus=[],
        waiting_requests=[{"request_id": 666666, "prompt_tokens": 100}], running_requests=[], cache_snapshot={}
    )
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeProtocolMismatch):
            client.schedule_step(state_input)


def test_client_invalid_decision_selected_not_in_input():
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # 555555 request ID triggers response with selected ID 12345 (which is not in input)
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=10, timestamp=0.1, gpus=[],
        waiting_requests=[{"request_id": 555555, "prompt_tokens": 100}], running_requests=[], cache_snapshot={}
    )
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeInvalidSchedulerDecision):
            client.schedule_step(state_input)


def test_client_oversized_payload_rejected():
    """Verify that inputs exceeding 10MB payload size are rejected immediately."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    # Large list of dummy requests to build > 10MB json payload
    large_list = [{"request_id": i, "prompt_tokens": 10, "payload": "x" * 10000} for i in range(1050)]
    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=10, timestamp=0.1, gpus=[],
        waiting_requests=large_list, running_requests=[], cache_snapshot={}
    )
    with AptServeSubprocessClient(config) as client:
        with pytest.raises(AptServeUnsupportedConfiguration, match="exceeds maximum limit"):
            client.schedule_step(state_input)


# ======================================================================
# 2. SOURCE VERIFICATION TESTS (Step 14)
# ======================================================================

def test_source_verification_missing_checkout_directory():
    config = AptServeAdapterConfig(
        checkout_path="/nonexistent/path/here",
        execution_mode="official"
    )
    client = AptServeSubprocessClient(config)
    with pytest.raises(AptServeSourceCheckoutMissing):
        client.initialize()


def test_source_verification_non_git_path(tmp_path):
    config = AptServeAdapterConfig(
        checkout_path=str(tmp_path),
        execution_mode="official"
    )
    client = AptServeSubprocessClient(config)
    with pytest.raises(AptServeSourceCheckoutMissing, match="not a git repository"):
        client.initialize()


def test_source_verification_wrong_git_commit(tmp_path):
    # Initialize fake git repo with a different commit hash
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
    # create dummy file
    (tmp_path / "dummy.txt").write_text("hello")
    subprocess.run(["git", "add", "dummy.txt"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, capture_output=True)
    
    config = AptServeAdapterConfig(
        checkout_path=str(tmp_path),
        execution_mode="official"
    )
    client = AptServeSubprocessClient(config)
    with pytest.raises(AptServeWrongCommit, match="mismatch"):
        client.initialize()


def test_source_verification_hash_mismatch(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
    (tmp_path / "dummy.txt").write_text("hello")
    subprocess.run(["git", "add", "dummy.txt"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, capture_output=True)
    
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()

    # Put incorrect content in expected files to trigger hash mismatches
    os.makedirs(tmp_path / "additional_designs" / "core", exist_ok=True)
    (tmp_path / "additional_designs" / "aptserve_block.py").write_text("wrong content")
    
    config = AptServeAdapterConfig(
        checkout_path=str(tmp_path),
        execution_mode="official"
    )
    client = AptServeSubprocessClient(config)
    # Override the pinned commit inside the client provenance to bypass checkout check and hit hash check!
    client.provenance = AptServeSourceProvenance(pinned_commit=commit)
    
    with pytest.raises(AptServeSourceHashMismatch, match="hash mismatch"):
        client.initialize()


# ======================================================================
# 3. ENVIRONMENT VERIFICATION TESTS (Step 14)
# ======================================================================

def test_environment_verification_missing_python(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path)
    (tmp_path / "dummy.txt").write_text("hello")
    subprocess.run(["git", "add", "dummy.txt"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, capture_output=True)
    
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()

    # Stub the 5 source files with their correct hashes so source verification succeeds!
    os.makedirs(tmp_path / "additional_designs" / "core", exist_ok=True)
    import hashlib
    # We can write contents that match the actual expected hashes!
    # vllm/block.py
    (tmp_path / "additional_designs" / "aptserve_block.py").write_text("dummy")
    # we can bypass the hash verification check by temporarily overriding the client's expected hashes or prov
    
    config = AptServeAdapterConfig(
        checkout_path=str(tmp_path),
        python_executable="/nonexistent/bin/python3.11_test",
        execution_mode="official"
    )
    client = AptServeSubprocessClient(config)
    # Bypass checkout and hash verification checks
    client.provenance = AptServeSourceProvenance(pinned_commit=commit)
    global APT_SERVE_EXPECTED_HASHES
    from llmserveopt.policies.apt_serve_faithful import APT_SERVE_EXPECTED_HASHES
    # stub expected hashes to be empty so it skips files
    import llmserveopt.policies.apt_serve_faithful
    llmserveopt.policies.apt_serve_faithful.APT_SERVE_EXPECTED_HASHES = {}
    
    with pytest.raises(AptServeEnvironmentMissing, match="not found"):
        client.initialize()


# ======================================================================
# 4. PROTOCOL & SERIALIZATION TESTS (Step 14)
# ======================================================================

def test_deterministic_input_serialization():
    inp1 = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=1, timestamp=0.5,
        gpus=[{"id": 0}], waiting_requests=[{"id": 10}, {"id": 20}], running_requests=[], cache_snapshot={}
    )
    inp2 = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=1, timestamp=0.5,
        gpus=[{"id": 0}], waiting_requests=[{"id": 10}, {"id": 20}], running_requests=[], cache_snapshot={}
    )
    assert inp1.serialize_json() == inp2.serialize_json()


def test_deterministic_output_serialization():
    out1 = AptServeSchedulerOutput(
        schema_version=1, request_id=1, selected_request_ids=[10, 20],
        cache_assignments={"10": "kv", "20": "hidden"}, evictions=[], deprioritized_requests=[], value_scores={"10": 1.5}
    )
    out2 = AptServeSchedulerOutput(
        schema_version=1, request_id=1, selected_request_ids=[10, 20],
        cache_assignments={"10": "kv", "20": "hidden"}, evictions=[], deprioritized_requests=[], value_scores={"10": 1.5}
    )
    assert out1.serialize_json() == out2.serialize_json()


# ======================================================================
# 5. RECORDED-TRACE COMPATIBILITY TESTS (Step 12)
# ======================================================================

def test_recorded_micro_traces_can_be_parsed():
    """Verify that the committed official micro-traces under results/provenance parse correctly."""
    trace_path = PROVENANCE_DIR / "micro_trace.json"
    assert trace_path.exists()
    
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["apt_serve_commit"] == "c953217988274a761da35cf06c01033b18dadf68"
    
    # 3/3 committed micro-traces are fully valid
    for scenario in data["scenarios"]:
        assert scenario["status"] == "OK"
        assert isinstance(scenario["scheduled_request_ids"], list)
        assert "name" in scenario


def test_recorded_scenarios_input_representation():
    """Verify that recorded scenarios can be fully represented by AptServeSchedulerInput."""
    trace_path = PROVENANCE_DIR / "micro_trace.json"
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    
    for s in data["scenarios"]:
        waiting_reqs = []
        for req in s["requests"]:
            waiting_reqs.append({
                "request_id": hash(req["request_id"]),
                "prompt_tokens": req["prompt_len"],
                "arrival_time": 0.0,
                "predicted_output_tokens": 16,
                "current_cache_tier": "none"
            })
            
        inp = AptServeSchedulerInput(
            schema_version=1,
            request_id=100,
            simulator_step=0,
            timestamp=0.0,
            gpus=[{"max_kv_tokens": 1024, "max_active_sequences": 16, "max_batch_tokens": 2048}],
            waiting_requests=waiting_reqs,
            running_requests=[],
            cache_snapshot={}
        )
        serialized = inp.serialize_json()
        deserialized = AptServeSchedulerInput.deserialize_json(serialized)
        assert deserialized.schema_version == 1
        assert len(deserialized.waiting_requests) == len(waiting_reqs)

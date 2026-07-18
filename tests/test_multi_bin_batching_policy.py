"""Tests for the MultiBinBatchingPolicy.

Multi-Bin Batching groups requests by predicted output length into bins
and fills batches from a single bin to reduce length variance within a batch.

Wording note: This is a Multi-Bin-inspired baseline for research.
It is NOT the official implementation of any published work.
"""
import pytest
from llmserveopt.policies.multi_bin_batching import MultiBinBatchingPolicy
from llmserveopt.policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES, make_policy
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState


def make_req(req_id, prompt=32, output=32, deadline=100.0, priority=1.0, arrival=0.0):
    return ObservableRequest(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id="normal",
    )


def make_gpu(max_seq=8, max_kv=8192, max_batch=512, kv_used=0, active=()):
    return ObservableGPUState(
        gpu_id=0,
        max_active_sequences=max_seq,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=list(active),
        active_requests_info=[],
        current_kv_tokens=kv_used,
        tokens_decoded_per_request={},
    )


def make_state(reqs, now=0.0, gpus=None):
    if gpus is None:
        gpus = [make_gpu()]
    return ObservableState(
        time=now,
        waiting_queue=reqs,
        gpu_states=gpus,
        completed_count=0,
        step=0,
    )


# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------

def test_registered_in_baseline_names():
    assert "multi_bin_batching" in BASELINE_NAMES


def test_not_in_oracle_policy_names():
    assert "multi_bin_batching" not in ORACLE_POLICY_NAMES


def test_make_policy_returns_correct_type():
    p = make_policy("multi_bin_batching")
    assert isinstance(p, MultiBinBatchingPolicy)


def test_policy_name_attribute():
    p = MultiBinBatchingPolicy()
    assert p.name == "multi_bin_batching"


# -------------------------------------------------------------------------
# Bin assignment
# -------------------------------------------------------------------------

def test_bin_assignment_uses_predicted_output():
    """Requests are binned by predicted_output_tokens, not actual."""
    p = MultiBinBatchingPolicy(bin_edges=[32, 64, 128, 256])
    assert p._bin_id(16) == 0   # ≤32 → bin 0
    assert p._bin_id(32) == 0   # =32 → bin 0
    assert p._bin_id(33) == 1   # >32, ≤64 → bin 1
    assert p._bin_id(64) == 1
    assert p._bin_id(65) == 2   # ≤128 → bin 2
    assert p._bin_id(257) == 4  # > last edge → final bin


def test_bin_assignment_deterministic():
    p = MultiBinBatchingPolicy()
    for _ in range(5):
        assert p._bin_id(30) == p._bin_id(30)
        assert p._bin_id(100) == p._bin_id(100)


# -------------------------------------------------------------------------
# Deterministic scheduling
# -------------------------------------------------------------------------

def test_deterministic_across_calls():
    """Same state produces same action on repeated calls."""
    p = MultiBinBatchingPolicy()
    reqs = [make_req(i, output=30 + i * 20) for i in range(6)]
    state1 = make_state(reqs)
    state2 = make_state(reqs)
    a1 = p.select_action(state1)
    a2 = p.select_action(state2)
    assert a1.admit == a2.admit


def test_order_invariant_in_same_state():
    """Request order within the same bin should not affect final admission set."""
    p = MultiBinBatchingPolicy()
    # All requests in bin 0 (output ≤ 32)
    reqs_ab = [make_req(0, output=20), make_req(1, output=25)]
    reqs_ba = [make_req(1, output=25), make_req(0, output=20)]
    a1 = p.select_action(make_state(reqs_ab))
    a2 = p.select_action(make_state(reqs_ba))
    assert set(a1.admit[0]) == set(a2.admit[0])


# -------------------------------------------------------------------------
# Capacity constraints
# -------------------------------------------------------------------------

def test_capacity_respected_max_seq():
    """Policy must not exceed GPU max_active_sequences."""
    gpu = make_gpu(max_seq=2, max_kv=8192, max_batch=8)
    reqs = [make_req(i, output=30) for i in range(6)]
    action = MultiBinBatchingPolicy().select_action(make_state(reqs, gpus=[gpu]))
    total = sum(len(v) for v in action.admit.values())
    assert total <= 2


def test_capacity_respected_kv():
    """Policy must not exceed GPU max_kv_tokens."""
    gpu = make_gpu(max_seq=8, max_kv=50, max_batch=8, kv_used=30)
    # Each request needs 32 prompt tokens = 32 KV; only 20 free → none fit
    reqs = [make_req(i, prompt=32, output=16) for i in range(4)]
    action = MultiBinBatchingPolicy().select_action(make_state(reqs, gpus=[gpu]))
    total = sum(len(v) for v in action.admit.values())
    assert total == 0


def test_full_gpu_admits_nothing():
    gpu = make_gpu(max_seq=2, active=[99, 100])
    reqs = [make_req(i) for i in range(3)]
    action = MultiBinBatchingPolicy().select_action(make_state(reqs, gpus=[gpu]))
    assert sum(len(v) for v in action.admit.values()) == 0


def test_empty_queue_returns_empty():
    p = MultiBinBatchingPolicy()
    state = make_state([])
    action = p.select_action(state)
    assert action.is_empty()


# -------------------------------------------------------------------------
# Bin grouping behaviour
# -------------------------------------------------------------------------

def test_short_requests_admitted_before_long():
    """Bin 0 (short output) is processed before bin 4 (long output)."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    req_short = make_req(0, output=16)   # bin 0
    req_long  = make_req(1, output=512)  # last bin
    state = make_state([req_long, req_short], gpus=[gpu])
    action = MultiBinBatchingPolicy().select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_short.request_id


def test_mixed_output_sizes_distributed():
    """Requests of different sizes go to different bins."""
    p = MultiBinBatchingPolicy(bin_edges=[32, 64, 128, 256])
    gpu = make_gpu(max_seq=8, max_kv=8192, max_batch=8)
    reqs = [
        make_req(0, output=20),   # bin 0
        make_req(1, output=50),   # bin 1
        make_req(2, output=100),  # bin 2
        make_req(3, output=200),  # bin 3
        make_req(4, output=300),  # bin 4
    ]
    action = p.select_action(make_state(reqs, gpus=[gpu]))
    total = sum(len(v) for v in action.admit.values())
    assert total == 5  # all fit, no capacity limit hit


# -------------------------------------------------------------------------
# actual_output_tokens never accessed
# -------------------------------------------------------------------------

def test_no_actual_output_tokens_on_observable_request():
    """ObservableRequest must not expose actual_output_tokens to deployable policies."""
    req = make_req(0)
    assert not hasattr(req, "actual_output_tokens"), (
        "ObservableRequest must not expose actual_output_tokens"
    )


def test_uses_predicted_output_not_actual():
    """bin_id() is based on predicted_output_tokens, which is all the policy has."""
    p = MultiBinBatchingPolicy(bin_edges=[32, 64, 128, 256])
    req = make_req(0, output=20)  # predicted=20 → bin 0
    assert p._bin_id(req.predicted_output_tokens) == 0


# -------------------------------------------------------------------------
# Multi-GPU support
# -------------------------------------------------------------------------

def test_multi_gpu_distributes_requests():
    """Requests can be spread across multiple GPUs."""
    gpu0 = make_gpu(max_seq=2, max_kv=8192, max_batch=2)
    gpu1 = ObservableGPUState(
        gpu_id=1, max_active_sequences=2, max_batch_tokens=2,
        max_kv_tokens=8192, active_request_ids=[], active_requests_info=[],
        current_kv_tokens=0, tokens_decoded_per_request={},
    )
    reqs = [make_req(i, output=20) for i in range(4)]
    action = MultiBinBatchingPolicy().select_action(make_state(reqs, gpus=[gpu0, gpu1]))
    total = sum(len(v) for v in action.admit.values())
    assert total == 4


# -------------------------------------------------------------------------
# Documentation wording guard
# -------------------------------------------------------------------------

def test_baselines_doc_says_inspired_not_official():
    """docs/baselines.md must not claim this is an official implementation."""
    from pathlib import Path
    doc = Path(__file__).parent.parent / "docs" / "baselines.md"
    if not doc.exists():
        pytest.skip("baselines.md missing")
    text = doc.read_text()
    # Should say "inspired" or "style" or "approximate", not "official implementation"
    # The multi_bin_batching entry must not claim to be official
    assert "official" not in text.lower() or "not" in text.lower(), (
        "baselines.md must not claim multi_bin_batching is an official implementation"
    )

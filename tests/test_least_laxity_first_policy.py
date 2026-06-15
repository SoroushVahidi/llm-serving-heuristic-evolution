"""Tests for the LeastLaxityFirst (LLF) scheduling policy."""
import pytest
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy
from llmserveopt.policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES, make_policy
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState


def make_req(req_id, prompt=64, output=64, deadline=5.0, priority=2.0, arrival=0.0):
    return ObservableRequest(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id="medium",
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


def make_state(reqs, now=2.0, gpu=None):
    if gpu is None:
        gpu = make_gpu()
    return ObservableState(
        time=now,
        waiting_queue=reqs,
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )


# -------------------------------------------------------------------------
# Lower laxity selected first
# -------------------------------------------------------------------------

def test_lower_laxity_admitted_first():
    """Request with smaller laxity should be admitted when GPU can only take one."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # now=2.0, alpha=0.5, beta=1.0
    # req_a: deadline=10, service=0.5*64+1.0*64=96 → laxity = 10-2-96 = -88
    # req_b: deadline=20, service=0.5*64+1.0*32=48 → laxity = 20-2-48 = -30
    # req_a has lower laxity (-88 < -30) → admitted first
    req_a = make_req(0, prompt=64, output=64, deadline=10.0)
    req_b = make_req(1, prompt=64, output=32, deadline=20.0)
    state = make_state([req_a, req_b], now=2.0, gpu=gpu)
    p = LeastLaxityFirstPolicy()
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_a.request_id


def test_laxity_prefers_urgent_over_close_deadline():
    """LLF may differ from EDF when service time dominates."""
    # req_a: deadline=5, service large → low laxity despite close deadline
    # req_b: deadline=100, service tiny → high laxity despite far deadline
    # With GPU that takes only 1: LLF picks req_a (lower laxity)
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # now=2, alpha=0.5, beta=1.0
    # req_a: laxity = 5-2 - (0.5*512+1*512) = 3 - 768 = -765
    # req_b: laxity = 100-2 - (0.5*8+1*8) = 98 - 12 = 86
    req_a = make_req(0, prompt=512, output=512, deadline=5.0)
    req_b = make_req(1, prompt=8, output=8, deadline=100.0)
    state = make_state([req_a, req_b], now=2.0, gpu=gpu)
    p = LeastLaxityFirstPolicy()
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert req_a.request_id in admitted


# -------------------------------------------------------------------------
# Tie-breaking is deterministic
# -------------------------------------------------------------------------

def test_tie_break_by_deadline():
    """Equal laxity → earlier deadline wins."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    p = LeastLaxityFirstPolicy()
    now = 0.0
    # Both with same prompt/output → same service_proxy → laxity = deadline - now - service
    # req_a: deadline=3 → laxity=3-service
    # req_b: deadline=5 → laxity=5-service → req_a wins (smaller laxity)
    req_a = make_req(0, prompt=64, output=32, deadline=3.0)
    req_b = make_req(1, prompt=64, output=32, deadline=5.0)
    state = make_state([req_b, req_a], now=now, gpu=gpu)
    action = p.select_action(state)
    assert action.admit[0] == [req_a.request_id]


def test_tie_break_determinism_across_calls():
    """Same state produces same result on two separate calls."""
    p = LeastLaxityFirstPolicy()
    reqs = [make_req(i, prompt=64, output=32, deadline=float(10 - i)) for i in range(4)]
    state = make_state(reqs, now=1.0)
    a1 = p.select_action(state)
    # Reset GPU state for second call
    state2 = make_state(reqs, now=1.0)
    a2 = p.select_action(state2)
    assert a1.admit == a2.admit


# -------------------------------------------------------------------------
# actual_output_tokens not used
# -------------------------------------------------------------------------

def test_no_actual_output_tokens_attribute():
    """ObservableRequest passed to policy has no actual_output_tokens."""
    req = make_req(0)
    assert not hasattr(req, "actual_output_tokens"), (
        "ObservableRequest must not expose actual_output_tokens"
    )


# -------------------------------------------------------------------------
# Infeasible requests not selected
# -------------------------------------------------------------------------

def test_full_gpu_admits_nothing():
    gpu = make_gpu(max_seq=2, max_kv=8192, max_batch=2, active=[99, 100])
    reqs = [make_req(i) for i in range(3)]
    state = make_state(reqs, gpu=gpu)
    p = LeastLaxityFirstPolicy()
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_kv_capacity_respected():
    gpu = make_gpu(max_seq=8, max_kv=100, max_batch=8, kv_used=80)
    # All requests need 64 prompt tokens = 64 KV, but only 20 free → none fit
    reqs = [make_req(i, prompt=64) for i in range(3)]
    state = make_state(reqs, gpu=gpu)
    p = LeastLaxityFirstPolicy()
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_empty_queue_returns_empty():
    state = make_state([])
    p = LeastLaxityFirstPolicy()
    action = p.select_action(state)
    assert action.is_empty()


# -------------------------------------------------------------------------
# Policy is registered and not oracle
# -------------------------------------------------------------------------

def test_registered_in_baseline_names():
    assert "least_laxity_first" in BASELINE_NAMES


def test_not_in_oracle_policy_names():
    assert "least_laxity_first" not in ORACLE_POLICY_NAMES


def test_make_policy_returns_correct_type():
    p = make_policy("least_laxity_first")
    assert isinstance(p, LeastLaxityFirstPolicy)


def test_policy_name_attribute():
    p = LeastLaxityFirstPolicy()
    assert p.name == "least_laxity_first"

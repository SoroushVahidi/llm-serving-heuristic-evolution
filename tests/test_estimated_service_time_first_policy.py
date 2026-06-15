"""Tests for EstimatedServiceTimeFirst (ESTF) policy — PARS-inspired SJF proxy."""
import pytest
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES, make_policy
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState


def make_req(req_id, prompt=64, output=64, deadline=10.0, priority=2.0, arrival=0.0):
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


def make_state(reqs, now=1.0, gpu=None):
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
# Shorter estimated service time chosen first
# -------------------------------------------------------------------------

def test_shorter_service_admitted_first_output():
    """Request with smaller predicted_output_tokens admitted first (beta dominates)."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # alpha=0.5, beta=1.0 (defaults)
    # req_a: 0.5*64 + 1.0*32 = 64  (shorter)
    # req_b: 0.5*64 + 1.0*128 = 160 (longer)
    req_a = make_req(0, prompt=64, output=32)
    req_b = make_req(1, prompt=64, output=128)
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert action.admit.get(0) == [req_a.request_id]


def test_shorter_service_admitted_first_prompt():
    """Request with smaller prompt_tokens (and same output) admitted first."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # alpha=0.5, beta=1.0
    # req_a: 0.5*32 + 1.0*64 = 80  (shorter)
    # req_b: 0.5*256 + 1.0*64 = 192 (longer)
    req_a = make_req(0, prompt=32, output=64)
    req_b = make_req(1, prompt=256, output=64)
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert action.admit.get(0) == [req_a.request_id]


def test_both_prompt_and_output_affect_ranking():
    """Both alpha*prompt and beta*output contribute to the score."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # req_a: 0.5*200 + 1.0*10 = 110  → shorter
    # req_b: 0.5*10  + 1.0*200 = 205 → longer
    req_a = make_req(0, prompt=200, output=10)
    req_b = make_req(1, prompt=10, output=200)
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert action.admit.get(0) == [req_a.request_id]


# -------------------------------------------------------------------------
# Tie-breaking is deterministic
# -------------------------------------------------------------------------

def test_tie_break_by_deadline():
    """Equal estimated service time → earlier deadline wins."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # Same prompt/output → same est
    req_a = make_req(0, prompt=64, output=64, deadline=3.0)
    req_b = make_req(1, prompt=64, output=64, deadline=7.0)
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert action.admit[0] == [req_a.request_id]


def test_tie_break_by_priority():
    """Equal service time and deadline → higher priority wins."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    req_a = make_req(0, prompt=64, output=64, deadline=5.0, priority=3.0)
    req_b = make_req(1, prompt=64, output=64, deadline=5.0, priority=1.0)
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert action.admit[0] == [req_a.request_id]


def test_tie_break_by_request_id():
    """All else equal → lower request_id wins."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    req_a = make_req(0, prompt=64, output=64, deadline=5.0, priority=2.0)
    req_b = make_req(5, prompt=64, output=64, deadline=5.0, priority=2.0)
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert action.admit[0] == [req_a.request_id]


def test_determinism_across_calls():
    p = EstimatedServiceTimeFirstPolicy()
    reqs = [make_req(i, prompt=64 + i * 10, output=64 - i * 5) for i in range(4)]
    state = make_state(reqs)
    a1 = p.select_action(state)
    state2 = make_state(reqs)
    a2 = p.select_action(state2)
    assert a1.admit == a2.admit


# -------------------------------------------------------------------------
# actual_output_tokens not used
# -------------------------------------------------------------------------

def test_no_actual_output_on_observable_request():
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
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_kv_limit_respected():
    gpu = make_gpu(max_seq=8, max_kv=100, max_batch=8, kv_used=80)
    reqs = [make_req(i, prompt=64) for i in range(3)]  # each needs 64 KV, only 20 free
    state = make_state(reqs, gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy()
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_empty_queue_returns_empty():
    state = make_state([])
    p = EstimatedServiceTimeFirstPolicy()
    assert p.select_action(state).is_empty()


# -------------------------------------------------------------------------
# Policy is registered and not oracle
# -------------------------------------------------------------------------

def test_registered_in_baseline_names():
    assert "estimated_service_time_first" in BASELINE_NAMES


def test_not_in_oracle_policy_names():
    assert "estimated_service_time_first" not in ORACLE_POLICY_NAMES


def test_make_policy_returns_correct_type():
    p = make_policy("estimated_service_time_first")
    assert isinstance(p, EstimatedServiceTimeFirstPolicy)


def test_policy_name_attribute():
    p = EstimatedServiceTimeFirstPolicy()
    assert p.name == "estimated_service_time_first"


# -------------------------------------------------------------------------
# Custom alpha/beta
# -------------------------------------------------------------------------

def test_custom_alpha_beta_changes_ranking():
    """With alpha=0, only output tokens matter."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    # alpha=0, beta=1 → only predicted_output matters
    # req_a: output=32 → est=32  (shorter)
    # req_b: output=200 → est=200 (longer), despite short prompt
    req_a = make_req(0, prompt=512, output=32)   # large prompt but short output
    req_b = make_req(1, prompt=8, output=200)    # tiny prompt, long output
    state = make_state([req_b, req_a], gpu=gpu)
    p = EstimatedServiceTimeFirstPolicy(alpha=0.0, beta=1.0)
    action = p.select_action(state)
    assert action.admit[0] == [req_a.request_id]

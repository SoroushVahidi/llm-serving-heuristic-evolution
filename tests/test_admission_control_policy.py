"""Tests for the AdmissionControlPolicy."""
import pytest
from llmserveopt.policies.admission_control import AdmissionControlPolicy
from llmserveopt.policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES, make_policy
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState


def make_req(req_id, prompt=64, output=64, deadline=100.0, priority=1.0, arrival=0.0):
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
# Basic registration and identity
# -------------------------------------------------------------------------

def test_registered_in_baseline_names():
    assert "admission_control" in BASELINE_NAMES


def test_not_in_oracle_policy_names():
    assert "admission_control" not in ORACLE_POLICY_NAMES


def test_make_policy_returns_correct_type():
    p = make_policy("admission_control")
    assert isinstance(p, AdmissionControlPolicy)


def test_policy_name_attribute():
    p = AdmissionControlPolicy()
    assert p.name == "admission_control"


# -------------------------------------------------------------------------
# Laxity filtering — requests with highly negative laxity are skipped
# -------------------------------------------------------------------------

def test_positive_laxity_request_admitted():
    """Request with positive laxity should be admitted when capacity allows."""
    # now=2.0, alpha=0.5, beta=1.0
    # est = 0.5*64 + 1.0*32 = 64; laxity = 200-2-64 = 134 (positive) → admitted
    req = make_req(0, prompt=64, output=32, deadline=200.0)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=0.0)
    action = p.select_action(state)
    assert req.request_id in action.admit[0]


def test_infeasible_by_laxity_not_admitted():
    """Request with laxity < -threshold should be skipped (filtered out).

    Uses threshold=500 which is a calibrated value for the default service
    model (step_size=0.001): est=768 steps, laxity=5-2-768=-765, -765 < -500.
    """
    # now=2.0, est=0.5*512+1.0*512=768; laxity = 5-2-768 = -765
    # threshold=500: min_laxity=-500 → -765 < -500 → filtered
    req = make_req(0, prompt=512, output=512, deadline=5.0)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=500.0)
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_threshold_controls_filter():
    """A loose threshold should allow requests with moderate negative laxity."""
    # now=2.0, est=0.5*64+1.0*64=96; laxity = 5-2-96 = -93
    # threshold=100.0: min_laxity=-100 → -93 >= -100 → admitted
    req = make_req(0, prompt=64, output=64, deadline=5.0)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=100.0)
    action = p.select_action(state)
    assert req.request_id in action.admit[0]


def test_mixed_laxity_only_viable_admitted():
    """Only requests passing the laxity filter should be admitted.

    threshold=500: req_ok passes (-67 > -500), req_bad filtered (-765 < -500).
    """
    gpu = make_gpu(max_seq=4, max_kv=8192, max_batch=4)
    now = 2.0
    # req_ok: est=0.5*64+1*32=64; laxity=200-2-64=134 → passes (threshold=500)
    # req_bad: est=0.5*512+1*512=768; laxity=5-2-768=-765 → filtered
    req_ok = make_req(0, prompt=64, output=32, deadline=200.0)
    req_bad = make_req(1, prompt=512, output=512, deadline=5.0)
    state = make_state([req_ok, req_bad], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=500.0)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert req_ok.request_id in admitted
    assert req_bad.request_id not in admitted


# -------------------------------------------------------------------------
# Ordering: most urgent (lowest laxity) first
# -------------------------------------------------------------------------

def test_lower_laxity_admitted_first_when_capacity_limited():
    """With capacity=1, request with lower laxity is admitted first."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    now = 2.0
    # req_a: laxity = 10-2-96 = -88 (lower, more urgent)
    # req_b: laxity = 20-2-48 = -30 (higher, less urgent)
    # Both pass threshold=100.0
    req_a = make_req(0, prompt=64, output=64, deadline=10.0)
    req_b = make_req(1, prompt=64, output=32, deadline=20.0)
    state = make_state([req_a, req_b], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=100.0)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_a.request_id


def test_priority_tiebreak():
    """Equal laxity → higher priority wins."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    now = 2.0
    # deadline=200: est=0.5*64+1*64=96, laxity=200-2-96=102 (positive) → passes filter
    # Same laxity for both since prompt/output/deadline identical; req_hi has higher priority
    req_lo = make_req(0, prompt=64, output=64, deadline=200.0, priority=1.0)
    req_hi = make_req(1, prompt=64, output=64, deadline=200.0, priority=3.0)
    state = make_state([req_lo, req_hi], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.0)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_hi.request_id


def test_request_id_tiebreak():
    """All else equal → lower request_id wins (deterministic)."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    now = 0.0
    req_a = make_req(0, prompt=64, output=64, deadline=200.0, priority=1.0)
    req_b = make_req(9, prompt=64, output=64, deadline=200.0, priority=1.0)
    state = make_state([req_b, req_a], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.0)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_a.request_id


# -------------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------------

def test_deterministic_across_calls():
    """Same state produces same action on two separate calls."""
    p = AdmissionControlPolicy(laxity_threshold=50.0)
    reqs = [make_req(i, prompt=64, output=32, deadline=float(100 + i * 10)) for i in range(6)]
    state1 = make_state(reqs, now=1.0)
    state2 = make_state(reqs, now=1.0)
    a1 = p.select_action(state1)
    a2 = p.select_action(state2)
    assert a1.admit == a2.admit


# -------------------------------------------------------------------------
# GPU capacity respected
# -------------------------------------------------------------------------

def test_full_gpu_admits_nothing():
    gpu = make_gpu(max_seq=2, max_kv=8192, max_batch=2, active=[99, 100])
    reqs = [make_req(i, deadline=1000.0) for i in range(3)]
    state = make_state(reqs, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=9999.0)
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_kv_capacity_respected():
    gpu = make_gpu(max_seq=8, max_kv=100, max_batch=8, kv_used=80)
    reqs = [make_req(i, prompt=64, deadline=1000.0) for i in range(3)]
    state = make_state(reqs, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=9999.0)
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_empty_queue_returns_empty():
    state = make_state([])
    p = AdmissionControlPolicy()
    assert p.select_action(state).is_empty()


# -------------------------------------------------------------------------
# actual_output_tokens is never accessed
# -------------------------------------------------------------------------

def test_no_actual_output_tokens_on_observable_request():
    req = make_req(0)
    assert not hasattr(req, "actual_output_tokens"), (
        "ObservableRequest must not expose actual_output_tokens"
    )


# -------------------------------------------------------------------------
# Custom alpha/beta
# -------------------------------------------------------------------------

def test_custom_alpha_beta_changes_estimate():
    """alpha=0 means only output tokens count in service estimate."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    now = 0.0
    # With alpha=0, beta=1: est = output_tokens
    # req_a: est=32, laxity=100-0-32=68 → passes threshold=50
    # req_b: est=200, laxity=100-0-200=-100 < -50 → filtered
    req_a = make_req(0, prompt=512, output=32, deadline=100.0)
    req_b = make_req(1, prompt=8, output=200, deadline=100.0)
    state = make_state([req_a, req_b], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=50.0, alpha=0.0, beta=1.0)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert req_a.request_id in admitted
    assert req_b.request_id not in admitted

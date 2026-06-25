"""Tests for the AdmissionControlPolicy.

Units note
----------
After the Phase 2B.7 unit fix, all laxity calculations are in **seconds**:
  laxity = slo_deadline - now - est_seconds
  est_seconds = step_size * (alpha * prompt + beta * output)
  default step_size = 0.001, alpha = 0.5, beta = 1.0

Example (step_size=0.001, alpha=0.5, beta=1.0):
  prompt=64, output=64 → est_steps=96, est_seconds=0.096 s
  deadline=100, now=2 → laxity = 100 - 2 - 0.096 = 97.904 s  (positive, feasible)
  deadline=2.05, now=2 → laxity = 0.05 - 0.096 = -0.046 s    (negative, infeasible)
"""
import pytest
from llmserveopt.policies.admission_control import AdmissionControlPolicy
from llmserveopt.policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES, make_policy
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState

STEP = 0.001  # default step_size


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
# Unit-consistent laxity calculation
# -------------------------------------------------------------------------

def test_laxity_in_seconds():
    """_laxity() must return a value in seconds."""
    p = AdmissionControlPolicy(step_size=STEP)
    # prompt=64, output=64 → est_steps=96 → est_s=0.096
    # deadline=5.0, now=2.0 → laxity = 5.0-2.0-0.096 = 2.904 s
    req = make_req(0, prompt=64, output=64, deadline=5.0)
    lax = p._laxity(req, now=2.0)
    assert lax == pytest.approx(5.0 - 2.0 - 96 * STEP, abs=1e-6)


def test_est_seconds_correct():
    """_est_seconds() must equal step_size * service_proxy."""
    p = AdmissionControlPolicy(step_size=STEP)
    req = make_req(0, prompt=128, output=64)
    # est_steps = 0.5*128 + 1.0*64 = 128; est_s = 0.128
    assert p._est_seconds(req) == pytest.approx(128 * STEP, abs=1e-9)


def test_step_size_scales_laxity():
    """Larger step_size → larger est_seconds → smaller laxity."""
    req = make_req(0, prompt=64, output=64, deadline=5.0)
    p1 = AdmissionControlPolicy(step_size=0.001)
    p2 = AdmissionControlPolicy(step_size=0.01)
    lax1 = p1._laxity(req, now=2.0)
    lax2 = p2._laxity(req, now=2.0)
    assert lax1 > lax2  # larger step_size → more service time → less slack


# -------------------------------------------------------------------------
# Laxity filtering — threshold in seconds
# -------------------------------------------------------------------------

def test_positive_laxity_request_admitted():
    """Request with positive laxity is admitted with threshold=0.0."""
    # prompt=64, output=32: est_steps=64, est_s=0.064
    # deadline=200, now=2: laxity=197.936 > 0 → passes
    req = make_req(0, prompt=64, output=32, deadline=200.0)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
    action = p.select_action(state)
    assert req.request_id in action.admit[0]


def test_infeasible_request_blocked_at_zero_threshold():
    """Request infeasible by seconds is blocked with threshold=0.0."""
    # prompt=64, output=64: est_steps=96, est_s=0.096
    # deadline=2.05, now=2.0: laxity=0.05-0.096=-0.046 < 0 → blocked
    req = make_req(0, prompt=64, output=64, deadline=2.05)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_threshold_gives_slack_for_prediction_error():
    """Positive threshold allows slightly infeasible requests (prediction buffer)."""
    # laxity = -0.046 s; threshold=0.1 → min_laxity=-0.1 → -0.046 >= -0.1 → admitted
    req = make_req(0, prompt=64, output=64, deadline=2.05)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=0.1, step_size=STEP)
    action = p.select_action(state)
    assert req.request_id in action.admit[0]


def test_infeasible_by_laxity_not_admitted():
    """Request with laxity < -threshold is not admitted."""
    # prompt=64, output=64: est_s=0.096; deadline=2.05, now=2.0: laxity=-0.046
    # threshold=0.02 → min_laxity=-0.02 → -0.046 < -0.02 → blocked
    req = make_req(0, prompt=64, output=64, deadline=2.05)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=0.02, step_size=STEP)
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_inf_threshold_disables_filtering():
    """laxity_threshold=inf admits all requests regardless of laxity."""
    # deadline=2.001 → laxity very negative; still admitted with threshold=inf
    req = make_req(0, prompt=64, output=64, deadline=2.001)
    state = make_state([req], now=2.0)
    p = AdmissionControlPolicy(laxity_threshold=float("inf"), step_size=STEP)
    action = p.select_action(state)
    assert req.request_id in action.admit[0]


def test_threshold_controls_filter():
    """Threshold in seconds correctly gates partial admission."""
    # req_ok: laxity=-0.046 s, threshold=0.1 → admitted
    # req_bad: deadline=2.02, laxity=0.02-0.096=-0.076 < -0.05 → blocked at threshold=0.05
    gpu = make_gpu(max_seq=4, max_kv=8192, max_batch=4)
    req_ok = make_req(0, prompt=64, output=64, deadline=2.05)   # laxity=-0.046
    req_bad = make_req(1, prompt=64, output=64, deadline=2.02)  # laxity=-0.076
    state = make_state([req_ok, req_bad], now=2.0, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.05, step_size=STEP)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert req_ok.request_id in admitted
    assert req_bad.request_id not in admitted


def test_mixed_laxity_only_viable_admitted():
    """Only requests passing the laxity filter should be admitted."""
    gpu = make_gpu(max_seq=4, max_kv=8192, max_batch=4)
    # req_ok: deadline=200 → laxity=~198s → passes (threshold=0)
    # req_bad: deadline=2.05 → laxity=-0.046 → blocked (threshold=0)
    req_ok = make_req(0, prompt=64, output=32, deadline=200.0)
    req_bad = make_req(1, prompt=64, output=64, deadline=2.05)
    state = make_state([req_ok, req_bad], now=2.0, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert req_ok.request_id in admitted
    assert req_bad.request_id not in admitted


def test_active_filtering_under_overload():
    """With many requests and a tight threshold, only feasible ones are admitted."""
    # 5 requests: 4 have tight deadlines (infeasible), 1 has a safe deadline
    now = 2.0
    safe = make_req(0, prompt=64, output=64, deadline=200.0)   # laxity ≈ 198s
    infeasible = [
        make_req(i + 1, prompt=64, output=64, deadline=2.05)   # laxity = -0.046s
        for i in range(4)
    ]
    state = make_state([safe] + infeasible, now=now)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
    action = p.select_action(state)
    admitted = set(action.admit.get(0, []))
    assert safe.request_id in admitted
    for req in infeasible:
        assert req.request_id not in admitted


# -------------------------------------------------------------------------
# Ordering: most urgent (lowest laxity) first
# -------------------------------------------------------------------------

def test_lower_laxity_admitted_first_when_capacity_limited():
    """With capacity=1, request with lower laxity (more urgent) is admitted first."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    now = 2.0
    # req_a: deadline=5, est_s=0.096, laxity=2.904 (lower)
    # req_b: deadline=10, est_s=0.048, laxity=7.952 (higher)
    # Both feasible (positive laxity with threshold=0)
    req_a = make_req(0, prompt=64, output=64, deadline=5.0)
    req_b = make_req(1, prompt=64, output=32, deadline=10.0)
    state = make_state([req_a, req_b], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_a.request_id


def test_priority_tiebreak():
    """Equal laxity → higher priority wins."""
    gpu = make_gpu(max_seq=1, max_kv=8192, max_batch=1)
    now = 2.0
    req_lo = make_req(0, prompt=64, output=64, deadline=200.0, priority=1.0)
    req_hi = make_req(1, prompt=64, output=64, deadline=200.0, priority=3.0)
    state = make_state([req_lo, req_hi], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
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
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1
    assert admitted[0] == req_a.request_id


# -------------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------------

def test_deterministic_across_calls():
    """Same state produces same action on two separate calls."""
    p = AdmissionControlPolicy(laxity_threshold=0.5, step_size=STEP)
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
    p = AdmissionControlPolicy(laxity_threshold=float("inf"), step_size=STEP)
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_kv_capacity_respected():
    gpu = make_gpu(max_seq=8, max_kv=100, max_batch=8, kv_used=80)
    reqs = [make_req(i, prompt=64, deadline=1000.0) for i in range(3)]
    state = make_state(reqs, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=float("inf"), step_size=STEP)
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
    now = 2.0
    # With alpha=0, beta=1, step_size=0.001:
    # req_a: prompt=512, output=32 → est_s=0.032; deadline=200 → laxity=197.968 > 0 → passes
    # req_b: prompt=8,   output=64 → est_s=0.064; deadline=2.05 → laxity=-0.014 < 0 → filtered
    req_a = make_req(0, prompt=512, output=32, deadline=200.0)
    req_b = make_req(1, prompt=8,   output=64, deadline=2.05)
    state = make_state([req_a, req_b], now=now, gpu=gpu)
    p = AdmissionControlPolicy(laxity_threshold=0.0, step_size=STEP, alpha=0.0, beta=1.0)
    action = p.select_action(state)
    admitted = action.admit.get(0, [])
    assert req_a.request_id in admitted
    assert req_b.request_id not in admitted


# -------------------------------------------------------------------------
# Oracle exclusion
# -------------------------------------------------------------------------

def test_not_in_oracle_names():
    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES
    assert "admission_control" not in ORACLE_POLICY_NAMES


def test_not_in_selector_candidates():
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES
    # admission_control is a deployable policy, so it SHOULD be a selector candidate
    assert "admission_control" in SELECTOR_CANDIDATES
    # And NOT in oracle names
    assert "admission_control" not in ORACLE_POLICY_NAMES

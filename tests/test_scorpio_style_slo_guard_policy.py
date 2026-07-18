"""Tests for ScorpioStyleSloGuardPolicy (Phase 2B.10)."""
from __future__ import annotations

import inspect


from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.registry import (
    BASELINE_NAMES,
    ORACLE_POLICY_NAMES,
    SELECTOR_CANDIDATE_NAMES,
    make_policy,
)
from llmserveopt.policies.scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES

STEP = 0.001


def make_req(
    req_id: int,
    prompt: int = 64,
    output: int = 64,
    deadline: float = 100.0,
    priority: float = 1.0,
    arrival: float = 0.0,
) -> ObservableRequest:
    return ObservableRequest(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id="medium",
    )


def make_gpu(
    max_seq: int = 8,
    max_kv: int = 8192,
    max_batch: int = 512,
    kv_used: int = 0,
    active: tuple = (),
    decoding: int = 0,
) -> ObservableGPUState:
    return ObservableGPUState(
        gpu_id=0,
        max_active_sequences=max_seq,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=list(active),
        active_requests_info=[],
        current_kv_tokens=kv_used,
        tokens_decoded_per_request={},
        decoding_count=decoding,
    )


def make_state(
    reqs: list[ObservableRequest],
    now: float = 2.0,
    gpu: ObservableGPUState | None = None,
) -> ObservableState:
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
# Registration
# -------------------------------------------------------------------------


def test_registered_in_baseline_names():
    assert "scorpio_style_slo_guard" in BASELINE_NAMES


def test_registered_in_selector_candidates():
    assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATES
    assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATE_NAMES


def test_not_in_oracle_policy_names():
    assert "scorpio_style_slo_guard" not in ORACLE_POLICY_NAMES


def test_make_policy_returns_correct_type():
    p = make_policy("scorpio_style_slo_guard")
    assert isinstance(p, ScorpioStyleSloGuardPolicy)


def test_policy_name_attribute():
    assert ScorpioStyleSloGuardPolicy().name == "scorpio_style_slo_guard"


def test_deployable_count_is_twenty():
    assert len(BASELINE_NAMES) == 20
    assert len(SELECTOR_CANDIDATES) == 20


# -------------------------------------------------------------------------
# Leakage and determinism
# -------------------------------------------------------------------------


def test_no_actual_output_tokens_in_source():
    src = inspect.getsource(ScorpioStyleSloGuardPolicy.select_action)
    assert "actual_output_tokens" not in src


def test_no_oracle_reference_in_source():
    src = inspect.getsource(ScorpioStyleSloGuardPolicy)
    assert "oracle_srtf" not in src


def test_deterministic_under_identical_state():
    p = ScorpioStyleSloGuardPolicy()
    reqs = [
        make_req(1, deadline=50.0, priority=2.0),
        make_req(2, deadline=40.0, priority=1.0, arrival=0.5),
    ]
    state = make_state(reqs, now=5.0)
    a1 = p.select_action(state)
    a2 = p.select_action(state)
    assert a1.admit == a2.admit


def test_reset_restores_admission_budget():
    p = ScorpioStyleSloGuardPolicy(admission_budget_max=4.0)
    p._admission_budget = 0.5
    p.reset()
    assert p._admission_budget == 4.0


# -------------------------------------------------------------------------
# TTFT / laxity guards
# -------------------------------------------------------------------------


def test_infeasible_laxity_request_not_admitted():
    """Request with negative laxity should be filtered when threshold=0."""
    p = ScorpioStyleSloGuardPolicy(laxity_threshold=0.0, step_size=STEP)
    # est_s = 0.001*(32+64)=0.096; deadline=2.05, now=2 → laxity negative
    req = make_req(1, prompt=64, output=64, deadline=2.05)
    state = make_state([req], now=2.0)
    action = p.select_action(state)
    assert action.admit[0] == []


def test_feasible_tight_slo_request_admitted():
    p = ScorpioStyleSloGuardPolicy(laxity_threshold=0.0, step_size=STEP)
    urgent = make_req(1, prompt=32, output=32, deadline=10.0, priority=3.0)
    state = make_state([urgent], now=1.0)
    action = p.select_action(state)
    assert 1 in action.admit[0]


def test_ttft_proxy_slack_filters_infeasible_prefill():
    p = ScorpioStyleSloGuardPolicy(
        laxity_threshold=float("inf"),
        ttft_slack_threshold=0.0,
        step_size=STEP,
        alpha=0.5,
    )
    # prefill_proxy = 1.024; deadline must exceed now + prefill for TTFT slack >= 0
    ok = make_req(1, prompt=2048, output=16, deadline=2.5)
    bad = make_req(2, prompt=2048, output=16, deadline=1.01)
    state = make_state([bad, ok], now=1.0)
    action = p.select_action(state)
    assert 1 in action.admit[0]
    assert 2 not in action.admit[0]


# -------------------------------------------------------------------------
# KV / decode pressure
# -------------------------------------------------------------------------


def test_high_kv_pressure_prefers_shorter_decode():
    p = ScorpioStyleSloGuardPolicy(
        kv_utilization_threshold=0.5,
        long_decode_token_threshold=128,
        laxity_threshold=float("inf"),
        ttft_slack_threshold=float("inf"),
    )
    short = make_req(1, output=64, deadline=100.0, priority=1.0)
    long = make_req(2, output=512, deadline=100.0, priority=1.0)
    gpu = make_gpu(kv_used=7000, max_kv=8192, decoding=6, max_seq=8)
    state = make_state([long, short], now=2.0, gpu=gpu)
    action = p.select_action(state)
    admitted = action.admit[0]
    assert 1 in admitted
    assert 2 not in admitted


def test_guard_mode_limits_admissions_under_pressure():
    p = ScorpioStyleSloGuardPolicy(
        admission_budget_max=1.0,
        admission_budget_refill=0.0,
        admission_cost=1.0,
        queue_overload_factor=1.0,
        laxity_threshold=float("inf"),
        ttft_slack_threshold=float("inf"),
    )
    p.reset()
    reqs = [make_req(i, deadline=100.0) for i in range(1, 6)]
    state = make_state(reqs, now=1.0, gpu=make_gpu(max_seq=4))
    action = p.select_action(state)
    total_admitted = sum(len(v) for v in action.admit.values())
    assert total_admitted <= 1


# -------------------------------------------------------------------------
# Priority and tie-breaking
# -------------------------------------------------------------------------


def test_higher_priority_admitted_first_when_slack_equal():
    p = ScorpioStyleSloGuardPolicy(
        laxity_threshold=float("inf"),
        ttft_slack_threshold=float("inf"),
        admission_budget_max=1.0,
        admission_budget_refill=0.0,
        admission_cost=1.0,
        queue_overload_factor=1.0,
    )
    p.reset()
    low = make_req(1, priority=1.0, deadline=50.0)
    high = make_req(2, priority=5.0, deadline=50.0)
    state = make_state([low, high], now=1.0, gpu=make_gpu(max_seq=1))
    action = p.select_action(state)
    assert action.admit[0] == [2]


def test_tie_break_by_arrival_then_id():
    p = ScorpioStyleSloGuardPolicy(
        laxity_threshold=float("inf"),
        ttft_slack_threshold=float("inf"),
    )
    r1 = make_req(2, priority=1.0, deadline=50.0, arrival=1.0)
    r2 = make_req(1, priority=1.0, deadline=50.0, arrival=0.5)
    state = make_state([r1, r2], now=2.0, gpu=make_gpu(max_seq=1))
    action = p.select_action(state)
    assert action.admit[0] == [1]


# -------------------------------------------------------------------------
# Capacity constraints
# -------------------------------------------------------------------------


def test_respects_gpu_capacity():
    p = ScorpioStyleSloGuardPolicy()
    gpu = make_gpu(max_seq=1, active=(99,))
    req = make_req(1, deadline=100.0)
    state = make_state([req], now=1.0, gpu=gpu)
    action = p.select_action(state)
    assert action.admit[0] == []


def test_returns_valid_action_type():
    p = ScorpioStyleSloGuardPolicy()
    action = p.select_action(make_state([]))
    assert isinstance(action, Action)


# -------------------------------------------------------------------------
# Overloaded hand-built trace
# -------------------------------------------------------------------------


def test_overloaded_trace_admits_feasible_subset():
    p = ScorpioStyleSloGuardPolicy(step_size=STEP, laxity_threshold=0.0)
    reqs = [
        make_req(1, prompt=128, output=128, deadline=20.0, priority=3.0, arrival=0.0),
        make_req(2, prompt=128, output=128, deadline=5.0, priority=2.0, arrival=0.1),
        make_req(3, prompt=128, output=128, deadline=3.0, priority=1.0, arrival=0.2),
    ]
    state = make_state(reqs, now=2.5, gpu=make_gpu(max_seq=4))
    action = p.select_action(state)
    admitted = set(action.admit[0])
    assert len(admitted) >= 1
    assert 3 in admitted or 2 in admitted


def test_prediction_noise_does_not_use_true_output():
    """Policy only reads predicted_output_tokens from ObservableRequest."""
    p = ScorpioStyleSloGuardPolicy()
    req = make_req(1, output=999, deadline=100.0)
    assert req.predicted_output_tokens == 999
    state = make_state([req], now=1.0)
    action = p.select_action(state)
    assert 1 in action.admit[0]

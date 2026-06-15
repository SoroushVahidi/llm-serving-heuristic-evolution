"""Tests for HeuristicPolicy integration with the simulator."""
import pytest
from llmserveopt.heuristics import build_heuristic_policy, compile_heuristic
from llmserveopt.heuristics.compiler import CompilationError
from llmserveopt.heuristics.examples import edf_like, fifo_like, slo_kv_balanced, throughput_oriented
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState


def make_request(req_id, arrival=0.0, prompt=128, output=64, deadline=10.0, priority=2.0):
    return ObservableRequest(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=arrival + deadline,
        priority=priority,
        class_id="medium",
    )


def make_gpu(gpu_id=0, active=(), max_seq=8, max_kv=8192, max_batch=512, kv_used=0):
    return ObservableGPUState(
        gpu_id=gpu_id,
        max_active_sequences=max_seq,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=list(active),
        active_requests_info=[],
        current_kv_tokens=kv_used,
        tokens_decoded_per_request={},
    )


def make_state(requests, gpu_id=0, active=()):
    gpu = make_gpu(gpu_id=gpu_id, active=active)
    return ObservableState(
        time=1.0,
        waiting_queue=requests,
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )


# ---------------------------------------------------------------------------
# Basic policy creation and action
# ---------------------------------------------------------------------------

def test_build_fifo_like_policy():
    p = build_heuristic_policy(fifo_like())
    assert p is not None
    assert "fifo_like" in p.name


def test_build_edf_like_policy():
    p = build_heuristic_policy(edf_like())
    assert p is not None


def test_policy_returns_action():
    p = build_heuristic_policy(fifo_like())
    reqs = [make_request(i) for i in range(3)]
    state = make_state(reqs)
    action = p.select_action(state)
    assert isinstance(action, Action)
    assert 0 in action.admit


def test_empty_queue_returns_empty_action():
    p = build_heuristic_policy(fifo_like())
    state = make_state([])
    action = p.select_action(state)
    assert action.is_empty()


def test_all_examples_return_valid_action():
    examples = [fifo_like(), edf_like(), slo_kv_balanced(), throughput_oriented()]
    reqs = [make_request(i, arrival=float(i) * 0.1) for i in range(5)]
    state = make_state(reqs)
    for doc in examples:
        p = build_heuristic_policy(doc)
        action = p.select_action(state)
        assert isinstance(action, Action)
        total = sum(len(v) for v in action.admit.values())
        assert total >= 0


# ---------------------------------------------------------------------------
# Feasibility constraints respected
# ---------------------------------------------------------------------------

def test_max_sequences_not_exceeded():
    p = build_heuristic_policy(fifo_like())
    gpu = make_gpu(max_seq=2, max_kv=8192, active=[99, 100])
    state = ObservableState(
        time=1.0,
        waiting_queue=[make_request(i) for i in range(5)],
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    # GPU is full (2/2 active), nothing can be admitted
    assert sum(len(v) for v in action.admit.values()) == 0


def test_kv_budget_not_exceeded():
    p = build_heuristic_policy(edf_like())
    # KV tokens already near capacity; request needs 128 prompt tokens
    gpu = make_gpu(max_seq=8, max_kv=200, kv_used=180)
    req = make_request(0, prompt=128)  # 128 > 200-180=20 remaining
    state = ObservableState(
        time=1.0,
        waiting_queue=[req],
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    assert sum(len(v) for v in action.admit.values()) == 0


def test_feasible_request_admitted():
    p = build_heuristic_policy(throughput_oriented())
    gpu = make_gpu(max_seq=8, max_kv=8192, kv_used=0)
    req = make_request(0, prompt=64)
    state = ObservableState(
        time=1.0,
        waiting_queue=[req],
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    assert req.request_id in action.admit.get(0, [])


# ---------------------------------------------------------------------------
# Compilation error on invalid heuristic
# ---------------------------------------------------------------------------

def test_compilation_fails_on_bad_heuristic():
    bad = {
        "name": "bad",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"var": "req.actual_output_tokens"}},
    }
    with pytest.raises(CompilationError):
        build_heuristic_policy(bad)


# ---------------------------------------------------------------------------
# Reset clears internal state
# ---------------------------------------------------------------------------

def test_reset_clears_violation_history():
    p = build_heuristic_policy(slo_kv_balanced())
    p.record_completion(True)
    p.record_completion(True)
    p.reset()
    # After reset, no violation history — should still produce valid action
    state = make_state([make_request(0)])
    action = p.select_action(state)
    assert isinstance(action, Action)

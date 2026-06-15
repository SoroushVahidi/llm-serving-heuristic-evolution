"""Tests that heuristic policies are deterministic — same input → same output."""
import pytest
from llmserveopt.heuristics import build_heuristic_policy
from llmserveopt.heuristics.examples import edf_like, fifo_like, slo_kv_balanced, throughput_oriented
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


def make_state(requests):
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=8,
        max_batch_tokens=512,
        max_kv_tokens=8192,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    return ObservableState(
        time=2.0,
        waiting_queue=requests,
        gpu_states=[gpu],
        completed_count=0,
        step=5,
    )


@pytest.fixture
def requests():
    return [
        make_request(i, arrival=i * 0.1, prompt=64 + i * 16, output=32 + i * 8, deadline=5.0, priority=float(i % 3 + 1))
        for i in range(6)
    ]


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_determinism_same_state(doc_fn, requests):
    """Calling select_action twice with the same state returns the same result."""
    p = build_heuristic_policy(doc_fn())
    state = make_state(requests)
    a1 = p.select_action(state)
    a2 = p.select_action(state)
    assert a1.admit == a2.admit, f"{doc_fn.__name__}: actions differ on same state"


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_determinism_across_instances(doc_fn, requests):
    """Two independently created policy instances return the same action."""
    state = make_state(requests)
    p1 = build_heuristic_policy(doc_fn())
    p2 = build_heuristic_policy(doc_fn())
    a1 = p1.select_action(state)
    a2 = p2.select_action(state)
    assert a1.admit == a2.admit, f"{doc_fn.__name__}: two instances differ"


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_determinism_after_reset(doc_fn, requests):
    """Policy returns the same action before and after reset (given same state)."""
    p = build_heuristic_policy(doc_fn())
    state = make_state(requests)
    a1 = p.select_action(state)
    p.reset()
    a2 = p.select_action(state)
    assert a1.admit == a2.admit, f"{doc_fn.__name__}: action changed after reset"

"""Tests that heuristic policies never violate GPU capacity constraints."""
import pytest
from llmserveopt.heuristics import build_heuristic_policy
from llmserveopt.heuristics.examples import edf_like, fifo_like, slo_kv_balanced, throughput_oriented
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState


def make_request(req_id, prompt=128, output=64, deadline=5.0, priority=2.0, arrival=0.0):
    return ObservableRequest(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=arrival + deadline,
        priority=priority,
        class_id="medium",
    )


def make_gpu(gpu_id=0, active=(), max_seq=4, max_kv=1024, max_batch=16, kv_used=0):
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


def check_action_feasible(action, gpus_by_id, queue_by_id):
    """Assert action doesn't exceed any GPU constraints."""
    for gpu_id, req_ids in action.admit.items():
        gpu = gpus_by_id[gpu_id]
        new_active = len(gpu.active_request_ids) + len(req_ids)
        assert new_active <= gpu.max_active_sequences, (
            f"GPU {gpu_id}: active {new_active} > max {gpu.max_active_sequences}"
        )
        kv_added = sum(queue_by_id[rid].prompt_tokens for rid in req_ids)
        new_kv = gpu.current_kv_tokens + kv_added
        assert new_kv <= gpu.max_kv_tokens, (
            f"GPU {gpu_id}: KV {new_kv} > max {gpu.max_kv_tokens}"
        )
        new_batch = new_active
        assert new_batch <= gpu.max_batch_tokens, (
            f"GPU {gpu_id}: batch {new_batch} > max {gpu.max_batch_tokens}"
        )


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_feasibility_fresh_gpu(doc_fn):
    """On an empty GPU, admitted requests must fit within constraints."""
    p = build_heuristic_policy(doc_fn())
    gpu = make_gpu(max_seq=4, max_kv=512, max_batch=4, kv_used=0)
    reqs = [make_request(i, prompt=64 + i * 16) for i in range(8)]  # more requests than GPU can take
    state = ObservableState(
        time=1.0,
        waiting_queue=reqs,
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    queue_by_id = {r.request_id: r for r in reqs}
    check_action_feasible(action, {0: gpu}, queue_by_id)


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_feasibility_nearly_full_gpu(doc_fn):
    """Near-capacity GPU: admitted requests must not overflow KV or sequence count."""
    p = build_heuristic_policy(doc_fn())
    # 3 of 4 slots used, 900 of 1024 KV used → only small requests fit
    gpu = make_gpu(max_seq=4, max_kv=1024, max_batch=4, kv_used=900, active=[1, 2, 3])
    reqs = [
        make_request(10, prompt=64),   # 64 ≤ 124 remaining → feasible (if seq slot free)
        make_request(11, prompt=256),  # 256 > 124 → KV overflow
        make_request(12, prompt=32),   # feasible
    ]
    state = ObservableState(
        time=1.0,
        waiting_queue=reqs,
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    queue_by_id = {r.request_id: r for r in reqs}
    check_action_feasible(action, {0: gpu}, queue_by_id)


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_feasibility_fully_loaded_gpu(doc_fn):
    """A full GPU must admit zero requests."""
    p = build_heuristic_policy(doc_fn())
    gpu = make_gpu(max_seq=4, max_kv=1024, max_batch=4, kv_used=512, active=[1, 2, 3, 4])
    reqs = [make_request(i) for i in range(5)]
    state = ObservableState(
        time=1.0,
        waiting_queue=reqs,
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    total = sum(len(v) for v in action.admit.values())
    assert total == 0, f"{doc_fn.__name__}: admitted {total} to a full GPU"


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_admitted_ids_are_from_queue(doc_fn):
    """Admitted request IDs must all be from the waiting queue."""
    p = build_heuristic_policy(doc_fn())
    gpu = make_gpu(max_seq=8, max_kv=8192, max_batch=512)
    reqs = [make_request(i) for i in range(5)]
    queue_ids = {r.request_id for r in reqs}
    state = ObservableState(
        time=1.0,
        waiting_queue=reqs,
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    for req_ids in action.admit.values():
        for rid in req_ids:
            assert rid in queue_ids, f"{doc_fn.__name__}: admitted unknown request_id {rid}"


@pytest.mark.parametrize("doc_fn", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_no_duplicate_admissions(doc_fn):
    """Each request may be admitted at most once across all GPUs."""
    p = build_heuristic_policy(doc_fn())
    gpus = [make_gpu(gpu_id=i, max_seq=4, max_kv=2048, max_batch=4) for i in range(2)]
    reqs = [make_request(i) for i in range(6)]
    state = ObservableState(
        time=1.0,
        waiting_queue=reqs,
        gpu_states=gpus,
        completed_count=0,
        step=1,
    )
    action = p.select_action(state)
    all_admitted = []
    for req_ids in action.admit.values():
        all_admitted.extend(req_ids)
    assert len(all_admitted) == len(set(all_admitted)), (
        f"{doc_fn.__name__}: duplicate request IDs in action"
    )

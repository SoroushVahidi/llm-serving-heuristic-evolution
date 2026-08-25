"""Tests for src/llmserveopt/policies/sarathi_faithful.py.

Covers chunked-prefill correctness, stall-free/decode-first scheduling
correctness, and KV-block memory correctness (the things this baseline
claims fidelity for -- see docs/sarathi_faithful_scheduler_reference.md).
Runtime/hardware performance reproduction is explicitly out of scope.

Some tests hand-derive the expected outcome directly from Sarathi-Serve's
SarathiScheduler._schedule() (commit ceaa0660, read live via the GitHub API
while building this baseline) -- these are marked "fidelity" in their
docstring.
"""
from __future__ import annotations

import warnings

import pytest

from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.simulator.service_model import ServiceModel


# ---------------------------------------------------------------------------
# Hand-constructed ObservableState helpers (matches this repo's established
# direct-state policy-test convention)
# ---------------------------------------------------------------------------

def make_req(req_id, prompt=100, output=8, deadline=1000.0, priority=1.0, arrival=0.0):
    return ObservableRequest(
        request_id=req_id, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline,
        priority=priority, class_id="medium",
    )


def make_gpu(gpu_id=0, max_seq=100, max_kv=100_000, max_batch=100_000, active_reqs=(), decoded=None):
    return ObservableGPUState(
        gpu_id=gpu_id, max_active_sequences=max_seq, max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=[r.request_id for r in active_reqs],
        active_requests_info=list(active_reqs),
        current_kv_tokens=sum(r.prompt_tokens for r in active_reqs),
        tokens_decoded_per_request=decoded or {},
    )


def make_state(waiting, gpu=None, now=0.0, step=0):
    if gpu is None:
        gpu = make_gpu()
    return ObservableState(time=now, waiting_queue=list(waiting), gpu_states=[gpu], completed_count=0, step=step)


# ---------------------------------------------------------------------------
# Chunked prefill: long prompt split across multiple chunks
# ---------------------------------------------------------------------------

def test_long_prompt_admitted_with_first_chunk_only():
    """Fidelity: admission reserves memory for the FULL prompt but only
    accounts chunk_size tokens of *progress* this iteration."""
    policy = SarathiFaithfulPolicy(chunk_size=300, watermark=0.0)
    req = make_req(0, prompt=1000)
    action = policy.select_action(make_state([req]))
    assert action.admit[0] == [0]
    state = policy._request_states[0][0]
    assert state.remaining_prefill == 1000 - 300  # first chunk = 300, 700 left


def test_chunk_boundaries_exact_multiple():
    """prompt exactly divisible by chunk_size: first chunk uses the full
    chunk_size, remaining is still a clean multiple."""
    policy = SarathiFaithfulPolicy(chunk_size=250, watermark=0.0)
    req = make_req(0, prompt=1000)  # 1000 / 250 = 4 exact chunks
    policy.select_action(make_state([req]))
    assert policy._request_states[0][0].remaining_prefill == 750


def test_final_partial_chunk():
    """A prompt not evenly divisible by chunk_size: the LAST chunk must be
    the exact remainder, never overshooting into 0-or-negative remaining."""
    policy = SarathiFaithfulPolicy(chunk_size=300, watermark=0.0)
    req = make_req(0, prompt=1000)
    gpu = make_gpu()
    policy.select_action(make_state([req], gpu=gpu))  # 1000 -> 700 remaining

    active = make_req(0, prompt=1000)
    for expected_remaining in (400, 100):
        gpu2 = make_gpu(active_reqs=[active])
        policy.select_action(make_state([], gpu=gpu2))
        assert policy._request_states[0][0].remaining_prefill == expected_remaining

    # Final call: only 100 tokens remain, less than a full chunk (300) --
    # the chunk must be exactly 100, not 300, and remaining must land at
    # exactly 0, never negative.
    gpu3 = make_gpu(active_reqs=[active])
    policy.select_action(make_state([], gpu=gpu3))
    assert policy._request_states[0][0].remaining_prefill == 0


def test_multiple_concurrent_prefills_share_budget_fcfs():
    """Fidelity: when multiple requests are mid-prefill simultaneously, the
    earlier-arrived one gets first claim on the shared chunk_size budget;
    the later one gets only what's left."""
    policy = SarathiFaithfulPolicy(chunk_size=100, watermark=0.0)
    req_a = make_req(0, prompt=200, arrival=0.0)
    req_b = make_req(1, prompt=200, arrival=0.001)
    gpu = make_gpu()
    action1 = policy.select_action(make_state([req_a, req_b], gpu=gpu))
    # req_a admitted first: gets full 100-token chunk (100 used, 0 left for req_b).
    assert action1.admit[0] == [0]
    assert 1 not in action1.admit[0]


def test_deterministic_chunk_order_by_arrival_then_id():
    """Fidelity: like the pinned reference itself ("we do not sort the
    waiting queue since the preempted sequence groups are added to the
    front and the new sequence groups are added to the back" -- read
    directly from SarathiScheduler._schedule's own comment), this policy
    trusts state.waiting_queue's given order rather than re-sorting. The
    simulator itself guarantees FCFS-with-preemption order (see
    docs/vllm_faithful_scheduler_reference.md); this test supplies that
    same guarantee by hand and checks only ONE request is admitted per
    call under max_num_seqs=1, in the order given."""
    policy = SarathiFaithfulPolicy(chunk_size=50, watermark=0.0, max_num_seqs=1)
    req_lo_id = make_req(2, prompt=200, arrival=0.0)
    req_hi_id = make_req(5, prompt=200, arrival=0.0)
    action = policy.select_action(make_state([req_lo_id, req_hi_id]))
    assert action.admit[0] == [2]  # front-of-queue request admitted; tie is
    # resolved upstream by the simulator's own FCFS insertion order, not by
    # this policy re-sorting -- exactly matching the pinned reference.


# ---------------------------------------------------------------------------
# Stall-free / decode-first scheduling
# ---------------------------------------------------------------------------

def test_decoding_request_always_kept_ahead_of_new_prefill_admission():
    """Fidelity: Phase 1a (already-decoding sequences) is processed and
    accounted BEFORE Phase 2 (new admissions) -- a decoding request's slot
    is never sacrificed to admit a new prefill."""
    policy = SarathiFaithfulPolicy(chunk_size=50, watermark=0.0, block_size=1)
    # Manually seed one decoding request (remaining_prefill == 0).
    decoding_req = make_req(0, prompt=10, arrival=0.0)
    gpu = make_gpu(max_kv=1000, active_reqs=[decoding_req])
    bm = policy._get_block_manager(gpu)
    bm.allocate(0, 10)
    from llmserveopt.policies.sarathi_faithful import _RequestState
    policy._get_request_states(0)[0] = _RequestState(remaining_prefill=0)

    waiting_req = make_req(1, prompt=200, arrival=0.001)
    action = policy.select_action(make_state([waiting_req], gpu=gpu))
    # The decoding request must still be "kept" (implicitly, by not being
    # preempted) and consume its 1 token of budget before the new
    # admission's chunk is computed.
    assert 0 not in action.preempt.get(0, [])
    # New admission still happens (budget=50-1=49 left for its first chunk).
    assert action.admit[0] == [1]
    assert policy._request_states[0][1].remaining_prefill == 200 - 49


def test_prefill_uses_only_leftover_budget_after_decode():
    """chunk_size=10, 3 decoding requests consume 3 tokens of budget ->
    only 7 tokens left for a new admission's first chunk."""
    policy = SarathiFaithfulPolicy(chunk_size=10, watermark=0.0, block_size=1)
    from llmserveopt.policies.sarathi_faithful import _RequestState

    decoding_reqs = [make_req(i, prompt=5, arrival=float(i) * 0.0001) for i in range(3)]
    gpu = make_gpu(max_kv=1000, active_reqs=decoding_reqs)
    bm = policy._get_block_manager(gpu)
    states = policy._get_request_states(0)
    for r in decoding_reqs:
        bm.allocate(r.request_id, r.prompt_tokens)
        states[r.request_id] = _RequestState(remaining_prefill=0)

    waiting_req = make_req(10, prompt=100, arrival=0.001)
    action = policy.select_action(make_state([waiting_req], gpu=gpu))
    assert action.admit[0] == [10]
    assert policy._request_states[0][10].remaining_prefill == 100 - 7


def test_no_budget_overflow_across_decode_and_new_admission():
    """Total tokens accounted for in one iteration (decode + new-admission
    chunk) must never exceed chunk_size."""
    policy = SarathiFaithfulPolicy(chunk_size=5, watermark=0.0, block_size=1)
    from llmserveopt.policies.sarathi_faithful import _RequestState

    decoding_reqs = [make_req(i, prompt=5, arrival=float(i) * 0.0001) for i in range(5)]
    gpu = make_gpu(max_kv=1000, active_reqs=decoding_reqs)
    bm = policy._get_block_manager(gpu)
    states = policy._get_request_states(0)
    for r in decoding_reqs:
        bm.allocate(r.request_id, r.prompt_tokens)
        states[r.request_id] = _RequestState(remaining_prefill=0)

    # Budget (5) is already fully consumed by the 5 decoding requests.
    waiting_req = make_req(10, prompt=100, arrival=0.001)
    action = policy.select_action(make_state([waiting_req], gpu=gpu))
    assert action.admit[0] == []  # no budget left for even 1 prefill token


def test_continuous_decode_progress_kept_every_step():
    policy = SarathiFaithfulPolicy(chunk_size=50, watermark=0.0, block_size=1)
    from llmserveopt.policies.sarathi_faithful import _RequestState

    req = make_req(0, prompt=5, arrival=0.0)
    gpu = make_gpu(max_kv=1000, active_reqs=[req])
    bm = policy._get_block_manager(gpu)
    bm.allocate(0, 5)
    policy._get_request_states(0)[0] = _RequestState(remaining_prefill=0)

    for _ in range(5):
        action = policy.select_action(make_state([], gpu=make_gpu(max_kv=1000, active_reqs=[req])))
        assert action.preempt.get(0, []) == []  # never preempted (ample capacity)
        assert 0 in policy._request_states[0]  # still tracked/kept running


# ---------------------------------------------------------------------------
# Scheduler fidelity: tight budget / capacity / max_num_seqs / preemption
# ---------------------------------------------------------------------------

def test_tight_token_budget_blocks_admission_entirely():
    """Fidelity: if the FIRST waiting request's chunk would be 0, admission
    stops entirely (break) -- reproduced directly from the pinned source,
    not inferred from its (misleading) inline comment."""
    policy = SarathiFaithfulPolicy(chunk_size=1, watermark=0.0, block_size=1)
    from llmserveopt.policies.sarathi_faithful import _RequestState
    decoding_req = make_req(0, prompt=5, arrival=0.0)
    gpu = make_gpu(max_kv=1000, active_reqs=[decoding_req])
    bm = policy._get_block_manager(gpu)
    bm.allocate(0, 5)
    policy._get_request_states(0)[0] = _RequestState(remaining_prefill=0)

    waiting_req = make_req(1, prompt=100, arrival=0.001)
    action = policy.select_action(make_state([waiting_req], gpu=gpu))
    assert action.admit[0] == []


def test_admission_stops_at_first_non_allocatable_request():
    """Fidelity: can_allocate failing for the front waiting request stops
    admission entirely, even if a LATER request would fit -- reproduced
    directly from the pinned source's actual code (break), not its comment
    claiming vLLM-style skip-and-continue behavior."""
    policy = SarathiFaithfulPolicy(chunk_size=1000, watermark=0.0, block_size=1)
    gpu = make_gpu(max_kv=10)  # only 10 blocks/tokens of capacity
    big_req = make_req(0, prompt=100, arrival=0.0)   # cannot be allocated
    small_req = make_req(1, prompt=2, arrival=0.001)  # would easily fit alone
    action = policy.select_action(make_state([big_req, small_req], gpu=gpu))
    assert action.admit[0] == []  # neither admitted -- big_req blocks the queue


def test_max_num_seqs_caps_concurrent_admissions():
    policy = SarathiFaithfulPolicy(chunk_size=10_000, watermark=0.0, max_num_seqs=1)
    gpu = make_gpu(max_kv=100_000)
    reqs = [make_req(0, prompt=50, arrival=0.0), make_req(1, prompt=50, arrival=0.001)]
    action = policy.select_action(make_state(reqs, gpu=gpu))
    assert action.admit[0] == [0]


def test_preempts_lowest_priority_decoding_sequence_when_out_of_blocks():
    """Fidelity: identical victim-selection algorithm to vllm_faithful --
    only applies to already-decoding (prefill-finished) sequences."""
    policy = SarathiFaithfulPolicy(chunk_size=100, watermark=0.0, block_size=1)
    from llmserveopt.policies.sarathi_faithful import _RequestState

    req_a = make_req(0, prompt=1, arrival=0.0)
    req_b = make_req(1, prompt=1, arrival=0.001)
    gpu = make_gpu(max_kv=4, active_reqs=[req_a, req_b])  # 4 blocks total
    bm = policy._get_block_manager(gpu)
    states = policy._get_request_states(0)
    bm.allocate(0, 2)  # 2 blocks
    bm.allocate(1, 2)  # 2 blocks -- capacity now fully used
    states[0] = _RequestState(remaining_prefill=0)
    states[1] = _RequestState(remaining_prefill=0)

    action = policy.select_action(make_state([], gpu=make_gpu(max_kv=4, active_reqs=[req_a, req_b])))
    # Both need a 3rd block to keep decoding; 0 free -> lower priority (req 1) preempted.
    assert action.preempt[0] == [1]


def test_no_starvation_moderate_load_end_to_end():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=2000)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=64, max_prefill_chunk_tokens=64, prefill_cost_per_token=1.0)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=20_000))
    reqs = [
        Request(request_id=i, arrival_time=float(i) * 0.002, prompt_tokens=40,
                predicted_output_tokens=15, actual_output_tokens=15,
                slo_deadline=1000.0, priority=1.0, class_id="medium")
        for i in range(20)
    ]
    sim.load_trace(reqs)
    policy = SarathiFaithfulPolicy(chunk_size=32, max_num_seqs=128, watermark=0.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="no-starvation")
    assert w == [], f"unexpected simulator warnings: {[str(x.message) for x in w]}"
    assert metrics.num_completed == 20
    assert metrics.num_dropped == 0


def test_deterministic_across_repeated_fresh_runs():
    def run():
        gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
        sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                           step_token_budget=64, max_prefill_chunk_tokens=64, prefill_cost_per_token=1.0)
        sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=10_000))
        reqs = [
            Request(request_id=i, arrival_time=float(i) * 0.003, prompt_tokens=30,
                    predicted_output_tokens=10, actual_output_tokens=10,
                    slo_deadline=1000.0, priority=1.0, class_id="medium")
            for i in range(15)
        ]
        sim.load_trace(reqs)
        policy = SarathiFaithfulPolicy(chunk_size=32, watermark=0.01)
        return sim.run(policy, workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.num_dropped == m2.num_dropped


# ---------------------------------------------------------------------------
# Regression: existing sarathi_style and legacy Phase 1 behavior unchanged
# ---------------------------------------------------------------------------

def test_phase1_legacy_mode_still_works_without_prefill_modeling():
    """sarathi_faithful must not require enable_prefill_modeling=True to
    run without crashing -- it degrades to instant-prefill admission
    decisions (documented limitation: chunked-prefill *behavior* is not
    visible in this mode, but the policy itself remains well-formed)."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))  # default ServiceModel, Phase 1
    reqs = [Request(request_id=i, arrival_time=float(i) * 0.001, prompt_tokens=20,
                     predicted_output_tokens=5, actual_output_tokens=5,
                     slo_deadline=1000.0, priority=1.0, class_id="medium") for i in range(5)]
    sim.load_trace(reqs)
    policy = SarathiFaithfulPolicy(chunk_size=64, watermark=0.0)
    metrics = sim.run(policy, workload_tag="phase1-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# Metrics/observability audit: TTFT and TPOT under explicit multi-step
# chunked-prefill execution (task requirement -- verified here rather than
# assumed; no new instrumentation needed, see
# docs/sarathi_faithful_scheduler_reference.md's infrastructure audit).
# ---------------------------------------------------------------------------

def test_ttft_reflects_full_multi_chunk_prefill_duration():
    """A prompt spanning multiple chunks must show TTFT approximately equal
    to the number of chunks needed (not 1 step, as it would if TTFT were
    (incorrectly) recorded at the first chunk instead of after the LAST one)."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=2000)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=100, max_prefill_chunk_tokens=100, prefill_cost_per_token=1.0)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=5000))
    # 350 prompt tokens / 100-token chunks = 4 chunks -> TTFT >= 4 steps.
    sim.load_trace([Request(request_id=0, arrival_time=0.0, prompt_tokens=350,
                             predicted_output_tokens=5, actual_output_tokens=5,
                             slo_deadline=1000.0, priority=1.0, class_id="medium")])
    policy = SarathiFaithfulPolicy(chunk_size=100, watermark=0.0)
    metrics = sim.run(policy, workload_tag="ttft-chunking")
    assert metrics.num_completed == 1
    # 4 prefill chunks + the step producing the first decode token.
    assert metrics.mean_ttft >= 4 * sm.step_size
    assert metrics.mean_ttft < 10 * sm.step_size  # sanity upper bound, not tight


def test_tpot_reflects_one_token_per_step_during_decode():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=2000)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=100, max_prefill_chunk_tokens=100, prefill_cost_per_token=1.0)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=5000))
    sim.load_trace([Request(request_id=0, arrival_time=0.0, prompt_tokens=50,
                             predicted_output_tokens=20, actual_output_tokens=20,
                             slo_deadline=1000.0, priority=1.0, class_id="medium")])
    policy = SarathiFaithfulPolicy(chunk_size=100, watermark=0.0)
    metrics = sim.run(policy, workload_tag="tpot-check")
    assert metrics.num_completed == 1
    assert metrics.mean_tpot == pytest.approx(sm.step_size)  # 1 token/step, undisturbed by chunking

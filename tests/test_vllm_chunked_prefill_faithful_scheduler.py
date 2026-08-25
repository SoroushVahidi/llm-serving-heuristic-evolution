"""Tests for src/llmserveopt/policies/vllm_chunked_prefill_faithful.py.

Covers chunked-prefill correctness, the "no explicit decode-priority phase"
scheduling behavior that structurally distinguishes this baseline from
sarathi_faithful, and KV-block memory correctness (the things this baseline
claims fidelity for -- see
docs/vllm_chunked_prefill_faithful_scheduler_reference.md). Runtime/
hardware performance reproduction is explicitly out of scope.

Each test is labeled EXACT-SOURCE FIDELITY (behavior read/derived directly
from vLLM's pinned v0.4.2 scheduler.py/block_manager_v1.py, commit
c7f2cf2b7f67bce5842fedfdba508440fe257375) or SIMULATOR ADAPTATION (a
disclosed, documented departure required by this project's simulator
abstraction -- e.g. the KV-block manager's single-sequence-per-request
model, or the shared GPUState._step_phase15 execution-layer limitation
documented in docs/vllm_chunked_prefill_faithful_root_cause_analysis.md).
"""
from __future__ import annotations

import warnings

import pytest

from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy, _RequestState
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.simulator.service_model import ServiceModel


# ---------------------------------------------------------------------------
# Hand-constructed ObservableState helpers (matches this repo's established
# direct-state policy-test convention -- see test_sarathi_faithful_scheduler.py)
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
# EXACT-SOURCE FIDELITY: scheduler.py _get_num_new_tokens/_schedule_prefills
# ---------------------------------------------------------------------------

def test_long_prompt_admitted_with_first_chunk_only():
    """Admission reserves memory for the FULL prompt but only accounts
    max_num_batched_tokens of *progress* this iteration."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=300, watermark=0.0)
    req = make_req(0, prompt=1000)
    action = policy.select_action(make_state([req]))
    assert action.admit[0] == [0]
    state = policy._request_states[0][0]
    assert state.remaining_prefill == 1000 - 300  # first chunk = 300, 700 left


def test_chunk_boundaries_exact_multiple():
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=250, watermark=0.0)
    req = make_req(0, prompt=1000)  # 1000 / 250 = 4 exact chunks
    policy.select_action(make_state([req]))
    assert policy._request_states[0][0].remaining_prefill == 750


def test_final_partial_chunk_lands_exactly_at_zero():
    """A prompt not evenly divisible by the budget: the LAST chunk must be
    the exact remainder, never overshooting into negative remaining."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=300, watermark=0.0)
    req = make_req(0, prompt=1000)
    gpu = make_gpu()
    policy.select_action(make_state([req], gpu=gpu))  # 1000 -> 700 remaining

    active = make_req(0, prompt=1000)
    for expected_remaining in (400, 100):
        gpu2 = make_gpu(active_reqs=[active])
        policy.select_action(make_state([], gpu=gpu2))
        assert policy._request_states[0][0].remaining_prefill == expected_remaining

    gpu3 = make_gpu(active_reqs=[active])
    policy.select_action(make_state([], gpu=gpu3))
    assert policy._request_states[0][0].remaining_prefill == 0


def test_multiple_waiting_prefills_share_budget_fcfs():
    """EXACT-SOURCE FIDELITY: multiple new admissions in the same iteration
    (_schedule_prefills, enable_chunking=True) share one budget, FCFS -- the
    earlier one's chunk is capped by its OWN prompt length (200 < 300), so
    it fully finishes prefill this very iteration (remaining=0), leaving
    the rest of the budget (100) for the second request's own chunk."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=300, watermark=0.0)
    req_a = make_req(0, prompt=200, arrival=0.0)
    req_b = make_req(1, prompt=200, arrival=0.001)
    action = policy.select_action(make_state([req_a, req_b]))
    assert action.admit[0] == [0, 1]
    assert policy._request_states[0][0].remaining_prefill == 0    # 200 <= 300 budget: fully chunked in
    assert policy._request_states[0][1].remaining_prefill == 100  # 300-200=100 leftover budget for its chunk


def test_second_waiting_request_gets_zero_chunk_stops_admission():
    """EXACT-SOURCE FIDELITY: if the shared budget is fully consumed by an
    earlier waiting request, a later one whose own chunk would be 0 closes
    admission entirely (break, matching `_schedule_prefills` lines 688-691:
    `if (num_new_tokens == 0 or not budget.can_schedule(...)): break`)."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=100, watermark=0.0)
    req_a = make_req(0, prompt=100, arrival=0.0)  # consumes the ENTIRE budget
    req_b = make_req(1, prompt=200, arrival=0.001)
    action = policy.select_action(make_state([req_a, req_b]))
    assert action.admit[0] == [0]
    assert 1 not in action.admit[0]


def test_deterministic_chunk_order_by_arrival_then_id():
    """EXACT-SOURCE FIDELITY: like vllm_faithful/sarathi_faithful, this
    policy trusts state.waiting_queue's given FCFS-with-preemption order
    rather than re-sorting."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=50, watermark=0.0, max_num_seqs=1)
    req_lo_id = make_req(2, prompt=200, arrival=0.0)
    req_hi_id = make_req(5, prompt=200, arrival=0.0)
    action = policy.select_action(make_state([req_lo_id, req_hi_id]))
    assert action.admit[0] == [2]


# ---------------------------------------------------------------------------
# The key structural difference from sarathi_faithful: NO explicit
# decode-priority phase -- decode and continuing-prefill share ONE
# FCFS-by-arrival budget (docs/vllm_chunked_prefill_faithful_scheduler_
# reference.md's algorithm section). EXACT-SOURCE FIDELITY.
# ---------------------------------------------------------------------------

def test_earlier_arriving_continuing_prefill_can_starve_later_decode_this_step():
    """This is the reference doc's headline structural finding, exercised
    directly: an earlier-arriving, still-prefilling request consumes the
    ENTIRE shared per-step budget before a LATER-arriving, already-decoding
    request gets any -- unlike sarathi_faithful, which unconditionally
    processes decode-phase requests first regardless of arrival order."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=1, watermark=0.0, block_size=1)
    # request 0: arrived FIRST, still mid-prefill (large remaining prompt).
    prefilling_req = make_req(0, prompt=1000, arrival=0.0)
    # request 1: arrived LATER, already decoding (prefill done).
    decoding_req = make_req(1, prompt=5, arrival=1.0)
    gpu = make_gpu(max_kv=10_000, active_reqs=[prefilling_req, decoding_req])
    bm = policy._get_block_manager(gpu)
    states = policy._get_request_states(0)
    bm.allocate(0, 1000)
    bm.allocate(1, 5)
    states[0] = _RequestState(remaining_prefill=500)  # still prefilling
    states[1] = _RequestState(remaining_prefill=0)     # decoding

    kv_before = bm.kv_tokens_for(1)
    action = policy.select_action(make_state([], gpu=gpu))
    # The decoding request is NOT preempted (there is ample KV capacity) --
    # it simply receives no budget/progress this iteration, because the
    # earlier-arriving continuing-prefill request consumed the entire
    # 1-token budget first.
    assert action.preempt.get(0, []) == []
    assert bm.kv_tokens_for(1) == kv_before  # append_slot was never called for it
    assert states[0].remaining_prefill == 499  # the prefill candidate DID progress


def test_contrast_sarathi_faithful_protects_the_same_decode_request():
    """Direct contrast with the test above, same setup, sarathi_faithful:
    Phase 1a unconditionally processes decode-phase requests BEFORE any
    prefill continuation gets a look at the budget, so the decoding request
    is never starved by an earlier-arriving prefill the way it is above."""
    policy = SarathiFaithfulPolicy(chunk_size=1, watermark=0.0, block_size=1)
    prefilling_req = make_req(0, prompt=1000, arrival=0.0)
    decoding_req = make_req(1, prompt=5, arrival=1.0)
    gpu = make_gpu(max_kv=10_000, active_reqs=[prefilling_req, decoding_req])
    bm = policy._get_block_manager(gpu)
    from llmserveopt.policies.sarathi_faithful import _RequestState as SarathiRequestState
    states = policy._get_request_states(0)
    bm.allocate(0, 1000)
    bm.allocate(1, 5)
    states[0] = SarathiRequestState(remaining_prefill=500)
    states[1] = SarathiRequestState(remaining_prefill=0)

    kv_before = bm.kv_tokens_for(1)
    policy.select_action(make_state([], gpu=gpu))
    # Sarathi's decode-first Phase 1a DOES grow the decoding request's slot
    # this iteration, unlike vllm_chunked_prefill_faithful above.
    assert bm.kv_tokens_for(1) == kv_before + 1
    # ...leaving 0 budget for the prefill continuation (chunk_size=1, fully
    # consumed by decode) -- it makes no progress this iteration.
    assert states[0].remaining_prefill == 500


# ---------------------------------------------------------------------------
# Continuing-prefill as an eligible preemption VICTIM (but never itself
# initiates a slot check) -- EXACT-SOURCE FIDELITY, see reference doc's
# "why continuing-prefill sequences never actually fail can_append_slots".
# ---------------------------------------------------------------------------

def test_continuing_prefill_request_eligible_as_preemption_victim():
    """A candidate already fully processed this iteration (added to
    kept_ids) can never be evicted by a LATER candidate's slot search --
    identical to vllm_faithful/sarathi_faithful (mirrors `running_queue.
    popleft()` removing it from the pool before any subsequent candidate's
    `running_queue.pop()` victim search runs). So for a continuing-prefill
    candidate to be evicted, it must arrive LATER (still unprocessed/
    pending) than the decode candidate that needs the slot it's holding."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=100, watermark=0.0, block_size=1)
    decoding_req = make_req(0, prompt=1, arrival=0.0)      # processed FIRST (needs a new block to decode)
    prefilling_req = make_req(1, prompt=1, arrival=0.001)  # still PENDING when decoding_req's slot search runs
    gpu = make_gpu(max_kv=2, active_reqs=[decoding_req, prefilling_req])  # 2 blocks total, both already used
    bm = policy._get_block_manager(gpu)
    states = policy._get_request_states(0)
    bm.allocate(0, 1)  # decoding_req's 1 existing block
    bm.allocate(1, 1)  # prefilling_req's 1 existing block -- 0 free left
    states[0] = _RequestState(remaining_prefill=0)     # decoding
    states[1] = _RequestState(remaining_prefill=500)   # still mid-prefill

    action = policy.select_action(make_state([], gpu=gpu))
    # decoding_req needs a NEW block (block_size=1: 1 token growth = 1 new
    # block) and none is free -> the only other candidate, prefilling_req
    # (still pending, later arrival = lower priority), is evicted.
    assert action.preempt[0] == [1]


def test_continuing_prefill_never_itself_fails_slot_check():
    """A continuing-prefill candidate's own KV blocks were fully reserved
    at admission (its full prompt token count) -- calling can_append_slot
    for it would be a SIMULATOR ADAPTATION concern only if this policy
    called it at all, which it deliberately does not (see module
    docstring). This test asserts the policy never shrinks available
    capacity because of a continuing-prefill candidate's own progress."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=1000, watermark=0.0, block_size=16)
    req = make_req(0, prompt=100, arrival=0.0)
    gpu = make_gpu(max_kv=1000, active_reqs=[req])
    bm = policy._get_block_manager(gpu)
    states = policy._get_request_states(0)
    bm.allocate(0, 100)
    states[0] = _RequestState(remaining_prefill=50)
    free_before = bm.num_free_blocks
    policy.select_action(make_state([], gpu=gpu))
    assert bm.num_free_blocks == free_before  # no new block consumed


# ---------------------------------------------------------------------------
# Scheduler fidelity: tight budget / capacity / max_num_seqs / preemption
# EXACT-SOURCE FIDELITY (mirrors vllm_faithful's/sarathi_faithful's own
# equivalent tests -- same underlying block_manager_v1 semantics, confirmed
# unchanged at this pin in the reference doc).
# ---------------------------------------------------------------------------

def test_tight_token_budget_blocks_admission_entirely():
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=1, watermark=0.0, block_size=1)
    decoding_req = make_req(0, prompt=5, arrival=0.0)
    gpu = make_gpu(max_kv=1000, active_reqs=[decoding_req])
    bm = policy._get_block_manager(gpu)
    bm.allocate(0, 5)
    policy._get_request_states(0)[0] = _RequestState(remaining_prefill=0)

    waiting_req = make_req(1, prompt=100, arrival=0.001)
    action = policy.select_action(make_state([waiting_req], gpu=gpu))
    assert action.admit[0] == []  # budget fully consumed by the 1 decoding request


def test_admission_stops_at_first_non_allocatable_request():
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=1000, watermark=0.0, block_size=1)
    gpu = make_gpu(max_kv=10)  # only 10 blocks/tokens of capacity
    big_req = make_req(0, prompt=100, arrival=0.0)   # cannot be allocated
    small_req = make_req(1, prompt=2, arrival=0.001)  # would easily fit alone
    action = policy.select_action(make_state([big_req, small_req], gpu=gpu))
    assert action.admit[0] == []  # neither admitted -- big_req blocks the queue


def test_max_num_seqs_caps_concurrent_admissions():
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=10_000, watermark=0.0, max_num_seqs=1)
    gpu = make_gpu(max_kv=100_000)
    reqs = [make_req(0, prompt=50, arrival=0.0), make_req(1, prompt=50, arrival=0.001)]
    action = policy.select_action(make_state(reqs, gpu=gpu))
    assert action.admit[0] == [0]


def test_preempts_lowest_priority_decoding_sequence_when_out_of_blocks():
    """Identical victim-selection algorithm to vllm_faithful/sarathi_faithful."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=100, watermark=0.0, block_size=1)
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
    assert action.preempt[0] == [1]


# ---------------------------------------------------------------------------
# End-to-end: no starvation, determinism, Phase-1 legacy compatibility
# ---------------------------------------------------------------------------

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
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=32, max_num_seqs=128, watermark=0.0)

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
        policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=32, watermark=0.01)
        return sim.run(policy, workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.num_dropped == m2.num_dropped


def test_phase1_legacy_mode_still_works_without_prefill_modeling():
    """SIMULATOR ADAPTATION: like sarathi_faithful, this policy must not
    require enable_prefill_modeling=True to run without crashing -- it
    degrades to instant-prefill admission decisions."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))  # default ServiceModel, Phase 1
    reqs = [Request(request_id=i, arrival_time=float(i) * 0.001, prompt_tokens=20,
                     predicted_output_tokens=5, actual_output_tokens=5,
                     slo_deadline=1000.0, priority=1.0, class_id="medium") for i in range(5)]
    sim.load_trace(reqs)
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=64, watermark=0.0)
    metrics = sim.run(policy, workload_tag="phase1-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# Historical regression requirement (task-mandated): vllm_faithful's
# historical all-or-nothing-admission behavior must be byte-for-byte
# unchanged; vllm_chunked_prefill_faithful must admit the SAME oversized
# prompt incrementally and complete it.
# ---------------------------------------------------------------------------

def test_historical_vllm_faithful_never_admits_oversized_prompt():
    """prompt_tokens (5000) > max_num_batched_tokens (2560): vllm_faithful
    must NEVER admit this request -- exact historical behavior, unchanged."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=256, max_batch_tokens=100_000, max_kv_tokens=131_072)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=512, max_prefill_chunk_tokens=512)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=2000))
    sim.load_trace([Request(request_id=0, arrival_time=0.0, prompt_tokens=5000,
                             predicted_output_tokens=32, actual_output_tokens=32,
                             slo_deadline=100_000.0, priority=1.0, class_id="medium")])
    policy = VLLMFaithfulPolicy(max_num_batched_tokens=2560)
    metrics = sim.run(policy, workload_tag="historical-regression-vllm-faithful")
    assert metrics.num_completed == 0
    assert metrics.completion_fraction == 0.0


def test_new_baseline_admits_and_completes_the_same_oversized_prompt():
    """The negative-image of the test above, same request shape, same
    max_num_batched_tokens=2560 budget: vllm_chunked_prefill_faithful must
    schedule it in multiple chunks, show monotonic prefill progress,
    eventually finish prefill, transition to decode, and complete."""
    gpu = GPUConfig(gpu_id=0, max_active_sequences=256, max_batch_tokens=100_000, max_kv_tokens=131_072)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=512, max_prefill_chunk_tokens=512)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=2000))
    sim.load_trace([Request(request_id=0, arrival_time=0.0, prompt_tokens=5000,
                             predicted_output_tokens=32, actual_output_tokens=32,
                             slo_deadline=100_000.0, priority=1.0, class_id="medium")])
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=2560)
    metrics = sim.run(policy, workload_tag="historical-regression-chunked")
    assert metrics.num_completed == 1
    assert metrics.completion_fraction == 1.0
    assert metrics.mean_ttft > 0.0  # took multiple prefill steps, not instant


def test_new_baseline_shows_monotonic_prefill_progress_across_chunks():
    """Direct, deterministic trace of the admitted request's shadow
    remaining_prefill strictly decreasing (never increasing) step over
    step, until it reaches 0 and the request transitions to decode."""
    policy = VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=2560, watermark=0.0)
    req = make_req(0, prompt=5000, arrival=0.0)
    action = policy.select_action(make_state([req]))
    assert action.admit[0] == [0]
    remaining_history = [policy._request_states[0][0].remaining_prefill]
    assert remaining_history[-1] == 5000 - 2560

    active = make_req(0, prompt=5000)
    while policy._request_states[0][0].remaining_prefill > 0:
        gpu = make_gpu(active_reqs=[active])
        policy.select_action(make_state([], gpu=gpu))
        remaining_history.append(policy._request_states[0][0].remaining_prefill)

    for earlier, later in zip(remaining_history, remaining_history[1:]):
        assert later < earlier  # strictly monotonic decrease
    assert remaining_history[-1] == 0
    assert len(remaining_history) == 2  # ceil(5000/2560) chunks


# ---------------------------------------------------------------------------
# vllm_faithful's own suite is unaffected -- byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_vllm_faithful_registry_metadata_unchanged():
    """Sanity: importing this new module must not have mutated
    vllm_faithful's own defaults/behavior in any way (separate file, no
    shared mutable state)."""
    assert VLLMFaithfulPolicy().max_num_batched_tokens == 2560
    assert VLLMFaithfulPolicy().max_num_seqs == 256
    assert VLLMFaithfulPolicy().name == "vllm_faithful"
    assert VLLMChunkedPrefillFaithfulPolicy().name == "vllm_chunked_prefill_faithful"
    assert VLLMChunkedPrefillFaithfulPolicy().max_num_batched_tokens == 512

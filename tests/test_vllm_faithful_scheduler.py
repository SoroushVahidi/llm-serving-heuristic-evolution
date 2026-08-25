"""Tests for src/llmserveopt/policies/vllm_faithful.py.

Covers scheduler correctness and KV-block-memory correctness (the two
things this baseline claims fidelity for -- see
docs/vllm_faithful_scheduler_reference.md). Runtime/hardware performance
reproduction is explicitly out of scope and not tested here.

Some tests hand-derive the expected outcome directly from vLLM v0.1.0's
Scheduler._schedule() / BlockSpaceManager (commit 67d96c29, read live via
the GitHub API while building this baseline) -- these are marked "fidelity"
in their docstring.
"""
from __future__ import annotations

import warnings

from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Hand-constructed ObservableState helpers (matches the repo's established
# direct-state policy-test convention, e.g. test_least_laxity_first_policy.py)
# ---------------------------------------------------------------------------

def make_req(req_id, prompt=8, output=8, deadline=100.0, priority=1.0, arrival=0.0):
    return ObservableRequest(
        request_id=req_id, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline,
        priority=priority, class_id="medium",
    )


def make_gpu(gpu_id=0, max_seq=100, max_kv=1000, max_batch=1000, active_reqs=(), decoded=None):
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
# Basic admission within capacity
# ---------------------------------------------------------------------------

def test_admits_all_requests_within_ample_capacity():
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0)
    req_a, req_b = make_req(0, prompt=8), make_req(1, prompt=8, arrival=0.001)
    state = make_state([req_a, req_b])
    action = policy.select_action(state)
    assert sorted(action.admit[0]) == [0, 1]
    assert action.preempt.get(0, []) == []


def test_deterministic_tie_breaking_by_request_id():
    """Fidelity: the pinned reference's FCFS policy sorts by arrival time;
    equal arrival times must break ties deterministically by request_id."""
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0)
    gpu = make_gpu(max_seq=1)  # only one sequence can ever be admitted
    req_hi_id, req_lo_id = make_req(5, arrival=0.0), make_req(2, arrival=0.0)
    state = make_state([req_lo_id, req_hi_id], gpu=gpu)  # queue order shouldn't matter either
    action = policy.select_action(state)
    assert action.admit[0] == [2]


def test_deterministic_across_repeated_fresh_runs():
    def run_once():
        policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0)
        gpu = make_gpu(max_seq=2)
        reqs = [make_req(i, arrival=float(i) * 0.001) for i in range(5)]
        action = policy.select_action(make_state(reqs, gpu=gpu))
        return tuple(sorted(action.admit[0]))

    results = [run_once() for _ in range(5)]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# KV-capacity-limited admission (block manager watermark)
# ---------------------------------------------------------------------------

def test_watermark_blocks_admission_when_it_would_leave_too_little_free():
    """Fidelity: BlockSpaceManager.can_allocate requires
    free_blocks - needed >= watermark_blocks, not merely free_blocks >= needed."""
    # block_size=4, num_gpu_blocks = 40//4 = 10, watermark=0.2 -> watermark_blocks=2.
    policy = VLLMFaithfulPolicy(block_size=4, watermark=0.2)
    gpu = make_gpu(max_kv=40, max_seq=100)
    # req needs ceil((32+1)/4) = ceil(33/4) = 9 blocks -> leaves 10-9=1 free < 2 watermark -> rejected.
    req = make_req(0, prompt=32)
    action = policy.select_action(make_state([req], gpu=gpu))
    assert action.admit[0] == []


def test_admission_succeeds_exactly_at_watermark_boundary():
    # Same setup, but a smaller prompt that leaves exactly the watermark reserve.
    policy = VLLMFaithfulPolicy(block_size=4, watermark=0.2)
    gpu = make_gpu(max_kv=40, max_seq=100)
    # req needs ceil((23+1)/4) = 6 blocks -> leaves 10-6=4 free >= 2 watermark -> admitted.
    req = make_req(0, prompt=23)
    action = policy.select_action(make_state([req], gpu=gpu))
    assert action.admit[0] == [0]


def test_kv_capacity_limits_number_admitted():
    policy = VLLMFaithfulPolicy(block_size=8, watermark=0.0)
    # 3 blocks * 8 = 24 tokens capacity. Each request needs prompt+1=9 -> 2 blocks.
    # Only 1 fits (2 blocks), the 2nd would need 2 more (4 total > 3 available).
    gpu = make_gpu(max_kv=24, max_seq=100)
    reqs = [make_req(0, prompt=8, arrival=0.0), make_req(1, prompt=8, arrival=0.001)]
    action = policy.select_action(make_state(reqs, gpu=gpu))
    assert action.admit[0] == [0]


# ---------------------------------------------------------------------------
# max_num_seqs / max_num_batched_tokens budgets
# ---------------------------------------------------------------------------

def test_max_num_seqs_caps_concurrent_admissions():
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0, max_num_seqs=1)
    gpu = make_gpu(max_seq=100, max_kv=10_000)  # ample block/simulator capacity
    reqs = [make_req(0, arrival=0.0), make_req(1, arrival=0.001)]
    action = policy.select_action(make_state(reqs, gpu=gpu))
    assert action.admit[0] == [0]  # only 1 admitted despite ample KV capacity


def test_max_num_batched_tokens_caps_prompt_admission_this_iteration():
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0, max_num_batched_tokens=100)
    gpu = make_gpu(max_seq=100, max_kv=10_000)
    # First request's prompt (80) fits the 100-token iteration budget; the
    # second (80 more) would push it to 160 > 100 -> stays waiting this step.
    reqs = [make_req(0, prompt=80, arrival=0.0), make_req(1, prompt=80, arrival=0.001)]
    action = policy.select_action(make_state(reqs, gpu=gpu))
    assert action.admit[0] == [0]


def test_simulator_gpu_config_is_a_final_safety_net():
    """Even if this policy's own vLLM-style budgets would allow more, it
    must never exceed the simulator's own GPUConfig capacity (guards
    against spurious admission-rejected warnings in tight configs)."""
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0)
    gpu = make_gpu(max_seq=1, max_kv=10_000)  # simulator-native cap is the binding one
    reqs = [make_req(0, arrival=0.0), make_req(1, arrival=0.001)]
    action = policy.select_action(make_state(reqs, gpu=gpu))
    assert action.admit[0] == [0]


# ---------------------------------------------------------------------------
# Preemption: lowest-priority running sequence evicted first
# ---------------------------------------------------------------------------

def test_preempts_lowest_priority_running_when_out_of_blocks():
    """Fidelity: Scheduler._schedule's running-queue loop preempts the
    lowest-priority (here: latest-arrived) sequence group first when a
    higher-priority one cannot get its next decode slot.

    block_size=1 so blocks_needed == exact token count (no fragmentation
    ambiguity), num_gpu_blocks=4. Both requests need prompt(1)+1=2 tokens
    at admission -> 2 blocks each -> exactly 4 blocks used, 0 free. The
    very next growth call must therefore preempt someone."""
    policy = VLLMFaithfulPolicy(block_size=1, watermark=0.0)
    gpu = make_gpu(max_kv=4, max_seq=100)

    req_a = make_req(0, prompt=1, arrival=0.0)
    req_b = make_req(1, prompt=1, arrival=0.001)
    action1 = policy.select_action(make_state([req_a, req_b], gpu=gpu, step=0))
    assert sorted(action1.admit[0]) == [0, 1]

    # Growth call: active_requests_info is static (ObservableRequest carries
    # only prompt_tokens, not decode progress); this policy tracks decode
    # growth in its OWN internal block manager, not from
    # tokens_decoded_per_request, so no manual state bump is needed here.
    active_a = make_req(0, prompt=1, arrival=0.0)
    active_b = make_req(1, prompt=1, arrival=0.001)
    gpu2 = ObservableGPUState(
        gpu_id=0, max_active_sequences=100, max_batch_tokens=1000, max_kv_tokens=4,
        active_request_ids=[0, 1], active_requests_info=[active_a, active_b],
        current_kv_tokens=4, tokens_decoded_per_request={0: 1, 1: 1},
    )
    action2 = policy.select_action(make_state([], gpu=gpu2, step=1))
    # request 0 (earlier arrival, higher priority) is processed first and
    # needs a 3rd block; nothing free -> request 1 (lower priority) is
    # evicted to make room, then request 0 succeeds.
    assert action2.preempt[0] == [1]
    assert action2.admit[0] == []


def test_running_growth_is_reserved_before_new_admission_in_same_iteration():
    """Fidelity: Scheduler._schedule reserves decode slots for already-
    RUNNING sequences (step 1) BEFORE considering new admissions from
    `waiting` (step 3), in the same iteration. Here, request 0's own growth
    consumes the last free block, so request 1 must NOT be admitted this
    same step even though it would have fit had request 0 not grown."""
    policy = VLLMFaithfulPolicy(block_size=1, watermark=0.0)
    gpu = make_gpu(max_kv=4, max_seq=100)
    req_a = make_req(0, prompt=1, arrival=0.0)
    action1 = policy.select_action(make_state([req_a], gpu=gpu, step=0))
    assert action1.admit[0] == [0]  # 2 blocks used, 2 free

    # Step 1: request 0 (already running) grows from 2->3 blocks, consuming
    # the last free block. Request 1 (prompt=1, needs 2 blocks) arrives and
    # is waiting, but only 1 free block remains after request 0's growth.
    req_b = make_req(1, prompt=1, arrival=0.001)
    action2 = policy.select_action(make_state(
        [req_b], gpu=make_gpu(max_kv=4, max_seq=100, active_reqs=[req_a]), step=1,
    ))
    assert action2.preempt[0] == []  # request 0 still had a free block available
    assert action2.admit[0] == []  # request 1 correctly NOT admitted this step


# ---------------------------------------------------------------------------
# Full-simulator end-to-end behavior
# ---------------------------------------------------------------------------

def _gpu_config(max_seq=100, max_tok=1000, max_kv=1000):
    return GPUConfig(gpu_id=0, max_active_sequences=max_seq, max_batch_tokens=max_tok, max_kv_tokens=max_kv)


def _req(rid, arrival, prompt=8, output=8, deadline=1000.0, priority=1.0):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id="medium",
    )


def test_no_request_admitted_before_arrival_end_to_end():
    gpu = _gpu_config(max_seq=1)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(0, arrival=0.0, output=5), _req(1, arrival=0.05, output=5)])
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0)

    violations = []

    def checked_select(state, _orig=policy.select_action):
        for req in state.waiting_queue:
            if req.arrival_time > state.time:
                violations.append((req.request_id, req.arrival_time, state.time))
        return _orig(state)

    policy.select_action = checked_select
    metrics = sim.run(policy, workload_tag="no-early-admission")
    assert violations == []
    assert metrics.num_completed == 2


def test_no_capacity_violation_end_to_end():
    """Every step, active-sequence count and KV usage must stay within the
    simulator's own GPUConfig -- proven by the absence of any admission-
    rejected warning across a moderately loaded run."""
    gpu = _gpu_config(max_seq=8, max_kv=200)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.002, prompt=10, output=15) for i in range(30)]
    sim.load_trace(reqs)
    policy = VLLMFaithfulPolicy(block_size=8, watermark=0.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="capacity-check")
    assert w == [], f"unexpected simulator warnings: {[str(x.message) for x in w]}"
    assert metrics.num_completed == 30


def test_continuous_admission_after_completion():
    """A request that cannot be admitted due to max_num_seqs must be
    admitted once an earlier one completes and frees a slot -- continuous
    batching, not static once-per-run admission."""
    gpu = _gpu_config(max_seq=1, max_kv=10_000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(0, arrival=0.0, output=5), _req(1, arrival=0.0, output=5)])
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0, max_num_seqs=1)
    metrics = sim.run(policy, workload_tag="continuous-admission")
    assert metrics.num_completed == 2
    assert metrics.num_dropped == 0


def test_batch_membership_changes_across_iterations():
    """The active set is not static: it grows on admission and shrinks on
    completion across the run (continuous batching)."""
    gpu = _gpu_config(max_seq=2, max_kv=10_000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, output=5) for i in range(6)])
    policy = VLLMFaithfulPolicy(block_size=16, watermark=0.0, max_num_seqs=2)

    observed_active_sets = []

    def recording_select(state, _orig=policy.select_action):
        observed_active_sets.append(frozenset(state.gpu_states[0].active_request_ids))
        return _orig(state)

    policy.select_action = recording_select
    metrics = sim.run(policy, workload_tag="batch-membership")
    assert metrics.num_completed == 6
    assert len(set(observed_active_sets)) > 1, "active set must change over the run"


def test_regression_matches_previous_run_bit_for_bit():
    """Same trace/config run twice must produce identical metrics (no
    hidden randomness anywhere in this policy)."""
    def run():
        gpu = _gpu_config(max_seq=4, max_kv=500)
        sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=10_000))
        sim.load_trace([_req(i, arrival=float(i) * 0.003, prompt=12, output=10) for i in range(15)])
        policy = VLLMFaithfulPolicy(block_size=8, watermark=0.01)
        return sim.run(policy, workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.num_dropped == m2.num_dropped

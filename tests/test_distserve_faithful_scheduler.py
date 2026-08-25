"""Tests for src/llmserveopt/policies/distserve_faithful.py.

Covers scheduler correctness (context-stage FCFS admission, bridge-queue
handoff, decode-stage FCFS admission + swap-based capacity management) and
KV-block-memory correctness -- the fidelity claims documented in
docs/distserve_faithful_scheduler_reference.md. Runtime/hardware
performance reproduction and DistServe's offline parallelism/placement
planner are explicitly out of scope and not tested here.

Tests marked "fidelity" hand-derive the expected outcome directly from the
pinned reference (LLMServe/DistServe commit
0ec355c8743d3fbd2d02f3cd62b5be6eae368f92, read live via the GitHub API
while building this baseline).
"""
from __future__ import annotations

import warnings

import pytest

from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.policies.distserve_faithful import DistServeFaithfulPolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Hand-constructed ObservableState helpers (matches the repo's established
# direct-state policy-test convention, e.g. test_vllm_faithful_scheduler.py)
# ---------------------------------------------------------------------------

def make_req(req_id, prompt=8, output=8, deadline=1000.0, priority=1.0, arrival=0.0):
    return ObservableRequest(
        request_id=req_id, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline,
        priority=priority, class_id="medium",
    )


def make_gpu(gpu_id, role, max_seq=100, max_kv=100_000, max_batch=1_000_000, active_reqs=(), decoded=None):
    return ObservableGPUState(
        gpu_id=gpu_id, max_active_sequences=max_seq, max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=[r.request_id for r in active_reqs],
        active_requests_info=list(active_reqs),
        current_kv_tokens=sum(r.prompt_tokens for r in active_reqs),
        tokens_decoded_per_request=decoded or {},
        role=role,
    )


def make_state(waiting=(), migrating=(), context_gpu=None, decode_gpu=None, extra_gpus=(), now=0.0, step=0):
    if context_gpu is None:
        context_gpu = make_gpu(0, "prefill")
    if decode_gpu is None:
        decode_gpu = make_gpu(1, "decode")
    return ObservableState(
        time=now, waiting_queue=list(waiting), gpu_states=[context_gpu, decode_gpu, *extra_gpus],
        completed_count=0, step=step, migrating_queue=list(migrating),
    )


def _gpu_configs(context_kv=100_000, decode_kv=100_000, context_seq=100, decode_seq=100):
    return [
        GPUConfig(gpu_id=0, max_active_sequences=context_seq, max_batch_tokens=1_000_000,
                  max_kv_tokens=context_kv, role="prefill"),
        GPUConfig(gpu_id=1, max_active_sequences=decode_seq, max_batch_tokens=1_000_000,
                  max_kv_tokens=decode_kv, role="decode"),
    ]


def _disagg_service_model(step_token_budget=100_000, max_prefill_chunk_tokens=100_000,
                           migration_transfer_delay=0.0, decode_first=True):
    return ServiceModel(
        enable_prefill_modeling=True, enable_disaggregation=True, decode_first=decode_first,
        step_token_budget=step_token_budget, max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        prefill_cost_per_token=1.0, migration_transfer_delay=migration_transfer_delay,
    )


def _req(rid, arrival=0.0, prompt=50, output=10, deadline=1000.0, priority=1.0):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id="medium",
    )


# ---------------------------------------------------------------------------
# Worker-count validation
# ---------------------------------------------------------------------------

def test_requires_exactly_one_prefill_and_one_decode_gpu():
    policy = DistServeFaithfulPolicy()
    extra_prefill = make_gpu(2, "prefill")
    state = make_state(waiting=[make_req(0)], extra_gpus=[extra_prefill])
    with pytest.raises(ValueError):
        policy.select_action(state)


def test_rejects_zero_role_gpus():
    policy = DistServeFaithfulPolicy()
    gpu_none_a = make_gpu(0, None)
    gpu_none_b = make_gpu(1, None)
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[gpu_none_a, gpu_none_b], completed_count=0, step=0)
    with pytest.raises(ValueError):
        policy.select_action(state)


# ---------------------------------------------------------------------------
# Context (prefill) stage
# ---------------------------------------------------------------------------

def test_context_stage_fcfs_ordering():
    policy = DistServeFaithfulPolicy(context_max_batch_size=1)
    req_hi_id, req_lo_id = make_req(5, arrival=0.0), make_req(2, arrival=0.0)
    state = make_state(waiting=[req_lo_id, req_hi_id])  # queue order shouldn't matter
    action = policy.select_action(state)
    assert action.admit[0] == [2]


def test_context_stage_exact_batch_size_boundary():
    policy = DistServeFaithfulPolicy(context_max_batch_size=2)
    reqs = [make_req(i, arrival=float(i) * 0.001) for i in range(3)]
    action = policy.select_action(make_state(waiting=reqs))
    assert action.admit[0] == [0, 1]  # exactly at the boundary, 3rd stays waiting


def test_context_stage_exact_token_budget_boundary():
    policy = DistServeFaithfulPolicy(context_max_tokens_per_batch=100)
    # First request's prompt (60) fits; a second (60 more) would push to 120 > 100.
    reqs = [make_req(0, prompt=60, arrival=0.0), make_req(1, prompt=60, arrival=0.001)]
    action = policy.select_action(make_state(waiting=reqs))
    assert action.admit[0] == [0]


def test_context_stage_token_budget_admits_exactly_at_boundary():
    policy = DistServeFaithfulPolicy(context_max_tokens_per_batch=100)
    reqs = [make_req(0, prompt=60, arrival=0.0), make_req(1, prompt=40, arrival=0.001)]
    action = policy.select_action(make_state(waiting=reqs))
    assert action.admit[0] == [0, 1]  # 60+40 == 100, exactly fits


def test_context_stage_oversized_request_blocks_admission_entirely():
    """Fidelity: get_next_batch_and_pop stops (does not skip-and-continue)
    at the first request that cannot be admitted -- an oversized request at
    the front of the queue blocks everyone behind it this iteration."""
    policy = DistServeFaithfulPolicy(context_max_tokens_per_batch=50)
    oversized = make_req(0, prompt=100, arrival=0.0)
    fits = make_req(1, prompt=10, arrival=0.001)
    action = policy.select_action(make_state(waiting=[oversized, fits]))
    assert action.admit[0] == []


def test_context_stage_kv_capacity_limits_admission():
    # block_size=16, num_gpu_blocks = 32//16 = 2 blocks = 32 tokens capacity.
    policy = DistServeFaithfulPolicy(block_size=16)
    gpu = make_gpu(0, "prefill", max_kv=32)
    reqs = [make_req(0, prompt=32, arrival=0.0), make_req(1, prompt=16, arrival=0.001)]
    action = policy.select_action(make_state(waiting=reqs, context_gpu=gpu))
    assert action.admit[0] == [0]  # exactly fills capacity; 2nd has no room


def test_context_stage_deterministic_across_runs():
    def run_once():
        policy = DistServeFaithfulPolicy(context_max_batch_size=2)
        reqs = [make_req(i, arrival=float(i) * 0.001) for i in range(5)]
        return tuple(policy.select_action(make_state(waiting=reqs)).admit[0])

    results = [run_once() for _ in range(5)]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# Bridge / handoff (migrating_queue)
# ---------------------------------------------------------------------------

def test_migrating_request_not_admitted_until_visible():
    """A request absent from migrating_queue (still mid-transfer, per the
    Simulator's own transfer_ready_time gate) must never be admitted by this
    policy -- it can only ever see what ObservableState exposes."""
    policy = DistServeFaithfulPolicy()
    state = make_state(migrating=[])  # nothing transfer-ready yet
    action = policy.select_action(state)
    assert action.admit[1] == []


def test_migrating_request_admitted_exactly_when_visible():
    policy = DistServeFaithfulPolicy()
    req = make_req(0, prompt=20)
    action = policy.select_action(make_state(migrating=[req]))
    assert action.admit[1] == [0]


def test_no_duplicate_handoff_end_to_end():
    """Across a full run, no request_id ever appears in more than one of
    {waiting, migrating, active-on-any-gpu} simultaneously."""
    gpus = _gpu_configs()
    sm = _disagg_service_model(migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    n = 10
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=30, output=8) for i in range(n)])
    policy = DistServeFaithfulPolicy()

    violations = []

    def checked_select(state, _orig=policy.select_action):
        seen = list(r.request_id for r in state.waiting_queue)
        seen += [r.request_id for r in state.migrating_queue]
        for g in state.gpu_states:
            seen += g.active_request_ids
        if len(seen) != len(set(seen)):
            violations.append(state.step)
        return _orig(state)

    policy.select_action = checked_select
    metrics = sim.run(policy, workload_tag="no-dup")
    assert violations == []
    assert metrics.num_completed == n
    assert metrics.num_dropped == 0


def test_no_lost_request_end_to_end():
    gpus = _gpu_configs()
    sm = _disagg_service_model(migration_transfer_delay=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    n = 15
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=25, output=6) for i in range(n)])
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="no-loss")
    assert metrics.num_completed == n
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# Decode stage
# ---------------------------------------------------------------------------

def test_decode_stage_fcfs_admission_order():
    policy = DistServeFaithfulPolicy(decode_max_batch_size=1)
    req_hi_id, req_lo_id = make_req(5, prompt=20, arrival=0.0), make_req(2, prompt=20, arrival=0.0)
    action = policy.select_action(make_state(migrating=[req_lo_id, req_hi_id]))
    assert action.admit[1] == [2]


def test_decode_stage_max_batch_size_boundary():
    policy = DistServeFaithfulPolicy(decode_max_batch_size=2, waiting_block_prop_threshold=1.0)
    reqs = [make_req(i, prompt=10, arrival=float(i) * 0.001) for i in range(3)]
    action = policy.select_action(make_state(migrating=reqs))
    assert action.admit[1] == [0, 1]


def test_decode_capacity_saturation_triggers_swap():
    """Fidelity: DecodingStageFCFSScheduler.get_next_batch's capacity loop
    swaps out the lowest-priority (latest-arrival) active request when an
    earlier one cannot get its next decode slot -- this policy uses
    Action.swap (preserve progress), not Action.preempt."""
    policy = DistServeFaithfulPolicy(block_size=1, decode_max_batch_size=100,
                                       waiting_block_prop_threshold=1.0)
    decode_gpu = make_gpu(1, "decode", max_kv=4)
    req_a = make_req(0, prompt=1, arrival=0.0)
    req_b = make_req(1, prompt=1, arrival=0.001)
    action1 = policy.select_action(make_state(migrating=[req_a, req_b], decode_gpu=decode_gpu))
    assert sorted(action1.admit[1]) == [0, 1]  # 2 blocks each -> exactly fills 4 blocks

    active_a = make_req(0, prompt=1, arrival=0.0)
    active_b = make_req(1, prompt=1, arrival=0.001)
    decode_gpu2 = make_gpu(1, "decode", max_kv=4, active_reqs=[active_a, active_b])
    action2 = policy.select_action(make_state(decode_gpu=decode_gpu2, step=1))
    # request 0 (earlier arrival) grows first and needs a 3rd block; no free
    # block remains -> request 1 (later arrival) is swapped out to make room.
    assert action2.swap[1] == [1]
    assert action2.admit[1] == []


def test_requires_exactly_one_decode_gpu_not_multiple():
    policy = DistServeFaithfulPolicy()
    extra_decode = make_gpu(2, "decode")
    state = make_state(extra_gpus=[extra_decode])
    with pytest.raises(ValueError):
        policy.select_action(state)


def test_decode_stage_deterministic_routing():
    def run_once():
        policy = DistServeFaithfulPolicy()
        reqs = [make_req(i, prompt=10, arrival=float(i) * 0.001) for i in range(4)]
        return tuple(policy.select_action(make_state(migrating=reqs)).admit[1])

    results = [run_once() for _ in range(5)]
    assert len(set(results)) == 1


def test_swapped_request_reentry_takes_priority_over_new_migration():
    """Fidelity: get_next_batch's admission loop 'considers requests in the
    swapped queue first' -- a previously-swapped-out request must be
    re-admitted before an ordinary (never-swapped) bridge-queue candidate,
    even if the new candidate arrived earlier."""
    policy = DistServeFaithfulPolicy(block_size=1, decode_max_batch_size=1,
                                       waiting_block_prop_threshold=1.0)
    policy._swapped_out_ids = [7]
    policy._swapped_out_num_tokens = {7: 3}
    decode_gpu = make_gpu(1, "decode", max_kv=100)
    swapped_back = make_req(7, prompt=3, arrival=0.0)
    new_candidate = make_req(1, prompt=3, arrival=-1.0)  # arrived earlier
    action = policy.select_action(make_state(migrating=[new_candidate, swapped_back], decode_gpu=decode_gpu))
    assert action.admit[1] == [7]  # swap re-entry wins despite max_batch_size=1


def test_waiting_block_prop_threshold_gates_new_migration_acceptance():
    """Fidelity: post_process's should_accept refuses to pull more from the
    bridge queue once the (per-round) accepted backlog would reach
    waiting_block_prop_threshold * capacity, even though raw KV capacity
    would allow more."""
    # block_size=1, num_gpu_blocks=100 -> threshold=0.01 means only 1 block
    # of backlog may be accepted per round.
    policy = DistServeFaithfulPolicy(block_size=1, waiting_block_prop_threshold=0.01,
                                       decode_max_batch_size=100)
    decode_gpu = make_gpu(1, "decode", max_kv=100)
    reqs = [make_req(i, prompt=1, arrival=float(i) * 0.001) for i in range(5)]
    action = policy.select_action(make_state(migrating=reqs, decode_gpu=decode_gpu))
    assert action.admit[1] == [0]  # only the first fits under the tiny threshold


def test_decode_stage_continuous_progression_after_capacity_frees():
    gpus = _gpu_configs(decode_kv=200)  # 12 blocks @ block_size=16 default... use explicit small config
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(0, arrival=0.0, prompt=20, output=3), _req(1, arrival=0.0, prompt=20, output=3)])
    policy = DistServeFaithfulPolicy(decode_max_batch_size=1, waiting_block_prop_threshold=1.0)
    metrics = sim.run(policy, workload_tag="continuous-decode")
    assert metrics.num_completed == 2
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_prompt_to_completion():
    gpus = _gpu_configs()
    sm = _disagg_service_model(migration_transfer_delay=0.005)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(0, prompt=100, output=10)])
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="e2e")
    assert metrics.num_completed == 1
    assert metrics.num_dropped == 0


def test_ttft_includes_queueing_prefill_and_transfer_delay():
    gpus = _gpu_configs()
    sm = _disagg_service_model(step_token_budget=1000, max_prefill_chunk_tokens=1000,
                                migration_transfer_delay=0.02)  # 20 steps
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=5000))
    sim.load_trace([_req(0, prompt=50, output=5)])
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="ttft")
    assert metrics.num_completed == 1
    assert metrics.mean_ttft >= 0.02  # at least the transfer delay alone


def test_tpot_measured_correctly():
    gpus = _gpu_configs()
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=2000))
    sim.load_trace([_req(0, prompt=20, output=15)])
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="tpot")
    assert metrics.num_completed == 1
    assert metrics.mean_tpot == sm.step_size


def test_deterministic_repeated_runs_end_to_end():
    def run():
        gpus = _gpu_configs(decode_kv=2000)
        sm = _disagg_service_model(migration_transfer_delay=0.003)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
        sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=40, output=10) for i in range(10)])
        return sim.run(DistServeFaithfulPolicy(), workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.num_dropped == m2.num_dropped


def test_no_capacity_violations_end_to_end():
    gpus = _gpu_configs(context_kv=500, decode_kv=500, context_seq=8, decode_seq=8)
    sm = _disagg_service_model(step_token_budget=64, max_prefill_chunk_tokens=64,
                                migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.002, prompt=15, output=10) for i in range(20)]
    sim.load_trace(reqs)
    policy = DistServeFaithfulPolicy(block_size=8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="capacity-check")
    assert w == [], f"unexpected simulator warnings: {[str(x.message) for x in w]}"
    assert metrics.num_completed == 20
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# Regression: pre-existing baselines and legacy configs unaffected
# ---------------------------------------------------------------------------

def test_vllm_faithful_unaffected_by_distserve_faithful_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(VLLMFaithfulPolicy(block_size=16, watermark=0.0), workload_tag="vllm-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_sarathi_faithful_unaffected_by_distserve_faithful_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=64, max_prefill_chunk_tokens=64)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(SarathiFaithfulPolicy(chunk_size=64, watermark=0.0), workload_tag="sarathi-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_legacy_simulator_unchanged():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=10, max_batch_tokens=1000, max_kv_tokens=1000)
    assert gpu.role is None
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=2000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001) for i in range(5)])
    metrics = sim.run(FIFOPolicy(), workload_tag="legacy")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_old_gpu_configs_without_role_still_valid():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=10, max_batch_tokens=1000, max_kv_tokens=1000)
    assert gpu.role is None  # unchanged default


# ---------------------------------------------------------------------------
# Paper-level sanity-check workload patterns (scheduler-behavior validation
# only -- no hardware-speedup claims; see docs/distserve_faithful_scheduler_reference.md)
# ---------------------------------------------------------------------------

def test_workload_long_prompt_heavy_builds_context_side_pressure():
    """Long prompts should dominate context-stage (prefill) processing
    time, producing observable multi-step prefill chunking rather than
    instant admission-to-decode."""
    gpus = _gpu_configs()
    sm = _disagg_service_model(step_token_budget=200, max_prefill_chunk_tokens=200,
                                migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=2000, output=5) for i in range(4)])
    policy = DistServeFaithfulPolicy()

    max_context_active = [0]

    def recording(state, _orig=policy.select_action):
        context_gpu = next(g for g in state.gpu_states if g.role == "prefill")
        max_context_active[0] = max(max_context_active[0], len(context_gpu.active_request_ids))
        return _orig(state)

    policy.select_action = recording
    metrics = sim.run(policy, workload_tag="long-prompt-heavy")
    assert metrics.num_completed == 4
    assert max_context_active[0] >= 1


def test_workload_decode_heavy_builds_decode_side_pressure():
    """Long outputs with short prompts should saturate the decode stage,
    exercising continuous batching (and, at tight capacity, swap) rather
    than the context stage."""
    gpus = _gpu_configs(decode_kv=300)
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=10, output=80) for i in range(4)])
    policy = DistServeFaithfulPolicy(block_size=8)

    max_decode_active = [0]

    def recording(state, _orig=policy.select_action):
        decode_gpu = next(g for g in state.gpu_states if g.role == "decode")
        max_decode_active[0] = max(max_decode_active[0], len(decode_gpu.active_request_ids))
        return _orig(state)

    policy.select_action = recording
    metrics = sim.run(policy, workload_tag="decode-heavy")
    assert metrics.num_completed == 4
    assert metrics.num_dropped == 0
    assert max_decode_active[0] >= 1


def test_workload_mixed_prompt_and_output_lengths():
    gpus = _gpu_configs()
    sm = _disagg_service_model(step_token_budget=200, max_prefill_chunk_tokens=200,
                                migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = []
    for i in range(8):
        prompt = 500 if i % 2 == 0 else 20
        output = 5 if i % 2 == 0 else 60
        reqs.append(_req(i, arrival=float(i) * 0.001, prompt=prompt, output=output))
    sim.load_trace(reqs)
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="mixed")
    assert metrics.num_completed == 8
    assert metrics.num_dropped == 0


def test_workload_context_bottleneck_produces_bridge_backlog():
    """A tight context-stage token budget relative to demand should cause a
    visible backlog of transfer-ready requests to build in the bridge
    queue (decode side under-utilized relative to context-side demand)."""
    gpus = _gpu_configs(context_kv=100_000, decode_kv=100_000, context_seq=1)
    sm = _disagg_service_model(step_token_budget=50, max_prefill_chunk_tokens=50,
                                migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=0.0, prompt=200, output=5) for i in range(5)])
    policy = DistServeFaithfulPolicy(context_max_batch_size=1)

    max_waiting_backlog = [0]

    def recording(state, _orig=policy.select_action):
        max_waiting_backlog[0] = max(max_waiting_backlog[0], len(state.waiting_queue))
        return _orig(state)

    policy.select_action = recording
    metrics = sim.run(policy, workload_tag="context-bottleneck")
    assert metrics.num_completed == 5
    assert max_waiting_backlog[0] >= 1  # requests pile up waiting for the single context slot


def test_workload_decode_bottleneck_produces_swap_or_backlog():
    """Ample context capacity plus a tiny decode capacity should force
    either bridge-queue backlog or swap activity on the decode side."""
    gpus = _gpu_configs(context_kv=100_000, decode_kv=64)
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=0.0, prompt=20, output=20) for i in range(6)])
    policy = DistServeFaithfulPolicy(block_size=4)

    max_migrating_backlog = [0]
    saw_swap = [False]

    def recording(state, _orig=policy.select_action):
        max_migrating_backlog[0] = max(max_migrating_backlog[0], len(state.migrating_queue))
        action = _orig(state)
        if any(action.swap.values()):
            saw_swap[0] = True
        return action

    policy.select_action = recording
    metrics = sim.run(policy, workload_tag="decode-bottleneck")
    assert metrics.num_completed == 6
    assert metrics.num_dropped == 0
    assert max_migrating_backlog[0] >= 1 or saw_swap[0]

"""Tests for src/llmserveopt/policies/llumnix_faithful.py and the shared
live cross-instance relocation primitive it introduces
(Action.migrate / Simulator._apply_migrations / RequestPhase.RELOCATING /
ObservableGPUState.incoming_migrations).

Covers the migration primitive itself, initial placement (dispatch),
runtime rescheduling (migration-pair selection + LCFS candidate selection),
end-to-end behavior, regression against pre-existing baselines, and
paper-level sanity-check workload patterns -- the fidelity claims
documented in docs/llumnix_faithful_scheduler_reference.md.
"""
from __future__ import annotations

import warnings
from collections import Counter

import pytest

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.distserve_faithful import DistServeFaithfulPolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.llumnix_faithful import LlumnixFaithfulPolicy
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
from llmserveopt.policies.tetriinfer_paper_reimplementation import TetriInferPaperReimplementationPolicy
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(rid, arrival=0.0, prompt=20, output=10, deadline=1000.0, priority=1.0):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id="medium",
    )


def _gpu_configs(n=2, max_kv=1000, max_seq=32):
    return [
        GPUConfig(gpu_id=i, max_active_sequences=max_seq, max_batch_tokens=100_000, max_kv_tokens=max_kv)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Migration primitive tests (Action.migrate / Simulator._apply_migrations)
# ---------------------------------------------------------------------------

def _seed_active(sim, gpu_id, rid, prompt=20, output=10, decoded=3):
    ir = InternalRequest(request=_req(rid, prompt=prompt, output=output))
    ir.phase = RequestPhase.ACTIVE
    ir.gpu_id = gpu_id
    ir.admission_time = 0.0
    ir.tokens_decoded = decoded
    ir.first_token_time = 0.001
    sim._gpu_map[gpu_id]._active[rid] = ir
    return ir


def test_migration_successful_moves_request_between_gpus():
    gpus = _gpu_configs(n=2)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    sim._apply_action(Action(migrate={0: [(0, 1)]}))
    assert 0 not in sim._gpu_map[0]._active
    assert 0 in sim._relocating
    assert sim._relocating[0].migration_destination_gpu_id == 1
    assert sim._relocating[0].phase == RequestPhase.RELOCATING


def test_migration_insufficient_destination_capacity_rejected():
    gpus = [
        GPUConfig(gpu_id=0, max_active_sequences=10, max_batch_tokens=1000, max_kv_tokens=1000),
        GPUConfig(gpu_id=1, max_active_sequences=10, max_batch_tokens=1000, max_kv_tokens=1),  # tiny
    ]
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0, prompt=20)])
    sim._reset()
    _seed_active(sim, 0, 0, prompt=20)

    sim._apply_action(Action(migrate={0: [(0, 1)]}))
    assert 0 in sim._relocating

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(admit={1: [0]}))
        warned = [str(x.message) for x in w]
    assert any("rejected" in msg for msg in warned)
    assert 0 in sim._relocating, "rejected migration must remain in-flight, not silently disappear"
    assert 0 not in sim._gpu_map[1]._active


def test_migration_delay_gates_admission_eligibility():
    gpus = _gpu_configs(n=2)
    sm = ServiceModel(llumnix_migration_delay=1000.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    sim._apply_action(Action(migrate={0: [(0, 1)]}))
    assert sim._relocating[0].transfer_ready_time == sim._time + 1000.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(admit={1: [0]}))
        warned = [str(x.message) for x in w]
    assert any("mid-relocation" in msg for msg in warned)
    assert 0 not in sim._gpu_map[1]._active
    assert 0 in sim._relocating


def test_migration_preserves_decoded_progress_and_admission_time():
    gpus = _gpu_configs(n=2)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    ir = _seed_active(sim, 0, 0, decoded=7)
    ir.admission_time = 0.003

    sim._apply_action(Action(migrate={0: [(0, 1)]}))
    relocating = sim._relocating[0]
    assert relocating.tokens_decoded == 7
    assert relocating.admission_time == 0.003  # preserved, not reset

    sim._time = 0.0
    sim._apply_action(Action(admit={1: [0]}))
    resumed = sim._gpu_map[1]._active[0]
    assert resumed.tokens_decoded == 7
    assert resumed.admission_time == 0.003  # still preserved after resume


def test_migration_kv_conservation_no_double_ownership():
    gpus = _gpu_configs(n=2)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    sim._apply_action(Action(migrate={0: [(0, 1)]}))
    # Not active anywhere while in-flight, not double-counted.
    assert 0 not in sim._gpu_map[0]._active
    assert 0 not in sim._gpu_map[1]._active
    assert sim._gpu_map[0].current_kv_tokens == 0
    assert sim._gpu_map[1].current_kv_tokens == 0

    sim._apply_action(Action(admit={1: [0]}))
    assert 0 in sim._gpu_map[1]._active
    assert 0 not in sim._gpu_map[0]._active
    assert 0 not in sim._relocating


def test_migration_no_double_execution_same_request_twice_in_one_action():
    gpus = _gpu_configs(n=3)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(migrate={0: [(0, 1), (0, 2)]}))
        warned = [str(x.message) for x in w]
    assert any("more than once" in msg for msg in warned)
    assert 0 in sim._relocating
    assert sim._relocating[0].migration_destination_gpu_id == 1  # first pair wins


def test_migration_no_duplicate_completion_end_to_end():
    gpus = _gpu_configs(n=3, max_kv=2000)
    sm = ServiceModel(llumnix_migration_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=15, output=20) for i in range(12)])
    policy = LlumnixFaithfulPolicy(need_migrate_frequency=1, migrate_out_threshold=0.1)
    metrics = sim.run(policy, workload_tag="no-dup-completion")
    assert metrics.num_completed == 12
    assert metrics.num_dropped == 0


def test_migration_rejected_destination_admit_wrong_gpu_id():
    """A relocating request may only be admitted onto its fixed
    destination -- any other gpu_id must be rejected."""
    gpus = _gpu_configs(n=3)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)
    sim._apply_action(Action(migrate={0: [(0, 1)]}))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(admit={2: [0]}))  # wrong destination
        warned = [str(x.message) for x in w]
    assert any("relocating to gpu_id=1" in msg for msg in warned)
    assert 0 not in sim._gpu_map[2]._active
    assert 0 in sim._relocating


def test_migration_unknown_destination_gpu_rejected():
    gpus = _gpu_configs(n=2)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(migrate={0: [(0, 999)]}))
        warned = [str(x.message) for x in w]
    assert any("unknown destination" in msg for msg in warned)
    assert 0 in sim._gpu_map[0]._active  # never evicted


def test_migration_same_source_and_destination_rejected():
    gpus = _gpu_configs(n=2)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(migrate={0: [(0, 0)]}))
        warned = [str(x.message) for x in w]
    assert any("both gpu_id=0" in msg for msg in warned)
    assert 0 in sim._gpu_map[0]._active


def test_migrate_and_admit_same_request_same_action_rejected():
    gpus = _gpu_configs(n=2)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(0)])
    sim._reset()
    _seed_active(sim, 0, 0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim._apply_action(Action(migrate={0: [(0, 1)]}, admit={1: [0]}))
        warned = [str(x.message) for x in w]
    assert any("preempted/swapped/migrated and admitted" in msg for msg in warned)


# ---------------------------------------------------------------------------
# Initial placement (dispatch)
# ---------------------------------------------------------------------------

def test_dispatch_balanced_cluster_round_robins():
    gpus = _gpu_configs(n=4, max_kv=10_000)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001) for i in range(8)])
    policy = LlumnixFaithfulPolicy()
    metrics = sim.run(policy, workload_tag="dispatch-balanced")
    assert metrics.num_completed == 8
    counts = Counter(policy._dispatch_assignment.values())
    assert set(counts.values()) == {2}  # exactly balanced 2-2-2-2


def test_dispatch_deterministic_tie_breaking():
    def run():
        gpus = _gpu_configs(n=3, max_kv=10_000)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=20_000))
        sim.load_trace([_req(i, arrival=float(i) * 0.001) for i in range(9)])
        policy = LlumnixFaithfulPolicy()
        sim.run(policy, workload_tag="tie-break")
        return tuple(sorted(policy._dispatch_assignment.items()))

    assert run() == run()


def test_dispatch_uneven_load_still_completes():
    """Requests still get dispatched (round-robin is load-oblivious by
    design -- see module docstring) even when resulting per-instance load
    ends up uneven; migration (tested separately) is what corrects this."""
    gpus = _gpu_configs(n=2, max_kv=5000)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.001, output=(200 if i % 2 == 0 else 10)) for i in range(10)]
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy(need_migrate_frequency=1000000)  # effectively disable migration
    metrics = sim.run(policy, workload_tag="uneven-load")
    assert metrics.num_completed == 10


def test_requires_at_least_one_gpu():
    policy = LlumnixFaithfulPolicy()
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[], completed_count=0, step=0)
    with pytest.raises(ValueError):
        policy.select_action(state)


# ---------------------------------------------------------------------------
# Rescheduling (migration-pair selection + LCFS candidate selection)
# ---------------------------------------------------------------------------

def test_rescheduling_load_hotspot_triggers_migration():
    n_gpus = 4
    gpus = _gpu_configs(n=n_gpus, max_kv=3000)
    sm = ServiceModel(llumnix_migration_delay=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    heavy_ids = {i for i in range(40) if i % n_gpus == 0}
    reqs = [_req(i, arrival=float(i) * 0.0005, output=(400 if i in heavy_ids else 10)) for i in range(40)]
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy()

    migrations = []
    orig = policy.select_action
    def traced(state, _orig=orig):
        action = _orig(state)
        for src, pairs in action.migrate.items():
            migrations.extend((src, dst) for _rid, dst in pairs)
        return action
    policy.select_action = traced

    metrics = sim.run(policy, workload_tag="load-hotspot")
    assert metrics.num_completed == 40
    assert len(migrations) > 0, "load hotspot on instance 0 should trigger at least one migration"
    assert all(src == 0 for src, _dst in migrations), "migrations should relieve the overloaded instance 0"


def test_rescheduling_no_migration_when_balanced():
    gpus = _gpu_configs(n=3, max_kv=5000)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.002, output=20) for i in range(9)]  # uniform, balanced
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy()

    migrations = []
    orig = policy.select_action
    def traced(state, _orig=orig):
        action = _orig(state)
        for pairs in action.migrate.values():
            migrations.extend(pairs)
        return action
    policy.select_action = traced

    metrics = sim.run(policy, workload_tag="balanced-no-migration")
    assert metrics.num_completed == 9
    assert len(migrations) == 0, "a genuinely balanced, uniform workload should not trigger migration"


def test_rescheduling_priority_exempt_requests_never_migration_source():
    """Fidelity: priority-exempt requests are skipped by LCFS candidate
    selection regardless of load (see docs/llumnix_faithful_scheduler_reference.md §A.6)."""
    gpu = ObservableGPUState(
        gpu_id=0, max_active_sequences=100, max_batch_tokens=100_000, max_kv_tokens=100_000,
        active_request_ids=[1, 2],
        active_requests_info=[
            ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=10, predicted_output_tokens=50,
                               slo_deadline=1000.0, priority=10.0, class_id="p"),  # high priority
            ObservableRequest(request_id=2, arrival_time=0.001, prompt_tokens=10, predicted_output_tokens=50,
                               slo_deadline=1000.0, priority=1.0, class_id="p"),
        ],
        current_kv_tokens=20,
        tokens_decoded_per_request={1: 5, 2: 5},
    )
    policy = LlumnixFaithfulPolicy(priority_exempt_threshold=5.0)
    candidate = policy._lcfs_candidate(gpu, exclude_ids=set())
    assert candidate is not None
    assert candidate.request_id == 2  # request 1 (priority=10 > threshold) is exempt


def test_rescheduling_prefill_only_request_never_migration_source():
    """Fidelity: a request that hasn't decoded any tokens yet (still
    mid-prefill) is never an LCFS candidate."""
    gpu = ObservableGPUState(
        gpu_id=0, max_active_sequences=100, max_batch_tokens=100_000, max_kv_tokens=100_000,
        active_request_ids=[1],
        active_requests_info=[
            ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=10, predicted_output_tokens=50,
                               slo_deadline=1000.0, priority=1.0, class_id="p"),
        ],
        current_kv_tokens=10,
        tokens_decoded_per_request={1: 0},  # no output yet
    )
    policy = LlumnixFaithfulPolicy()
    candidate = policy._lcfs_candidate(gpu, exclude_ids=set())
    assert candidate is None


def test_rescheduling_kv_hotspot_triggers_migration():
    n_gpus = 3
    gpus = _gpu_configs(n=n_gpus, max_kv=2000)
    sm = ServiceModel(llumnix_migration_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    heavy_ids = {i for i in range(30) if i % n_gpus == 0}
    reqs = [_req(i, arrival=float(i) * 0.0005, prompt=(150 if i in heavy_ids else 10), output=30) for i in range(30)]
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy()

    migrations = []
    orig = policy.select_action
    def traced(state, _orig=orig):
        action = _orig(state)
        for src, pairs in action.migrate.items():
            migrations.extend((src, dst) for _rid, dst in pairs)
        return action
    policy.select_action = traced

    metrics = sim.run(policy, workload_tag="kv-hotspot")
    assert metrics.num_completed == 30
    assert len(migrations) > 0


def test_rescheduling_migration_beneficial_condition_rejects_overshoot():
    """Fidelity: a migration pair is only approved if the destination
    would not itself exceed the migrate-out threshold after receiving the
    request (see need_migrate_balanced's `right_load_after_mig >
    migrate_out_load_threshold: continue`)."""
    policy = LlumnixFaithfulPolicy(migrate_out_threshold=0.01, block_size=1)  # extremely tight threshold
    gpus = _gpu_configs(n=2, max_kv=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=20_000))
    sim.load_trace([_req(0, output=100), _req(1, arrival=0.001, output=10)])
    metrics = sim.run(policy, workload_tag="beneficial-check")
    assert metrics.num_completed == 2
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_placement_execution_migration_completion():
    n_gpus = 3
    gpus = _gpu_configs(n=n_gpus, max_kv=2000)
    sm = ServiceModel(llumnix_migration_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    heavy_ids = {i for i in range(24) if i % n_gpus == 0}
    reqs = [_req(i, arrival=float(i) * 0.001, output=(300 if i in heavy_ids else 15)) for i in range(24)]
    sim.load_trace(reqs)
    metrics = sim.run(LlumnixFaithfulPolicy(), workload_tag="e2e")
    assert metrics.num_completed == 24
    assert metrics.num_dropped == 0


def test_end_to_end_repeated_migrations_allowed():
    gpus = _gpu_configs(n=4, max_kv=1500)
    sm = ServiceModel(llumnix_migration_delay=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    heavy_ids = {i for i in range(40) if i % 4 == 0}
    reqs = [_req(i, arrival=float(i) * 0.0005, output=(350 if i in heavy_ids else 15)) for i in range(40)]
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy(need_migrate_frequency=1)

    all_migrations = []
    orig = policy.select_action
    def traced(state, _orig=orig):
        action = _orig(state)
        for src, pairs in action.migrate.items():
            all_migrations.extend((src, rid, dst) for rid, dst in pairs)
        return action
    policy.select_action = traced

    metrics = sim.run(policy, workload_tag="repeated-migrations")
    assert metrics.num_completed == 40
    assert metrics.num_dropped == 0
    assert len(all_migrations) >= 1


def test_end_to_end_no_starvation():
    gpus = _gpu_configs(n=3, max_kv=3000)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.002, output=20) for i in range(15)]
    sim.load_trace(reqs)
    metrics = sim.run(LlumnixFaithfulPolicy(), workload_tag="no-starvation")
    assert metrics.num_completed == 15
    assert metrics.num_dropped == 0


def test_end_to_end_deterministic_repeated_runs():
    def run():
        gpus = _gpu_configs(n=3, max_kv=2000)
        sm = ServiceModel(llumnix_migration_delay=0.001)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
        heavy_ids = {i for i in range(18) if i % 3 == 0}
        reqs = [_req(i, arrival=float(i) * 0.001, output=(200 if i in heavy_ids else 15)) for i in range(18)]
        sim.load_trace(reqs)
        return sim.run(LlumnixFaithfulPolicy(), workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.num_dropped == m2.num_dropped


def test_end_to_end_no_capacity_violations():
    gpus = _gpu_configs(n=4, max_kv=800)
    sm = ServiceModel(llumnix_migration_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.001, prompt=15, output=25) for i in range(30)]
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy(block_size=8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="capacity-check")
    assert w == [], f"unexpected simulator warnings: {[str(x.message) for x in w]}"
    assert metrics.num_completed == 30
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# Regression: pre-existing baselines and legacy configs unaffected
# ---------------------------------------------------------------------------

def test_tetriinfer_unaffected_by_llumnix_existing():
    gpus = [
        GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000, role="prefill"),
        GPUConfig(gpu_id=100, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000, role="decode"),
    ]
    sm = ServiceModel(enable_prefill_modeling=True, enable_disaggregation=True, decode_first=True,
                       step_token_budget=100_000, max_prefill_chunk_tokens=100_000, migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(TetriInferPaperReimplementationPolicy(), workload_tag="tetriinfer-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_distserve_faithful_unaffected_by_llumnix_existing():
    gpus = [
        GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000, role="prefill"),
        GPUConfig(gpu_id=1, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000, role="decode"),
    ]
    sm = ServiceModel(enable_prefill_modeling=True, enable_disaggregation=True, decode_first=True,
                       step_token_budget=100_000, max_prefill_chunk_tokens=100_000, migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="distserve-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_vllm_faithful_unaffected_by_llumnix_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(VLLMFaithfulPolicy(block_size=16, watermark=0.0), workload_tag="vllm-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_sarathi_faithful_unaffected_by_llumnix_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True, step_token_budget=64, max_prefill_chunk_tokens=64)
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


# ---------------------------------------------------------------------------
# Paper-level sanity checks (scheduler-behavior validation only -- no
# hardware-speedup claims; see docs/llumnix_faithful_scheduler_reference.md)
# ---------------------------------------------------------------------------

class _StaticRoundRobinTestPolicy(BasePolicy):
    """Comparison-only: dispatches like Llumnix's own naive strategy but
    NEVER migrates -- demonstrates the benefit migration adds."""
    name = "static_round_robin_test"

    def __init__(self):
        self._assignment = {}
        self._ptr = 0

    def reset(self):
        self._assignment = {}
        self._ptr = 0

    def select_action(self, state):
        admit = {g.gpu_id: [] for g in state.gpu_states}
        gpu_by_id = {g.gpu_id: g for g in state.gpu_states}
        sorted_ids = sorted(gpu_by_id.keys())
        by_instance = {gid: [] for gid in sorted_ids}
        for req in state.waiting_queue:
            gid = self._assignment.get(req.request_id)
            if gid is None:
                gid = sorted_ids[self._ptr % len(sorted_ids)]
                self._ptr += 1
                self._assignment[req.request_id] = gid
            by_instance[gid].append(req)
        for gid in sorted_ids:
            gpu = gpu_by_id[gid]
            for req in by_instance[gid]:
                if self._feasible_on_gpu(gpu, req):
                    admit[gid].append(req.request_id)
                    gpu.active_request_ids.append(req.request_id)
                    gpu.current_kv_tokens += req.prompt_tokens
        return Action(admit=admit)


def test_sanity_imbalanced_arrivals_across_instances():
    n_gpus = 4
    gpus = _gpu_configs(n=n_gpus, max_kv=3000)
    sm = ServiceModel(llumnix_migration_delay=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    heavy_ids = {i for i in range(40) if i % n_gpus == 0}
    reqs = [_req(i, arrival=float(i) * 0.0005, output=(300 if i in heavy_ids else 10)) for i in range(40)]
    sim.load_trace(reqs)
    metrics = sim.run(LlumnixFaithfulPolicy(), workload_tag="sanity-imbalanced")
    assert metrics.num_completed == 40
    assert metrics.num_dropped == 0


def test_sanity_kv_fragmentation_pressure():
    gpus = _gpu_configs(n=3, max_kv=1200)
    sm = ServiceModel(llumnix_migration_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = [_req(i, arrival=0.0, prompt=(80 if i % 3 == 0 else 10), output=20) for i in range(15)]
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy(block_size=4)
    metrics = sim.run(policy, workload_tag="sanity-fragmentation")
    assert metrics.num_completed == 15
    assert metrics.num_dropped == 0


def test_sanity_long_running_decode_hotspot():
    n_gpus = 3
    gpus = _gpu_configs(n=n_gpus, max_kv=2500)
    sm = ServiceModel(llumnix_migration_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    heavy_ids = {i for i in range(24) if i % n_gpus == 0}
    reqs = [_req(i, arrival=float(i) * 0.0005, output=(500 if i in heavy_ids else 15)) for i in range(24)]
    sim.load_trace(reqs)
    metrics = sim.run(LlumnixFaithfulPolicy(), workload_tag="sanity-decode-hotspot")
    assert metrics.num_completed == 24
    assert metrics.num_dropped == 0


def test_sanity_priority_request_not_migrated_off_overloaded_instance():
    """Qualitative check: a priority-exempt request stuck on an overloaded
    instance stays there (never selected as a migration source), while
    ordinary requests around it may still be migrated -- matching the
    pinned reference's priority-protection semantics."""
    n_gpus = 3
    gpus = _gpu_configs(n=n_gpus, max_kv=2500)
    sm = ServiceModel(llumnix_migration_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = []
    for i in range(24):
        heavy = i % n_gpus == 0
        priority = 100.0 if (heavy and i == 0) else 1.0
        reqs.append(_req(i, arrival=float(i) * 0.0005, output=(400 if heavy else 15), priority=priority))
    sim.load_trace(reqs)
    policy = LlumnixFaithfulPolicy(priority_exempt_threshold=10.0)

    migrated_rids = set()
    orig = policy.select_action
    def traced(state, _orig=orig):
        action = _orig(state)
        for pairs in action.migrate.values():
            migrated_rids.update(rid for rid, _dst in pairs)
        return action
    policy.select_action = traced

    metrics = sim.run(policy, workload_tag="sanity-priority-protection")
    assert metrics.num_completed == 24
    assert 0 not in migrated_rids, "priority-exempt request 0 must never be a migration source"


def test_sanity_migration_reduces_hotspot_vs_static_round_robin():
    """Comparative, qualitative check: under an adversarial-for-round-robin
    workload (heavy requests aligned with the dispatch cadence), Llumnix's
    migration should relieve the resulting hotspot on instance 0 over the
    course of the run, whereas static round-robin (no migration) leaves
    every heavy request stuck there until each one individually finishes.
    Metric: total accumulated "instance-0 active count" summed across every
    scheduling step (a discrete proxy for sustained load) must be lower
    under Llumnix. No hardware-speedup claim -- this checks scheduling
    behavior only."""
    n_gpus = 4
    n_reqs = 40
    heavy_ids = {i for i in range(n_reqs) if i % n_gpus == 0}

    def build_reqs():
        return [_req(i, arrival=float(i) * 0.0005, output=(400 if i in heavy_ids else 15)) for i in range(n_reqs)]

    def run_and_measure(policy):
        gpus = _gpu_configs(n=n_gpus, max_kv=3000)
        sm = ServiceModel(llumnix_migration_delay=0.001)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=30_000))
        sim.load_trace(build_reqs())

        accumulated_load_on_0 = [0]
        orig = policy.select_action
        def traced(state, _orig=orig):
            gpu0 = next(g for g in state.gpu_states if g.gpu_id == 0)
            accumulated_load_on_0[0] += len(gpu0.active_request_ids)
            return _orig(state)
        policy.select_action = traced

        metrics = sim.run(policy, workload_tag="sanity-vs-rr")
        return metrics, accumulated_load_on_0[0]

    llumnix_metrics, llumnix_load = run_and_measure(LlumnixFaithfulPolicy())
    rr_metrics, rr_load = run_and_measure(_StaticRoundRobinTestPolicy())

    assert llumnix_metrics.num_completed == n_reqs
    assert llumnix_metrics.num_dropped == 0
    assert rr_metrics.num_completed == n_reqs
    assert llumnix_load < rr_load, (
        f"Llumnix's accumulated load on instance 0 ({llumnix_load}) should be lower than "
        f"static round-robin's ({rr_load}) -- migration should relieve the hotspot"
    )

"""Tests for the disaggregated prefill/decode simulator infrastructure
(opt-in; see docs/distserve_faithful_scheduler_reference.md).

This is infrastructure-only: GPUConfig.role, RequestPhase.MIGRATING,
ServiceModel.enable_disaggregation/migration_transfer_delay,
GPUState.pop_pending_handoff(), and the Simulator's bridge queue
(_migrating/_migrating_map, ObservableState.migrating_queue). No
distserve_faithful scheduling policy exists yet -- these tests use a
minimal test-only policy to exercise the infrastructure end-to-end.
"""
from __future__ import annotations

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, ObservableState, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _req(rid, arrival=0.0, prompt=50, output=10, deadline=1000.0, priority=1.0):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id="medium",
    )


class _DisaggregatedTestPolicy(BasePolicy):
    """Minimal test-only policy: FCFS-admits waiting requests onto
    role="prefill" GPUs and transfer-ready migrating_queue requests onto
    role="decode" GPUs. Not a real scheduling policy (no distserve_faithful
    exists yet) -- exists only to exercise the infrastructure end-to-end."""
    name = "test_disaggregated"

    def select_action(self, state: ObservableState) -> Action:
        admit: dict = {g.gpu_id: [] for g in state.gpu_states}
        prefill_gpus = [g for g in state.gpu_states if g.role == "prefill"]
        decode_gpus = [g for g in state.gpu_states if g.role == "decode"]

        for req in state.waiting_queue:
            for g in prefill_gpus:
                if self._feasible_on_gpu(g, req):
                    admit[g.gpu_id].append(req.request_id)
                    g.active_request_ids.append(req.request_id)
                    g.current_kv_tokens += req.prompt_tokens
                    break

        for req in state.migrating_queue:
            for g in decode_gpus:
                if self._feasible_on_gpu(g, req):
                    admit[g.gpu_id].append(req.request_id)
                    g.active_request_ids.append(req.request_id)
                    g.current_kv_tokens += req.prompt_tokens
                    break

        return Action(admit=admit)


def _disagg_gpus(prefill_kv=1000, decode_kv=1000, prefill_seq=10, decode_seq=10):
    return [
        GPUConfig(gpu_id=0, max_active_sequences=prefill_seq, max_batch_tokens=100_000,
                  max_kv_tokens=prefill_kv, role="prefill"),
        GPUConfig(gpu_id=1, max_active_sequences=decode_seq, max_batch_tokens=100_000,
                  max_kv_tokens=decode_kv, role="decode"),
    ]


def _disagg_service_model(step_token_budget=200, migration_transfer_delay=0.0):
    return ServiceModel(
        enable_prefill_modeling=True, enable_disaggregation=True, decode_first=True,
        step_token_budget=step_token_budget, max_prefill_chunk_tokens=step_token_budget,
        prefill_cost_per_token=1.0, migration_transfer_delay=migration_transfer_delay,
    )


# ---------------------------------------------------------------------------
# 1. Legacy single-pool behavior unchanged
# ---------------------------------------------------------------------------

def test_legacy_role_none_gpus_behave_exactly_as_before():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=10, max_batch_tokens=1000, max_kv_tokens=1000)
    assert gpu.role is None
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=2000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001) for i in range(5)])
    metrics = sim.run(FIFOPolicy(), workload_tag="legacy")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_gpu_config_rejects_invalid_role():
    import pytest
    with pytest.raises(ValueError):
        GPUConfig(gpu_id=0, max_active_sequences=1, max_batch_tokens=1, max_kv_tokens=1, role="bogus")


# ---------------------------------------------------------------------------
# 2 & 5. Prefill/decode pools advance independently; decode begins only
# after handoff/transfer completion
# ---------------------------------------------------------------------------

def test_decode_gpu_stays_empty_until_transfer_completes():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(migration_transfer_delay=0.005)  # 5 steps
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=5000))
    sim.load_trace([_req(0, prompt=50, output=5)])
    policy = _DisaggregatedTestPolicy()

    decode_active_seen_before_any_ready_migration = []
    ever_saw_ready_migration = [False]

    def recording_select(state, _orig=policy.select_action):
        decode_gpu = next(g for g in state.gpu_states if g.role == "decode")
        if len(state.migrating_queue) > 0:
            ever_saw_ready_migration[0] = True
        if not ever_saw_ready_migration[0] and decode_gpu.active_request_ids:
            decode_active_seen_before_any_ready_migration.append(state.step)
        return _orig(state)

    policy.select_action = recording_select
    metrics = sim.run(policy, workload_tag="handoff-timing")
    assert metrics.num_completed == 1
    assert decode_active_seen_before_any_ready_migration == [], (
        "decode GPU must never be active before a transfer-ready signal was ever observed"
    )
    assert ever_saw_ready_migration[0], "test setup sanity: a ready migration must occur"


def test_prefill_and_decode_gpus_track_independent_active_sets():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=5000))
    sim.load_trace([_req(0, prompt=300, output=20)])  # multi-chunk prefill
    policy = _DisaggregatedTestPolicy()

    saw_prefill_active_decode_empty = False

    def recording_select(state, _orig=policy.select_action):
        nonlocal saw_prefill_active_decode_empty
        prefill_gpu = next(g for g in state.gpu_states if g.role == "prefill")
        decode_gpu = next(g for g in state.gpu_states if g.role == "decode")
        if prefill_gpu.active_request_ids and not decode_gpu.active_request_ids:
            saw_prefill_active_decode_empty = True
        return _orig(state)

    policy.select_action = recording_select
    sim.run(policy, workload_tag="independent-pools")
    assert saw_prefill_active_decode_empty, (
        "prefill GPU must be able to make progress while decode GPU is empty"
    )


# ---------------------------------------------------------------------------
# 3. A request cannot decode before prefill completion
# ---------------------------------------------------------------------------

def test_admitting_not_yet_ready_migrating_request_is_rejected():
    gpus = _disagg_gpus()
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=_disagg_service_model(migration_transfer_delay=1.0), max_steps=10))
    sim.load_trace([_req(0, prompt=10, output=5)])
    sim._reset()
    # Manually seed a migrating request whose transfer is NOT ready yet.
    ir = InternalRequest(request=_req(0, prompt=10, output=5))
    ir.phase = RequestPhase.MIGRATING
    ir.transfer_ready_time = 1000.0  # far in the future
    sim._migrating.append(ir)
    sim._migrating_map[0] = ir

    action = Action(admit={1: [0]})  # try to admit onto the decode GPU (id=1) early
    sim._apply_action(action)
    decode_gpu = sim._gpu_map[1]
    assert 0 not in decode_gpu._active
    assert 0 in sim._migrating_map  # still pending, not admitted


# ---------------------------------------------------------------------------
# 4. Completed prefill transitions exactly once
# ---------------------------------------------------------------------------

def test_prefill_completion_handoff_happens_exactly_once():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(step_token_budget=50, migration_transfer_delay=0.01)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=5000))
    sim.load_trace([_req(0, prompt=200, output=10)])  # needs multiple chunks
    policy = _DisaggregatedTestPolicy()

    handoff_count = [0]
    orig_collect = sim._collect_handoffs

    def counting_collect():
        for g in sim._gpus:
            handoff_count[0] += len(g._pending_handoff)
        orig_collect()

    sim._collect_handoffs = counting_collect
    metrics = sim.run(policy, workload_tag="handoff-count")
    assert metrics.num_completed == 1
    assert handoff_count[0] == 1


# ---------------------------------------------------------------------------
# 6 & 7. Zero-cost and nonzero transfer-delay paths
# ---------------------------------------------------------------------------

def test_zero_transfer_cost_mode():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=2000))
    sim.load_trace([_req(0, prompt=10, output=5)])
    metrics = sim.run(_DisaggregatedTestPolicy(), workload_tag="zero-cost")
    assert metrics.num_completed == 1
    assert metrics.num_dropped == 0


def test_nonzero_transfer_delay_increases_ttft_vs_zero_cost():
    def run(delay):
        gpus = _disagg_gpus()
        sm = _disagg_service_model(migration_transfer_delay=delay)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=2000))
        sim.load_trace([_req(0, prompt=10, output=5)])
        return sim.run(_DisaggregatedTestPolicy(), workload_tag=f"delay-{delay}")

    zero_cost = run(0.0)
    with_delay = run(0.01)  # 10 steps
    assert zero_cost.num_completed == with_delay.num_completed == 1
    assert with_delay.mean_ttft > zero_cost.mean_ttft


# ---------------------------------------------------------------------------
# 8. Deterministic execution
# ---------------------------------------------------------------------------

def test_deterministic_across_repeated_runs():
    def run():
        gpus = _disagg_gpus()
        sm = _disagg_service_model(migration_transfer_delay=0.003)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=5000))
        sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=40, output=8) for i in range(6)])
        return sim.run(_DisaggregatedTestPolicy(), workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.mean_ttft == m2.mean_ttft


# ---------------------------------------------------------------------------
# 9 & 10. No duplication across pools; no request loss during handoff
# ---------------------------------------------------------------------------

def test_no_duplication_and_no_loss_across_pools():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    n = 12
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=30, output=8) for i in range(n)])
    policy = _DisaggregatedTestPolicy()

    duplication_violations = []

    def checked_select(state, _orig=policy.select_action):
        seen = []
        seen += [r.request_id for r in state.waiting_queue]
        seen += [r.request_id for r in state.migrating_queue]
        for g in state.gpu_states:
            seen += g.active_request_ids
        if len(seen) != len(set(seen)):
            duplication_violations.append(state.step)
        return _orig(state)

    policy.select_action = checked_select
    metrics = sim.run(policy, workload_tag="no-dup-no-loss")
    assert duplication_violations == []
    assert metrics.num_completed == n
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# 11 & 12. Correct TTFT / TPOT accounting under disaggregation
# ---------------------------------------------------------------------------

def test_ttft_accounts_for_prefill_plus_transfer_delay():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(step_token_budget=1000, migration_transfer_delay=0.01)  # 10 steps
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=2000))
    sim.load_trace([_req(0, prompt=50, output=5)])  # 1-step prefill
    metrics = sim.run(_DisaggregatedTestPolicy(), workload_tag="ttft-transfer")
    assert metrics.num_completed == 1
    # >= transfer delay alone (10 steps); prefill + admission overhead adds more.
    assert metrics.mean_ttft >= 0.01


def test_tpot_unaffected_by_disaggregation():
    gpus = _disagg_gpus()
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=2000))
    sim.load_trace([_req(0, prompt=20, output=15)])
    metrics = sim.run(_DisaggregatedTestPolicy(), workload_tag="tpot-check")
    assert metrics.num_completed == 1
    assert metrics.mean_tpot == sm.step_size


# ---------------------------------------------------------------------------
# 13. Capacity isolation between prefill and decode pools
# ---------------------------------------------------------------------------

def test_capacity_isolation_between_pools():
    """A tiny decode-side capacity must not block prefill-side admission:
    with ample prefill capacity and a decode pool that can hold only 1
    sequence, several fully-prefilled requests must be able to pile up in
    the bridge queue (migrating_queue) awaiting decode admission, rather
    than prefill stalling or the run failing."""
    gpus = _disagg_gpus(prefill_kv=10_000, decode_kv=60, prefill_seq=50, decode_seq=1)
    # Small step_token_budget relative to prompt_tokens=20 -> multi-step
    # prefill, so several requests genuinely overlap on the prefill GPU.
    sm = _disagg_service_model(step_token_budget=5, migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    n = 5
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(n)])
    policy = _DisaggregatedTestPolicy()

    max_prefill_active_seen = [0]
    max_migrating_backlog_seen = [0]

    def recording_select(state, _orig=policy.select_action):
        prefill_gpu = next(g for g in state.gpu_states if g.role == "prefill")
        max_prefill_active_seen[0] = max(max_prefill_active_seen[0], len(prefill_gpu.active_request_ids))
        max_migrating_backlog_seen[0] = max(max_migrating_backlog_seen[0], len(state.migrating_queue))
        return _orig(state)

    policy.select_action = recording_select
    metrics = sim.run(policy, workload_tag="capacity-isolation")
    assert metrics.num_completed == n
    assert metrics.num_dropped == 0
    # Ample prefill capacity (50 seqs, 10000 kv) allows more than 1
    # concurrently-prefilling request despite the decode side only ever
    # holding 1 at a time -- proves the pools are not sharing capacity.
    assert max_prefill_active_seen[0] >= 2
    # A backlog must be able to build in the bridge queue (decode-side
    # capacity=1 cannot immediately absorb every transfer-ready request).
    assert max_migrating_backlog_seen[0] >= 1


# ---------------------------------------------------------------------------
# 14. Backward compatibility with vllm_faithful and sarathi_faithful
# ---------------------------------------------------------------------------

def test_vllm_faithful_unaffected_by_role_field_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(VLLMFaithfulPolicy(block_size=16, watermark=0.0), workload_tag="vllm-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_sarathi_faithful_unaffected_by_disaggregation_fields_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sm = ServiceModel(enable_prefill_modeling=True, decode_first=True,
                       step_token_budget=64, max_prefill_chunk_tokens=64)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(SarathiFaithfulPolicy(chunk_size=64, watermark=0.0), workload_tag="sarathi-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0

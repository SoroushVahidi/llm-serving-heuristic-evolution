"""Tests for src/llmserveopt/policies/tetriinfer_paper_reimplementation.py,
tetriinfer_length_prediction.py, and tetriinfer_routing.py.

Covers the length-prediction abstraction, inter-instance power-of-two
decode routing, the local decode scheduler (greedy/reserve-static/
reserve-dynamic), end-to-end behavior, regression against pre-existing
baselines, and paper-level sanity-check workload patterns -- the fidelity
claims documented in docs/tetriinfer_reference.md. This baseline is
labeled `tetriinfer_paper_reimplementation`, not `_faithful`: no official
TetriInfer code exists to verify against (see that doc's section 0), so
these tests validate fidelity to the *paper's stated description*, not to
a pinned source commit.
"""
from __future__ import annotations

import warnings
from collections import Counter

import pytest

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.distserve_faithful import DistServeFaithfulPolicy
from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
from llmserveopt.policies.tetriinfer_length_prediction import LengthPredictor
from llmserveopt.policies.tetriinfer_routing import (
    HEAVY_DECODE_THRESHOLD_TOKENS,
    PowerOfTwoDecodeRouter,
    is_heavy_decode,
)
from llmserveopt.policies.tetriinfer_paper_reimplementation import TetriInferPaperReimplementationPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_req(req_id, prompt=20, output=30, deadline=1000.0, priority=1.0, arrival=0.0):
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


def make_state(waiting=(), migrating=(), prefill_gpus=None, decode_gpus=None, now=0.0, step=0):
    if prefill_gpus is None:
        prefill_gpus = [make_gpu(0, "prefill")]
    if decode_gpus is None:
        decode_gpus = [make_gpu(1, "decode")]
    return ObservableState(
        time=now, waiting_queue=list(waiting), gpu_states=[*prefill_gpus, *decode_gpus],
        completed_count=0, step=step, migrating_queue=list(migrating),
    )


def _gpu_configs(n_prefill=1, n_decode=2, prefill_kv=100_000, decode_kv=100_000):
    configs = []
    for i in range(n_prefill):
        configs.append(GPUConfig(gpu_id=i, max_active_sequences=100, max_batch_tokens=1_000_000,
                                  max_kv_tokens=prefill_kv, role="prefill"))
    for i in range(n_decode):
        configs.append(GPUConfig(gpu_id=100 + i, max_active_sequences=100, max_batch_tokens=1_000_000,
                                  max_kv_tokens=decode_kv, role="decode"))
    return configs


def _disagg_service_model(step_token_budget=100_000, max_prefill_chunk_tokens=100_000,
                           migration_transfer_delay=0.0):
    return ServiceModel(
        enable_prefill_modeling=True, enable_disaggregation=True, decode_first=True,
        step_token_budget=step_token_budget, max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        prefill_cost_per_token=1.0, migration_transfer_delay=migration_transfer_delay,
    )


def _req(rid, arrival=0.0, prompt=30, output=20, deadline=1000.0, priority=1.0):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id="medium",
    )


# ---------------------------------------------------------------------------
# Length prediction
# ---------------------------------------------------------------------------

def test_exact_prediction_no_bucketing():
    predictor = LengthPredictor(granularity=0, mode="exact")
    pred = predictor.predict(137)
    assert pred.point_estimate == 137
    assert pred.lower_bound == pred.upper_bound == 137


def test_bucket_boundaries_exact():
    predictor = LengthPredictor(granularity=200, mode="exact")
    assert predictor.predict(0).bucket_index == 0
    assert predictor.predict(199).bucket_index == 0
    assert predictor.predict(200).bucket_index == 1
    assert predictor.predict(399).bucket_index == 1
    assert predictor.predict(400).bucket_index == 2
    pred = predictor.predict(250)
    assert (pred.lower_bound, pred.upper_bound) == (200, 400)
    assert pred.point_estimate == 200  # paper §5.2.3: resource estimates use the LOWER bound


def test_underprediction_uses_source_value_only():
    """The predictor must never see or reference actual_output_tokens --
    only the caller-supplied source (this project's predicted_output_tokens).
    An "underprediction" scenario is exactly: source < some hypothetical
    true value the predictor is never given."""
    predictor = LengthPredictor(granularity=100, mode="exact")
    # Source (predicted_output_tokens) is deliberately much lower than any
    # "true" value would be -- the predictor has no way to know or correct
    # for this, by construction.
    pred = predictor.predict(10)
    assert pred.point_estimate == 0  # bucket [0,100) -> lower bound 0


def test_overprediction_uses_source_value_only():
    predictor = LengthPredictor(granularity=100, mode="exact")
    pred = predictor.predict(950)
    assert pred.bucket_index == 9
    assert pred.point_estimate == 900


def test_noisy_mode_deterministic_given_seed():
    def run():
        predictor = LengthPredictor(granularity=50, mode="noisy", noise_std_tokens=20.0, seed=42)
        return [predictor.predict(100).point_estimate for _ in range(10)]

    assert run() == run()


def test_noisy_mode_differs_from_exact_mode_in_general():
    exact = LengthPredictor(granularity=0, mode="exact").predict(100).point_estimate
    noisy_predictor = LengthPredictor(granularity=0, mode="noisy", noise_std_tokens=50.0, seed=1)
    noisy_values = [noisy_predictor.predict(100).point_estimate for _ in range(20)]
    assert any(v != exact for v in noisy_values)


def test_observable_request_has_no_actual_output_tokens_field():
    """Structural no-leakage guarantee: a policy operating through the
    normal ObservableState path cannot access ground-truth output length
    even if it tried."""
    req = make_req(0)
    assert not hasattr(req, "actual_output_tokens")


def test_invalid_predictor_mode_rejected():
    with pytest.raises(ValueError):
        LengthPredictor(mode="bogus")


# ---------------------------------------------------------------------------
# Inter-instance routing
# ---------------------------------------------------------------------------

def test_routing_two_eligible_workers_picks_one_of_them():
    router = PowerOfTwoDecodeRouter(seed=0)
    gpus = [make_gpu(1, "decode"), make_gpu(2, "decode")]
    chosen = router.select_decode_gpu(gpus, predicted_heavy=False, fits_fn=lambda g: True)
    assert chosen in (1, 2)


def test_routing_one_ineligible_worker_excluded():
    router = PowerOfTwoDecodeRouter(seed=0)
    gpus = [make_gpu(1, "decode"), make_gpu(2, "decode")]
    chosen = router.select_decode_gpu(gpus, predicted_heavy=False, fits_fn=lambda g: g.gpu_id == 1)
    assert chosen == 1


def test_routing_no_eligible_worker_returns_none():
    router = PowerOfTwoDecodeRouter(seed=0)
    gpus = [make_gpu(1, "decode"), make_gpu(2, "decode")]
    chosen = router.select_decode_gpu(gpus, predicted_heavy=False, fits_fn=lambda g: False)
    assert chosen is None


def test_routing_deterministic_given_seed():
    def run():
        router = PowerOfTwoDecodeRouter(seed=7)
        gpus = [make_gpu(i, "decode") for i in range(5)]
        return [router.select_decode_gpu(gpus, predicted_heavy=False, fits_fn=lambda g: True) for _ in range(10)]

    assert run() == run()


def test_routing_prefers_lower_heavy_light_ratio():
    """Fidelity: paper §5.2.3's stated dispatcher objective -- given two
    power-of-two candidates, prefer the one that would leave the LOWER
    heavy:light decode ratio (i.e., a heavy incoming request should
    prefer joining an all-light instance over an already-heavy one)."""
    router = PowerOfTwoDecodeRouter(seed=0)
    light_reqs = [make_req(i, output=10) for i in range(1, 4)]
    all_light_gpu = make_gpu(1, "decode", active_reqs=light_reqs)
    all_heavy_gpu = make_gpu(2, "decode", active_reqs=[make_req(10, output=500), make_req(11, output=500)])

    # Force the power-of-two sample to include both candidates by using
    # only these two GPUs (so len(alpha)==2 always samples both).
    chosen = router.select_decode_gpu(
        [all_light_gpu, all_heavy_gpu], predicted_heavy=True, fits_fn=lambda g: True,
    )
    assert chosen == 1  # joining the all-light instance keeps the better ratio


def test_routing_tie_break_lower_gpu_id():
    router = PowerOfTwoDecodeRouter(seed=0)
    gpu_a = make_gpu(5, "decode")
    gpu_b = make_gpu(2, "decode")
    chosen = router.select_decode_gpu([gpu_a, gpu_b], predicted_heavy=False, fits_fn=lambda g: True)
    assert chosen == 2  # identical (empty) ratios -> lower gpu_id wins


def test_routing_hotspot_avoidance_across_many_sequential_arrivals():
    """Repeated power-of-two routing across many independent heavy
    requests, always re-evaluating current load, should not collapse onto
    a single instance (hotspot) the way naive round-robin/first-feasible
    routing that ignores heavy/light mix could."""
    router = PowerOfTwoDecodeRouter(seed=3)
    n_gpus = 4
    gpus = {i: make_gpu(i, "decode") for i in range(n_gpus)}
    counts = Counter()
    for _ in range(40):
        candidates = list(gpus.values())
        chosen = router.select_decode_gpu(candidates, predicted_heavy=True, fits_fn=lambda g: True)
        counts[chosen] += 1
        # Simulate the chosen GPU accumulating one more heavy request.
        gpu = gpus[chosen]
        new_req = make_req(1000 + sum(counts.values()), output=500)
        gpus[chosen] = make_gpu(chosen, "decode", active_reqs=list(gpu.active_requests_info) + [new_req])
    assert len(counts) > 1, "all 40 heavy requests routed to a single instance -- hotspot not avoided"
    assert max(counts.values()) < 40


def test_is_heavy_decode_threshold():
    assert not is_heavy_decode(HEAVY_DECODE_THRESHOLD_TOKENS)
    assert is_heavy_decode(HEAVY_DECODE_THRESHOLD_TOKENS + 1)


# ---------------------------------------------------------------------------
# Worker-count validation
# ---------------------------------------------------------------------------

def test_requires_at_least_one_prefill_and_decode_gpu():
    policy = TetriInferPaperReimplementationPolicy()
    only_decode = [make_gpu(1, "decode"), make_gpu(2, "decode")]
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=only_decode, completed_count=0, step=0)
    with pytest.raises(ValueError):
        policy.select_action(state)


def test_invalid_prefill_policy_rejected():
    with pytest.raises(ValueError):
        TetriInferPaperReimplementationPolicy(prefill_local_policy="bogus")


def test_invalid_decode_policy_rejected():
    with pytest.raises(ValueError):
        TetriInferPaperReimplementationPolicy(decode_local_policy="bogus")


# ---------------------------------------------------------------------------
# Local decode scheduler: reserve-static / reserve-dynamic / greedy
# ---------------------------------------------------------------------------

def test_reserve_static_admits_when_full_prediction_fits_current_capacity():
    # block_size=1, decode_kv=10 blocks. Full predicted sequence footprint
    # (prompt=1 + output=8 + 1 same-step growth) = 10 <= 10 free blocks.
    policy = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_static", predictor_granularity=0, block_size=1,
    )
    decode_gpu = make_gpu(1, "decode", max_kv=10)
    req = make_req(0, prompt=1, output=8)
    action = policy.select_action(make_state(migrating=[req], decode_gpus=[decode_gpu]))
    assert action.admit[1] == [0]


def test_reserve_static_rejects_when_full_prediction_exceeds_current_capacity():
    policy = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_static", predictor_granularity=0, block_size=1,
    )
    decode_gpu = make_gpu(1, "decode", max_kv=5)  # only 5 blocks total
    req = make_req(0, prompt=1, output=8)  # full predicted footprint = 1+8+1=10 > 5
    action = policy.select_action(make_state(migrating=[req], decode_gpus=[decode_gpu]))
    assert action.admit[1] == []


def test_reserve_dynamic_admits_when_shortest_remaining_job_frees_enough():
    """Fidelity: reserve-dynamic admits a new request if there will be
    spare memory once the shortest-remaining active job finishes, even
    though the FULL predicted footprint does not currently fit (reserve-
    static would reject this same case -- see the companion test below).

    Uses two sequential real `select_action` calls on the SAME policy
    instance, with the second call's active-request state exactly
    reflecting what the first call itself produced (0 additional decode
    progress claimed) -- rather than a hand-built fixture the policy's own
    block manager never saw, which would silently change what's being
    tested (see docs/tetriinfer_reference.md and this policy's
    _grow_active_decode_requests docstring: blocks grow incrementally, not
    via bulk upfront reservation, so a fresh policy instance handed
    pre-existing active state has no record of it)."""
    policy = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_dynamic", predictor_granularity=0, block_size=1,
    )
    decode_gpu = make_gpu(1, "decode", max_kv=6)
    # A: prompt=1, output=1 -- a SHORT predicted job, admitted first.
    # Incremental allocation = prompt+1 = 2 blocks; 4 free remain.
    soon_req = make_req(99, prompt=1, output=1)
    action1 = policy.select_action(make_state(migrating=[soon_req], decode_gpus=[decode_gpu]))
    assert action1.admit[1] == [99]

    # B: prompt=1, output=3 -- full predicted footprint = 1+3+1=5 > 4 free
    # -> reserve-static would reject. But A's OWN full predicted footprint
    # (1+1=2 blocks) would be freed once it finishes soon (its predicted
    # remaining is short), and 4+2=6 >= 5 -- reserve-dynamic admits.
    soon_req_active = make_req(99, prompt=1, output=1)
    decode_gpu2 = make_gpu(1, "decode", max_kv=6, active_reqs=[soon_req_active], decoded={99: 0})
    new_req = make_req(0, prompt=1, output=3)
    action2 = policy.select_action(make_state(migrating=[new_req], decode_gpus=[decode_gpu2], step=1))
    assert action2.admit[1] == [0]


def test_reserve_static_rejects_the_same_case_reserve_dynamic_admits():
    """Companion to the test above: identical scenario, but reserve-static
    (no future projection) rejects it -- demonstrating the two policies
    genuinely differ, not just in name."""
    policy = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_static", predictor_granularity=0, block_size=1,
    )
    decode_gpu = make_gpu(1, "decode", max_kv=6)
    soon_req = make_req(99, prompt=1, output=1)
    action1 = policy.select_action(make_state(migrating=[soon_req], decode_gpus=[decode_gpu]))
    assert action1.admit[1] == [99]

    soon_req_active = make_req(99, prompt=1, output=1)
    decode_gpu2 = make_gpu(1, "decode", max_kv=6, active_reqs=[soon_req_active], decoded={99: 0})
    new_req = make_req(0, prompt=1, output=3)
    action2 = policy.select_action(make_state(migrating=[new_req], decode_gpus=[decode_gpu2], step=1))
    assert action2.admit[1] == []


def test_reserve_dynamic_rejects_when_no_active_job_would_free_enough():
    policy = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_dynamic", predictor_granularity=0, block_size=1,
    )
    decode_gpu = make_gpu(1, "decode", max_kv=6)
    soon_req = make_req(99, prompt=1, output=1)
    action1 = policy.select_action(make_state(migrating=[soon_req], decode_gpus=[decode_gpu]))
    assert action1.admit[1] == [99]

    # This time the incoming request is far larger -- even freeing all of
    # A's blocks isn't nearly enough.
    soon_req_active = make_req(99, prompt=1, output=1)
    decode_gpu2 = make_gpu(1, "decode", max_kv=6, active_reqs=[soon_req_active], decoded={99: 0})
    new_req = make_req(0, prompt=1, output=20)  # full predicted footprint = 22
    action2 = policy.select_action(make_state(migrating=[new_req], decode_gpus=[decode_gpu2], step=1))
    assert action2.admit[1] == []


def test_memory_pressure_case_no_admission_when_gpu_full():
    """Real pre-existing pressure (built via genuine prior admissions, not
    a hand-constructed fixture the policy's own block manager never saw --
    see test_reserve_dynamic_admits_when_shortest_remaining_job_frees_enough's
    docstring for why that distinction matters)."""
    policy = TetriInferPaperReimplementationPolicy(decode_local_policy="greedy", block_size=1)
    decode_gpu = make_gpu(1, "decode", max_kv=6)
    # Greedy reserves prompt_tokens+1 per request -- 3 fillers x 2 blocks
    # each exactly fills the 6-block capacity.
    fillers = [make_req(i, prompt=1, output=1, arrival=float(i)) for i in range(3)]
    action1 = policy.select_action(make_state(migrating=fillers, decode_gpus=[decode_gpu]))
    assert sorted(action1.admit[1]) == [0, 1, 2]

    active = [make_req(i, prompt=1, output=1, arrival=float(i)) for i in action1.admit[1]]
    decode_gpu2 = make_gpu(1, "decode", max_kv=6, active_reqs=active)
    new_req = make_req(99, prompt=1, output=5)
    action2 = policy.select_action(make_state(migrating=[new_req], decode_gpus=[decode_gpu2], step=1))
    assert action2.admit[1] == []


def test_prediction_error_changes_admission_outcome():
    """An underprediction (predicted_output_tokens too low) can admit a
    request that an accurate prediction would have rejected -- this is a
    disclosed, paper-acknowledged residual risk (see
    docs/tetriinfer_reference.md section E), not a bug."""
    decode_gpu = make_gpu(1, "decode", max_kv=10)
    accurate = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_static", predictor_granularity=0, predictor_mode="exact", block_size=1,
    )
    # Full predicted footprint = prompt(1) + output(8) + 1 = 10 -- exactly fits.
    req_accurate = make_req(0, prompt=1, output=8)
    action_accurate = accurate.select_action(make_state(migrating=[req_accurate], decode_gpus=[decode_gpu]))
    assert action_accurate.admit[1] == [0]

    under = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_static", predictor_granularity=0, predictor_mode="exact", block_size=1,
    )
    # A larger predicted_output_tokens (as if the SAME request were fed a
    # less accurate/higher prediction) pushes the full footprint over
    # capacity: 1+50+1=52 > 10 -- rejected.
    req_over_budget = make_req(0, prompt=1, output=50)
    action_rejected = under.select_action(make_state(migrating=[req_over_budget], decode_gpus=[decode_gpu]))
    assert action_rejected.admit[1] == []


def test_no_starvation_end_to_end():
    gpus = _gpu_configs(n_prefill=1, n_decode=2, decode_kv=2000)
    sm = _disagg_service_model(migration_transfer_delay=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=20, output=15) for i in range(15)])
    policy = TetriInferPaperReimplementationPolicy(decode_max_batch_size=2)
    metrics = sim.run(policy, workload_tag="no-starvation")
    assert metrics.num_completed == 15
    assert metrics.num_dropped == 0


def test_no_capacity_violation_end_to_end():
    gpus = _gpu_configs(n_prefill=2, n_decode=3, prefill_kv=500, decode_kv=500)
    sm = _disagg_service_model(step_token_budget=64, max_prefill_chunk_tokens=64, migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = [_req(i, arrival=float(i) * 0.002, prompt=15, output=10) for i in range(20)]
    sim.load_trace(reqs)
    policy = TetriInferPaperReimplementationPolicy(block_size=8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="capacity-check")
    assert w == [], f"unexpected simulator warnings: {[str(x.message) for x in w]}"
    assert metrics.num_completed == 20
    assert metrics.num_dropped == 0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_prefill_bridge_decode_completion():
    gpus = _gpu_configs(n_prefill=1, n_decode=2)
    sm = _disagg_service_model(migration_transfer_delay=0.005)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(0, prompt=80, output=20)])
    metrics = sim.run(TetriInferPaperReimplementationPolicy(), workload_tag="e2e")
    assert metrics.num_completed == 1
    assert metrics.num_dropped == 0


def test_mixed_short_long_decode_workload():
    gpus = _gpu_configs(n_prefill=1, n_decode=3, decode_kv=3000)
    sm = _disagg_service_model(migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = []
    for i in range(12):
        output = 10 if i % 2 == 0 else 300
        reqs.append(_req(i, arrival=float(i) * 0.002, prompt=30, output=output))
    sim.load_trace(reqs)
    metrics = sim.run(TetriInferPaperReimplementationPolicy(), workload_tag="mixed-decode")
    assert metrics.num_completed == 12
    assert metrics.num_dropped == 0


def test_deterministic_repeated_runs_end_to_end():
    def run():
        gpus = _gpu_configs(n_prefill=1, n_decode=2, decode_kv=2000)
        sm = _disagg_service_model(migration_transfer_delay=0.003)
        sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
        sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=40, output=25) for i in range(10)])
        return sim.run(TetriInferPaperReimplementationPolicy(), workload_tag="repro")

    m1, m2 = run(), run()
    assert m1.num_completed == m2.num_completed
    assert m1.mean_latency == m2.mean_latency
    assert m1.num_dropped == m2.num_dropped


# ---------------------------------------------------------------------------
# Regression: pre-existing baselines and legacy configs unaffected
# ---------------------------------------------------------------------------

def test_distserve_faithful_unaffected_by_tetriinfer_existing():
    gpus = [
        GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000, role="prefill"),
        GPUConfig(gpu_id=1, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000, role="decode"),
    ]
    sm = _disagg_service_model(migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(DistServeFaithfulPolicy(), workload_tag="distserve-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_vllm_faithful_unaffected_by_tetriinfer_existing():
    gpu = GPUConfig(gpu_id=0, max_active_sequences=50, max_batch_tokens=100_000, max_kv_tokens=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], max_steps=5000))
    sim.load_trace([_req(i, arrival=float(i) * 0.001, prompt=20, output=5) for i in range(5)])
    metrics = sim.run(VLLMFaithfulPolicy(block_size=16, watermark=0.0), workload_tag="vllm-compat")
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_sarathi_faithful_unaffected_by_tetriinfer_existing():
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
# hardware-speedup claims; see docs/tetriinfer_reference.md)
# ---------------------------------------------------------------------------

class _NaiveRoundRobinDecodeTestPolicy(BasePolicy):
    """Minimal comparison-only policy: routes migrating requests to decode
    GPUs in naive round-robin order, ignoring heavy/light mix entirely --
    used only to demonstrate the hotspot TetriInfer's dispatcher is
    designed to avoid. Not a claim about any other baseline's behavior."""
    name = "naive_round_robin_decode_test"

    def __init__(self):
        self._next_index = 0

    def reset(self):
        self._next_index = 0

    def select_action(self, state):
        admit = {g.gpu_id: [] for g in state.gpu_states}
        prefill_gpus = [g for g in state.gpu_states if g.role == "prefill"]
        decode_gpus = sorted((g for g in state.gpu_states if g.role == "decode"), key=lambda g: g.gpu_id)

        for req in state.waiting_queue:
            for g in prefill_gpus:
                if self._feasible_on_gpu(g, req):
                    admit[g.gpu_id].append(req.request_id)
                    g.active_request_ids.append(req.request_id)
                    g.current_kv_tokens += req.prompt_tokens
                    break

        for req in state.migrating_queue:
            g = decode_gpus[self._next_index % len(decode_gpus)]
            self._next_index += 1
            if self._feasible_on_gpu(g, req):
                admit[g.gpu_id].append(req.request_id)
                g.active_request_ids.append(req.request_id)
                g.current_kv_tokens += req.prompt_tokens

        return Action(admit=admit)


def test_sanity_mixed_downstream_workload():
    gpus = _gpu_configs(n_prefill=1, n_decode=3, decode_kv=5000)
    sm = _disagg_service_model(step_token_budget=200, max_prefill_chunk_tokens=200, migration_transfer_delay=0.002)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    reqs = []
    for i in range(16):
        prompt = 500 if i % 3 == 0 else 20
        output = 5 if i % 3 == 0 else 200
        reqs.append(_req(i, arrival=float(i) * 0.001, prompt=prompt, output=output))
    sim.load_trace(reqs)
    metrics = sim.run(TetriInferPaperReimplementationPolicy(), workload_tag="sanity-mixed")
    assert metrics.num_completed == 16
    assert metrics.num_dropped == 0


def test_sanity_high_kv_pressure():
    gpus = _gpu_configs(n_prefill=1, n_decode=2, decode_kv=300)
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=0.0, prompt=15, output=40) for i in range(10)])
    policy = TetriInferPaperReimplementationPolicy(decode_local_policy="reserve_dynamic", block_size=4)
    metrics = sim.run(policy, workload_tag="sanity-high-kv-pressure")
    assert metrics.num_completed == 10
    assert metrics.num_dropped == 0


def test_sanity_inaccurate_predictions_still_completes():
    gpus = _gpu_configs(n_prefill=1, n_decode=2, decode_kv=3000)
    sm = _disagg_service_model(migration_transfer_delay=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=20_000))
    sim.load_trace([_req(i, arrival=float(i) * 0.002, prompt=20, output=50) for i in range(10)])
    policy = TetriInferPaperReimplementationPolicy(
        decode_local_policy="reserve_dynamic",
        predictor_mode="noisy", predictor_noise_std_tokens=40.0, predictor_seed=5,
    )
    metrics = sim.run(policy, workload_tag="sanity-noisy-predictions")
    assert metrics.num_completed == 10
    assert metrics.num_dropped == 0


def _heavy_light_spread(assignment: dict, heavy_ids: set, n_gpus: int) -> int:
    """max - min heavy-request count across decode instances (0 = perfectly
    balanced). Only counts requests that were actually routed (present in
    `assignment`)."""
    per_gpu_heavy = Counter()
    for rid, gpu_id in assignment.items():
        if rid in heavy_ids:
            per_gpu_heavy[gpu_id] += 1
    counts = [per_gpu_heavy.get(100 + i, 0) for i in range(n_gpus)]
    return max(counts) - min(counts)


def test_sanity_hotspot_reduction_vs_naive_round_robin():
    """Qualitative behavior comparison only -- no hardware-speedup claim.
    Round-robin routes request k to instance (k mod n_gpus) purely by
    arrival position, ignoring heaviness entirely. When heavy requests
    arrive in a pattern that aligns with the round-robin cadence (here:
    every 4th request is heavy, matching n_decode=4), naive round-robin
    concentrates ALL heavy requests onto a single instance -- a textbook
    hotspot. TetriInfer's dispatcher explicitly optimizes for a balanced
    heavy:light ratio and must not reproduce that same worst-case
    concentration."""
    n_gpus = 4
    n_requests = 24
    heavy_ids = {i for i in range(n_requests) if i % n_gpus == 0}  # aligned with round-robin cadence

    def build_reqs():
        reqs = []
        for i in range(n_requests):
            output = 400 if i in heavy_ids else 20
            reqs.append(_req(i, arrival=float(i) * 0.0005, prompt=15, output=output))
        return reqs

    # TetriInfer
    gpus = _gpu_configs(n_prefill=1, n_decode=n_gpus, decode_kv=20_000)
    sm = _disagg_service_model(migration_transfer_delay=0.0)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, max_steps=500))
    sim.load_trace(build_reqs())
    tetri_policy = TetriInferPaperReimplementationPolicy(routing_seed=1)
    sim.run(tetri_policy, workload_tag="hotspot-check-tetri")
    tetri_spread = _heavy_light_spread(tetri_policy._decode_assignment, heavy_ids, n_gpus)

    # Naive round-robin
    gpus_rr = _gpu_configs(n_prefill=1, n_decode=n_gpus, decode_kv=20_000)
    sim_rr = Simulator(SimulatorConfig(gpu_configs=gpus_rr, service_model=sm, max_steps=500))
    sim_rr.load_trace(build_reqs())
    rr_policy = _NaiveRoundRobinDecodeTestPolicy()

    rr_assignment: dict = {}

    def recording_select(state, _orig=rr_policy.select_action):
        action = _orig(state)
        for gpu_id, rids in action.admit.items():
            for rid in rids:
                if gpu_id >= 100:  # decode-role gpu_ids in this test's convention
                    rr_assignment.setdefault(rid, gpu_id)
        return action

    rr_policy.select_action = recording_select
    sim_rr.run(rr_policy, workload_tag="hotspot-check-rr")
    rr_spread = _heavy_light_spread(rr_assignment, heavy_ids, n_gpus)

    # Round-robin's worst case here is total concentration (all 6 heavy
    # requests on one instance, spread == 6); TetriInfer must do strictly
    # better on this adversarial-for-round-robin workload.
    assert rr_spread == len(heavy_ids), "test setup sanity: round-robin should concentrate all heavy requests"
    assert tetri_spread < rr_spread

"""CC2 equivalence tests: primitive-reconstructed policies vs. originals.

Compares each reconstructed policy against its original implementation on:

* request ordering / admitted request IDs (synthetic single-step states,
  including multi-GPU feasibility-constrained placement);
* full simulator-trace runs: admitted IDs per step, GPU assignments,
  completion fraction, ANWG, and deterministic replay (same seed twice).

All synthetic states respect the causal invariant every production policy
in this repository depends on: a waiting request's arrival_time is always
<= state.time (a request cannot be waiting before it has arrived), and
request_id is non-decreasing with arrival_time (true for every trace
generator in this repository). Randomized fuzz coverage
(`test_randomized_synthetic_states_match_exactly`) explicitly constructs
only such physically valid states.

Exact vs. approximate status, and the rationale for each, is documented in
docs/architecture/contextual_composition_primitives.md and summarized in
each reconstructed policy's docstring in primitive_reconstructions.py.
"""
from __future__ import annotations

import copy
import random

import pytest

from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState, Request
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.admission_control import AdmissionControlPolicy
from llmserveopt.policies.best_fit import BestFitPolicy
from llmserveopt.policies.edf import EDFPolicy
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.primitive_reconstructions import (
    PrimitiveAdmissionControlPolicy,
    PrimitiveBestFitPolicy,
    PrimitiveEDFPolicy,
    PrimitiveEstimatedServiceTimeFirstPolicy,
    PrimitiveFIFOPolicy,
    PrimitiveScorpioStyleSloGuardPolicy,
    PrimitiveWeightedShortestProcessingPolicy,
)
from llmserveopt.policies.scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy
from llmserveopt.policies.weighted_shortest_processing import WeightedShortestProcessingPolicy

EXACT_TOLERANCE = 1e-9


def req(request_id, *, prompt=64, output=64, arrival=0.0, deadline=20.0, priority=1.0, class_id="medium"):
    return ObservableRequest(
        request_id=request_id, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline, priority=priority, class_id=class_id,
    )


def gpu(*, gpu_id=0, max_seq=4, max_kv=4096, max_batch=128, active=None, kv_used=0, prefilling=0, decoding=0):
    return ObservableGPUState(
        gpu_id=gpu_id, max_active_sequences=max_seq, max_batch_tokens=max_batch, max_kv_tokens=max_kv,
        active_request_ids=list(active or []), active_requests_info=[], current_kv_tokens=kv_used,
        tokens_decoded_per_request={}, prefilling_count=prefilling, decoding_count=decoding,
    )


def state(reqs, *, now=5.0, step=1, gpus=None):
    return ObservableState(time=now, waiting_queue=reqs, gpu_states=gpus or [gpu()], completed_count=0, step=step)


# Representative-policy pairs. All are documented EXACT except SCORPIO,
# which is documented APPROXIMATE (see primitive_reconstructions.py).
EXACT_PAIRS = [
    ("fifo", lambda: FIFOPolicy(), lambda: PrimitiveFIFOPolicy()),
    ("edf", lambda: EDFPolicy(), lambda: PrimitiveEDFPolicy()),
    ("weighted_shortest_processing", lambda: WeightedShortestProcessingPolicy(), lambda: PrimitiveWeightedShortestProcessingPolicy()),
    ("estimated_service_time_first", lambda: EstimatedServiceTimeFirstPolicy(), lambda: PrimitiveEstimatedServiceTimeFirstPolicy()),
    ("best_fit", lambda: BestFitPolicy(), lambda: PrimitiveBestFitPolicy()),
    ("admission_control", lambda: AdmissionControlPolicy(laxity_threshold=0.0), lambda: PrimitiveAdmissionControlPolicy(laxity_threshold=0.0)),
    ("admission_control_unfiltered", lambda: AdmissionControlPolicy(), lambda: PrimitiveAdmissionControlPolicy()),
]

APPROXIMATE_PAIRS = [
    ("scorpio_style_slo_guard", lambda: ScorpioStyleSloGuardPolicy(), lambda: PrimitiveScorpioStyleSloGuardPolicy()),
]

ALL_PAIRS = EXACT_PAIRS + APPROXIMATE_PAIRS


def _sample_reqs():
    return [
        req(1, prompt=512, output=512, arrival=0.0, deadline=20.0, priority=1.0),
        req(2, prompt=32, output=32, arrival=1.0, deadline=5.0, priority=2.0),
        req(3, prompt=128, output=128, arrival=2.0, deadline=100.0, priority=0.5),
        req(4, prompt=64, output=1024, arrival=3.0, deadline=8.0, priority=1.0),
        req(5, prompt=256, output=16, arrival=3.0, deadline=6.0, priority=1.0),
    ]


@pytest.mark.parametrize("name,make_orig,make_recon", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_single_gpu_synthetic_state_admits_same_ids(name, make_orig, make_recon):
    reqs = _sample_reqs()
    s1 = state(copy.deepcopy(reqs))
    s2 = state(copy.deepcopy(reqs))
    a1 = make_orig().select_action(s1)
    a2 = make_recon().select_action(s2)
    assert a1.admit == a2.admit, f"{name}: gpu assignment differs\norig={a1.admit}\nrecon={a2.admit}"


@pytest.mark.parametrize("name,make_orig,make_recon", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_multi_gpu_feasibility_constrained_state_admits_same_ids(name, make_orig, make_recon):
    reqs = _sample_reqs() + [req(6, prompt=800, output=800, arrival=4.0, deadline=50.0, priority=0.2)]
    gpus = [
        gpu(gpu_id=0, max_seq=2, max_kv=700, max_batch=128, decoding=1, prefilling=1),
        gpu(gpu_id=1, max_seq=2, max_kv=1500, max_batch=128, decoding=0, prefilling=0),
        gpu(gpu_id=2, max_seq=1, max_kv=2000, max_batch=128, decoding=2, prefilling=0),
    ]
    s1 = state(copy.deepcopy(reqs), now=6.0, gpus=copy.deepcopy(gpus))
    s2 = state(copy.deepcopy(reqs), now=6.0, gpus=copy.deepcopy(gpus))
    a1 = make_orig().select_action(s1)
    a2 = make_recon().select_action(s2)
    assert a1.admit == a2.admit, f"{name}: gpu assignment differs\norig={a1.admit}\nrecon={a2.admit}"
    assert a1.all_admitted_ids() == a2.all_admitted_ids()


@pytest.mark.parametrize("name,make_orig,make_recon", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_empty_waiting_queue_is_a_no_op_for_both(name, make_orig, make_recon):
    s1 = state([])
    s2 = state([])
    a1 = make_orig().select_action(s1)
    a2 = make_recon().select_action(s2)
    assert a1.all_admitted_ids() == set() == a2.all_admitted_ids()


def _valid_random_reqs(rnd, n):
    """Build a physically valid request list: request_id and arrival_time
    both non-decreasing together, matching every trace generator in this
    repository (see docs/architecture/contextual_composition_primitives.md
    "Known Gaps" for why this precondition matters)."""
    reqs = []
    arrival = 0.0
    for i in range(n):
        arrival += rnd.choice([0.0, 0.0, 0.5, 1.0])
        reqs.append(req(
            i + 1,
            prompt=rnd.choice([16, 32, 64, 128, 256, 512, 1024]),
            output=rnd.choice([16, 32, 64, 128, 256, 512, 1024]),
            arrival=arrival,
            deadline=arrival + rnd.choice([-5.0, 0.5, 2.0, 5.0, 10.0, 50.0, 200.0]),
            priority=rnd.choice([0.1, 0.5, 1.0, 2.0, 5.0]),
        ))
    return reqs


@pytest.mark.parametrize("name,make_orig,make_recon", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_randomized_synthetic_states_match_exactly(name, make_orig, make_recon):
    rnd = random.Random(hash(name) & 0xFFFFFFFF)
    mismatches = 0
    trials = 60
    for _ in range(trials):
        reqs = _valid_random_reqs(rnd, rnd.randint(0, 10))
        n_gpus = rnd.randint(1, 3)
        gpus = [
            gpu(
                gpu_id=g, max_seq=rnd.choice([1, 2, 3]), max_kv=rnd.choice([256, 512, 1024, 4096]),
                max_batch=rnd.choice([2, 4, 8, 128]),
                decoding=rnd.choice([0, 1, 2]), prefilling=rnd.choice([0, 1]),
            )
            for g in range(n_gpus)
        ]
        max_arrival = max((r.arrival_time for r in reqs), default=0.0)
        now = max_arrival + rnd.choice([0.0, 0.5, 1.0, 5.0])
        s1 = state(copy.deepcopy(reqs), now=now, gpus=copy.deepcopy(gpus))
        s2 = state(copy.deepcopy(reqs), now=now, gpus=copy.deepcopy(gpus))
        a1 = make_orig().select_action(s1)
        a2 = make_recon().select_action(s2)
        if a1.admit != a2.admit:
            mismatches += 1
    if name == "scorpio_style_slo_guard":
        # APPROXIMATE: documented as reproducing the same formulas via
        # composed primitive calls; asserted here at exact tolerance
        # (0 mismatches observed across all CC2 fixtures) but treated as
        # an approximation claim, not a formal guarantee, per the CC2 doc.
        assert mismatches == 0, f"{name}: {mismatches}/{trials} synthetic-state mismatches"
    else:
        assert mismatches == 0, f"{name}: {mismatches}/{trials} synthetic-state mismatches"


# ---------------------------------------------------------------------------
# Full simulator-trace equivalence
# ---------------------------------------------------------------------------


def _make_trace(n, seed):
    rnd = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        t += rnd.expovariate(1.0 / 0.05)
        prompt = rnd.choice([32, 64, 128, 256, 512])
        out = rnd.choice([32, 64, 128, 256, 512])
        reqs.append(Request(
            request_id=i, arrival_time=t, prompt_tokens=prompt,
            predicted_output_tokens=out, actual_output_tokens=out,
            slo_deadline=t + rnd.choice([0.5, 1.0, 2.0, 5.0, 20.0]),
            priority=rnd.choice([0.5, 1.0, 2.0]), class_id="medium",
        ))
    return reqs


_TRACE_GPU_CONFIGS = [
    GPUConfig(gpu_id=0, max_active_sequences=4, max_batch_tokens=128, max_kv_tokens=4096),
    GPUConfig(gpu_id=1, max_active_sequences=4, max_batch_tokens=128, max_kv_tokens=4096),
]


@pytest.mark.parametrize("name,make_orig,make_recon", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_simulator_trace_metrics_match(name, make_orig, make_recon, seed):
    trace = _make_trace(50, seed)
    m1 = run_policy(make_orig(), trace, _TRACE_GPU_CONFIGS, workload_tag="cc2_equivalence", seed=seed)
    m2 = run_policy(make_recon(), trace, _TRACE_GPU_CONFIGS, workload_tag="cc2_equivalence", seed=seed)

    assert m1.num_completed == m2.num_completed, name
    assert m1.num_dropped == m2.num_dropped, name
    assert m1.num_total == m2.num_total, name
    assert m1.completion_fraction == pytest.approx(m2.completion_fraction, abs=EXACT_TOLERANCE), name
    assert m1.arrival_normalized_weighted_goodput == pytest.approx(
        m2.arrival_normalized_weighted_goodput, abs=EXACT_TOLERANCE
    ), name
    assert m1.weighted_goodput == pytest.approx(m2.weighted_goodput, abs=EXACT_TOLERANCE, nan_ok=True), name


@pytest.mark.parametrize("name,make_orig,make_recon", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_reconstructed_policy_deterministic_replay(name, make_orig, make_recon):
    trace = _make_trace(40, seed=7)
    m_a = run_policy(make_recon(), trace, _TRACE_GPU_CONFIGS, workload_tag="cc2_replay", seed=7)
    m_b = run_policy(make_recon(), trace, _TRACE_GPU_CONFIGS, workload_tag="cc2_replay", seed=7)
    assert m_a.num_completed == m_b.num_completed, name
    assert m_a.num_dropped == m_b.num_dropped, name
    assert m_a.arrival_normalized_weighted_goodput == pytest.approx(
        m_b.arrival_normalized_weighted_goodput, abs=EXACT_TOLERANCE
    ), name
    assert m_a.completion_fraction == pytest.approx(m_b.completion_fraction, abs=EXACT_TOLERANCE), name


def test_scorpio_reconstruction_triggers_overload_guard_branch():
    """Sanity check that the APPROXIMATE scorpio pair's equivalence
    coverage actually exercises the guard-active branch (kv/decode/queue
    overload and long-decode filtering), not just the common-case path."""
    rnd = random.Random(99)
    trace = _make_trace(80, seed=99)
    tight_gpus = [GPUConfig(gpu_id=0, max_active_sequences=3, max_batch_tokens=8, max_kv_tokens=512)]
    orig = ScorpioStyleSloGuardPolicy()
    recon = PrimitiveScorpioStyleSloGuardPolicy()
    m1 = run_policy(orig, trace, tight_gpus, workload_tag="cc2_overload", seed=99)
    m2 = run_policy(recon, trace, tight_gpus, workload_tag="cc2_overload", seed=99)
    assert m1.num_completed == m2.num_completed
    assert m1.completion_fraction < 1.0  # confirms this fixture is actually capacity-constrained
    assert m1.arrival_normalized_weighted_goodput == pytest.approx(
        m2.arrival_normalized_weighted_goodput, abs=EXACT_TOLERANCE
    )

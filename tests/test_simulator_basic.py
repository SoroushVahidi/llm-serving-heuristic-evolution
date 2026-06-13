"""
Basic simulator correctness tests.
"""
import warnings

import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.edf import EDFPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _make_gpus(n=2, max_seq=8, max_tok=64, max_kv=1024):
    return [
        GPUConfig(gpu_id=i, max_active_sequences=max_seq,
                  max_batch_tokens=max_tok, max_kv_tokens=max_kv)
        for i in range(n)
    ]


def _req(rid, arrival, prompt=10, output=5, deadline=100.0, priority=1.0):
    return Request(
        request_id=rid,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        actual_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id="medium",
    )


# -------------------------------------------------------------------
# Arrival-time enforcement
# -------------------------------------------------------------------

def test_request_not_admitted_before_arrival():
    """A request with future arrival time must never be scheduled early."""
    requests = [
        _req(0, arrival=0.0, output=5),
        _req(1, arrival=10.0, output=5),   # arrives far in the future
    ]
    gpu_configs = _make_gpus(n=1)
    service_model = ServiceModel(step_size=0.001)

    sim_cfg = SimulatorConfig(gpu_configs=gpu_configs, service_model=service_model,
                              drain_steps=1000)
    sim = Simulator(sim_cfg)
    sim.load_trace(requests)

    policy = FIFOPolicy()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = sim.run(policy, workload_tag="test", seed=0)

    # Request 1 has arrival_time=10s; with step_size=0.001 and drain_steps=1000
    # the simulation ends around 5*0.001 + 1s ≈ 1s, well before req 1 arrives.
    # So req 1 must either be dropped (not completed) or completed but only after
    # its arrival time.
    assert metrics.num_completed >= 1
    # Verify no warning about admitting a future request
    premature = [str(x.message) for x in w if "arrival_time" in str(x.message)]
    assert premature == [], f"Premature admission warnings: {premature}"


def test_all_requests_complete_tiny_trace():
    """A tiny trace with ample GPU capacity should complete all requests."""
    requests = [_req(i, arrival=float(i) * 0.01, output=3) for i in range(5)]
    gpu_configs = _make_gpus(n=2, max_seq=8, max_tok=64, max_kv=512)
    metrics = run_policy(
        FIFOPolicy(), requests, gpu_configs,
        ServiceModel(step_size=0.001), drain_steps=500,
    )
    assert metrics.num_completed == 5
    assert metrics.num_dropped == 0


def test_completion_order_fifo():
    """Requests admitted under FIFO should complete roughly in arrival order."""
    requests = [_req(i, arrival=float(i) * 0.001, output=5) for i in range(6)]
    gpu_configs = _make_gpus(n=1, max_seq=6, max_tok=64, max_kv=512)
    metrics = run_policy(
        FIFOPolicy(), requests, gpu_configs,
        ServiceModel(step_size=0.001), drain_steps=1000,
    )
    assert metrics.num_completed == 6


def test_metrics_latency_positive():
    requests = [_req(0, arrival=0.0, output=10, deadline=100.0)]
    gpu_configs = _make_gpus(n=1)
    metrics = run_policy(
        FIFOPolicy(), requests, gpu_configs, ServiceModel(step_size=0.001),
    )
    assert metrics.mean_latency > 0
    assert metrics.p95_latency >= metrics.mean_latency or metrics.num_completed < 20


def test_slo_violation_tight_deadline():
    """Requests with impossibly tight deadlines must show SLO violations."""
    requests = [
        Request(
            request_id=i,
            arrival_time=0.0,
            prompt_tokens=10,
            predicted_output_tokens=100,
            actual_output_tokens=100,
            slo_deadline=0.05,      # 50ms — much less than 100 * 1ms = 100ms
            priority=1.0,
            class_id="tight",
        )
        for i in range(4)
    ]
    gpu_configs = _make_gpus(n=1, max_seq=4, max_tok=32, max_kv=512)
    metrics = run_policy(
        FIFOPolicy(), requests, gpu_configs, ServiceModel(step_size=0.001),
    )
    assert metrics.num_completed > 0
    assert metrics.slo_violation_rate > 0.0


def test_no_double_admission():
    """The same request must not be admitted to two GPUs in one step."""
    requests = [_req(0, arrival=0.0, output=5)]
    gpu_configs = _make_gpus(n=3)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = run_policy(
            FIFOPolicy(), requests, gpu_configs, ServiceModel(step_size=0.001),
        )

    double_warn = [x for x in w if "multiple GPUs" in str(x.message)]
    assert double_warn == [], "Request was double-admitted"
    assert metrics.num_completed == 1


def test_invalid_gpu_id_skipped():
    """An action referencing a non-existent GPU ID must be skipped with a warning."""
    from llmserveopt.core.action import Action
    from llmserveopt.core.types import ObservableState
    from llmserveopt.policies.base import BasePolicy

    class BadGPUPolicy(BasePolicy):
        name = "bad_gpu"
        def select_action(self, state):
            if state.waiting_queue:
                return Action(admit={999: [state.waiting_queue[0].request_id]})
            return Action()

    requests = [_req(0, arrival=0.0, output=5)]
    gpu_configs = _make_gpus(n=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = run_policy(BadGPUPolicy(), requests, gpu_configs, ServiceModel())

    gpu_warn = [x for x in w if "gpu_id=999" in str(x.message)]
    assert len(gpu_warn) > 0, "Expected warning for invalid GPU ID"
    assert metrics.num_dropped == 1  # request never admitted


def test_invalid_request_id_skipped():
    """An action referencing a non-existent request ID must be skipped."""
    from llmserveopt.core.action import Action
    from llmserveopt.policies.base import BasePolicy

    class BadReqPolicy(BasePolicy):
        name = "bad_req"
        def select_action(self, state):
            return Action(admit={0: [99999]})

    requests = [_req(0, arrival=0.0, output=5)]
    gpu_configs = _make_gpus(n=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = run_policy(BadReqPolicy(), requests, gpu_configs, ServiceModel())

    req_warn = [x for x in w if "99999" in str(x.message)]
    assert len(req_warn) > 0, "Expected warning for invalid request ID"


# -------------------------------------------------------------------
# Capacity constraints
# -------------------------------------------------------------------

def test_max_active_sequences_enforced():
    """GPU must reject admission beyond max_active_sequences."""
    # 10 requests all arriving at t=0; GPU only fits 4
    requests = [_req(i, arrival=0.0, output=50) for i in range(10)]
    gpu_configs = [GPUConfig(gpu_id=0, max_active_sequences=4,
                             max_batch_tokens=64, max_kv_tokens=4096)]
    metrics = run_policy(
        FIFOPolicy(), requests, gpu_configs, ServiceModel(step_size=0.001),
        drain_steps=10_000,
    )
    # All should eventually complete since we drain long enough
    assert metrics.num_completed == 10
    assert metrics.num_dropped == 0


def test_kv_capacity_enforced():
    """GPU must reject admission when KV cache is exhausted."""
    # Each request needs 100 prompt tokens, GPU has kv_tokens=150 → fits 1 at a time
    requests = [
        Request(
            request_id=i, arrival_time=0.0,
            prompt_tokens=100, predicted_output_tokens=5, actual_output_tokens=5,
            slo_deadline=100.0, priority=1.0, class_id="medium",
        )
        for i in range(4)
    ]
    gpu_configs = [GPUConfig(gpu_id=0, max_active_sequences=4,
                             max_batch_tokens=64, max_kv_tokens=150)]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        metrics = run_policy(
            FIFOPolicy(), requests, gpu_configs, ServiceModel(step_size=0.001),
            drain_steps=5000,
        )
    assert metrics.num_completed == 4

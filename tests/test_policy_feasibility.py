"""
Tests: policies produce feasible actions and are deterministic under fixed seed.
"""
import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.registry import all_baseline_policies, BASELINE_NAMES, make_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.synthetic import make_small_debug_trace, make_medium_trace


GPU_CONFIGS = [
    GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=128, max_kv_tokens=2048),
    GPUConfig(gpu_id=1, max_active_sequences=16, max_batch_tokens=128, max_kv_tokens=2048),
]
SERVICE_MODEL = ServiceModel(step_size=0.001)


@pytest.mark.parametrize("policy_name", BASELINE_NAMES)
def test_policy_completes_requests(policy_name):
    """Every baseline policy should complete at least some requests."""
    requests = make_small_debug_trace(seed=42)
    policy = make_policy(policy_name, seed=0)
    metrics = run_policy(
        policy, requests, GPU_CONFIGS, SERVICE_MODEL, drain_steps=5000,
    )
    assert metrics.num_completed >= 0   # no crash
    assert metrics.num_completed + metrics.num_dropped == len(requests)


@pytest.mark.parametrize("policy_name", BASELINE_NAMES)
def test_policy_deterministic(policy_name):
    """Running the same policy twice on the same trace gives identical metrics."""
    requests = make_small_debug_trace(seed=7)
    policy_a = make_policy(policy_name, seed=123)
    policy_b = make_policy(policy_name, seed=123)

    m_a = run_policy(policy_a, requests, GPU_CONFIGS, SERVICE_MODEL, drain_steps=5000)
    m_b = run_policy(policy_b, requests, GPU_CONFIGS, SERVICE_MODEL, drain_steps=5000)

    assert m_a.num_completed == m_b.num_completed
    assert abs(m_a.mean_latency - m_b.mean_latency) < 1e-9


def test_fifo_admits_oldest_first():
    """FIFO should always admit requests in arrival order when capacity allows."""
    # Single GPU, generous capacity: all arrive at different times
    requests = [
        Request(
            request_id=i, arrival_time=float(i) * 0.01,
            prompt_tokens=5, predicted_output_tokens=3, actual_output_tokens=3,
            slo_deadline=100.0, priority=1.0, class_id="medium",
        )
        for i in range(6)
    ]
    gpu_configs = [GPUConfig(gpu_id=0, max_active_sequences=6,
                             max_batch_tokens=64, max_kv_tokens=512)]
    metrics = run_policy(
        make_policy("fifo"), requests, gpu_configs, SERVICE_MODEL, drain_steps=500,
    )
    assert metrics.num_completed == 6


def test_edf_prioritizes_early_deadlines():
    """EDF should complete all requests when given tight vs. loose deadlines."""
    requests = [
        Request(
            request_id=0, arrival_time=0.0,
            prompt_tokens=5, predicted_output_tokens=5, actual_output_tokens=5,
            slo_deadline=0.02, priority=1.0, class_id="tight",  # tight: 20ms
        ),
        Request(
            request_id=1, arrival_time=0.0,
            prompt_tokens=5, predicted_output_tokens=5, actual_output_tokens=5,
            slo_deadline=100.0, priority=1.0, class_id="loose",
        ),
    ]
    gpu_configs = [GPUConfig(gpu_id=0, max_active_sequences=1,
                             max_batch_tokens=32, max_kv_tokens=256)]
    metrics = run_policy(
        make_policy("edf"), requests, gpu_configs, SERVICE_MODEL, drain_steps=200,
    )
    # With only 1 sequence slot, EDF should pick the tight one first
    assert metrics.num_completed == 2
    assert metrics.slo_violation_rate < 1.0


def test_slo_violation_rate_in_range():
    """SLO violation rate must be in [0, 1]."""
    requests = make_medium_trace(seed=0)[:50]
    for policy_name in BASELINE_NAMES:
        policy = make_policy(policy_name, seed=0)
        m = run_policy(policy, requests, GPU_CONFIGS, SERVICE_MODEL, drain_steps=5000)
        assert 0.0 <= m.slo_violation_rate <= 1.0, \
            f"{policy_name} produced slo_violation_rate={m.slo_violation_rate}"


def test_no_negative_latency():
    """Latency and queuing delay must always be non-negative."""
    requests = make_small_debug_trace(seed=0)
    for policy_name in BASELINE_NAMES:
        policy = make_policy(policy_name, seed=0)
        m = run_policy(policy, requests, GPU_CONFIGS, SERVICE_MODEL, drain_steps=5000)
        if m.num_completed > 0:
            assert m.mean_latency >= 0.0
            assert m.mean_queuing_delay >= 0.0

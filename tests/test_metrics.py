"""
Tests: metric computation correctness.
"""
import math
import pytest

from llmserveopt.core.metrics import compute_metrics, metrics_to_dict
from llmserveopt.core.types import CompletedRequest, Request


def _make_completed(rid, arrival, admission, completion, output_tokens=10,
                    slo_deadline=100.0):
    req = Request(
        request_id=rid, arrival_time=arrival,
        prompt_tokens=20, predicted_output_tokens=output_tokens,
        actual_output_tokens=output_tokens,
        slo_deadline=slo_deadline, priority=1.0, class_id="medium",
    )
    return CompletedRequest(
        request=req,
        admission_time=admission,
        completion_time=completion,
        gpu_id=0,
    )


def test_latency_calculation():
    c = _make_completed(0, arrival=0.0, admission=1.0, completion=5.0)
    assert c.latency == pytest.approx(5.0)
    assert c.queuing_delay == pytest.approx(1.0)
    assert c.service_time == pytest.approx(4.0)


def test_slo_violated():
    c_ok = _make_completed(0, arrival=0.0, admission=0.0, completion=1.0, slo_deadline=2.0)
    c_viol = _make_completed(1, arrival=0.0, admission=0.0, completion=3.0, slo_deadline=2.0)
    assert not c_ok.slo_violated
    assert c_viol.slo_violated


def test_metrics_single_request():
    completed = [_make_completed(0, arrival=0.0, admission=0.1, completion=1.0)]
    m = compute_metrics(
        completed=completed, dropped=[], sim_duration=2.0,
        gpu_utilization_history=[0.5, 0.5], active_batch_history=[1.0, 1.0],
    )
    assert m.num_completed == 1
    assert m.num_dropped == 0
    assert m.mean_latency == pytest.approx(1.0)
    assert m.slo_violation_rate == pytest.approx(0.0)
    assert m.request_throughput == pytest.approx(0.5)


def test_metrics_mixed_slo():
    completed = [
        _make_completed(0, arrival=0.0, admission=0.0, completion=1.0, slo_deadline=2.0),
        _make_completed(1, arrival=0.0, admission=0.0, completion=3.0, slo_deadline=2.0),
        _make_completed(2, arrival=0.0, admission=0.0, completion=0.5, slo_deadline=2.0),
        _make_completed(3, arrival=0.0, admission=0.0, completion=4.0, slo_deadline=2.0),
    ]
    m = compute_metrics(
        completed=completed, dropped=[], sim_duration=4.0,
        gpu_utilization_history=[], active_batch_history=[],
    )
    assert m.num_slo_violated == 2
    assert m.slo_violation_rate == pytest.approx(0.5)


def test_metrics_p95_p99():
    import numpy as np
    n = 100
    latencies = list(range(1, n + 1))  # 1..100 seconds
    completed = [
        _make_completed(i, arrival=0.0, admission=0.0, completion=float(l))
        for i, l in enumerate(latencies)
    ]
    m = compute_metrics(
        completed=completed, dropped=[], sim_duration=100.0,
        gpu_utilization_history=[], active_batch_history=[],
    )
    assert m.p95_latency == pytest.approx(np.percentile(latencies, 95))
    assert m.p99_latency == pytest.approx(np.percentile(latencies, 99))


def test_metrics_no_completed():
    m = compute_metrics(
        completed=[], dropped=[], sim_duration=10.0,
        gpu_utilization_history=[], active_batch_history=[],
    )
    assert m.num_completed == 0
    assert math.isnan(m.mean_latency)
    assert math.isnan(m.slo_violation_rate)


def test_metrics_to_dict_no_nan():
    completed = [_make_completed(0, arrival=0.0, admission=0.0, completion=1.0)]
    m = compute_metrics(
        completed=completed, dropped=[], sim_duration=2.0,
        gpu_utilization_history=[0.5], active_batch_history=[1.0],
    )
    d = metrics_to_dict(m)
    # NaN/Inf should be converted to None
    for k, v in d.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"NaN in metrics_to_dict key={k}"


# -------------------------------------------------------------------------
# completion_fraction and num_total tests
# -------------------------------------------------------------------------

def test_completion_fraction_all_complete():
    completed = [_make_completed(i, arrival=0.0, admission=0.0, completion=1.0) for i in range(5)]
    m = compute_metrics(
        completed=completed, dropped=[], sim_duration=2.0,
        gpu_utilization_history=[], active_batch_history=[],
        num_total=5,
    )
    assert m.num_total == 5
    assert m.completion_fraction == pytest.approx(1.0)


def test_completion_fraction_partial():
    completed = [_make_completed(i, arrival=0.0, admission=0.0, completion=1.0) for i in range(3)]
    dropped = [object(), object()]  # 2 dropped stubs
    m = compute_metrics(
        completed=completed, dropped=dropped, sim_duration=2.0,
        gpu_utilization_history=[], active_batch_history=[],
        num_total=10,
    )
    assert m.num_total == 10
    assert m.num_completed == 3
    assert m.completion_fraction == pytest.approx(0.3)


def test_completion_fraction_none_complete():
    m = compute_metrics(
        completed=[], dropped=[], sim_duration=1.0,
        gpu_utilization_history=[], active_batch_history=[],
        num_total=4,
    )
    assert m.num_total == 4
    assert m.completion_fraction == pytest.approx(0.0)


def test_completion_fraction_fallback_when_no_num_total():
    """When num_total=0, falls back to num_completed + num_dropped."""
    completed = [_make_completed(i, arrival=0.0, admission=0.0, completion=1.0) for i in range(2)]
    dropped = [object()]
    m = compute_metrics(
        completed=completed, dropped=dropped, sim_duration=2.0,
        gpu_utilization_history=[], active_batch_history=[],
        num_total=0,
    )
    assert m.num_total == 3  # 2 completed + 1 dropped
    assert m.completion_fraction == pytest.approx(2 / 3)


def test_completion_fraction_in_metrics_to_dict():
    completed = [_make_completed(i, arrival=0.0, admission=0.0, completion=1.0) for i in range(4)]
    m = compute_metrics(
        completed=completed, dropped=[], sim_duration=2.0,
        gpu_utilization_history=[], active_batch_history=[],
        num_total=8,
    )
    d = metrics_to_dict(m)
    assert "completion_fraction" in d
    assert d["completion_fraction"] == pytest.approx(0.5)
    assert "num_total" in d
    assert d["num_total"] == 8


def test_completion_fraction_zero_num_total():
    """num_total=0 with no completions/drops → NaN fraction, num_total fallback=0."""
    m = compute_metrics(
        completed=[], dropped=[], sim_duration=1.0,
        gpu_utilization_history=[], active_batch_history=[],
        num_total=0,
    )
    # fallback: num_total = 0+0 = 0; fraction = NaN
    assert m.num_total == 0
    assert math.isnan(m.completion_fraction)

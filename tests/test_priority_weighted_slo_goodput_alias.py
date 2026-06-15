"""Tests for priority_weighted_slo_goodput alias in RunMetrics."""
import pytest
from llmserveopt.core.metrics import RunMetrics, compute_metrics, metrics_to_dict
from llmserveopt.core.types import CompletedRequest, Request


def make_completed(req_id, priority, slo_deadline, completion_time, arrival=0.0, prompt=64, output=32):
    req = Request(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        actual_output_tokens=output,
        slo_deadline=slo_deadline,
        priority=priority,
        class_id="medium",
    )
    return CompletedRequest(
        request=req,
        admission_time=arrival,
        completion_time=completion_time,
        gpu_id=0,
    )


def run_compute(completeds):
    return compute_metrics(
        completed=completeds,
        dropped=[],
        sim_duration=10.0,
        gpu_utilization_history=[0.5] * 10,
        active_batch_history=[1.0] * 10,
    )


# -------------------------------------------------------------------------
# Property alias exists and equals weighted_goodput
# -------------------------------------------------------------------------

def test_alias_property_exists():
    m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
    m.weighted_goodput = 0.85
    assert hasattr(m, "priority_weighted_slo_goodput")


def test_alias_equals_weighted_goodput():
    m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
    m.weighted_goodput = 0.73
    assert m.priority_weighted_slo_goodput == m.weighted_goodput


def test_alias_equals_weighted_goodput_zero():
    m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
    m.weighted_goodput = 0.0
    assert m.priority_weighted_slo_goodput == 0.0


def test_alias_nan_when_goodput_nan():
    import math
    m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
    assert math.isnan(m.priority_weighted_slo_goodput)


# -------------------------------------------------------------------------
# metrics_to_dict includes alias
# -------------------------------------------------------------------------

def test_metrics_to_dict_includes_alias():
    m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
    m.weighted_goodput = 0.9
    d = metrics_to_dict(m)
    assert "priority_weighted_slo_goodput" in d, "alias missing from metrics_to_dict output"


def test_metrics_to_dict_alias_equals_weighted_goodput():
    m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
    m.weighted_goodput = 0.65
    d = metrics_to_dict(m)
    assert d["priority_weighted_slo_goodput"] == d["weighted_goodput"]


# -------------------------------------------------------------------------
# compute_metrics produces consistent alias
# -------------------------------------------------------------------------

def test_compute_metrics_alias_consistent():
    cs = [
        make_completed(0, priority=2.0, slo_deadline=5.0, completion_time=3.0),  # met
        make_completed(1, priority=1.0, slo_deadline=5.0, completion_time=6.0),  # violated
    ]
    m = run_compute(cs)
    assert m.priority_weighted_slo_goodput == m.weighted_goodput
    # Expected: (2.0 * 1) / (2.0 + 1.0) = 2/3
    assert abs(m.weighted_goodput - 2.0 / 3.0) < 1e-9


# -------------------------------------------------------------------------
# All-priority-1 case reduces to ordinary SLO goodput rate
# -------------------------------------------------------------------------

def test_all_priority_one_reduces_to_slo_rate():
    cs = [
        make_completed(0, priority=1.0, slo_deadline=5.0, completion_time=3.0),  # met
        make_completed(1, priority=1.0, slo_deadline=5.0, completion_time=3.0),  # met
        make_completed(2, priority=1.0, slo_deadline=5.0, completion_time=7.0),  # violated
    ]
    m = run_compute(cs)
    # 2 met out of 3 → 2/3 ≈ 0.666...
    assert abs(m.weighted_goodput - 2.0 / 3.0) < 1e-9
    assert m.weighted_goodput == m.priority_weighted_slo_goodput
    expected_slo_rate = 2.0 / 3.0
    assert abs(m.weighted_goodput - expected_slo_rate) < 1e-9


def test_all_met_slo_gives_goodput_one():
    cs = [make_completed(i, priority=float(i + 1), slo_deadline=10.0, completion_time=5.0) for i in range(4)]
    m = run_compute(cs)
    assert abs(m.weighted_goodput - 1.0) < 1e-9
    assert abs(m.priority_weighted_slo_goodput - 1.0) < 1e-9


def test_all_violated_gives_goodput_zero():
    cs = [make_completed(i, priority=2.0, slo_deadline=1.0, completion_time=5.0) for i in range(3)]
    m = run_compute(cs)
    assert m.weighted_goodput == pytest.approx(0.0)
    assert m.priority_weighted_slo_goodput == pytest.approx(0.0)

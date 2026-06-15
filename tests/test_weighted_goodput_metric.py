"""
Tests: weighted_goodput metric correctness.

weighted_goodput = sum(priority_i * 1[met SLO_i]) / sum(priority_i)
Priority falls back to 1.0 when request.priority == 0.
"""
import math
import pytest

from llmserveopt.core.metrics import compute_metrics, metrics_to_dict
from llmserveopt.core.types import CompletedRequest, Request


def _req(rid, *, priority=1.0, slo_deadline=10.0, actual_output_tokens=5):
    return Request(
        request_id=rid,
        arrival_time=0.0,
        prompt_tokens=10,
        predicted_output_tokens=actual_output_tokens,
        actual_output_tokens=actual_output_tokens,
        slo_deadline=slo_deadline,
        priority=priority,
        class_id="medium",
    )


def _completed(req, *, completion_time):
    return CompletedRequest(
        request=req,
        admission_time=0.0,
        completion_time=completion_time,
        gpu_id=0,
    )


def _run(completed_list):
    return compute_metrics(
        completed=completed_list,
        dropped=[],
        sim_duration=20.0,
        gpu_utilization_history=[],
        active_batch_history=[],
    )


class TestWeightedGoodput:
    def test_all_meet_slo_returns_one(self):
        reqs = [
            _completed(_req(0, slo_deadline=10.0), completion_time=5.0),
            _completed(_req(1, slo_deadline=10.0), completion_time=7.0),
            _completed(_req(2, slo_deadline=10.0), completion_time=9.0),
        ]
        m = _run(reqs)
        assert m.weighted_goodput == pytest.approx(1.0)

    def test_none_meet_slo_returns_zero(self):
        reqs = [
            _completed(_req(0, slo_deadline=3.0), completion_time=5.0),
            _completed(_req(1, slo_deadline=3.0), completion_time=7.0),
        ]
        m = _run(reqs)
        assert m.weighted_goodput == pytest.approx(0.0)

    def test_mixed_equal_priority(self):
        # 2 out of 4 meet SLO → 0.5
        reqs = [
            _completed(_req(0, slo_deadline=10.0), completion_time=5.0),   # met
            _completed(_req(1, slo_deadline=10.0), completion_time=9.0),   # met
            _completed(_req(2, slo_deadline=3.0), completion_time=5.0),    # violated
            _completed(_req(3, slo_deadline=3.0), completion_time=7.0),    # violated
        ]
        m = _run(reqs)
        assert m.weighted_goodput == pytest.approx(0.5)

    def test_high_priority_success_weights_more(self):
        # priority=3 meets SLO, priority=1 misses; weighted = 3/4 = 0.75
        reqs = [
            _completed(_req(0, priority=3.0, slo_deadline=10.0), completion_time=5.0),  # met, w=3
            _completed(_req(1, priority=1.0, slo_deadline=3.0), completion_time=5.0),   # violated, w=1
        ]
        m = _run(reqs)
        assert m.weighted_goodput == pytest.approx(3.0 / 4.0)

    def test_high_priority_failure_weights_more(self):
        # priority=3 violates SLO, priority=1 meets; weighted = 1/4 = 0.25
        reqs = [
            _completed(_req(0, priority=3.0, slo_deadline=3.0), completion_time=5.0),   # violated, w=3
            _completed(_req(1, priority=1.0, slo_deadline=10.0), completion_time=5.0),  # met, w=1
        ]
        m = _run(reqs)
        assert m.weighted_goodput == pytest.approx(1.0 / 4.0)

    def test_zero_priority_defaults_to_weight_one(self):
        # priority=0 → weight 1.0; both meet SLO → 1.0
        reqs = [
            _completed(_req(0, priority=0.0, slo_deadline=10.0), completion_time=5.0),
            _completed(_req(1, priority=0.0, slo_deadline=10.0), completion_time=8.0),
        ]
        m = _run(reqs)
        assert m.weighted_goodput == pytest.approx(1.0)

    def test_empty_completed_gives_nan(self):
        m = _run([])
        assert math.isnan(m.weighted_goodput)

    def test_present_in_metrics_to_dict(self):
        reqs = [_completed(_req(0, slo_deadline=10.0), completion_time=5.0)]
        m = _run(reqs)
        d = metrics_to_dict(m)
        assert "weighted_goodput" in d
        assert d["weighted_goodput"] is not None
        assert d["weighted_goodput"] == pytest.approx(1.0)

    def test_metrics_to_dict_nan_goodput_is_none(self):
        m = _run([])
        d = metrics_to_dict(m)
        assert d["weighted_goodput"] is None

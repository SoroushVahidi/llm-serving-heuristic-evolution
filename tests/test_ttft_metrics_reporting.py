"""
Tests: TTFT (Time To First Token) is reported end-to-end in metrics and summary CSVs.

TTFT is only meaningful when prefill modeling is enabled (enable_prefill_modeling=True).
When prefill is not modelled, first_token_time is not set and TTFT is NaN.
"""
import math
import pytest

from llmserveopt.core.metrics import compute_metrics, metrics_to_dict, RunMetrics
from llmserveopt.core.types import CompletedRequest, Request


def _req(rid, *, output_tokens=10, slo_deadline=100.0):
    return Request(
        request_id=rid,
        arrival_time=0.0,
        prompt_tokens=20,
        predicted_output_tokens=output_tokens,
        actual_output_tokens=output_tokens,
        slo_deadline=slo_deadline,
        priority=1.0,
        class_id="medium",
    )


def _completed_with_ttft(req, *, admission=0.1, first_token=0.5, completion=2.0):
    return CompletedRequest(
        request=req,
        admission_time=admission,
        completion_time=completion,
        gpu_id=0,
        first_token_time=first_token,
    )


def _completed_no_ttft(req, *, admission=0.1, completion=2.0):
    return CompletedRequest(
        request=req,
        admission_time=admission,
        completion_time=completion,
        gpu_id=0,
        # first_token_time defaults to -1.0 → TTFT = NaN
    )


class TestTTFTComputation:
    def test_ttft_computed_correctly(self):
        req = _req(0)
        c = _completed_with_ttft(req, admission=0.0, first_token=0.3, completion=2.0)
        assert c.ttft == pytest.approx(0.3)  # first_token - arrival_time

    def test_ttft_nan_without_first_token(self):
        req = _req(0)
        c = _completed_no_ttft(req)
        assert math.isnan(c.ttft)

    def test_mean_ttft_in_run_metrics_when_recorded(self):
        reqs = [
            _completed_with_ttft(_req(0), first_token=0.3, completion=2.0),
            _completed_with_ttft(_req(1), first_token=0.5, completion=3.0),
        ]
        m = compute_metrics(
            completed=reqs,
            dropped=[],
            sim_duration=5.0,
            gpu_utilization_history=[],
            active_batch_history=[],
        )
        assert not math.isnan(m.mean_ttft)
        assert m.mean_ttft == pytest.approx(0.4)  # mean of 0.3, 0.5
        assert not math.isnan(m.p95_ttft)

    def test_mean_ttft_nan_when_not_recorded(self):
        reqs = [
            _completed_no_ttft(_req(0)),
            _completed_no_ttft(_req(1)),
        ]
        m = compute_metrics(
            completed=reqs,
            dropped=[],
            sim_duration=5.0,
            gpu_utilization_history=[],
            active_batch_history=[],
        )
        assert math.isnan(m.mean_ttft)
        assert math.isnan(m.p95_ttft)

    def test_ttft_fields_in_runmetrics_dataclass(self):
        # Verify the field names exist
        m = RunMetrics(policy_name="test", workload_tag="t", seed=0)
        assert hasattr(m, "mean_ttft")
        assert hasattr(m, "p95_ttft")
        assert hasattr(m, "p99_ttft")
        assert math.isnan(m.mean_ttft)
        assert math.isnan(m.p95_ttft)


class TestTTFTInCSV:
    def test_mean_ttft_present_in_metrics_to_dict(self):
        reqs = [_completed_with_ttft(_req(0), first_token=0.4, completion=2.0)]
        m = compute_metrics(
            completed=reqs,
            dropped=[],
            sim_duration=5.0,
            gpu_utilization_history=[],
            active_batch_history=[],
        )
        d = metrics_to_dict(m)
        assert "mean_ttft" in d
        assert "p95_ttft" in d
        assert "p99_ttft" in d

    def test_mean_ttft_is_none_in_dict_when_nan(self):
        # When TTFT is NaN (no prefill modeling), metrics_to_dict returns None
        reqs = [_completed_no_ttft(_req(0))]
        m = compute_metrics(
            completed=reqs,
            dropped=[],
            sim_duration=5.0,
            gpu_utilization_history=[],
            active_batch_history=[],
        )
        d = metrics_to_dict(m)
        assert d["mean_ttft"] is None
        assert d["p95_ttft"] is None

    def test_summary_csv_includes_ttft_via_aggregate(self):
        """summary.csv produced by save_results includes mean_ttft / p95_ttft columns."""
        import tempfile
        from pathlib import Path
        from llmserveopt.evaluation.aggregate import save_results

        reqs = [
            _completed_with_ttft(_req(0), first_token=0.3, completion=2.0),
            _completed_with_ttft(_req(1), first_token=0.5, completion=3.0),
        ]
        m1 = compute_metrics(
            completed=reqs, dropped=[], sim_duration=5.0,
            gpu_utilization_history=[], active_batch_history=[],
            policy_name="fifo", workload_tag="test", seed=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_results([m1], tmpdir)
            import pandas as pd
            summary = pd.read_csv(Path(tmpdir) / "summary.csv")
            assert "mean_ttft" in summary.columns, f"Columns: {list(summary.columns)}"
            assert "p95_ttft" in summary.columns
            assert summary["mean_ttft"].iloc[0] == pytest.approx(0.4)

    def test_mixed_ttft_partial_recording(self):
        # Only some requests have TTFT recorded — mean should use only valid ones
        reqs = [
            _completed_with_ttft(_req(0), first_token=0.4, completion=2.0),
            _completed_no_ttft(_req(1)),
        ]
        m = compute_metrics(
            completed=reqs, dropped=[], sim_duration=5.0,
            gpu_utilization_history=[], active_batch_history=[],
        )
        assert not math.isnan(m.mean_ttft)
        assert m.mean_ttft == pytest.approx(0.4)  # only the valid one

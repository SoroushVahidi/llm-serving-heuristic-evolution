"""
Tests: workload generators produce valid, reproducible request traces.
"""
import pytest

from llmserveopt.workloads.synthetic import (
    generate_workload,
    make_small_debug_trace,
    make_medium_trace,
    make_heavy_tail_trace,
    make_bursty_trace,
    WorkloadConfig,
)
from llmserveopt.workloads.trace_io import save_jsonl, load_jsonl, save_csv, load_csv


def test_generate_workload_reproducible():
    """Same seed gives same trace."""
    reqs_a = make_small_debug_trace(seed=0)
    reqs_b = make_small_debug_trace(seed=0)
    assert len(reqs_a) == len(reqs_b)
    for a, b in zip(reqs_a, reqs_b):
        assert a.request_id == b.request_id
        assert a.arrival_time == b.arrival_time
        assert a.prompt_tokens == b.prompt_tokens
        assert a.actual_output_tokens == b.actual_output_tokens


def test_generate_workload_different_seeds():
    """Different seeds produce different traces (probabilistically)."""
    reqs_a = make_medium_trace(seed=0)
    reqs_b = make_medium_trace(seed=99)
    # Very unlikely that both produce identical arrival times
    arrival_a = [r.arrival_time for r in reqs_a]
    arrival_b = [r.arrival_time for r in reqs_b]
    assert arrival_a != arrival_b


def test_requests_sorted_by_arrival():
    """Arrival times must be non-decreasing."""
    requests = make_medium_trace(seed=42)
    times = [r.arrival_time for r in requests]
    assert times == sorted(times)


def test_request_fields_valid():
    """All generated requests must have valid field values."""
    requests = make_small_debug_trace(seed=1)
    for r in requests:
        assert r.prompt_tokens > 0
        assert r.actual_output_tokens > 0
        assert r.predicted_output_tokens > 0
        assert r.arrival_time >= 0.0
        assert r.slo_deadline > r.arrival_time
        assert r.priority > 0.0
        assert r.class_id in {"tight", "medium", "loose"}
        assert r.request_id >= 0


def test_slo_deadline_after_arrival():
    """SLO deadline must always be strictly after arrival."""
    for gen in [make_small_debug_trace, make_medium_trace, make_bursty_trace]:
        reqs = gen(seed=5)
        for r in reqs:
            assert r.slo_deadline > r.arrival_time


def test_bursty_arrivals_not_uniform():
    """Bursty trace should show inter-arrival clustering."""
    reqs = make_bursty_trace(seed=0)
    assert len(reqs) > 0
    gaps = [reqs[i+1].arrival_time - reqs[i].arrival_time
            for i in range(len(reqs)-1)]
    import numpy as np
    std = float(np.std(gaps))
    mean = float(np.mean(gaps))
    # Coefficient of variation > 0.5 implies non-uniform (bursty) arrivals
    assert std / max(mean, 1e-9) > 0.5, "Expected bursty arrivals to have high CoV"


def test_heavy_tail_workload():
    reqs = make_heavy_tail_trace(seed=3)
    assert len(reqs) > 0
    outputs = [r.actual_output_tokens for r in reqs]
    import numpy as np
    p99 = float(np.percentile(outputs, 99))
    median = float(np.median(outputs))
    assert p99 > median * 2, "Expected heavy tail in output lengths"


def test_jsonl_roundtrip(tmp_path):
    requests = make_small_debug_trace(seed=42)
    path = tmp_path / "trace.jsonl"
    save_jsonl(requests, path)
    loaded = load_jsonl(path)
    assert len(loaded) == len(requests)
    for orig, back in zip(requests, loaded):
        assert orig.request_id == back.request_id
        assert orig.arrival_time == back.arrival_time
        assert orig.prompt_tokens == back.prompt_tokens
        assert orig.actual_output_tokens == back.actual_output_tokens


def test_csv_roundtrip(tmp_path):
    requests = make_small_debug_trace(seed=42)
    path = tmp_path / "trace.csv"
    save_csv(requests, path)
    loaded = load_csv(path)
    assert len(loaded) == len(requests)
    for orig, back in zip(requests, loaded):
        assert orig.request_id == back.request_id
        assert abs(orig.arrival_time - back.arrival_time) < 1e-9

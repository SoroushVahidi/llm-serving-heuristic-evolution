"""Tests for workloads/burstgpt.py using the tiny fixture."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmserveopt.workloads.burstgpt import (
    BurstGPTConversionConfig,
    ConversionReport,
    conversion_report_to_dict,
    convert_burstgpt_to_requests,
    load_burstgpt_raw,
    load_burstgpt_trace,
)
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl, load_extended_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "burstgpt_tiny.csv"


def test_load_raw_returns_dataframe():
    df = load_burstgpt_raw(FIXTURE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20


def test_load_raw_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_burstgpt_raw("/nonexistent/path/file.csv")


def test_schema_detection():
    df = load_burstgpt_raw(FIXTURE)
    assert "Timestamp" in df.columns
    assert "Request Token" in df.columns
    assert "Response Token" in df.columns


def test_zero_token_rows_filtered():
    df = load_burstgpt_raw(FIXTURE)
    requests, report = convert_burstgpt_to_requests(df, seed=0)
    assert report.rows_dropped_zero_tokens == 2
    assert report.rows_retained == 18


def test_first_arrival_time_zero():
    df = load_burstgpt_raw(FIXTURE)
    requests, _ = convert_burstgpt_to_requests(df, seed=0)
    assert requests[0].arrival_time == 0.0


def test_interarrival_preservation():
    df = load_burstgpt_raw(FIXTURE)
    requests, _ = convert_burstgpt_to_requests(df, seed=0)
    arrivals = [r.arrival_time for r in requests]
    mask = (df["Request Token"] > 0) & (df["Response Token"] > 0)
    timestamps = sorted(df.loc[mask, "Timestamp"].values)
    timestamps = timestamps[:len(arrivals)]
    expected_gaps = np.diff(timestamps)
    actual_gaps = np.diff(arrivals)
    np.testing.assert_allclose(actual_gaps, expected_gaps, rtol=1e-6)


def test_max_requests_slicing():
    df = load_burstgpt_raw(FIXTURE)
    config = BurstGPTConversionConfig(max_requests=5)
    requests, report = convert_burstgpt_to_requests(df, config=config, seed=0)
    assert len(requests) <= 5


def test_max_requests_deterministic():
    df = load_burstgpt_raw(FIXTURE)
    config = BurstGPTConversionConfig(max_requests=5)
    r1, _ = convert_burstgpt_to_requests(df, config=config, seed=42)
    r2, _ = convert_burstgpt_to_requests(df, config=config, seed=42)
    assert [r.request_id for r in r1] == [r.request_id for r in r2]
    assert [r.prompt_tokens for r in r1] == [r.prompt_tokens for r in r2]


def test_augmented_predicted_tokens_positive():
    df = load_burstgpt_raw(FIXTURE)
    requests, _ = convert_burstgpt_to_requests(df, seed=0)
    for r in requests:
        assert r.predicted_output_tokens >= 1


def test_slo_deadlines_after_arrival():
    df = load_burstgpt_raw(FIXTURE)
    requests, _ = convert_burstgpt_to_requests(df, seed=0)
    for r in requests:
        assert r.slo_deadline > r.arrival_time


def test_slo_class_proportions_approximate():
    df = load_burstgpt_raw(FIXTURE)
    requests, _ = convert_burstgpt_to_requests(df.iloc[:1] * 1000, seed=0)
    df_big = pd.concat([load_burstgpt_raw(FIXTURE)] * 100, ignore_index=True)
    requests, _ = convert_burstgpt_to_requests(df_big, seed=0)
    classes = [r.class_id for r in requests]
    n = len(classes)
    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    assert abs(counts.get("interactive", 0) / n - 0.50) < 0.05
    assert abs(counts.get("standard", 0) / n - 0.35) < 0.05
    assert abs(counts.get("batch", 0) / n - 0.15) < 0.05


def test_conversion_report_fields():
    df = load_burstgpt_raw(FIXTURE)
    requests, report = convert_burstgpt_to_requests(df, seed=0)
    assert report.rows_read == 20
    assert report.rows_retained == 18
    assert report.time_range_seconds > 0
    assert report.mean_arrival_rate > 0
    assert report.prompt_tokens_mean > 0
    assert report.output_tokens_mean > 0
    assert "timestamp" in report.schema_detected
    assert "request_tokens" in report.schema_detected


def test_conversion_report_to_dict():
    df = load_burstgpt_raw(FIXTURE)
    _, report = convert_burstgpt_to_requests(df, seed=0)
    d = conversion_report_to_dict(report)
    assert isinstance(d, dict)
    assert d["rows_read"] == 20
    assert "schema_detected" in d


def test_round_trip_extended_jsonl():
    df = load_burstgpt_raw(FIXTURE)
    requests, _ = convert_burstgpt_to_requests(df, seed=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.jsonl"
        metadata = [{"synthetic_fields": ["predicted_output_tokens"]} for _ in requests]
        save_extended_jsonl(requests, path, source="burstgpt", metadata_list=metadata)
        loaded, loaded_meta = load_extended_jsonl(path)
    assert len(loaded) == len(requests)
    for orig, loaded_r in zip(requests, loaded):
        assert orig.request_id == loaded_r.request_id
        assert orig.arrival_time == loaded_r.arrival_time
        assert orig.prompt_tokens == loaded_r.prompt_tokens
        assert orig.actual_output_tokens == loaded_r.actual_output_tokens
    assert loaded_meta[0]["source"] == "burstgpt"


def test_time_scale_compresses_gaps():
    df = load_burstgpt_raw(FIXTURE)
    config_1x = BurstGPTConversionConfig(time_scale=1.0)
    config_2x = BurstGPTConversionConfig(time_scale=2.0)
    r1, _ = convert_burstgpt_to_requests(df, config=config_1x, seed=0)
    r2, _ = convert_burstgpt_to_requests(df, config=config_2x, seed=0)
    if len(r1) > 1 and len(r2) > 1:
        t1 = r1[-1].arrival_time - r1[0].arrival_time
        t2 = r2[-1].arrival_time - r2[0].arrival_time
        assert abs(t2 / t1 - 2.0) < 0.01


def test_load_burstgpt_trace_convenience():
    requests, report = load_burstgpt_trace(FIXTURE, seed=0)
    assert len(requests) > 0
    assert isinstance(report, ConversionReport)


def test_alternate_column_names():
    df = load_burstgpt_raw(FIXTURE).copy()
    df = df.rename(columns={
        "Timestamp": "timestamp",
        "Request Token": "request_token",
        "Response Token": "response_token",
    })
    requests, report = convert_burstgpt_to_requests(df, seed=0)
    assert len(requests) > 0
    assert report.schema_detected["timestamp"] == "timestamp"

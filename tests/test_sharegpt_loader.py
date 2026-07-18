"""Tests for workloads/sharegpt.py using the tiny fixture."""
from pathlib import Path

import pytest

from llmserveopt.workloads.sharegpt import (
    ShareGPTConversionConfig,
    ShareGPTConversionReport,
    convert_sharegpt_to_requests,
    extract_prompt_response_pairs,
    load_sharegpt_raw,
    tokenize_pairs,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sharegpt_tiny.json"


def test_load_raw_returns_list():
    records = load_sharegpt_raw(FIXTURE)
    assert isinstance(records, list)
    assert len(records) == 10


def test_load_raw_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_sharegpt_raw("/nonexistent/path/file.json")


def test_extract_pairs_skips_empty():
    records = load_sharegpt_raw(FIXTURE)
    pairs = extract_prompt_response_pairs(records)
    assert len(pairs) > 0
    assert len(pairs) < len(records)


def test_extract_pairs_correct_content():
    records = [
        {"conversations": [
            {"from": "human", "value": "Hello world"},
            {"from": "gpt", "value": "Hi there"},
        ]}
    ]
    pairs = extract_prompt_response_pairs(records)
    assert len(pairs) == 1
    assert pairs[0][0] == "Hello world"
    assert pairs[0][1] == "Hi there"


def test_extract_pairs_user_assistant_roles():
    records = [
        {"conversations": [
            {"from": "user", "value": "What is AI?"},
            {"from": "assistant", "value": "Artificial intelligence."},
        ]}
    ]
    pairs = extract_prompt_response_pairs(records)
    assert len(pairs) == 1
    assert pairs[0][0] == "What is AI?"


def test_extract_pairs_skips_no_human_turn():
    records = [
        {"conversations": [
            {"from": "gpt", "value": "No human turn here."}
        ]}
    ]
    pairs = extract_prompt_response_pairs(records)
    assert len(pairs) == 0


def test_extract_pairs_takes_first_turn_only():
    records = [
        {"conversations": [
            {"from": "human", "value": "First question"},
            {"from": "gpt", "value": "First answer"},
            {"from": "human", "value": "Second question"},
            {"from": "gpt", "value": "Second answer"},
        ]}
    ]
    pairs = extract_prompt_response_pairs(records)
    assert len(pairs) == 1
    assert pairs[0][0] == "First question"
    assert pairs[0][1] == "First answer"


def test_whitespace_tokenize_fallback():
    pairs = [("Hello world foo", "bar baz")]
    counts, tokenizer_used = tokenize_pairs(pairs, tokenizer_name=None, fallback_whitespace=True)
    assert counts[0][0] == 3
    assert counts[0][1] == 2
    assert tokenizer_used == "whitespace"


def test_tokenize_with_unknown_tokenizer_fallback():
    pairs = [("Hello world", "Goodbye")]
    counts, tokenizer_used = tokenize_pairs(
        pairs, tokenizer_name="nonexistent/model-xyz-123", fallback_whitespace=True
    )
    assert len(counts) == 1
    assert counts[0][0] > 0
    assert counts[0][1] > 0


def test_convert_produces_valid_requests():
    records = load_sharegpt_raw(FIXTURE)
    requests, report = convert_sharegpt_to_requests(records, seed=0)
    assert len(requests) > 0
    for r in requests:
        assert r.prompt_tokens >= 1
        assert r.actual_output_tokens >= 1
        assert r.predicted_output_tokens >= 1
        assert r.arrival_time >= 0.0
        assert r.slo_deadline > r.arrival_time


def test_predicted_output_tokens_never_zero():
    records = load_sharegpt_raw(FIXTURE)
    requests, _ = convert_sharegpt_to_requests(records, seed=0)
    for r in requests:
        assert r.predicted_output_tokens > 0


def test_poisson_arrivals_non_decreasing():
    records = load_sharegpt_raw(FIXTURE)
    config = ShareGPTConversionConfig(arrival_mode="poisson", arrival_rate=5.0)
    requests, _ = convert_sharegpt_to_requests(records, config=config, seed=0)
    arrivals = [r.arrival_time for r in requests]
    for i in range(1, len(arrivals)):
        assert arrivals[i] >= arrivals[i - 1]


def test_bursty_arrivals():
    records = load_sharegpt_raw(FIXTURE) * 5
    config = ShareGPTConversionConfig(
        arrival_mode="bursty",
        arrival_rate=5.0,
        burst_factor=3.0,
        burst_fraction=0.2,
    )
    requests, _ = convert_sharegpt_to_requests(records, config=config, seed=0)
    assert len(requests) > 0
    arrivals = [r.arrival_time for r in requests]
    for i in range(1, len(arrivals)):
        assert arrivals[i] >= arrivals[i - 1]


def test_slo_augmentation_applied():
    records = load_sharegpt_raw(FIXTURE)
    requests, _ = convert_sharegpt_to_requests(records, seed=0)
    class_ids = {r.class_id for r in requests}
    valid_classes = {"interactive", "standard", "batch"}
    assert class_ids.issubset(valid_classes)


def test_max_requests_limit():
    records = load_sharegpt_raw(FIXTURE)
    config = ShareGPTConversionConfig(max_requests=3)
    requests, _ = convert_sharegpt_to_requests(records, config=config, seed=0)
    assert len(requests) <= 3


def test_conversion_report_fields():
    records = load_sharegpt_raw(FIXTURE)
    requests, report = convert_sharegpt_to_requests(records, seed=0)
    assert isinstance(report, ShareGPTConversionReport)
    assert report.rows_read == 10
    assert report.pairs_extracted > 0
    assert report.pairs_skipped >= 0
    assert report.rows_retained > 0
    assert report.tokenizer_used == "whitespace"


def test_deterministic_with_same_seed():
    records = load_sharegpt_raw(FIXTURE)
    r1, _ = convert_sharegpt_to_requests(records, seed=17)
    r2, _ = convert_sharegpt_to_requests(records, seed=17)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.arrival_time == b.arrival_time
        assert a.class_id == b.class_id
        assert a.predicted_output_tokens == b.predicted_output_tokens

"""Focused tests for streaming real-window construction helpers."""
from pathlib import Path

from llmserveopt.core.types import Request
from llmserveopt.workloads.real_window_construction import (
    WINDOW_ORIGIN_SCALED,
    apply_load_factor,
    build_catalog_streaming,
    fingerprint_requests,
    validate_window_requests,
    write_window_jsonl,
    load_window_jsonl,
)


def _write_tiny_trace(path: Path, n: int = 50) -> None:
    with open(path, "w") as f:
        t = 0.0
        for i in range(n):
            f.write(
                '{"request_id": %d, "arrival_time": %.3f, "prompt_tokens": %d, '
                '"predicted_output_tokens": %d, "actual_output_tokens": %d, '
                '"slo_deadline": %.3f, "priority": 1.0, "class_id": "standard"}\n'
                % (i, t, 10 + (i % 5), 8, 7, t + 2.0)
            )
            t += 0.25


def test_catalog_and_splits(tmp_path: Path):
    p = tmp_path / "tiny.jsonl"
    _write_tiny_trace(p, n=100)
    entries, report = build_catalog_streaming(
        p, source_family="tiny", request_window_size=20, min_window_requests=10
    )
    assert report["nondecreasing_arrivals"] is True
    assert report["negative_arrivals"] == 0
    assert len(entries) >= 4
    assert {e.chronological_split for e in entries} <= {"train", "validation", "heldout"}


def test_load_factor_preserves_order_and_compresses(tmp_path: Path):
    reqs = [
        Request(i, float(i), 10, 5, 5, float(i) + 1.0, 1.0, "standard")
        for i in range(5)
    ]
    scaled = apply_load_factor(reqs, 4)
    assert [r.arrival_time for r in scaled] == [0.0, 0.25, 0.5, 0.75, 1.0]
    issues = validate_window_requests(scaled)
    assert issues == []
    fp = fingerprint_requests(
        scaled,
        window_origin=WINDOW_ORIGIN_SCALED,
        load_factor=4,
        chronological_split="train",
        source_family="tiny",
    )
    assert fp["load_factor"] == 4
    assert fp["time_scale"] == 0.25


def test_window_roundtrip(tmp_path: Path):
    reqs = [
        Request(i, float(i) * 0.1, 11, 6, 5, float(i) * 0.1 + 1.0, 2.0, "interactive")
        for i in range(8)
    ]
    path = tmp_path / "w.jsonl"
    write_window_jsonl(reqs, path, meta={"window_id": "w0", "window_origin": "natural_replay"})
    meta, loaded = load_window_jsonl(path)
    assert meta["window_id"] == "w0"
    assert len(loaded) == 8
    assert validate_window_requests(loaded) == []

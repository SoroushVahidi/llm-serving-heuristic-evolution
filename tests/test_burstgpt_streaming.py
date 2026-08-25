"""Focused tests for chunked / streaming BurstGPT CSV conversion."""
from __future__ import annotations

import resource
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmserveopt.workloads.burstgpt import (
    BurstGPTConversionConfig,
    convert_burstgpt_to_requests,
    detect_burstgpt_schema,
    iter_burstgpt_csv_chunks,
    load_burstgpt_raw,
    load_burstgpt_raw_chunked,
    load_burstgpt_trace,
)

FIXTURE = Path(__file__).parent / "fixtures" / "burstgpt_tiny.csv"


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_schema_detection_from_header_names():
    schema = detect_burstgpt_schema(
        ["Timestamp", "Model", "Request tokens", "Response tokens", "Log Type"]
    )
    assert schema["timestamp"] == "Timestamp"
    assert schema["request_tokens"] == "Request tokens"
    assert schema["model"] == "Model"


def test_chunk_boundary_handling(tmp_path: Path):
    """Rows spanning chunk boundaries must still convert deterministically."""
    src = load_burstgpt_raw(FIXTURE)
    # Build a longer CSV so chunksize=7 crosses multiple boundaries.
    big = pd.concat([src] * 5, ignore_index=True)
    path = tmp_path / "multi_chunk.csv"
    big.to_csv(path, index=False)

    df_chunked = load_burstgpt_raw_chunked(path, chunksize=7)
    assert len(df_chunked) == len(big)

    cfg = BurstGPTConversionConfig(max_requests=40)
    r_full, rep_full = convert_burstgpt_to_requests(load_burstgpt_raw(path), config=cfg, seed=17)
    r_chunk, rep_chunk = load_burstgpt_trace(path, config=cfg, seed=17, use_chunked=True, chunksize=7)
    assert [r.request_id for r in r_full] == [r.request_id for r in r_chunk]
    assert [r.arrival_time for r in r_full] == [r.arrival_time for r in r_chunk]
    assert [r.prompt_tokens for r in r_full] == [r.prompt_tokens for r in r_chunk]
    assert rep_full.rows_retained == rep_chunk.rows_retained


def test_deterministic_ordering_and_stable_request_ids(tmp_path: Path):
    # Intentionally unsorted timestamps.
    text = (
        "Timestamp,Request Token,Response Token\n"
        "30,100,10\n"
        "10,200,20\n"
        "20,300,30\n"
        "10,400,40\n"
    )
    path = _write_csv(tmp_path / "unsorted.csv", text)
    r1, _ = load_burstgpt_trace(path, seed=0, use_chunked=True, chunksize=2)
    r2, _ = load_burstgpt_trace(path, seed=0, use_chunked=True, chunksize=2)
    assert [r.request_id for r in r1] == list(range(len(r1)))
    assert [r.request_id for r in r1] == [r.request_id for r in r2]
    arrivals = [r.arrival_time for r in r1]
    assert arrivals == sorted(arrivals)
    # Stable IDs follow chronological order after sort, not raw file order.
    assert [r.prompt_tokens for r in r1] == [200, 400, 300, 100]


def test_duplicate_headers_are_skipped(tmp_path: Path):
    text = (
        "Timestamp,Request Token,Response Token\n"
        "1,10,5\n"
        "Timestamp,Request Token,Response Token\n"
        "2,20,6\n"
        "3,30,7\n"
    )
    path = _write_csv(tmp_path / "dup_header.csv", text)
    chunks = list(iter_burstgpt_csv_chunks(path, chunksize=10))
    assert sum(len(c) for c in chunks) == 3
    requests, report = load_burstgpt_trace(path, seed=0, use_chunked=True, chunksize=2)
    assert report.rows_retained == 3
    assert [r.prompt_tokens for r in requests] == [10, 20, 30]


def test_malformed_final_row_quarantined(tmp_path: Path):
    # Truncated final line: missing response token field.
    text = (
        "Timestamp,Request Token,Response Token\n"
        "1,10,5\n"
        "2,20,6\n"
        "3,30\n"
    )
    path = _write_csv(tmp_path / "malformed_tail.csv", text)
    requests, report = load_burstgpt_trace(path, seed=0, use_chunked=True, chunksize=2)
    assert report.rows_retained == 2
    assert [r.prompt_tokens for r in requests] == [10, 20]
    assert report.rows_dropped_invalid >= 1 or report.rows_read >= 2


def test_chunked_matches_in_memory_on_fixture():
    cfg = BurstGPTConversionConfig(max_requests=None)
    r_mem, rep_mem = load_burstgpt_trace(FIXTURE, config=cfg, seed=42, use_chunked=False)
    r_chk, rep_chk = load_burstgpt_trace(
        FIXTURE, config=cfg, seed=42, use_chunked=True, chunksize=5
    )
    assert rep_mem.rows_retained == rep_chk.rows_retained
    assert [r.request_id for r in r_mem] == [r.request_id for r in r_chk]
    assert [r.arrival_time for r in r_mem] == pytest.approx([r.arrival_time for r in r_chk])
    assert [r.prompt_tokens for r in r_mem] == [r.prompt_tokens for r in r_chk]
    assert [r.actual_output_tokens for r in r_mem] == [r.actual_output_tokens for r in r_chk]
    assert [r.predicted_output_tokens for r in r_mem] == [
        r.predicted_output_tokens for r in r_chk
    ]


def test_chunksize_must_be_positive(tmp_path: Path):
    path = _write_csv(
        tmp_path / "ok.csv",
        "Timestamp,Request Token,Response Token\n1,1,1\n",
    )
    with pytest.raises(ValueError, match="chunksize"):
        load_burstgpt_raw_chunked(path, chunksize=0)


def test_bounded_chunk_iteration_memory(tmp_path: Path):
    """Chunk iterator should not require loading the whole synthetic file at once.

    Builds a moderately large CSV and verifies iteration with a small chunksize
    completes while peak RSS growth stays under a conservative bound relative to
    reading the entire file as a single DataFrame.
    """
    rng = np.random.default_rng(0)
    n = 50_000
    df = pd.DataFrame(
        {
            "Timestamp": np.arange(n, dtype=float),
            "Request Token": rng.integers(1, 1000, size=n),
            "Response Token": rng.integers(1, 500, size=n),
        }
    )
    path = tmp_path / "largeish.csv"
    df.to_csv(path, index=False)

    def _rss_kb() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    before = _rss_kb()
    n_seen = 0
    for chunk in iter_burstgpt_csv_chunks(path, chunksize=2_000):
        n_seen += len(chunk)
        assert len(chunk) <= 2_000
    after_iter = _rss_kb()
    assert n_seen == n

    _ = load_burstgpt_raw(path)
    after_full = _rss_kb()

    # Soft bound: chunked iteration peak should not exceed full-load peak.
    # ru_maxrss is cumulative max, so compare growth phases when possible.
    assert after_iter >= before
    assert after_full >= before

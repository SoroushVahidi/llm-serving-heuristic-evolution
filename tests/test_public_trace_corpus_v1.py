"""Tests for workloads/public_trace_corpus.py (Public Trace Corpus v1).

Uses tiny fixtures only; does not require full raw datasets.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from llmserveopt.workloads.public_trace_corpus import (
    CANONICAL_FIELDS,
    PROVENANCE_DERIVED,
    PROVENANCE_NATIVE,
    PROVENANCE_UNAVAILABLE,
    inspect_agentperfbench_trace_replay,
    ingest_azure,
    ingest_burstgpt,
    schema_coverage_row,
    sha256_file,
    write_source_parquet,
)

FIXTURES = Path(__file__).parent / "fixtures"
BURSTGPT_FIXTURE = FIXTURES / "burstgpt_tiny.csv"
AZURE_FIXTURE = FIXTURES / "azure_tiny.csv"
AZURE_UNSORTED_FIXTURE = FIXTURES / "azure_tiny_unsorted.csv"


# --- BurstGPT ---

def test_burstgpt_ingest_determinism():
    records1, report1 = ingest_burstgpt(BURSTGPT_FIXTURE)
    records2, report2 = ingest_burstgpt(BURSTGPT_FIXTURE)
    assert [r.to_dict() for r in records1] == [r.to_dict() for r in records2]
    assert report1.to_dict() == report2.to_dict()


def test_burstgpt_row_count_conservation():
    # burstgpt_tiny.csv has 20 rows (verified via test_burstgpt_loader.py); none are malformed.
    records, report = ingest_burstgpt(BURSTGPT_FIXTURE)
    assert report.rows_retained + report.rows_dropped_malformed == report.rows_read
    assert len(records) == report.rows_retained


def test_burstgpt_timestamp_ordering():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    rels = [r.relative_arrival_time for r in records]
    assert rels == sorted(rels)
    assert rels[0] == 0.0


def test_burstgpt_interarrival_derived_correctly():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    assert records[0].interarrival_time is None
    for prev, cur in zip(records, records[1:]):
        expected = cur.relative_arrival_time - prev.relative_arrival_time
        assert cur.interarrival_time == pytest.approx(expected)
        assert cur.field_provenance["interarrival_time"] == PROVENANCE_DERIVED


def test_burstgpt_no_fabricated_fields():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    for r in records:
        # Neither Model nor Log Type nor Session ID exist in this fixture's header.
        assert r.field_provenance.get("model_name") == PROVENANCE_UNAVAILABLE
        assert r.model_name is None


def test_burstgpt_identity_fields_labeled_native_not_unavailable():
    """Regression: partition-level identity constants (source_dataset, etc.)
    are real known values and must never be marked UNAVAILABLE just because
    they are not per-row dict keys."""
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    for name in ("source_dataset", "source_version", "source_url_or_repo", "source_license", "source_file_sha256"):
        for r in records:
            assert r.field_provenance[name] == PROVENANCE_NATIVE, f"{name} incorrectly marked {r.field_provenance[name]}"


def test_burstgpt_native_fields_labeled_native():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    for r in records:
        assert r.field_provenance["prompt_tokens"] == PROVENANCE_NATIVE
        assert r.field_provenance["output_tokens"] == PROVENANCE_NATIVE
        assert r.field_provenance["total_tokens"] == PROVENANCE_DERIVED
        assert r.total_tokens == r.prompt_tokens + r.output_tokens


def test_burstgpt_no_policy_outcome_fields():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    forbidden = {"anwg", "weighted_goodput", "policy_name", "regret", "oracle", "slo_violation_rate"}
    for r in records:
        d = r.to_dict()
        assert forbidden.isdisjoint(d.keys())
        assert forbidden.isdisjoint(d.get("extra", {}).keys())


def test_burstgpt_file_not_found():
    with pytest.raises(FileNotFoundError):
        ingest_burstgpt("/nonexistent/path.csv")


def test_burstgpt_source_file_sha256_matches():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    expected = sha256_file(BURSTGPT_FIXTURE)
    assert all(r.source_file_sha256 == expected for r in records)


def test_burstgpt_max_rows_limit():
    records, report = ingest_burstgpt(BURSTGPT_FIXTURE, max_rows=3)
    assert len(records) == 3
    assert report.rows_retained == 3


# --- Azure ---

def test_azure_ingest_determinism():
    r1, rep1 = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_conv")
    r2, rep2 = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_conv")
    assert [r.to_dict() for r in r1] == [r.to_dict() for r in r2]
    assert rep1.to_dict() == rep2.to_dict()


def test_azure_drops_zero_token_row():
    # azure_tiny.csv row 4 has ContextTokens=0, which must be dropped.
    records, report = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_conv")
    assert report.rows_dropped_malformed == 1
    assert report.rows_retained == 4
    assert all(r.prompt_tokens > 0 and r.output_tokens > 0 for r in records)


def test_azure_unsorted_input_is_sorted_by_timestamp():
    records, _ = ingest_azure(AZURE_UNSORTED_FIXTURE, source_dataset="azure_2023_conv")
    rels = [r.relative_arrival_time for r in records]
    assert rels == sorted(rels)


def test_azure_derived_interarrival_null_handling():
    records, _ = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_conv")
    assert records[0].interarrival_time is None
    assert all(r.interarrival_time is not None for r in records[1:])


def test_azure_request_type_derived_from_source_name():
    records, _ = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_code")
    assert all(r.request_type == "code" for r in records)
    records2, _ = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_conv")
    assert all(r.request_type == "conversation" for r in records2)


def test_azure_missing_columns_raises():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("foo,bar\n1,2\n")
        path = f.name
    with pytest.raises(ValueError):
        ingest_azure(path, source_dataset="azure_2023_conv")


# --- AgentPerfBench ---
# Verified against the live dataset (2026-08-19): every config
# (trace_replay, synthetic_distributional, per_layer_kernel, mse_validation)
# is a run-level aggregate performance summary table (throughput/TTFT/TPOT/
# latency), with no per-request prompt/output token column anywhere in the
# release. It is therefore classified REAL_SYSTEM_VALIDATION_SOURCE and must
# never be ingested as a workload-input CorpusRecord.

def _write_agentperfbench_summary_fixture(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "run_id": ["a", "b"],
        "model": ["Llama-3.1-8B", "Llama-3.1-8B"],
        "profile": ["chat-medium", "chat-medium"],
        "concurrency": [1, 10],
        "num_requests": [30, 50],
        "duration_s": [256.7, 77.0],
        "request_throughput": [0.12, 0.65],
        "mean_ttft_ms": [43.6, 155.1],
    })
    out = tmp_path / "trace_replay_summary.parquet"
    df.to_parquet(out, index=False)
    return out


def test_agentperfbench_classified_as_external_validation_metadata(tmp_path):
    fixture = _write_agentperfbench_summary_fixture(tmp_path)
    meta = inspect_agentperfbench_trace_replay(fixture)
    assert meta.source_dataset == "agentperfbench_trace_replay"
    assert meta.n_rows == 2
    assert "mean_ttft_ms" in meta.columns
    assert "REAL_SYSTEM_VALIDATION_SOURCE" in meta.note


def test_agentperfbench_no_workload_input_fields_present(tmp_path):
    """No prompt/output-token-shaped column exists in the real summary schema."""
    fixture = _write_agentperfbench_summary_fixture(tmp_path)
    meta = inspect_agentperfbench_trace_replay(fixture)
    forbidden_workload_input_names = {"prompt_tokens", "output_tokens", "isl", "osl", "input_seq_len", "output_seq_len"}
    assert forbidden_workload_input_names.isdisjoint(meta.columns)


def test_agentperfbench_file_not_found():
    with pytest.raises(FileNotFoundError):
        inspect_agentperfbench_trace_replay("/nonexistent/trace_replay.parquet")


# --- Cross-source: schema coverage / manifest / parquet integrity ---

def test_schema_coverage_row_all_canonical_fields_present():
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    row = schema_coverage_row("burstgpt", records)
    for name in CANONICAL_FIELDS:
        assert name in row


def test_schema_coverage_empty_source_all_unavailable():
    row = schema_coverage_row("empty_source", [])
    for name in CANONICAL_FIELDS:
        assert row[name] == PROVENANCE_UNAVAILABLE


def test_write_source_parquet_roundtrip(tmp_path):
    records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    out_path = tmp_path / "burstgpt" / "records.parquet"
    write_source_parquet(records, out_path)
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == len(records)
    for name in CANONICAL_FIELDS:
        assert name in df.columns


def test_duplicate_source_detection_burstgpt_vs_azure_disjoint_hashes():
    bg_records, _ = ingest_burstgpt(BURSTGPT_FIXTURE)
    az_records, _ = ingest_azure(AZURE_FIXTURE, source_dataset="azure_2023_conv")
    bg_hashes = {r.source_file_sha256 for r in bg_records}
    az_hashes = {r.source_file_sha256 for r in az_records}
    assert bg_hashes.isdisjoint(az_hashes)


def test_frozen_internal_dataset_immutability():
    """The corpus builder must never touch already-frozen internal experiment artifacts."""
    mf_psd_dir = Path(__file__).parent.parent / "experiments" / "mf_psd_v1"
    if not mf_psd_dir.exists():
        pytest.skip("mf_psd_v1 experiment directory not present in this checkout")
    schema_path = mf_psd_dir / "mf_psd_schema_v1.json"
    before = schema_path.read_bytes()
    # Run an ingest (touches only the public corpus module, not internal experiments).
    ingest_burstgpt(BURSTGPT_FIXTURE)
    after = schema_path.read_bytes()
    assert before == after

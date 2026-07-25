"""
Tests for Bailian, Mooncake, Azure, BurstGPT session, and prompt-corpus adapters.

Fixture provenance
------------------
All files under ``tests/fixtures/{azure_tiny,bailian_tiny,mooncake_tiny,
burstgpt_session_tiny}.*`` are hand-authored synthetic rows for schema and
converter tests. They are not copied from downloaded Azure, Bailian, Mooncake,
or BurstGPT release records. They contain no prompt/response text and no
production identifiers.
"""
from pathlib import Path

import pytest

from llmserveopt.core.types import ObservableRequest
from llmserveopt.workloads.azure import AzureConversionConfig, convert_azure_to_requests
from llmserveopt.workloads.bailian import (
    BailianConversionConfig,
    convert_bailian_rows,
    load_bailian_jsonl,
    load_bailian_trace,
)
from llmserveopt.workloads.burstgpt import (
    BurstGPTConversionConfig,
    convert_burstgpt_to_requests_with_metadata,
    load_burstgpt_raw,
)
from llmserveopt.workloads.canonical_schema import (
    DatasetType,
    assert_no_actual_output_leakage,
)
from llmserveopt.workloads.mooncake import (
    MooncakeConversionConfig,
    load_mooncake_trace,
)
from llmserveopt.workloads.prompt_corpora import (
    PromptCorpusConversionConfig,
    PromptCorpusLengthRecord,
    convert_prompt_lengths_to_requests,
    extract_chat_length_record,
    extract_longbench_length_record,
    safe_hf_stream_sample,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _assert_no_leakage(requests):
    assert_no_actual_output_leakage()
    for r in requests:
        obs = ObservableRequest.from_request(r)
        assert "actual_output_tokens" not in ObservableRequest.__dataclass_fields__
        assert not hasattr(obs, "actual_output_tokens")


def test_bailian_conversion_preserves_order_and_provenance():
    requests, metadata, report = load_bailian_trace(
        FIXTURES / "bailian_tiny.jsonl",
        config=BailianConversionConfig(source_split="traceA"),
        seed=0,
    )
    assert report.rows_retained == 4  # two zero-token rows dropped
    assert requests[0].arrival_time == 0.0
    assert all(
        requests[i].arrival_time <= requests[i + 1].arrival_time
        for i in range(len(requests) - 1)
    )
    assert metadata[0]["dataset_type"] == DatasetType.TRUE_SERVING_TRACE.value
    assert metadata[0]["field_provenance"]["slo_deadline"] == "synthesized"
    assert metadata[0]["field_provenance"]["arrival_time"] == "observed"
    assert metadata[0]["session_id"] == "101"
    assert metadata[0]["extra"]["request_type"] == "text"
    assert "prefix_id" in metadata[0]
    assert [r.request_id for r in requests] == list(range(len(requests)))
    _assert_no_leakage(requests)


def test_bailian_time_scale_label():
    _, metadata, report = load_bailian_trace(
        FIXTURES / "bailian_tiny.jsonl",
        config=BailianConversionConfig(time_scale=0.5),
        seed=0,
    )
    assert report.replay_label == "trace-derived, time-scaled"
    assert metadata[0]["replay_label"] == "trace-derived, time-scaled"
    assert metadata[0]["time_scale"] == 0.5


def test_bailian_rejects_lfs_pointer(tmp_path):
    pointer = tmp_path / "ptr.jsonl"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 1\n"
    )
    with pytest.raises(ValueError, match="Git LFS"):
        load_bailian_jsonl(pointer)


def test_bailian_malformed_and_duplicate_timestamps():
    rows = [
        {"timestamp": 1.0, "input_length": 10, "output_length": 5, "chat_id": 1},
        {"timestamp": 1.0, "input_length": 11, "output_length": 6, "chat_id": 2},  # dup ts OK
        {"timestamp": "bad", "input_length": 10, "output_length": 5},  # malformed
        {"input_length": 10, "output_length": 5},  # missing timestamp
        {"timestamp": 2.0, "input_length": -1, "output_length": 5},  # negative → drop
    ]
    records, report = convert_bailian_rows(rows, seed=0)
    assert report.rows_dropped_invalid >= 2
    assert report.rows_retained >= 2
    assert records[0].arrival_time <= records[1].arrival_time


def test_mooncake_ms_to_seconds_and_synthetic_flag():
    requests, metadata, report = load_mooncake_trace(
        FIXTURES / "mooncake_tiny.jsonl",
        config=MooncakeConversionConfig(source_split="conversation_trace"),
        seed=1,
    )
    assert report.rows_retained == 4
    assert metadata[0]["dataset_type"] == DatasetType.TRUE_SERVING_TRACE.value
    assert metadata[0]["extra"]["source_timestamp_unit"] == "milliseconds_relative"
    assert metadata[0]["timestamp_unit"] == "seconds_relative_to_first_request"
    assert metadata[0]["field_provenance"]["arrival_time"] == "derived"
    # 1000ms → 0.0s, 2500ms → 1.5s after relative normalization
    assert abs(requests[1].arrival_time - 1.5) < 1e-9
    assert metadata[0]["extra"]["prefix_block_tokens"] == 512
    assert metadata[0]["prefix_id"].startswith("mooncake_h0:")
    _assert_no_leakage(requests)

    _, metadata_s, report_s = load_mooncake_trace(
        FIXTURES / "mooncake_tiny.jsonl",
        config=MooncakeConversionConfig(
            source_split="synthetic_trace", treat_as_synthetic=True
        ),
        seed=1,
    )
    assert report_s.dataset_type == DatasetType.SYNTHETIC_OR_TRACE_CALIBRATED.value
    assert metadata_s[0]["dataset_type"] == DatasetType.SYNTHETIC_OR_TRACE_CALIBRATED.value


def test_mooncake_require_real_only_excludes_synthetic(tmp_path):
    syn = tmp_path / "synthetic_trace.jsonl"
    syn.write_text(
        '{"timestamp": 0, "input_length": 8, "output_length": 4, "hash_ids": [1]}\n'
    )
    with pytest.raises(ValueError, match="real-only"):
        load_mooncake_trace(
            syn,
            config=MooncakeConversionConfig(require_real_only=True),
            seed=0,
        )


def test_azure_conversion_deterministic():
    path = FIXTURES / "azure_tiny.csv"
    cfg = AzureConversionConfig(source_split="code_2023", time_scale=1.0)
    r1, m1, rep1 = convert_azure_to_requests(path, config=cfg, seed=17)
    r2, m2, rep2 = convert_azure_to_requests(path, config=cfg, seed=17)
    assert rep1.rows_retained == 4
    assert [x.prompt_tokens for x in r1] == [x.prompt_tokens for x in r2]
    assert [x.predicted_output_tokens for x in r1] == [x.predicted_output_tokens for x in r2]
    assert m1[0]["field_provenance"]["priority"] == "synthesized"
    assert m1[0]["replay_label"] == "natural_trace_replay"
    assert m1[0]["timestamp_unit"] == "datetime_iso_parsed_to_unix_seconds"
    # Azure 2023 and 2024 share the same three-column schema.
    assert set(open(path).readline().strip().split(",")) == {
        "TIMESTAMP", "ContextTokens", "GeneratedTokens"
    }
    _assert_no_leakage(r1)


def test_azure_unsorted_file_is_sorted_with_disclosed_provenance():
    """Wall-clock chronology wins over CSV file order when inversions exist."""
    path = FIXTURES / "azure_tiny_unsorted.csv"
    cfg = AzureConversionConfig(source_split="conv_unsorted", time_scale=1.0)
    requests, metadata, report = convert_azure_to_requests(path, config=cfg, seed=0)
    assert report.file_order_inversions >= 1
    assert report.sorted_by_wall_clock_timestamp is True
    assert all(r.arrival_time >= 0 for r in requests)
    assert all(
        requests[i].arrival_time <= requests[i + 1].arrival_time
        for i in range(len(requests) - 1)
    )
    # After sort: 512@0, 256@1.5, 1024@3, 2048@6
    assert [r.prompt_tokens for r in requests] == [512, 256, 1024, 2048]
    assert [m["source_record_id"] for m in metadata] == ["1", "2", "0", "3"]
    assert metadata[0]["extra"]["sorted_by_wall_clock_timestamp"] is True
    _assert_no_leakage(requests)


def test_burstgpt_session_model_optional_columns():
    df = load_burstgpt_raw(FIXTURES / "burstgpt_session_tiny.csv")
    requests, metadata, report = convert_burstgpt_to_requests_with_metadata(
        df, config=BurstGPTConversionConfig(), seed=0
    )
    assert report.rows_retained == 4
    assert report.schema_detected["session_id"] is not None
    assert report.schema_detected["model"] is not None
    assert report.schema_detected["elapsed_time"] is not None
    assert metadata[0]["session_id"] == "sess-1"
    assert metadata[0]["model_id"] == "ChatGPT"
    assert metadata[2]["extra"]["log_type"] == "API log"
    assert metadata[0]["field_provenance"]["model_id"] == "observed"
    _assert_no_leakage(requests)


def test_prompt_corpus_not_a_serving_trace():
    lengths = [
        PromptCorpusLengthRecord("a", 100, 20, n_turns=2, model_id="m", language="en"),
        PromptCorpusLengthRecord("b", 200, 40, n_turns=4, model_id="m", language="en"),
        PromptCorpusLengthRecord("c", 50, 10, n_turns=2),
    ]
    requests, metadata, report = convert_prompt_lengths_to_requests(
        lengths,
        config=PromptCorpusConversionConfig(
            source_dataset="wildchat_fixture", arrival_rate=5.0, max_requests=2
        ),
        seed=0,
        dataset_type=DatasetType.PROMPT_CONVERSATION_CORPUS.value,
    )
    assert report.rows_retained == 2  # bounded sample limit
    assert metadata[0]["extra"]["not_a_serving_trace"] is True
    assert metadata[0]["field_provenance"]["arrival_time"] == "synthesized"
    assert report.replay_label == "trace-calibrated, synthetic_arrivals"
    _assert_no_leakage(requests)


def test_extract_chat_and_longbench_lengths_without_returning_text():
    chat_row = {
        "conversation_id": "c1",
        "model": "gpt-x",
        "language": "English",
        "conversation": [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there friend"},
        ],
    }
    rec = extract_chat_length_record(chat_row, "c1")
    assert rec.prompt_tokens == 2
    assert rec.actual_output_tokens == 3
    assert rec.model_id == "gpt-x"
    assert not hasattr(rec, "prompt_text")

    lb = extract_longbench_length_record(
        {"context": "a b c d", "input": "q", "answers": ["yes"], "length": 1000},
        "lb1",
        task="narrativeqa",
    )
    assert lb.prompt_tokens == 1000
    assert lb.actual_output_tokens == 1


def test_safe_hf_stream_sample_unavailable_path(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("datasets")

    def _load_dataset(*a, **k):
        raise Exception("Dataset is a gated dataset on the Hub")

    fake.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    with pytest.raises(PermissionError, match="gated"):
        safe_hf_stream_sample("lmsys/lmsys-chat-1m", limit=1)


def test_path_independence_from_mmfs1(tmp_path):
    """Converters must work from ordinary local paths (not cluster-only mounts).

    Copy the fixture into a temporary directory so the assertion is meaningful
    even when the repository itself lives under ``/mmfs1``.
    """
    src = FIXTURES / "bailian_tiny.jsonl"
    path = tmp_path / "bailian_tiny.jsonl"
    path.write_bytes(src.read_bytes())
    assert "mmfs1" not in str(path)
    requests, _, _ = load_bailian_trace(path, seed=0)
    assert len(requests) > 0


def test_public_converters_need_no_credentials():
    # Public fixtures convert without HF tokens or env credentials.
    load_bailian_trace(FIXTURES / "bailian_tiny.jsonl", seed=0)
    load_mooncake_trace(FIXTURES / "mooncake_tiny.jsonl", seed=0)
    convert_azure_to_requests(FIXTURES / "azure_tiny.csv", seed=0)

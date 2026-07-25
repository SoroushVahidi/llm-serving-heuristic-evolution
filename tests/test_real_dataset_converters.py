"""Tests for Bailian, Mooncake, Azure, BurstGPT session, and prompt-corpus adapters."""
from pathlib import Path

import pytest

from llmserveopt.core.types import ObservableRequest
from llmserveopt.workloads.azure import AzureConversionConfig, convert_azure_to_requests
from llmserveopt.workloads.bailian import (
    BailianConversionConfig,
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
    assert metadata[0]["session_id"] == "1"
    assert "prefix_id" in metadata[0]
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


def test_mooncake_conversion_and_synthetic_flag():
    requests, metadata, report = load_mooncake_trace(
        FIXTURES / "mooncake_tiny.jsonl",
        config=MooncakeConversionConfig(source_split="conversation_trace"),
        seed=1,
    )
    assert report.rows_retained == 4
    assert metadata[0]["dataset_type"] == DatasetType.TRUE_SERVING_TRACE.value
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
    _assert_no_leakage(r1)


def test_burstgpt_session_model_optional_columns():
    df = load_burstgpt_raw(FIXTURES / "burstgpt_session_tiny.csv")
    requests, metadata, report = convert_burstgpt_to_requests_with_metadata(
        df, config=BurstGPTConversionConfig(), seed=0
    )
    assert report.rows_retained == 4
    assert report.schema_detected["session_id"] is not None
    assert report.schema_detected["model"] is not None
    assert metadata[0]["session_id"] == "sess-1"
    assert metadata[0]["model_id"] == "ChatGPT"
    assert metadata[2]["extra"]["log_type"] == "API log"
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
            source_dataset="wildchat_fixture", arrival_rate=5.0
        ),
        seed=0,
        dataset_type=DatasetType.PROMPT_CONVERSATION_CORPUS.value,
    )
    assert report.rows_retained == 3
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
    # Ensure we did not stash text on the record
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


def test_path_independence_from_mmfs1():
    # Converters must accept arbitrary local paths (portable fixtures).
    path = FIXTURES / "bailian_tiny.jsonl"
    assert "mmfs1" not in str(path)
    requests, _, _ = load_bailian_trace(path, seed=0)
    assert len(requests) > 0

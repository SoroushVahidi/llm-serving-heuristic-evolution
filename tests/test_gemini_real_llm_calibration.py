"""Tests for the Gemini/Vertex live-call parsing logic in
scripts/run_gemini_real_llm_calibration.py.

These use lightweight fake response objects shaped like the real
google-genai SDK's response types (verified against actual API responses
during development — see docs/real_llm_multi_provider_plan.md) rather than
a real network call, so response-parsing bugs are caught before any live
credit is spent. Full dry-run/mock/resume/cap coverage lives in
tests/test_real_llm_provider_skeletons.py (parametrized across all
providers); this file covers only what's specific to Gemini's response
shapes, plus the v2 length-targeted workload CLI (mirroring
tests/test_cohere_api_calibration.py's v2 section).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_gemini_real_llm_calibration",
        ROOT / "scripts" / "run_gemini_real_llm_calibration.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class FakePart:
    text: Optional[str]


@dataclass
class FakeContent:
    parts: List[FakePart]


@dataclass
class FakeCandidate:
    content: Optional[FakeContent]
    finish_reason: Optional[str] = None


@dataclass
class FakeUsage:
    prompt_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None


@dataclass
class FakePlanned:
    model: str = "gemini-3.1-flash-lite"
    prompt_text: str = "hello"
    max_tokens: int = 64


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

def test_extract_text_joins_multiple_parts():
    mod = _load()
    candidates = [FakeCandidate(content=FakeContent(parts=[FakePart("Hello "), FakePart("world.")]))]
    assert mod._extract_text(candidates) == "Hello world."


def test_extract_text_empty_candidates():
    mod = _load()
    assert mod._extract_text([]) == ""


def test_extract_text_no_content():
    mod = _load()
    assert mod._extract_text([FakeCandidate(content=None)]) == ""


def test_extract_text_empty_parts():
    mod = _load()
    assert mod._extract_text([FakeCandidate(content=FakeContent(parts=[]))]) == ""


def test_extract_text_none_text_part_skipped():
    """Mirrors the real SDK: some parts (e.g. thought_signature-only) have
    text=None and must not raise or inject 'None' into the joined string."""
    mod = _load()
    candidates = [FakeCandidate(content=FakeContent(parts=[FakePart(None), FakePart("OK.")]))]
    assert mod._extract_text(candidates) == "OK."


# ---------------------------------------------------------------------------
# _call_gemini_non_streaming (fake client, no network)
# ---------------------------------------------------------------------------

class _FakeModelsNonStreaming:
    def __init__(self, response):
        self._response = response
        self.last_call_kwargs = None

    def generate_content(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


class _FakeClientNonStreaming:
    def __init__(self, response):
        self.models = _FakeModelsNonStreaming(response)


@dataclass
class FakeNonStreamingResponse:
    candidates: List[FakeCandidate]
    usage_metadata: Optional[FakeUsage]


def test_call_non_streaming_parses_text_and_usage():
    mod = _load()
    response = FakeNonStreamingResponse(
        candidates=[FakeCandidate(
            content=FakeContent(parts=[FakePart("OK.")]),
            finish_reason="FinishReason.STOP",
        )],
        usage_metadata=FakeUsage(prompt_token_count=3, candidates_token_count=2),
    )
    client = _FakeClientNonStreaming(response)
    out = mod._call_gemini_non_streaming(client, FakePlanned(), timeout_s=90)
    assert out["text"] == "OK."
    assert out["prompt_tokens"] == 3.0
    assert out["output_tokens"] == 2.0
    assert out["finish_reason"] == "FinishReason.STOP"
    assert out["ttft_seconds"] is None


def test_call_non_streaming_passes_model_and_max_tokens(monkeypatch):
    mod = _load()
    response = FakeNonStreamingResponse(
        candidates=[FakeCandidate(content=FakeContent(parts=[FakePart("x")]))],
        usage_metadata=None,
    )
    client = _FakeClientNonStreaming(response)
    planned = FakePlanned(model="gemini-3.1-flash-lite", prompt_text="a prompt", max_tokens=128)
    mod._call_gemini_non_streaming(client, planned, timeout_s=42)
    kwargs = client.models.last_call_kwargs
    assert kwargs["model"] == "gemini-3.1-flash-lite"
    assert kwargs["contents"] == "a prompt"
    assert kwargs["config"].max_output_tokens == 128
    assert kwargs["config"].http_options.timeout == 42_000


def test_call_non_streaming_handles_missing_usage():
    mod = _load()
    response = FakeNonStreamingResponse(
        candidates=[FakeCandidate(content=FakeContent(parts=[FakePart("x")]))],
        usage_metadata=None,
    )
    client = _FakeClientNonStreaming(response)
    out = mod._call_gemini_non_streaming(client, FakePlanned(), timeout_s=90)
    assert out["prompt_tokens"] is None
    assert out["output_tokens"] is None


# ---------------------------------------------------------------------------
# _call_gemini_streaming (fake client, no network)
# ---------------------------------------------------------------------------

@dataclass
class FakeStreamChunk:
    text: Optional[str]
    usage_metadata: Optional[FakeUsage] = None
    candidates: Optional[List[FakeCandidate]] = None


class _FakeModelsStreaming:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_call_kwargs = None

    def generate_content_stream(self, **kwargs):
        self.last_call_kwargs = kwargs
        return iter(self._chunks)


class _FakeClientStreaming:
    def __init__(self, chunks):
        self.models = _FakeModelsStreaming(chunks)


def test_call_streaming_measures_ttft_and_joins_chunks():
    mod = _load()
    chunks = [
        FakeStreamChunk(text="OK"),
        FakeStreamChunk(text="."),
        FakeStreamChunk(
            text=None,
            usage_metadata=FakeUsage(prompt_token_count=3, candidates_token_count=2),
            candidates=[FakeCandidate(content=None, finish_reason="FinishReason.STOP")],
        ),
    ]
    client = _FakeClientStreaming(chunks)
    out = mod._call_gemini_streaming(client, FakePlanned(), timeout_s=90)
    assert out["text"] == "OK."
    assert out["ttft_seconds"] is not None
    assert out["ttft_seconds"] >= 0
    assert out["prompt_tokens"] == 3.0
    assert out["output_tokens"] == 2.0
    assert out["finish_reason"] == "FinishReason.STOP"


def test_call_streaming_no_text_chunks_gives_none_ttft():
    mod = _load()
    chunks = [FakeStreamChunk(text=None, usage_metadata=FakeUsage(prompt_token_count=3, candidates_token_count=0))]
    client = _FakeClientStreaming(chunks)
    out = mod._call_gemini_streaming(client, FakePlanned(), timeout_s=90)
    assert out["text"] == ""
    assert out["ttft_seconds"] is None


# ---------------------------------------------------------------------------
# _build_client credential precedence (no real network — only checks which
# branch is taken and that a clear error is raised when nothing is set)
# ---------------------------------------------------------------------------

def test_build_client_raises_clearly_with_no_credentials(monkeypatch):
    mod = _load()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    try:
        mod._build_client()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "GOOGLE_API_KEY" in str(e) or "GOOGLE_CLOUD_PROJECT" in str(e)


def test_build_client_prefers_api_key_when_present(monkeypatch):
    mod = _load()
    import google.genai as genai

    captured = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return "fake-client"

    monkeypatch.setattr(genai, "Client", _fake_client)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-not-real")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    result = mod._build_client()
    assert result == "fake-client"
    assert captured == {"api_key": "fake-not-real"}


def test_build_client_falls_back_to_vertex(monkeypatch):
    mod = _load()
    import google.genai as genai

    captured = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return "fake-client"

    monkeypatch.setattr(genai, "Client", _fake_client)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east1")
    result = mod._build_client()
    assert result == "fake-client"
    assert captured == {"vertexai": True, "project": "some-project", "location": "us-east1"}


# ---------------------------------------------------------------------------
# v2 length-targeted workload CLI (mirrors test_cohere_api_calibration.py)
# ---------------------------------------------------------------------------

V2_GRID_ARGS = [
    "--stream",
    "--model", "gemini-3.1-flash-lite",
    "--workload-version", "v2",
    "--prompt-buckets", "short,medium,long",
    "--target-output-tokens-list", "64,128,256",
    "--concurrency-list", "1,2,4,8",
    "--requests-per-cell", "3",
    "--timeout-seconds", "120",
    "--rpm-limit", "20",
    "--max-total-requests", "108",
    "--max-total-input-tokens", "250000",
    "--max-total-output-tokens", "50000",
    "--max-estimated-cost-usd", "5",
    "--seed", "20260703",
    "--fail-fast",
]


def test_v2_dry_run_plans_108_requests(tmp_path):
    mod = _load()
    result = mod.main(["--dry-run", *V2_GRID_ARGS, "--output-dir", str(tmp_path)])
    assert result == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_requests"] == 3 * 3 * 4 * 3 == 108
    assert not (tmp_path / "requests.jsonl").exists()


def test_v2_dry_run_records_target_output_tokens_in_plan(tmp_path):
    mod = _load()
    mod.main(["--dry-run", *V2_GRID_ARGS, "--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    targets = {r["target_output_tokens"] for r in manifest["requests_preview"]}
    assert targets <= {64, 128, 256}
    assert all(r["workload_version"] == "v2" for r in manifest["requests_preview"])


def test_v2_requires_target_output_tokens_list(tmp_path):
    mod = _load()
    result = mod.main([
        "--dry-run", "--workload-version", "v2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 2


def test_v2_mock_run_schema_has_v2_fields_and_corrected_latency(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project-for-test")
    mod = _load()
    result = mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert len(lines) == 1 * 2 * 1 * 2  # 4
    for row in lines:
        assert row["workload_version"] == "v2"
        assert row["target_output_tokens"] in (64, 128)
        assert row["status"] == "success"
        # Corrected latency schema, not the pre-fix elapsed_seconds field.
        assert "rate_limiter_wait_seconds" in row
        assert "provider_request_latency_seconds" in row
        assert "ttft_seconds" in row
        assert "total_wall_time_seconds" in row


def test_v2_provider_latency_not_polluted_by_rate_limiter_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project-for-test")
    mod = _load()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    for row in lines:
        assert row["rate_limiter_wait_seconds"] == 0.0
        assert row["provider_request_latency_seconds"] is not None
        assert row["provider_request_latency_seconds"] < 1.0


def test_v2_summary_reports_output_token_distribution_by_target(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project-for-test")
    mod = _load()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert (tmp_path / "aggregate_by_target_output_tokens.csv").exists()
    import pandas as pd
    by_target = pd.read_csv(tmp_path / "aggregate_by_target_output_tokens.csv")
    assert set(by_target["target_output_tokens"]) == {64, 128}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["by_target_output_tokens"]) == 2


def test_v2_resume_skips_completed_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project-for-test")
    mod = _load()
    base_args = [
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--seed", "5",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(base_args) == 0
    first_lines = (tmp_path / "requests.jsonl").read_text().strip().splitlines()
    assert len(first_lines) == 1 * 2 * 1 * 2  # 4

    expanded_args = [
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2", "--resume",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128,256",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--seed", "5",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(expanded_args) == 0
    all_lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    ids = [r["request_id"] for r in all_lines]
    assert len(ids) == len(set(ids)), "resume must not duplicate request_ids"
    assert len(all_lines) == 1 * 3 * 1 * 2  # 6


def test_v2_refuses_to_overwrite_nonempty_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project-for-test")
    mod = _load()
    args = [
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(args) == 0
    assert mod.main(args) == 3


def test_v2_api_key_never_written_to_output_files(tmp_path, monkeypatch):
    # See the equivalent Cohere test for why git_diff.patch is excluded: it
    # is an intentional full working-tree diff snapshot, not harness output,
    # and would otherwise flag this test's own uncommitted source line.
    secret = "fake-project-SECRET-TEST-VALUE-12345"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", secret)
    mod = _load()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file() and f.name != "git_diff.patch":
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"

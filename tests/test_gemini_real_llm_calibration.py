"""Tests for the Gemini/Vertex live-call parsing logic in
scripts/run_gemini_real_llm_calibration.py.

These use lightweight fake response objects shaped like the real
google-genai SDK's response types (verified against actual API responses
during development — see docs/real_llm_multi_provider_plan.md) rather than
a real network call, so response-parsing bugs are caught before any live
credit is spent. Full dry-run/mock/resume/cap coverage lives in
tests/test_real_llm_provider_skeletons.py (parametrized across all
providers); this file covers only what's specific to Gemini's response
shapes.
"""
from __future__ import annotations

import importlib.util
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

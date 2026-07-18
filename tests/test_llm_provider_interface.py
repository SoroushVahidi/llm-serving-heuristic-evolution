"""Tests for LLM provider interface and mock provider."""
import json
from llmserveopt.llm_generation.provider_base import LLMProvider, LLMResponse
from llmserveopt.llm_generation.providers import MockProvider, build_providers


def test_llm_response_is_dataclass():
    r = LLMResponse(provider="mock", model="mock-v1", text="hello")
    assert r.provider == "mock"
    assert r.model == "mock-v1"
    assert r.text == "hello"
    assert r.prompt_tokens is None


def test_mock_provider_is_available():
    p = MockProvider()
    assert p.is_available()


def test_mock_provider_name():
    assert MockProvider().name == "mock"


def test_mock_provider_returns_response():
    p = MockProvider()
    messages = [{"role": "user", "content": "generate a heuristic"}]
    resp = p.generate(messages)
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "mock"
    assert len(resp.text) > 0


def test_mock_provider_response_is_valid_json():
    p = MockProvider()
    messages = [{"role": "user", "content": "test"}]
    resp = p.generate(messages)
    doc = json.loads(resp.text)
    assert isinstance(doc, dict)
    assert "name" in doc


def test_mock_provider_cycles_through_responses():
    p = MockProvider()
    messages = [{"role": "user", "content": "x"}]
    names = []
    for _ in range(4):
        resp = p.generate(messages)
        doc = json.loads(resp.text)
        names.append(doc.get("name", ""))
    # Should get 4 responses (cycles through the 4 mock responses)
    assert len(names) == 4


def test_mock_provider_repair_mode():
    p = MockProvider()
    messages = [{"role": "user", "content": "repair"}]
    resp = p.generate(messages, _is_repair=True)
    doc = json.loads(resp.text)
    assert "name" in doc
    # The expression part must not reference actual_output_tokens
    import json as _json
    expr_text = _json.dumps(doc.get("default", {}))
    assert "actual_output" not in expr_text


def test_build_providers_mock():
    providers = build_providers(["mock"])
    assert len(providers) == 1
    assert providers[0].name == "mock"


def test_build_providers_unknown_skips():
    providers = build_providers(["nonexistent_provider"])
    assert len(providers) == 0


def test_build_providers_mixed():
    providers = build_providers(["mock", "nonexistent"])
    assert len(providers) == 1
    assert providers[0].name == "mock"


def test_provider_protocol_satisfied():
    p = MockProvider()
    assert isinstance(p, LLMProvider)


def test_mock_unavailable_if_required_env_missing():
    """Non-mock providers report unavailability when keys are absent (using env override)."""
    import os
    from llmserveopt.llm_generation.providers import CloudRiftProvider
    # Save original
    orig = os.environ.pop("CLOUDRIFT_API_KEY", None)
    orig_url = os.environ.pop("CLOUDRIFT_BASE_URL", None)
    try:
        p = CloudRiftProvider()
        assert not p.is_available()
    finally:
        if orig:
            os.environ["CLOUDRIFT_API_KEY"] = orig
        if orig_url:
            os.environ["CLOUDRIFT_BASE_URL"] = orig_url

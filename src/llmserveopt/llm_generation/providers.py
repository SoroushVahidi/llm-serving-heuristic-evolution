"""
LLM provider wrappers.

All providers call external APIs for offline heuristic generation.
No provider is called at runtime during request scheduling.

Usage
-----
providers = build_providers(["cloudrift", "cohere", "mistral"])
available = [p for p in providers if p.is_available()]

Environment variables (never hardcoded):
  CLOUDRIFT_API_KEY, CLOUDRIFT_BASE_URL
  COHERE_API_KEY
  MISTRAL_API_KEY
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from .provider_base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# CloudRift (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class CloudRiftProvider:
    """CloudRift via OpenAI-compatible API.

    Uses explicit CLOUDRIFT_BASE_URL — never assumes api.openai.com.
    """

    name = "cloudrift"

    def __init__(self) -> None:
        self._api_key = os.environ.get("CLOUDRIFT_API_KEY", "")
        self._base_url = os.environ.get("CLOUDRIFT_BASE_URL", "")
        self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key and self._base_url)

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                return None
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str = "auto",
    ) -> LLMResponse:
        client = self._get_client()
        if client is None:
            return LLMResponse(
                provider=self.name, model=model, text="",
                raw_metadata={"error": "openai SDK not installed"},
            )
        # Resolve "auto" to the current CloudRift default model
        actual_model = model if model != "auto" else "Qwen/Qwen3.6-35B-A3B-FP8"
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.monotonic() - t0
            usage = resp.usage
            msg = resp.choices[0].message
            # Thinking models (e.g. Qwen3) put response in reasoning when content is None
            text = msg.content or getattr(msg, "reasoning", None) or ""
            return LLMResponse(
                provider=self.name,
                model=actual_model,
                text=text,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                raw_metadata={"elapsed_s": elapsed},
            )
        except Exception as e:
            return LLMResponse(
                provider=self.name, model=actual_model, text="",
                raw_metadata={"error": str(e)},
            )


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------

class CohereProvider:
    """Cohere API via cohere Python SDK."""

    name = "cohere"

    def __init__(self) -> None:
        self._api_key = os.environ.get("COHERE_API_KEY", "")
        self._client = None

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import cohere  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str = "auto",
    ) -> LLMResponse:
        actual_model = model if model != "auto" else "command-r-plus-08-2024"
        try:
            client = self._get_client()
            t0 = time.monotonic()
            resp = client.chat(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.monotonic() - t0
            text = resp.message.content[0].text if resp.message.content else ""
            usage = resp.usage
            return LLMResponse(
                provider=self.name,
                model=actual_model,
                text=text,
                prompt_tokens=usage.billed_units.input_tokens if usage and usage.billed_units else None,
                completion_tokens=usage.billed_units.output_tokens if usage and usage.billed_units else None,
                raw_metadata={"elapsed_s": elapsed},
            )
        except Exception as e:
            return LLMResponse(
                provider=self.name, model=actual_model, text="",
                raw_metadata={"error": str(e)},
            )


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

class MistralProvider:
    """Mistral API via mistralai Python SDK."""

    name = "mistral"

    def __init__(self) -> None:
        self._api_key = os.environ.get("MISTRAL_API_KEY", "")
        self._client = None

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import mistralai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self._api_key)
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str = "auto",
    ) -> LLMResponse:
        actual_model = model if model != "auto" else "mistral-large-latest"
        try:
            client = self._get_client()
            t0 = time.monotonic()
            resp = client.chat.complete(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.monotonic() - t0
            text = resp.choices[0].message.content or "" if resp.choices else ""
            usage = resp.usage
            return LLMResponse(
                provider=self.name,
                model=actual_model,
                text=text,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                raw_metadata={"elapsed_s": elapsed},
            )
        except Exception as e:
            return LLMResponse(
                provider=self.name, model=actual_model, text="",
                raw_metadata={"error": str(e)},
            )


# ---------------------------------------------------------------------------
# Mock provider (dry-run)
# ---------------------------------------------------------------------------

_MOCK_VALID_HEURISTIC = {
    "name": "mock_slo_aware_sjf",
    "description": "Mock: SLO-aware SJF — urgency × inverse service time.",
    "tie_breaker": "earliest_deadline",
    "regimes": [
        {
            "condition": {
                "op": "sub",
                "args": [{"var": "sys.kv_utilization"}, {"const": 0.7}],
            },
            "request_score": {
                "op": "weighted_sum",
                "terms": [
                    [{"var": "req.deadline_urgency"}, 0.8],
                    [
                        {
                            "op": "div_safe",
                            "args": [{"const": 1.0}, {"var": "req.estimated_kv_cost"}],
                        },
                        0.2,
                    ],
                ],
            },
        }
    ],
    "default": {
        "request_score": {
            "op": "weighted_sum",
            "terms": [
                [{"var": "req.deadline_urgency"}, 0.6],
                [{"op": "neg", "args": [{"var": "req.estimated_decode_cost"}]}, 0.4],
            ],
        }
    },
}

_MOCK_INVALID_HEURISTIC = {
    "name": "mock_invalid_needs_repair",
    "description": "Mock: uses a forbidden variable — should fail verification.",
    "tie_breaker": "earliest_deadline",
    "default": {
        "request_score": {"var": "req.actual_output_tokens"},
    },
}

_MOCK_REPAIRED_HEURISTIC = {
    "name": "mock_repaired",
    "description": "Mock: repaired version — replaced actual_output_tokens with predicted.",
    "tie_breaker": "earliest_deadline",
    "default": {
        "request_score": {
            "op": "neg",
            "args": [{"var": "req.predicted_output_tokens"}],
        }
    },
}

_MOCK_THROUGHPUT_HEURISTIC = {
    "name": "mock_throughput_sjf",
    "description": "Mock: pure throughput focus — shortest predicted output first.",
    "tie_breaker": "shortest_output",
    "default": {
        "request_score": {
            "op": "neg",
            "args": [{"var": "req.predicted_output_tokens"}],
        }
    },
}


class MockProvider:
    """Deterministic mock provider for dry-run testing.

    Returns known heuristics in a fixed order: valid, invalid (triggers repair),
    then valid again. This exercises the full verify → repair → archive loop.
    """

    name = "mock"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._call_count = 0
        # Sequence of responses: index → heuristic dict
        self._responses = [
            _MOCK_VALID_HEURISTIC,
            _MOCK_INVALID_HEURISTIC,
            _MOCK_THROUGHPUT_HEURISTIC,
            _MOCK_VALID_HEURISTIC,   # extra valid
        ]
        self._repair_response = _MOCK_REPAIRED_HEURISTIC

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str = "mock",
        _is_repair: bool = False,
    ) -> LLMResponse:
        if _is_repair:
            doc = self._repair_response
        else:
            idx = self._call_count % len(self._responses)
            doc = self._responses[idx]
            self._call_count += 1
        text = json.dumps(doc, indent=2)
        return LLMResponse(
            provider=self.name,
            model="mock-v1",
            text=text,
            prompt_tokens=100,
            completion_tokens=len(text) // 4,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_CLASSES = {
    "cloudrift": CloudRiftProvider,
    "cohere": CohereProvider,
    "mistral": MistralProvider,
    "mock": MockProvider,
}


def build_providers(names: List[str]) -> List:
    """Instantiate providers by name. Unknown names are skipped with a warning."""
    result = []
    for name in names:
        cls = _PROVIDER_CLASSES.get(name.strip().lower())
        if cls is None:
            print(f"  [WARN] Unknown provider '{name}' — skipping")
            continue
        result.append(cls())
    return result

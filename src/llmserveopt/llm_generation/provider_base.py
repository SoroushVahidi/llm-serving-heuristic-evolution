"""
Common provider interface for LLM API calls.

All providers are used offline only — no API is called during
request scheduling at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    provider: str
    model: str
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_estimate: Optional[float] = None
    raw_metadata: Optional[Dict[str, Any]] = field(default=None, repr=False)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface all LLM provider wrappers must satisfy."""

    name: str

    def is_available(self) -> bool:
        """Return True iff the provider SDK and credentials are present."""
        ...

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Call the API and return the raw text response.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...} dicts.
        temperature : float — sampling temperature.
        max_tokens : int — maximum completion tokens.

        Returns
        -------
        LLMResponse — never raises; returns a response with empty text on error.
        """
        ...

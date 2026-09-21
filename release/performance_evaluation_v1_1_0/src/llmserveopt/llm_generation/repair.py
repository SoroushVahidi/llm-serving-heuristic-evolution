"""
Repair logic: extract JSON from LLM response and attempt re-verification.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..heuristics.verifier import VerificationResult, verify_heuristic


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from raw LLM response text.

    Handles: raw JSON, JSON inside ```json...``` fences, JSON after preamble.
    """
    text = text.strip()

    # 1. Strip code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Try full parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 3. Find the first { ... } block
    start = text.find("{")
    if start == -1:
        return None

    # Scan for matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                fragment = text[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break

    return None


def verify_and_collect_errors(candidate: Dict) -> Tuple[bool, List[Tuple[str, str]]]:
    """Run verifier and return (valid, errors)."""
    result: VerificationResult = verify_heuristic(candidate)
    return result.valid, result.errors


def run_repair_loop(
    candidate: Dict,
    provider,
    max_attempts: int,
    messages_builder,
    save_attempt_fn,
    *,
    temperature: float = 0.4,
    max_tokens: int = 2000,
) -> Tuple[Optional[Dict], bool, int]:
    """Attempt to repair an invalid candidate via LLM re-generation.

    Parameters
    ----------
    candidate : dict — the initial invalid candidate.
    provider — LLMProvider instance.
    max_attempts : int.
    messages_builder : callable(candidate, errors) -> messages list.
    save_attempt_fn : callable(n, raw_text, cand_or_None) — archives each attempt.
    temperature : float.
    max_tokens : int.

    Returns
    -------
    (repaired_candidate_or_None, repaired_ok, n_attempts)
    """
    current = candidate
    _, errors = verify_and_collect_errors(current)

    for attempt in range(1, max_attempts + 1):
        messages = messages_builder(current, errors)
        # Signal repair mode to mock provider
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if hasattr(provider, "generate") and "_is_repair" in provider.generate.__code__.co_varnames:
            resp = provider.generate(messages, _is_repair=True, **kwargs)
        else:
            resp = provider.generate(messages, **kwargs)

        extracted = extract_json(resp.text)
        save_attempt_fn(attempt, resp.text, extracted)

        if extracted is None:
            continue

        valid, errors = verify_and_collect_errors(extracted)
        current = extracted
        if valid:
            return current, True, attempt

    return current, False, max_attempts

"""Offline scoring pipeline: turn real prompt text into a cached
``{request_id: score}`` map, entirely outside the live simulator loop.

Mirrors ``baselines/vllm_ltr/adapter/offline_scoring.py`` exactly (same
cache format, same prompt-hash staleness guard) -- see that module's
docstring for the full rationale. PARS has no live per-step path either:
its `ObservableRequest` input requirement (real prompt text) is not
carried by this simulator's data model, exactly the same structural gap
vLLM-LTR has.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Mapping

from .checkpoint_loader import PARSPredictorHandle
from .errors import PARSAdapterError


class StaleScoreCacheError(PARSAdapterError):
    """Raised when a cached score's stored prompt hash does not match the
    prompt text currently associated with that request id."""


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def score_prompts_offline(
    handle: PARSPredictorHandle,
    id_to_prompt: Mapping[int, str],
    batch_size: int = 8,
) -> Dict[int, dict]:
    """Score every (request_id, prompt_text) pair once. Returns
    ``{request_id: {"score": float, "prompt_sha256": str}}``. Uses only
    prompt text -- no arrival time, no output length, no
    actual_output_tokens. Deterministic: delegates to
    ``PARSPredictorHandle.score_batch``, itself ``eval()``/``no_grad()``."""
    ids = list(id_to_prompt.keys())
    prompts = [id_to_prompt[i] for i in ids]
    scores = handle.score_batch(prompts, batch_size=batch_size)
    return {
        rid: {"score": score, "prompt_sha256": _prompt_hash(prompt)}
        for rid, prompt, score in zip(ids, prompts, scores)
    }


def save_score_cache(cache: Mapping[int, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cache.items()}, f, indent=2)


def load_score_cache(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def scores_only(cache: Mapping[int, dict], id_to_prompt: Mapping[int, str] = None) -> Dict[int, float]:
    """Extract the plain ``{request_id: score}`` map
    ``PARSSemanticReferencePolicy`` expects. If ``id_to_prompt`` is given,
    verifies every entry's stored prompt hash still matches -- raises
    ``StaleScoreCacheError`` on any mismatch."""
    if id_to_prompt is not None:
        for rid, entry in cache.items():
            if rid not in id_to_prompt:
                continue
            expected = _prompt_hash(id_to_prompt[rid])
            if entry.get("prompt_sha256") != expected:
                raise StaleScoreCacheError(
                    f"Cached score for request_id={rid} was computed from "
                    "different prompt text than the one supplied now "
                    f"(cached hash {entry.get('prompt_sha256')!r} != "
                    f"current hash {expected!r}). Refusing to serve a stale "
                    "score."
                )
    return {rid: entry["score"] for rid, entry in cache.items()}

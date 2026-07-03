#!/usr/bin/env python3
"""
Gemini / Vertex AI real-LLM calibration: black-box latency/TTFT measurement.

This is a **separate** script from scripts/run_gemini_api_calibration.py
(which predates the shared multi-provider schema and has its own
config-file-driven CLI, still used for the Phase 2C.4 dry-run). This script
uses the shared grid/output schema from
src/llmserveopt/real_llm/calibration_common.py — the same one
scripts/run_cohere_api_calibration.py uses — so a Gemini live pilot produces
byte-for-byte comparable output (requests.jsonl, summary.json/md,
aggregate_by_*.csv, manifest.json, reproducibility.md) to the Cohere pilot.

Like the Cohere script, this is a **black-box** measurement: request
submitted -> first streamed token -> last token, with no visibility into or
control over Google's internal batching/scheduling.

Auth: uses the `google-genai` SDK. Prefers a direct Gemini Developer API key
(GOOGLE_API_KEY) if set; otherwise falls back to Vertex AI via
GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION and Application Default
Credentials (`gcloud auth application-default login`) — the mode this
project's environment is actually configured for. Live-mode credential
gating checks GOOGLE_CLOUD_PROJECT specifically (see API_KEY_ENV_VAR below)
since that is what this project uses; only its *presence*, never its value,
is ever recorded.

DEFAULT MODE (no flags): refuse to run anything.
--dry-run:        plan the request grid, validate hard caps, write manifest
                   and reproducibility metadata, never import the genai SDK
                   or contact any API.
--allow-live-api:  actually issue calls under hard caps (requires Vertex
                   credentials and passing all cap checks).
--mock:            replace live API calls with a local deterministic stub
                   (for tests only; no network, no credentials needed).

Usage (dry-run):
    python scripts/run_gemini_real_llm_calibration.py \\
        --dry-run \\
        --model gemini-3.1-flash-lite \\
        --prompt-buckets short,medium,long \\
        --max-tokens-list 64,128,256 \\
        --concurrency-list 1,2,4,8 \\
        --requests-per-cell 5 \\
        --output-dir experiments/real_llm/gemini_pilot_DRYRUN

Usage (live pilot — review hard caps first):
    python scripts/run_gemini_real_llm_calibration.py \\
        --allow-live-api --stream \\
        --model gemini-3.1-flash-lite \\
        --prompt-buckets short,medium,long \\
        --max-tokens-list 64,128,256 \\
        --concurrency-list 1,2,4,8 \\
        --requests-per-cell 5 \\
        --timeout-seconds 90 --rpm-limit 20 \\
        --max-total-requests 180 --max-total-input-tokens 250000 \\
        --max-total-output-tokens 50000 --max-estimated-cost-usd 5 \\
        --seed 20260703 --fail-fast \\
        --output-dir experiments/real_llm/gemini_pilot_20260703T000000Z

Usage (v2 length-targeted workload — see docs/real_llm_v2_workload_proposal.md):
    python scripts/run_gemini_real_llm_calibration.py \\
        --allow-live-api --stream \\
        --model gemini-3.1-flash-lite \\
        --workload-version v2 \\
        --prompt-buckets short,medium,long \\
        --target-output-tokens-list 64,128,256 \\
        --concurrency-list 1,2,4,8 \\
        --requests-per-cell 3 \\
        --timeout-seconds 120 --rpm-limit 20 \\
        --max-total-requests 108 --max-total-input-tokens 250000 \\
        --max-total-output-tokens 50000 --max-estimated-cost-usd 5 \\
        --seed 20260703 --fail-fast \\
        --output-dir experiments/real_llm/gemini_v2_length_targeted_20260703T000000Z

See docs/real_llm_multi_provider_plan.md for the rollout plan.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402

# ---------------------------------------------------------------------------
# Gemini-specific constants
# ---------------------------------------------------------------------------

# Cheapest/fastest Flash-Lite class model confirmed available on this
# project's Vertex AI endpoint (queried via client.models.list(), a free
# metadata call — see docs/real_llm_multi_provider_plan.md). Deliberately
# not a Pro/reasoning model.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Placeholder pricing (USD per 1M tokens) — conservative round numbers, NOT
# verified against a current Google Cloud invoice for gemini-3.1-flash-lite
# specifically (released after this assistant's training cutoff). Used only
# for pre-flight/post-hoc cap estimation; the actual pilot's tiny token
# volume keeps real cost far below any plausible per-token rate regardless.
# Verify against your Cloud Billing console before scaling beyond this pilot.
_PRICE_PER_M_INPUT_USD = 0.10
_PRICE_PER_M_OUTPUT_USD = 0.40

# Credential gate: this project is configured for Vertex AI (ADC), not a
# direct Gemini Developer API key, so GOOGLE_CLOUD_PROJECT is what we check
# for presence before allowing a live call. _build_client() prefers
# GOOGLE_API_KEY if present, for portability to a Gemini-Developer-API setup.
API_KEY_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
SDK_PACKAGE_NAME = "google-genai"


# ---------------------------------------------------------------------------
# Client / API calls
# ---------------------------------------------------------------------------

def _build_client():
    import google.genai as genai

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        return genai.Client(api_key=api_key)

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "Neither GOOGLE_API_KEY nor GOOGLE_CLOUD_PROJECT is set. "
            "Set one of them before running live mode."
        )
    return genai.Client(vertexai=True, project=project, location=location)


def _extract_text(candidates) -> str:
    if not candidates:
        return ""
    content = candidates[0].content
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts if getattr(part, "text", None))


def _call_gemini_non_streaming(client, planned, timeout_s: int) -> Dict[str, Any]:
    from google.genai import types

    resp = client.models.generate_content(
        model=planned.model,
        contents=planned.prompt_text,
        config=types.GenerateContentConfig(
            max_output_tokens=planned.max_tokens,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        ),
    )
    usage = resp.usage_metadata
    finish_reason = None
    if resp.candidates and resp.candidates[0].finish_reason:
        finish_reason = str(resp.candidates[0].finish_reason)
    return {
        "text": _extract_text(resp.candidates),
        "finish_reason": finish_reason,
        "prompt_tokens": float(usage.prompt_token_count) if usage and usage.prompt_token_count else None,
        "output_tokens": float(usage.candidates_token_count) if usage and usage.candidates_token_count else None,
        "ttft_seconds": None,
    }


def _call_gemini_streaming(client, planned, timeout_s: int) -> Dict[str, Any]:
    from google.genai import types

    t0 = time.monotonic()
    first_token_t = None
    chunks: List[str] = []
    finish_reason = None
    prompt_tokens = None
    output_tokens = None
    for chunk in client.models.generate_content_stream(
        model=planned.model,
        contents=planned.prompt_text,
        config=types.GenerateContentConfig(
            max_output_tokens=planned.max_tokens,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        ),
    ):
        if chunk.text:
            if first_token_t is None:
                first_token_t = time.monotonic()
            chunks.append(chunk.text)
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
            if usage.prompt_token_count:
                prompt_tokens = float(usage.prompt_token_count)
            if usage.candidates_token_count:
                output_tokens = float(usage.candidates_token_count)
        if chunk.candidates and chunk.candidates[0].finish_reason:
            finish_reason = str(chunk.candidates[0].finish_reason)
    ttft = (first_token_t - t0) if first_token_t is not None else None
    return {
        "text": "".join(chunks),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_seconds": ttft,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini/Vertex real-LLM calibration — dry-run and live black-box latency/TTFT harness."
    )
    cc.add_common_arguments(parser, default_model=DEFAULT_MODEL)
    parser.add_argument("--rpm-limit", type=int, default=20)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=False)
    parser.add_argument(
        "--workload-version", choices=["v1", "v2"], default="v1",
        help=(
            "v1: original build_prompt() grid, swept over --max-tokens-list "
            "(output length does not vary with max_tokens — see "
            "docs/real_llm_v2_workload_proposal.md). v2: length-targeted "
            "build_length_targeted_prompt() grid, swept over "
            "--target-output-tokens-list, with max_tokens set to "
            f"{cc.DEFAULT_MAX_TOKENS_HEADROOM_MULTIPLIER}x each target for headroom."
        ),
    )
    parser.add_argument(
        "--target-output-tokens-list", type=cc.csv_int_list, default=None,
        help="Comma-separated target output token counts (v2 only), e.g. 64,128,256.",
    )
    parser.add_argument(
        "--min-output-token-ratio", type=float, default=0.70,
        help=(
            "v2 only: a request's reached_target_output_range is True when "
            "output_tokens >= this ratio x target_output_tokens."
        ),
    )
    parser.add_argument(
        "--record-output-text-preview-chars", type=int, default=80,
        help=(
            "Max characters of generated text to store per request as "
            "output_text_preview (0 disables preview storage). Full model "
            "output is never persisted regardless of this setting."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return cc.run_calibration_main(
        args,
        root=ROOT,
        provider_display_name="Gemini",
        api_key_env_var=API_KEY_ENV_VAR,
        sdk_package_name=SDK_PACKAGE_NAME,
        price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
        price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
        live_implemented=True,
        build_client_fn=_build_client,
        call_streaming_fn=_call_gemini_streaming,
        call_non_streaming_fn=_call_gemini_non_streaming,
    )


if __name__ == "__main__":
    raise SystemExit(main())

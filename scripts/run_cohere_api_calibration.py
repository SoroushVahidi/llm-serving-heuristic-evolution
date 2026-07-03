#!/usr/bin/env python3
"""
Cohere API Calibration: real-LLM black-box latency/TTFT measurement.

This script measures observed latency and time-to-first-token (TTFT) from the
Cohere hosted Chat API across a grid of (prompt_bucket, max_tokens,
concurrency_level) cells. It is a **black-box** measurement: we only observe
what a client sees (request submitted -> first token -> last token). We have
no visibility into and no control over Cohere's internal batching, scheduling,
or GPU allocation. This calibrates simulator assumptions against an external
observed latency distribution; it does not validate or falsify any claim
about how Cohere's scheduler works internally.

Shared grid construction, hard-cap enforcement, JSONL logging, aggregation,
and reproducibility metadata live in
src/llmserveopt/real_llm/calibration_common.py, reused by every provider's
calibration script (see docs/real_llm_multi_provider_plan.md). This module
supplies only the Cohere-specific pieces: how to build a client and how to
make one streaming/non-streaming call.

DEFAULT MODE (no flags): refuse to run anything.
--dry-run:        plan the request grid, validate hard caps, write manifest
                   and reproducibility metadata, never import the cohere SDK
                   or contact any API.
--allow-live-api:  actually issue calls under hard caps (requires
                   COHERE_API_KEY and passing all cap checks).
--mock:            replace live API calls with a local deterministic stub
                   (for tests only; no network, no credentials needed).

Usage (dry-run):
    python scripts/run_cohere_api_calibration.py \\
        --dry-run \\
        --model command-r7b-12-2024 \\
        --prompt-buckets short,medium,long \\
        --max-tokens-list 64,128,256 \\
        --concurrency-list 1,2,4,8 \\
        --requests-per-cell 5 \\
        --output-dir experiments/real_llm/cohere_pilot_DRYRUN

Usage (live pilot — review hard caps first):
    python scripts/run_cohere_api_calibration.py \\
        --allow-live-api --stream \\
        --model command-r7b-12-2024 \\
        --prompt-buckets short,medium,long \\
        --max-tokens-list 64,128,256 \\
        --concurrency-list 1,2,4,8 \\
        --requests-per-cell 5 \\
        --timeout-seconds 90 --rpm-limit 20 \\
        --max-total-requests 180 --max-total-input-tokens 250000 \\
        --max-total-output-tokens 50000 --max-estimated-cost-usd 5 \\
        --seed 20260703 --fail-fast \\
        --output-dir experiments/real_llm/cohere_pilot_20260703T000000Z

See docs/cohere_api_calibration.md for the full design and safety notes.
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
# Cohere-specific constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "command-r7b-12-2024"  # cheapest current Command model

# Approximate Cohere pricing (USD per 1M tokens) for command-r7b-12-2024.
# Placeholder estimate based on published rate-card ratios at time of writing;
# used ONLY for pre-flight cost caps and post-hoc estimates, never for
# billing. Verify against your Cohere account/invoice before relying on it
# for larger runs.
_PRICE_PER_M_INPUT_USD = 0.0375
_PRICE_PER_M_OUTPUT_USD = 0.15

# ---------------------------------------------------------------------------
# Re-exports of shared plan/schema types and functions, bound to Cohere's
# pricing where the shared signature requires it. Kept at module level (not
# just accessed via `cc.`) for backward compatibility with call sites and
# tests written against the original single-file script.
# ---------------------------------------------------------------------------

KNOWN_PROMPT_BUCKETS = cc.KNOWN_PROMPT_BUCKETS
PlannedRequest = cc.PlannedRequest
RequestResult = cc.RequestResult
build_prompt = cc.build_prompt
approx_token_count = cc.approx_token_count
expand_call_plan = cc.expand_call_plan
RpmLimiter = cc.RpmLimiter
JsonlWriter = cc.JsonlWriter
load_completed_request_ids = cc.load_completed_request_ids
make_skipped_result = cc.make_skipped_result
execute_one_request = cc.execute_one_request


def estimate_cost_usd(total_input_tokens: float, total_output_tokens: float) -> float:
    return cc.estimate_cost_usd(
        total_input_tokens, total_output_tokens,
        _PRICE_PER_M_INPUT_USD, _PRICE_PER_M_OUTPUT_USD,
    )


def validate_call_plan(plan: List[cc.PlannedRequest], args: argparse.Namespace) -> List[str]:
    return cc.validate_call_plan(
        plan, args,
        price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
        price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
    )


class BudgetTracker(cc.BudgetTracker):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            args,
            price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
            price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
        )


FailFastTracker = cc.FailFastTracker


def aggregate_results(out_dir: Path) -> Dict[str, Any]:
    return cc.aggregate_results(
        out_dir,
        price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
        price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
    )


def write_summary(out_dir: Path, overall: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    cc.write_summary(out_dir, overall, cfg, provider_display_name="Cohere")


# ---------------------------------------------------------------------------
# Mock call: kept as a module-level, monkeypatchable name (tests replace
# `_mock_call` directly to simulate failures for fail-fast testing).
# ---------------------------------------------------------------------------

_mock_call = cc.mock_call


def _dispatch_mock_call(planned: cc.PlannedRequest, stream: bool) -> Dict[str, Any]:
    return _mock_call(planned, stream)


# ---------------------------------------------------------------------------
# Cohere-specific API calls
# ---------------------------------------------------------------------------

def _build_client():
    import cohere
    return cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


def _call_cohere_non_streaming(client, planned: cc.PlannedRequest, timeout_s: int):
    resp = client.chat(
        model=planned.model,
        messages=[{"role": "user", "content": planned.prompt_text}],
        max_tokens=planned.max_tokens,
        temperature=0.0,
        request_options={"timeout_in_seconds": timeout_s, "max_retries": 0},
    )
    text = resp.message.content[0].text if resp.message.content else ""
    usage = resp.usage
    billed = usage.billed_units if usage else None
    return {
        "text": text,
        "finish_reason": str(resp.finish_reason) if resp.finish_reason else None,
        "prompt_tokens": billed.input_tokens if billed else None,
        "output_tokens": billed.output_tokens if billed else None,
        "ttft_seconds": None,
    }


def _call_cohere_streaming(client, planned: cc.PlannedRequest, timeout_s: int):
    t0 = time.monotonic()
    first_token_t = None
    chunks: List[str] = []
    finish_reason = None
    prompt_tokens = None
    output_tokens = None
    for event in client.chat_stream(
        model=planned.model,
        messages=[{"role": "user", "content": planned.prompt_text}],
        max_tokens=planned.max_tokens,
        temperature=0.0,
        request_options={"timeout_in_seconds": timeout_s, "max_retries": 0},
    ):
        if event.type == "content-delta":
            if first_token_t is None:
                first_token_t = time.monotonic()
            delta = event.delta
            if delta and delta.message and delta.message.content:
                chunks.append(delta.message.content.text or "")
        elif event.type == "message-end" and event.delta:
            finish_reason = str(event.delta.finish_reason) if event.delta.finish_reason else None
            usage = event.delta.usage
            if usage and usage.billed_units:
                prompt_tokens = usage.billed_units.input_tokens
                output_tokens = usage.billed_units.output_tokens
    ttft = (first_token_t - t0) if first_token_t is not None else None
    return {
        "text": "".join(chunks),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_seconds": ttft,
    }


def run_requests(plan: List[cc.PlannedRequest], args: argparse.Namespace, out_dir: Path, *, mock: bool) -> None:
    cc.run_requests(
        plan, args, out_dir, mock=mock,
        build_client_fn=_build_client,
        call_streaming_fn=_call_cohere_streaming,
        call_non_streaming_fn=_call_cohere_non_streaming,
        price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
        price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
        mock_call_fn=_dispatch_mock_call,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cohere API calibration — dry-run and live black-box latency/TTFT harness."
    )
    cc.add_common_arguments(parser, default_model=DEFAULT_MODEL)
    parser.add_argument("--rpm-limit", type=int, default=20)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return cc.run_calibration_main(
        args,
        root=ROOT,
        provider_display_name="Cohere",
        api_key_env_var="COHERE_API_KEY",
        sdk_package_name="cohere",
        price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
        price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
        live_implemented=True,
        build_client_fn=_build_client,
        call_streaming_fn=_call_cohere_streaming,
        call_non_streaming_fn=_call_cohere_non_streaming,
        mock_call_fn=_dispatch_mock_call,
    )


if __name__ == "__main__":
    raise SystemExit(main())

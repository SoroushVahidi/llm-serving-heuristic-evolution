#!/usr/bin/env python3
"""
Cohere API connectivity/latency smoke test.

This is a minimal sanity check for the real-LLM calibration experiment: it
confirms that COHERE_API_KEY is valid, that the `cohere` SDK talks to the
current v2 chat API correctly, and (optionally) that streaming/TTFT
measurement works. It is NOT the calibration experiment itself — it issues
exactly one (or two, with --stream) tiny request and reports timing/shape,
nothing about accuracy or representative workload.

Never prints the API key or full raw response text unless --debug is passed.

Usage:
    python scripts/smoke_test_cohere_api.py
    python scripts/smoke_test_cohere_api.py --stream
    python scripts/smoke_test_cohere_api.py --model command-r7b-12-2024 --max-tokens 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODEL = "command-r7b-12-2024"  # smallest/cheapest current Command model
DEFAULT_PROMPT = "Say OK."
DEFAULT_MAX_TOKENS = 5


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Cohere API connectivity/latency smoke test."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Cohere model to test (default: {DEFAULT_MODEL}, the cheapest current Command model).",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Prompt text (kept short by default to minimize cost).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Use streaming chat and measure time-to-first-token.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print full raw response text (never the API key). Off by default.",
    )
    return parser.parse_args(argv)


def check_streaming_supported() -> bool:
    try:
        import cohere
        return hasattr(cohere.ClientV2, "chat_stream")
    except ImportError:
        return False


def run_non_streaming(client, model: str, prompt: str, max_tokens: int, debug: bool) -> int:
    t0 = time.monotonic()
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  result:          FAILURE ({type(e).__name__})")
        print(f"  elapsed_s:       {elapsed:.3f}")
        print(f"  error:           {e}")
        return 1

    elapsed = time.monotonic() - t0
    text = resp.message.content[0].text if resp.message.content else ""
    usage = resp.usage
    prompt_tokens = usage.billed_units.input_tokens if usage and usage.billed_units else None
    completion_tokens = usage.billed_units.output_tokens if usage and usage.billed_units else None

    print(f"  result:          SUCCESS")
    print(f"  model:           {model}")
    print(f"  elapsed_s:       {elapsed:.3f}")
    print(f"  prompt_tokens:   {prompt_tokens}")
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  response_length: {len(text)} chars")
    print(f"  streaming_supported: {check_streaming_supported()}")
    if debug:
        print(f"  raw_text:        {text!r}")
    return 0


def run_streaming(client, model: str, prompt: str, max_tokens: int, debug: bool) -> int:
    t0 = time.monotonic()
    first_token_t: Optional[float] = None
    chunks = []
    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    try:
        for event in client.chat_stream(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        ):
            if event.type == "content-delta":
                if first_token_t is None:
                    first_token_t = time.monotonic()
                delta = event.delta
                if delta and delta.message and delta.message.content:
                    chunks.append(delta.message.content.text or "")
            elif event.type == "message-end":
                if event.delta:
                    finish_reason = event.delta.finish_reason
                    usage = event.delta.usage
                    if usage and usage.billed_units:
                        prompt_tokens = usage.billed_units.input_tokens
                        completion_tokens = usage.billed_units.output_tokens
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  result:          FAILURE ({type(e).__name__})")
        print(f"  elapsed_s:       {elapsed:.3f}")
        print(f"  error:           {e}")
        return 1

    total_elapsed = time.monotonic() - t0
    ttft = (first_token_t - t0) if first_token_t is not None else None
    text = "".join(chunks)

    print(f"  result:          SUCCESS")
    print(f"  model:           {model}")
    print(f"  streaming:       True")
    print(f"  ttft_s:          {ttft:.3f}" if ttft is not None else "  ttft_s:          n/a (no content-delta events)")
    print(f"  total_elapsed_s: {total_elapsed:.3f}")
    print(f"  finish_reason:   {finish_reason}")
    print(f"  prompt_tokens:   {prompt_tokens}")
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  response_length: {len(text)} chars")
    if debug:
        print(f"  raw_text:        {text!r}")
    return 0


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    print("Cohere API smoke test")
    print(f"  requested_model: {args.model}")
    print(f"  stream_mode:     {args.stream}")

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        print("  result:          FAILURE")
        print("  error:           COHERE_API_KEY is not set in the environment.")
        print("  fix:             export COHERE_API_KEY=... (see docs/api_provider_setup.md)")
        return 1

    try:
        import cohere
    except ImportError:
        print("  result:          FAILURE")
        print("  error:           cohere SDK not installed. Run: pip install cohere")
        return 1

    client = cohere.ClientV2(api_key=api_key)

    if args.stream:
        if not check_streaming_supported():
            print("  result:          FAILURE")
            print("  error:           Installed cohere SDK does not support chat_stream.")
            return 1
        return run_streaming(client, args.model, args.prompt, args.max_tokens, args.debug)
    else:
        return run_non_streaming(client, args.model, args.prompt, args.max_tokens, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())

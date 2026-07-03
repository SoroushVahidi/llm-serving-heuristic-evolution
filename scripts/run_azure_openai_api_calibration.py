#!/usr/bin/env python3
"""
Azure OpenAI real-LLM calibration — dry-run/mock skeleton.

Uses the shared grid/output schema from
src/llmserveopt/real_llm/calibration_common.py, the same one
scripts/run_cohere_api_calibration.py uses, so a future Azure OpenAI live
pilot produces output directly comparable to the Cohere pilot.

LIVE MODE IS NOT YET IMPLEMENTED. `--allow-live-api` without `--mock` will
refuse with a clear error (exit code 6). Only `--dry-run` and `--mock` are
supported today. See docs/real_llm_multi_provider_plan.md — Azure OpenAI is
planned as a smaller pilot given a $100 total credit budget.

Usage (dry-run):
    python scripts/run_azure_openai_api_calibration.py \\
        --dry-run \\
        --model gpt-4o-mini \\
        --prompt-buckets short,medium,long \\
        --max-tokens-list 64,128,256 \\
        --concurrency-list 1,2,4 \\
        --requests-per-cell 2 \\
        --output-dir experiments/real_llm/azure_openai_pilot_DRYRUN

Required environment (when live mode is implemented): AZURE_OPENAI_API_KEY,
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT (deployment name may differ
from --model, since Azure routes by deployment, not raw model name).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402

DEFAULT_MODEL = "gpt-4o-mini"

# Approximate OpenAI-family small-model pricing (USD per 1M tokens) used only
# for pre-flight cap estimation. Azure OpenAI pricing can differ by region
# and commitment tier — verify against your actual Azure invoice/rate card
# before a real live run, especially given the limited $100 credit budget.
_PRICE_PER_M_INPUT_USD = 0.15
_PRICE_PER_M_OUTPUT_USD = 0.60

API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"
SDK_PACKAGE_NAME = "openai"

# Given the smaller credit budget for this provider, default caps are tighter
# than Cohere's 180-request pilot.
_DEFAULT_MAX_TOTAL_REQUESTS = 60
_DEFAULT_MAX_TOTAL_INPUT_TOKENS = 80_000
_DEFAULT_MAX_TOTAL_OUTPUT_TOKENS = 16_000
_DEFAULT_MAX_ESTIMATED_COST_USD = 2.0


def parse_args(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Azure OpenAI real-LLM calibration (dry-run/mock only; live not yet implemented)."
    )
    cc.add_common_arguments(
        parser,
        default_model=DEFAULT_MODEL,
        default_max_total_requests=_DEFAULT_MAX_TOTAL_REQUESTS,
        default_max_total_input_tokens=_DEFAULT_MAX_TOTAL_INPUT_TOKENS,
        default_max_total_output_tokens=_DEFAULT_MAX_TOTAL_OUTPUT_TOKENS,
        default_max_estimated_cost_usd=_DEFAULT_MAX_ESTIMATED_COST_USD,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return cc.run_calibration_main(
        args,
        root=ROOT,
        provider_display_name="Azure OpenAI",
        api_key_env_var=API_KEY_ENV_VAR,
        sdk_package_name=SDK_PACKAGE_NAME,
        price_per_m_input_usd=_PRICE_PER_M_INPUT_USD,
        price_per_m_output_usd=_PRICE_PER_M_OUTPUT_USD,
        live_implemented=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Download Azure LLM Inference Dataset 2024 (large files — user-initiated only).

File sizes:
  code trace: ~692 MB
  conv trace: ~1.1 GB

License: CC-BY Attribution License
Citation: Stojkovic et al., "DynamoLLM" HPCA 2025.

Usage
-----
python scripts/data/download_azure_llm_2024.py \\
    --output-dir data/raw/azure \\
    [--code-only] \\
    [--conv-only]

After download, convert with:
  python scripts/data/convert_azure_llm_trace.py \\
      --input data/raw/azure/AzureLLMInferenceTrace_code_2024.csv \\
      --output data/processed/azure/azure_llm_2024_code.jsonl \\
      --source azure_2024_code \\
      --time-scale 0.01 \\
      --max-requests 10000

IMPORTANT: These files are NOT committed to git (data/raw/* is gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

AZURE_2024_URLS = {
    "code": "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv",
    "conv": "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv",
}
AZURE_2024_SIZES = {
    "code": 691_989_454,  # bytes (~692 MB)
    "conv": 1_135_195_393,  # bytes (~1.08 GB)
}
AZURE_2024_LOCAL_NAMES = {
    "code": "AzureLLMInferenceTrace_code_2024.csv",
    "conv": "AzureLLMInferenceTrace_conv_2024.csv",
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {pct:5.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)
    else:
        mb = downloaded / 1e6
        print(f"\r  {mb:.0f} MB downloaded", end="", flush=True)


def download_azure_2024(output_dir: Path, traces: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        url = AZURE_2024_URLS[trace]
        local_name = AZURE_2024_LOCAL_NAMES[trace]
        dest = output_dir / local_name
        expected_size = AZURE_2024_SIZES[trace]

        if dest.exists() and dest.stat().st_size == expected_size:
            print(f"  {local_name}: already downloaded ({dest.stat().st_size / 1e6:.0f} MB)")
            continue

        print(f"\nDownloading {local_name} (~{expected_size / 1e9:.2f} GB)...")
        print(f"  URL: {url}")
        print(f"  Destination: {dest}")
        try:
            urllib.request.urlretrieve(url, dest, _progress_hook)
            print()
            actual_size = dest.stat().st_size
            if actual_size != expected_size:
                print(f"  WARNING: expected {expected_size} bytes, got {actual_size} bytes")
            else:
                print(f"  Download complete: {actual_size / 1e6:.0f} MB")
        except Exception as e:
            print(f"\n  ERROR: {e}", file=sys.stderr)
            if dest.exists():
                dest.unlink()
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Azure LLM Inference Dataset 2024 (large files)"
    )
    parser.add_argument("--output-dir", default="data/raw/azure",
                        help="Directory to save downloaded files")
    parser.add_argument("--code-only", action="store_true",
                        help="Download only the code trace (~692 MB)")
    parser.add_argument("--conv-only", action="store_true",
                        help="Download only the conv trace (~1.1 GB)")
    args = parser.parse_args()

    traces = []
    if args.code_only:
        traces = ["code"]
    elif args.conv_only:
        traces = ["conv"]
    else:
        traces = ["code", "conv"]

    print("Azure LLM Inference Dataset 2024")
    print("  License:  CC-BY Attribution License")
    print("  Citation: DynamoLLM, HPCA 2025")
    print(f"  Traces:   {traces}")

    download_azure_2024(Path(args.output_dir), traces)
    print("\nDone. Convert with:")
    for trace in traces:
        local = AZURE_2024_LOCAL_NAMES[trace]
        print(f"  python scripts/data/convert_azure_llm_trace.py \\")
        print(f"      --input data/raw/azure/{local} \\")
        print(f"      --output data/processed/azure/azure_llm_2024_{trace}.jsonl \\")
        print(f"      --source azure_2024_{trace} --time-scale 0.01 --max-requests 10000")


if __name__ == "__main__":
    main()

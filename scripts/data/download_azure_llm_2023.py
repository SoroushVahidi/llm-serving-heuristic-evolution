#!/usr/bin/env python3
"""
Download Azure LLM Inference Dataset 2023 code + conversation CSVs.

Official pin: Azure/AzurePublicDataset AzureLLMInferenceDataset2023.md
License: CC-BY Attribution. No function-calling subset is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

FILES = {
    "AzureLLMInferenceTrace_code_2023.csv": (
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv"
    ),
    "AzureLLMInferenceTrace_conv_2023.csv": (
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv"
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(name: str, url: str, dest_dir: Path) -> dict:
    dest = dest_dir / name
    partial = dest_dir / f"{name}.partial"
    if dest.exists() and dest.stat().st_size > 0:
        return {
            "file": name,
            "status": "already_present",
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "url": url,
        }
    print(f"Downloading {name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "llmserveopt-azure2023/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(partial, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if partial.stat().st_size <= 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: empty download")
    digest = sha256_file(partial)
    partial.replace(dest)
    return {
        "file": name,
        "status": "downloaded",
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "url": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = [download_one(n, u, out_dir) for n, u in FILES.items()]
    for r in results:
        print(f"  OK {r['file']} bytes={r['bytes']} sha256={r['sha256']}")
    # Explicitly document absence of function-calling subset.
    manifest = {
        "dataset": "Azure_LLM_Inference_2023",
        "official_pin": "AzureLLMInferenceDataset2023.md",
        "official_source": "https://github.com/Azure/AzurePublicDataset",
        "code_repo_license": "CC-BY-4.0",
        "data_license": "CC-BY-4.0",
        "access_utc": started,
        "files": results,
        "missing_splits_note": "No public function-calling subset (do not fabricate)",
    }
    man_path = Path(args.manifest_out) if args.manifest_out else out_dir / "download_manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

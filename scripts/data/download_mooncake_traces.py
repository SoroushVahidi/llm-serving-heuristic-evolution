#!/usr/bin/env python3
"""
Download Mooncake FAST25 traces with real/synthetic classification.

CODE_REPO_LICENSE = Apache-2.0
DATA_LICENSE = NOT_EXPLICITLY_SPECIFIED
REDISTRIBUTION = PROHIBITED_UNTIL_CLARIFIED

Real files → --real-dir
Synthetic → --synthetic-dir (quarantine; not converted as real)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

TRACE_BASE = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces"
)

CLASSIFICATION = {
    "conversation_trace.jsonl": {
        "class": "real_production_derived",
        "expected_requests": 12031,
        "destination": "real",
    },
    "toolagent_trace.jsonl": {
        "class": "real_production_derived",
        "expected_requests": 23608,
        "destination": "real",
    },
    "synthetic_trace.jsonl": {
        "class": "synthetic",
        "expected_requests": 3993,
        "destination": "synthetic",
        "arrival": "poisson",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_nonempty_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def download_one(name: str, dest_dir: Path) -> dict:
    meta = CLASSIFICATION[name]
    dest = dest_dir / name
    partial = dest_dir / f"{name}.partial"
    url = f"{TRACE_BASE}/{name}"
    if dest.exists() and dest.stat().st_size > 0:
        rows = count_nonempty_lines(dest)
        return {
            "file": name,
            "status": "already_present",
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "url": url,
            "class": meta["class"],
            "rows": rows,
            "expected_requests": meta["expected_requests"],
        }
    print(f"Downloading {name} ({meta['class']})...")
    req = urllib.request.Request(url, headers={"User-Agent": "llmserveopt-mooncake/1.0"})
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
    rows = count_nonempty_lines(dest)
    if rows != meta["expected_requests"]:
        raise RuntimeError(
            f"{name}: expected {meta['expected_requests']} rows, got {rows}"
        )
    return {
        "file": name,
        "status": "downloaded",
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "url": url,
        "class": meta["class"],
        "rows": rows,
        "expected_requests": meta["expected_requests"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Do not download synthetic_trace.jsonl",
    )
    args = parser.parse_args()
    real_dir = Path(args.real_dir)
    syn_dir = Path(args.synthetic_dir)
    real_dir.mkdir(parents=True, exist_ok=True)
    syn_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = []
    for name, meta in CLASSIFICATION.items():
        if meta["destination"] == "synthetic" and args.skip_synthetic:
            results.append(
                {
                    "file": name,
                    "status": "skipped",
                    "class": meta["class"],
                }
            )
            continue
        dest_dir = real_dir if meta["destination"] == "real" else syn_dir
        results.append(download_one(name, dest_dir))
        print(f"  OK {name} class={meta['class']} sha256={results[-1].get('sha256')}")

    ambiguous = []  # none known under FAST25-release/traces/
    manifest = {
        "dataset": "Mooncake_Kimi_Traces",
        "official_pin": "FAST25-release/traces",
        "official_source": "https://github.com/kvcache-ai/Mooncake",
        "code_repo_license": "Apache-2.0",
        "data_license": "NOT_EXPLICITLY_SPECIFIED",
        "redistribution": "PROHIBITED_UNTIL_CLARIFIED",
        "access_utc": started,
        "real_files": [r for r in results if r.get("class") == "real_production_derived"],
        "synthetic_files": [r for r in results if r.get("class") == "synthetic"],
        "ambiguous_files": ambiguous,
        "files": results,
        "real_only_enforcement": True,
    }
    man_path = (
        Path(args.manifest_out)
        if args.manifest_out
        else real_dir.parent / "manifests" / "download_manifest.json"
    )
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path}")
    if ambiguous:
        print("ERROR: ambiguous Mooncake files present; refusing success", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

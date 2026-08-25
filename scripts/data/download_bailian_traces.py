#!/usr/bin/env python3
"""
Download Bailian/Qwen anonymized usage traces from pinned commit via GitHub LFS media.

Pinned commit: 5f7439c51ec248a0c585f7d90a41a6f57773b912
DATA_LICENSE = Apache-2.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

PINNED_COMMIT = "5f7439c51ec248a0c585f7d90a41a6f57773b912"
MEDIA_BASE = (
    "https://media.githubusercontent.com/media/alibaba-edu/"
    f"qwen-bailian-usagetraces-anon/{PINNED_COMMIT}"
)

# Expected sizes from repository status JSON (bytes).
ASSETS = {
    "qwen_traceA_blksz_16.jsonl": 56_354_493,
    "qwen_traceB_blksz_16.jsonl": 96_209_982,
    "qwen_thinking_blksz_16.jsonl": 27_901_454,
    "qwen_coder_blksz_16.jsonl": 132_054_902,
}

SPLIT_MAP = {
    "qwen_traceA_blksz_16.jsonl": "to_c_traceA",
    "qwen_traceB_blksz_16.jsonl": "to_b_traceB",
    "qwen_thinking_blksz_16.jsonl": "thinking",
    "qwen_coder_blksz_16.jsonl": "coder",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(name: str, dest_dir: Path) -> dict:
    expected = ASSETS[name]
    dest = dest_dir / name
    partial = dest_dir / f"{name}.partial"
    url = f"{MEDIA_BASE}/{name}"
    if dest.exists() and dest.stat().st_size == expected:
        return {
            "file": name,
            "status": "already_present",
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "url": url,
            "source_split": SPLIT_MAP[name],
        }
    print(f"Downloading {name} ({expected} bytes)...")
    req = urllib.request.Request(url, headers={"User-Agent": "llmserveopt-bailian/1.0"})
    with urllib.request.urlopen(req, timeout=1200) as resp, open(partial, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    actual = partial.stat().st_size
    if actual != expected:
        # Treat published sizes as strong expectations; fail closed.
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: expected {expected} bytes, got {actual}")
    # Reject Git LFS pointer stubs (~100 bytes).
    head = open(dest if False else partial, "rb").read(100)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: received Git LFS pointer instead of payload")
    digest = sha256_file(partial)
    partial.replace(dest)
    return {
        "file": name,
        "status": "downloaded",
        "bytes": actual,
        "sha256": digest,
        "url": url,
        "source_split": SPLIT_MAP[name],
        "expected_bytes": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = [download_one(name, out_dir) for name in ASSETS]
    for r in results:
        print(f"  OK {r['file']} sha256={r['sha256']}")
    # Also fetch LICENSE + README for provenance (small).
    for meta in ("LICENSE", "README.md"):
        url = (
            "https://raw.githubusercontent.com/alibaba-edu/qwen-bailian-usagetraces-anon/"
            f"{PINNED_COMMIT}/{meta}"
        )
        dest = out_dir / meta
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
    manifest = {
        "dataset": "Qwen_Bailian_Anonymous_Traces",
        "official_pin": f"commit:{PINNED_COMMIT}",
        "official_source": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
        "code_repo_license": "Apache-2.0",
        "data_license": "Apache-2.0",
        "access_utc": started,
        "files": results,
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

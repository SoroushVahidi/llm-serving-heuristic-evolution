#!/usr/bin/env python3
"""
Download BurstGPT release v2.0 assets (official HPMLL/BurstGPT).

License: CC-BY-4.0 (code and data).
Never prints credentials. Writes to .partial then atomically renames.

Usage:
  python scripts/data/download_burstgpt_v2.py \\
      --output-dir /mmfs1/.../datasets/burstgpt_v2/raw \\
      --files without_fails
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

RELEASE_TAG = "v2.0"
RELEASE_BASE = f"https://github.com/HPMLL/BurstGPT/releases/download/{RELEASE_TAG}"

ASSETS = {
    "BurstGPT_1.csv": 52_283_111,
    "BurstGPT_2.csv": 144_819_209,
    "BurstGPT_3.csv": 231_682_327,
    "BurstGPT_without_fails_1.csv": 51_429_517,
    "BurstGPT_without_fails_2.csv": 142_376_815,
    "BurstGPT_without_fails_3.csv": 217_312_026,
}

FILE_GROUPS = {
    "all": list(ASSETS.keys()),
    "without_fails": [
        "BurstGPT_without_fails_1.csv",
        "BurstGPT_without_fails_2.csv",
        "BurstGPT_without_fails_3.csv",
    ],
    "raw": ["BurstGPT_1.csv", "BurstGPT_2.csv", "BurstGPT_3.csv"],
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
    url = f"{RELEASE_BASE}/{name}"
    if dest.exists() and dest.stat().st_size == expected:
        digest = sha256_file(dest)
        return {
            "file": name,
            "status": "already_present",
            "bytes": dest.stat().st_size,
            "sha256": digest,
            "url": url,
        }

    print(f"Downloading {name} ({expected} bytes)...")
    print(f"  URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "llmserveopt-burstgpt-v2/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(partial, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    actual = partial.stat().st_size
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: expected {expected} bytes, got {actual}")
    digest = sha256_file(partial)
    partial.replace(dest)
    return {
        "file": name,
        "status": "downloaded",
        "bytes": actual,
        "sha256": digest,
        "url": url,
        "expected_bytes": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BurstGPT v2.0 release assets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--files",
        choices=sorted(FILE_GROUPS.keys()),
        default="without_fails",
        help="Asset group to download (default: without_fails)",
    )
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for name in FILE_GROUPS[args.files]:
        results.append(download_one(name, out_dir))
        print(f"  OK {name} sha256={results[-1]['sha256']}")

    manifest = {
        "dataset": "BurstGPT",
        "release_tag": RELEASE_TAG,
        "official_source": "https://github.com/HPMLL/BurstGPT",
        "code_repo_license": "CC-BY-4.0",
        "data_license": "CC-BY-4.0",
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

#!/usr/bin/env python3
"""
Download BurstGPT dataset from the official HuggingFace repository.

Source: https://huggingface.co/datasets/HKUDS/BurstGPT
Paper:  arXiv 2401.17644 (SIGMETRICS 2025)
License: MIT

Usage:
    python scripts/download_burstgpt.py --output data/raw/burstgpt/
    python scripts/download_burstgpt.py --output data/raw/burstgpt/ --dry-run
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATASET_URLS = [
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv",
    "https://raw.githubusercontent.com/HKUDS/BurstGPT/main/data/BurstGPT_without_fails.csv",
    "https://huggingface.co/datasets/HKUDS/BurstGPT/resolve/main/data/BurstGPT_without_fails.csv",
]

# The primary file in HPMLL/BurstGPT is BurstGPT_1.csv
# HKUDS/BurstGPT (original) used BurstGPT_without_fails.csv but that repo
# may be renamed or moved. Both CSVs use identical column schema.
FILENAME = "BurstGPT_1.csv"
LICENSE = "MIT"
PAPER = "arXiv 2401.17644 (SIGMETRICS 2025)"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksum(checksum_file: Path, filename: str) -> str | None:
    if not checksum_file.exists():
        return None
    with open(checksum_file) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[1] == filename:
                return parts[0]
    return None


def save_checksum(checksum_file: Path, filename: str, checksum: str) -> None:
    lines = []
    if checksum_file.exists():
        with open(checksum_file) as f:
            lines = [ln for ln in f.readlines() if not ln.strip().endswith(filename)]
    lines.append(f"{checksum}  {filename}\n")
    with open(checksum_file, "w") as f:
        f.writelines(lines)


def download_file(url: str, dest: Path) -> bool:
    print(f"  Trying: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 BurstGPT-Downloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  Downloaded: {downloaded:,} / {total:,} bytes ({pct:.1f}%)", end="", flush=True)
            print()
        return True
    except urllib.error.URLError as e:
        print(f"  Failed: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download BurstGPT dataset")
    parser.add_argument("--output", default="data/raw/burstgpt/", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without downloading")
    args = parser.parse_args()

    print(f"\nBurstGPT Dataset Downloader")
    print(f"  Paper   : {PAPER}")
    print(f"  License : {LICENSE}")
    print(f"  Source  : https://github.com/HKUDS/BurstGPT")
    print()

    out_dir = Path(args.output)
    dest = out_dir / FILENAME
    checksum_file = out_dir / "checksums.sha256"

    if args.dry_run:
        print(f"[dry-run] Would download to: {dest}")
        print(f"[dry-run] URLs to try:")
        for url in DATASET_URLS:
            print(f"  {url}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        existing_checksum = sha256sum(dest)
        saved_checksum = load_checksum(checksum_file, FILENAME)
        if saved_checksum and existing_checksum == saved_checksum:
            print(f"File already exists and checksum matches: {dest}")
            print(f"SHA256: {existing_checksum}")
            return
        else:
            print(f"File exists but checksum mismatch or no saved checksum. Re-downloading.")

    print(f"Downloading to: {dest}")
    success = False
    for url in DATASET_URLS:
        if download_file(url, dest):
            success = True
            break

    if not success:
        if dest.exists():
            dest.unlink()
        print("\nERROR: All download attempts failed.")
        print("\nTo download manually:")
        for url in DATASET_URLS:
            print(f"  wget '{url}' -O '{dest}'")
        print(f"\nOr clone from HuggingFace:")
        print(f"  git lfs install")
        print(f"  git clone https://huggingface.co/datasets/HKUDS/BurstGPT {out_dir}/hf/")
        sys.exit(1)

    checksum = sha256sum(dest)
    save_checksum(checksum_file, FILENAME, checksum)
    file_size = dest.stat().st_size
    print(f"\nDownload complete!")
    print(f"  File    : {dest}")
    print(f"  Size    : {file_size:,} bytes")
    print(f"  SHA256  : {checksum}")
    print(f"\nNext step:")
    print(f"  python scripts/convert_burstgpt.py --input {dest} --output data/processed/burstgpt/burstgpt_10k.jsonl")


if __name__ == "__main__":
    main()

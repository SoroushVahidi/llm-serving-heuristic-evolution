#!/usr/bin/env python3
"""Verify Tier 1 dataset checksum files when present under a datasets root."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_one(ds_dir: Path) -> list[str]:
    errors: list[str] = []
    ck = ds_dir / "checksums.sha256"
    if not ck.exists():
        return [f"{ds_dir.name}: missing checksums.sha256 (skip)"]
    for line in ck.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, rel = parts[0], parts[-1].lstrip("*")
        target = ds_dir / rel
        if not target.exists():
            errors.append(f"{ds_dir.name}: missing {rel}")
            continue
        got = sha256_file(target)
        if got != digest:
            errors.append(f"{ds_dir.name}: mismatch {rel}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.datasets_root)
    if not root.exists():
        print(f"missing datasets root: {root}", file=sys.stderr)
        return 2
    all_err: list[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "global_manifests"):
        if args.dry_run:
            print(f"would verify {d}")
            continue
        errs = verify_one(d)
        for e in errs:
            print(e)
        # "missing checksums" is informational skip, not hard fail unless mismatches
        all_err.extend([e for e in errs if "mismatch" in e or "missing " in e and "checksums" not in e])
    if all_err:
        print(f"FAILED {len(all_err)} issues", file=sys.stderr)
        return 1
    print("OK (or skipped datasets without checksums.sha256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

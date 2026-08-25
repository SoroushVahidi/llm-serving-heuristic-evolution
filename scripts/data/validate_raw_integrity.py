#!/usr/bin/env python3
"""
Raw-file integrity: sha256, bytes, row counts, header signature, first/last timestamps.

Does not print request content / prompts / responses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_azure_ts(ts: str) -> float:
    ts = ts.strip()
    if "." in ts:
        base, frac = ts.rsplit(".", 1)
        ts = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(ts.replace(" ", "T")).timestamp()


def inspect_csv(path: Path, timestamp_col: Optional[str] = None) -> Dict[str, Any]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"rows": 0, "header": [], "error": "empty"}
        rows = 0
        first_ts = None
        last_ts = None
        ts_idx = None
        if timestamp_col and timestamp_col in header:
            ts_idx = header.index(timestamp_col)
        elif "TIMESTAMP" in header:
            ts_idx = header.index("TIMESTAMP")
        elif "Timestamp" in header:
            ts_idx = header.index("Timestamp")
        elif "timestamp" in header:
            ts_idx = header.index("timestamp")
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            # Skip duplicate header rows.
            if row == header:
                continue
            rows += 1
            if ts_idx is not None and ts_idx < len(row):
                raw = row[ts_idx].strip()
                try:
                    if "T" in raw or "-" in raw[:5]:
                        val = _parse_azure_ts(raw)
                    else:
                        val = float(raw)
                except Exception:
                    continue
                if first_ts is None:
                    first_ts = val
                last_ts = val
    return {
        "rows": rows,
        "header": header,
        "header_signature": "|".join(header),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "timestamp_unit_hint": "iso_or_numeric",
    }


def inspect_jsonl(path: Path, timestamp_key: str = "timestamp") -> Dict[str, Any]:
    rows = 0
    first_ts = None
    last_ts = None
    keys = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows += 1
            if keys is None:
                keys = sorted(obj.keys())
            if timestamp_key in obj:
                val = float(obj[timestamp_key])
                if first_ts is None:
                    first_ts = val
                last_ts = val
    return {
        "rows": rows,
        "header": keys or [],
        "header_signature": "|".join(keys or []),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "timestamp_unit_hint": "numeric_jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--format", choices=["csv", "jsonl", "auto"], default="auto")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: missing {path}", file=sys.stderr)
        sys.exit(1)
    fmt = args.format
    if fmt == "auto":
        fmt = "jsonl" if path.suffix == ".jsonl" else "csv"

    info = inspect_jsonl(path) if fmt == "jsonl" else inspect_csv(path)
    record = {
        "path": str(path),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": fmt,
        "source_url": args.source_url,
        "source_revision": args.source_revision,
        "acquisition_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **info,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote integrity record → {out}")
    print(
        f"  bytes={record['bytes']} rows={record['rows']} "
        f"sha256={record['sha256'][:16]}..."
    )


if __name__ == "__main__":
    main()

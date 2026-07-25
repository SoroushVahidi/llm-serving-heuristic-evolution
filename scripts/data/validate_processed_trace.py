#!/usr/bin/env python3
"""
Validate processed extended JSONL traces for schema, chronology, duplicates,
and actual_output_tokens leakage into scheduler-visible objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import ObservableRequest
from llmserveopt.workloads.canonical_schema import assert_no_actual_output_leakage
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl


def validate(path: Path) -> dict:
    requests, metadata = load_extended_jsonl(path)
    n = len(requests)
    issues = []
    assert_no_actual_output_leakage()
    obs_fields = {f.name for f in ObservableRequest.__dataclass_fields__.values()}
    if "actual_output_tokens" in obs_fields:
        issues.append("ObservableRequest exposes actual_output_tokens")

    # Chronology
    prev = -1.0
    chrono_ok = True
    for r in requests:
        if r.arrival_time < prev:
            chrono_ok = False
            break
        prev = r.arrival_time
    if not chrono_ok:
        issues.append("not_chronological")

    neg_ts = sum(1 for r in requests if r.arrival_time < 0)
    neg_prompt = sum(1 for r in requests if r.prompt_tokens < 0)
    neg_out = sum(1 for r in requests if r.actual_output_tokens < 0)
    zero_out = sum(1 for r in requests if r.actual_output_tokens == 0)
    dup_ids = [i for i, c in Counter(r.request_id for r in requests).items() if c > 1]

    source_ids = []
    for md in metadata:
        sid = md.get("source_record_id")
        if sid is not None:
            source_ids.append(str(sid))
    dup_source = [i for i, c in Counter(source_ids).items() if c > 1]

    # Timestamp ties
    arrivals = [r.arrival_time for r in requests]
    ties = sum(1 for a, b in zip(arrivals, arrivals[1:]) if a == b)

    # Leakage check on serialized observable view
    for r in requests[: min(100, n)]:
        obs = ObservableRequest(
            request_id=r.request_id,
            arrival_time=r.arrival_time,
            prompt_tokens=r.prompt_tokens,
            predicted_output_tokens=r.predicted_output_tokens,
            slo_deadline=r.slo_deadline,
            priority=r.priority,
            class_id=r.class_id,
        )
        payload = obs.__dict__
        if "actual_output_tokens" in payload:
            issues.append("observable_payload_leak")
            break

    report = {
        "path": str(path),
        "n_requests": n,
        "chronological": chrono_ok,
        "negative_timestamps": neg_ts,
        "negative_prompt_tokens": neg_prompt,
        "negative_output_tokens": neg_out,
        "zero_output_requests": zero_out,
        "duplicate_request_ids": len(dup_ids),
        "duplicate_source_record_ids": len(dup_source),
        "timestamp_ties": ties,
        "max_prompt_tokens": max((r.prompt_tokens for r in requests), default=None),
        "max_output_tokens": max((r.actual_output_tokens for r in requests), default=None),
        "actual_output_leakage_protected": "actual_output_tokens" not in obs_fields,
        "issues": issues,
        "ok": not issues and neg_ts == 0 and neg_prompt == 0 and len(dup_ids) == 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = validate(Path(args.input))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("ok", "n_requests", "chronological", "issues")}, indent=2))
    if not report["ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

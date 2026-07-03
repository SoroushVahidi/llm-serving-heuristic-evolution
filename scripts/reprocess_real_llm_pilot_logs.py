#!/usr/bin/env python3
"""
Reprocess an existing real-LLM calibration pilot's requests.jsonl to produce
a corrected-vs-raw latency summary, without making any API calls.

Motivation: logs written before the rate_limiter_wait_seconds /
provider_request_latency_seconds split in
src/llmserveopt/real_llm/calibration_common.py (see execute_one_request)
recorded a single elapsed_seconds/total_latency_seconds field that could
include time the request spent blocked in the local RPM limiter, not just
provider response time. This script reads the already-completed
requests.jsonl for a pilot, heuristically flags requests whose latency looks
RPM-wait-polluted (see reprocess_legacy_summary docstring for the exact
rule), and writes summary_corrected.json/.md into the same output directory.

Never contacts any provider API. Safe to run against any existing pilot
directory, live or mock.

Usage:
    python scripts/reprocess_real_llm_pilot_logs.py \\
        --input-dir experiments/real_llm/cohere_pilot_20260703T040421Z
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True,
        help="Pilot output directory containing requests.jsonl (also where "
        "summary_corrected.json/.md are written).",
    )
    parser.add_argument(
        "--min-ttft-gap-seconds", type=float, default=5.0,
        help="Flag a streaming request as likely RPM-wait-polluted if "
        "latency - ttft exceeds this many seconds.",
    )
    parser.add_argument(
        "--min-absolute-latency-seconds", type=float, default=10.0,
        help="Flag a non-streaming request (no ttft) as likely RPM-wait-"
        "polluted if latency alone exceeds this many seconds.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = ROOT / in_dir
    requests_path = in_dir / "requests.jsonl"
    if not requests_path.exists():
        print(f"ERROR: {requests_path} does not exist.", file=sys.stderr)
        return 2

    reprocessed = cc.reprocess_legacy_summary(
        requests_path,
        min_ttft_gap_seconds=args.min_ttft_gap_seconds,
        min_absolute_latency_seconds=args.min_absolute_latency_seconds,
    )
    cc.write_legacy_reprocessed_summary(in_dir, reprocessed)

    print(f"Reprocessed {reprocessed['n_success']} successful requests from {requests_path}")
    print(f"  flagged likely-RPM-wait outliers: {reprocessed['n_flagged_likely_rate_limiter_wait']}")
    print(f"  raw p50/p95/p99 latency (s):       "
          f"{reprocessed['raw_stats']['p50_latency_s']} / "
          f"{reprocessed['raw_stats']['p95_latency_s']} / "
          f"{reprocessed['raw_stats']['p99_latency_s']}")
    print(f"  corrected p50/p95/p99 latency (s): "
          f"{reprocessed['corrected_stats_excluding_flagged']['p50_latency_s']} / "
          f"{reprocessed['corrected_stats_excluding_flagged']['p95_latency_s']} / "
          f"{reprocessed['corrected_stats_excluding_flagged']['p99_latency_s']}")
    print(f"  wrote {in_dir / 'summary_corrected.json'}")
    print(f"  wrote {in_dir / 'summary_corrected.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

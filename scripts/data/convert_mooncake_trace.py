#!/usr/bin/env python3
"""
Convert Mooncake / Kimi serving JSONL to extended simulator JSONL.

Official source: https://github.com/kvcache-ai/Mooncake (FAST25-release/traces)
License: Apache-2.0 (repository)

Mark synthetic_trace.jsonl with --treat-as-synthetic.

Usage
-----
python scripts/data/convert_mooncake_trace.py \\
    --input /path/to/conversation_trace.jsonl \\
    --output /path/to/processed/mooncake_conversation.jsonl \\
    --source-split conversation_trace \\
    --time-scale 1.0 \\
    --max-requests 10000 \\
    --seed 17
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.workloads.augmentation import AugmentationConfig
from llmserveopt.workloads.mooncake import MooncakeConversionConfig, load_mooncake_trace
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Mooncake JSONL trace")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-split", default="conversation_trace")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--treat-as-synthetic",
        action="store_true",
        help="Label as synthetic/trace-calibrated (required for synthetic_trace.jsonl)",
    )
    args = parser.parse_args()

    cfg = MooncakeConversionConfig(
        max_requests=args.max_requests,
        time_scale=args.time_scale,
        source_split=args.source_split,
        treat_as_synthetic=args.treat_as_synthetic,
    )
    requests, metadata, report = load_mooncake_trace(
        args.input, config=cfg, seed=args.seed, augmentation_config=AugmentationConfig()
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_extended_jsonl(
        requests,
        out,
        source=f"mooncake_{args.source_split}",
        metadata_list=metadata,
    )
    report_path = out.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Saved {len(requests)} requests → {out}")
    print(f"Report → {report_path}")
    print(f"dataset_type={report.dataset_type} replay_label={report.replay_label}")


if __name__ == "__main__":
    main()

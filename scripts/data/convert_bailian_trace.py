#!/usr/bin/env python3
"""
Convert Bailian/Qwen anonymized serving JSONL to extended simulator JSONL.

Official source: https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
License: Apache-2.0

Files are Git LFS objects (~28–132 MB each). Do not commit downloaded rows.

Usage
-----
python scripts/data/convert_bailian_trace.py \\
    --input /path/to/qwen_traceA_blksz_16.jsonl \\
    --output /path/to/processed/bailian_traceA.jsonl \\
    --source-split to_c_traceA \\
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
from llmserveopt.workloads.bailian import BailianConversionConfig, load_bailian_trace
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Bailian/Qwen JSONL trace")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-split", default="traceA")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cfg = BailianConversionConfig(
        max_requests=args.max_requests,
        time_scale=args.time_scale,
        source_split=args.source_split,
    )
    requests, metadata, report = load_bailian_trace(
        args.input, config=cfg, seed=args.seed, augmentation_config=AugmentationConfig()
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_extended_jsonl(
        requests,
        out,
        source=f"bailian_{args.source_split}",
        metadata_list=metadata,
    )
    report_path = out.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Saved {len(requests)} requests → {out}")
    print(f"Report → {report_path}")
    print(f"replay_label={report.replay_label} time_scale={report.time_scale}")


if __name__ == "__main__":
    main()

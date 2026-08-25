#!/usr/bin/env python3
"""
Convert raw BurstGPT CSV to simulator JSONL format.

Usage:
    python scripts/convert_burstgpt.py \
        --input data/raw/burstgpt/BurstGPT_without_fails.csv \
        --output data/processed/burstgpt/burstgpt_10k.jsonl \
        --max-requests 10000 \
        --seed 17 \
        --config configs/traces/burstgpt_conversion.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.workloads.burstgpt import (
    BurstGPTConversionConfig,
    conversion_report_to_dict,
    load_burstgpt_trace_with_metadata,
)
from llmserveopt.workloads.augmentation import load_augmentation_config
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl


def main():
    parser = argparse.ArgumentParser(description="Convert BurstGPT CSV to simulator JSONL")
    parser.add_argument("--input", required=True, help="Path to raw BurstGPT CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-requests", type=int, default=None, help="Max requests to include")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for augmentation")
    parser.add_argument("--config", default=None, help="Path to YAML conversion config")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Multiply interarrivals by this factor")
    parser.add_argument("--start-time", type=float, default=None, help="Filter: start timestamp")
    parser.add_argument("--end-time", type=float, default=None, help="Filter: end timestamp")
    parser.add_argument(
        "--chunked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream CSV in chunks (default: true). Use --no-chunked for legacy full read_csv.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per chunk when --chunked is enabled",
    )
    args = parser.parse_args()

    cfg_dict = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)

    conv_cfg_dict = cfg_dict.get("conversion", {})
    max_requests = args.max_requests or conv_cfg_dict.get("max_requests", None)
    time_scale = args.time_scale if args.time_scale != 1.0 else float(conv_cfg_dict.get("time_scale", 1.0))

    conversion_config = BurstGPTConversionConfig(
        start_time=args.start_time,
        end_time=args.end_time,
        max_requests=max_requests,
        time_scale=time_scale,
        min_prompt_tokens=int(conv_cfg_dict.get("min_prompt_tokens", 1)),
        min_output_tokens=int(conv_cfg_dict.get("min_output_tokens", 1)),
        max_prompt_tokens=int(conv_cfg_dict.get("max_prompt_tokens", 32768)),
        max_output_tokens=int(conv_cfg_dict.get("max_output_tokens", 32768)),
    )

    aug_config = load_augmentation_config(cfg_dict) if cfg_dict else None

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting BurstGPT trace:")
    print(f"  Input : {input_path}")
    print(f"  Output: {args.output}")
    print(f"  Seed  : {args.seed}")

    requests, metadata, report = load_burstgpt_trace_with_metadata(
        input_path,
        conversion_config,
        args.seed,
        aug_config,
        use_chunked=args.chunked,
        chunksize=args.chunksize,
    )

    print(f"\nConversion report:")
    print(f"  Rows read         : {report.rows_read}")
    print(f"  Rows retained     : {report.rows_retained}")
    print(f"  Rows dropped zero : {report.rows_dropped_zero_tokens}")
    print(f"  Time range        : {report.time_range_seconds:.2f} s")
    print(f"  Mean arrival rate : {report.mean_arrival_rate:.2f} req/s")
    print(f"  Prompt tokens p95 : {report.prompt_tokens_p95:.0f}")
    print(f"  Output tokens p95 : {report.output_tokens_p95:.0f}")

    output_path = Path(args.output)
    save_extended_jsonl(
        requests, output_path, source="burstgpt", metadata_list=metadata
    )
    print(f"\nSaved {len(requests)} requests to {output_path}")

    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(conversion_report_to_dict(report), f, indent=2)
    print(f"Saved conversion report to {report_path}")


if __name__ == "__main__":
    main()

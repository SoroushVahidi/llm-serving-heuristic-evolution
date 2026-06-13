#!/usr/bin/env python3
"""
Convert raw ShareGPT JSON to simulator JSONL format.

Usage:
    python scripts/convert_sharegpt.py \
        --input data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
        --output data/processed/sharegpt/sharegpt_10k.jsonl \
        --arrival-mode poisson \
        --arrival-rate 10.0 \
        --duration 3600.0 \
        --seed 17 \
        --config configs/traces/sharegpt_conversion.yaml
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.workloads.sharegpt import (
    ShareGPTConversionConfig,
    ShareGPTConversionReport,
    convert_sharegpt_to_requests,
    load_sharegpt_raw,
)
from llmserveopt.workloads.augmentation import load_augmentation_config
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl


def report_to_dict(report: ShareGPTConversionReport) -> dict:
    return {
        "rows_read": report.rows_read,
        "pairs_extracted": report.pairs_extracted,
        "pairs_skipped": report.pairs_skipped,
        "rows_retained": report.rows_retained,
        "time_range_seconds": report.time_range_seconds,
        "mean_arrival_rate": report.mean_arrival_rate,
        "prompt_tokens_mean": report.prompt_tokens_mean,
        "prompt_tokens_p95": report.prompt_tokens_p95,
        "output_tokens_mean": report.output_tokens_mean,
        "output_tokens_p95": report.output_tokens_p95,
        "tokenizer_used": report.tokenizer_used,
        "seed": report.seed,
        "augmentation_config_summary": report.augmentation_config_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT JSON to simulator JSONL")
    parser.add_argument("--input", required=True, help="Path to raw ShareGPT JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--arrival-mode", default="poisson", choices=["poisson", "bursty", "mmpp"],
                        help="Arrival process model")
    parser.add_argument("--arrival-rate", type=float, default=10.0, help="Mean arrival rate (req/s)")
    parser.add_argument("--duration", type=float, default=3600.0, help="Trace duration (s)")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--config", default=None, help="Path to YAML conversion config")
    parser.add_argument("--max-requests", type=int, default=None, help="Max requests to include")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name")
    args = parser.parse_args()

    cfg_dict = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)

    arrival_cfg = cfg_dict.get("arrival", {})
    conv_cfg = cfg_dict.get("conversion", {})

    config = ShareGPTConversionConfig(
        arrival_mode=arrival_cfg.get("mode", args.arrival_mode),
        arrival_rate=float(arrival_cfg.get("rate", args.arrival_rate)),
        duration=float(arrival_cfg.get("duration", args.duration)),
        burst_factor=float(arrival_cfg.get("burst_factor", 5.0)),
        burst_fraction=float(arrival_cfg.get("burst_fraction", 0.2)),
        tokenizer_name=args.tokenizer or conv_cfg.get("tokenizer_name", None),
        fallback_whitespace=bool(conv_cfg.get("fallback_whitespace", True)),
        max_requests=args.max_requests or conv_cfg.get("max_requests", None),
        min_prompt_tokens=int(conv_cfg.get("min_prompt_tokens", 1)),
        min_output_tokens=int(conv_cfg.get("min_output_tokens", 1)),
        max_prompt_tokens=int(conv_cfg.get("max_prompt_tokens", 8192)),
        max_output_tokens=int(conv_cfg.get("max_output_tokens", 8192)),
    )

    aug_config = load_augmentation_config(cfg_dict) if cfg_dict else None

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting ShareGPT trace:")
    print(f"  Input        : {input_path}")
    print(f"  Output       : {args.output}")
    print(f"  Arrival mode : {config.arrival_mode} @ {config.arrival_rate} req/s")
    print(f"  Seed         : {args.seed}")

    records = load_sharegpt_raw(input_path)
    requests, report = convert_sharegpt_to_requests(records, config, args.seed, aug_config)

    print(f"\nConversion report:")
    print(f"  Rows read         : {report.rows_read}")
    print(f"  Pairs extracted   : {report.pairs_extracted}")
    print(f"  Pairs skipped     : {report.pairs_skipped}")
    print(f"  Rows retained     : {report.rows_retained}")
    print(f"  Tokenizer         : {report.tokenizer_used}")
    print(f"  Time range        : {report.time_range_seconds:.2f} s")
    print(f"  Mean arrival rate : {report.mean_arrival_rate:.2f} req/s")
    print(f"  Prompt tokens p95 : {report.prompt_tokens_p95:.0f}")
    print(f"  Output tokens p95 : {report.output_tokens_p95:.0f}")

    output_path = Path(args.output)
    save_extended_jsonl(requests, output_path, source="sharegpt")
    print(f"\nSaved {len(requests)} requests to {output_path}")

    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    print(f"Saved conversion report to {report_path}")


if __name__ == "__main__":
    main()

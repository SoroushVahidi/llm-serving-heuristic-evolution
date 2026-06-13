#!/usr/bin/env python3
"""
Generate and save synthetic traces to disk.

Usage:
    python scripts/generate_synthetic_traces.py --out-dir traces/ --seeds 0 1 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.workloads.synthetic import (
    make_small_debug_trace,
    make_medium_trace,
    make_heavy_tail_trace,
    make_bursty_trace,
)
from llmserveopt.workloads.trace_io import save_jsonl, save_csv


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traces")
    parser.add_argument("--out-dir", default="traces", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds to generate")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generators = {
        "small_debug": make_small_debug_trace,
        "medium": make_medium_trace,
        "heavy_tail": make_heavy_tail_trace,
        "bursty": make_bursty_trace,
    }

    for name, gen in generators.items():
        for seed in args.seeds:
            requests = gen(seed=seed)
            jsonl_path = out / f"{name}_seed{seed}.jsonl"
            csv_path = out / f"{name}_seed{seed}.csv"
            save_jsonl(requests, jsonl_path)
            save_csv(requests, csv_path)
            print(f"  {jsonl_path}  ({len(requests)} requests)")

    print(f"\nDone. Traces saved to: {out}")


if __name__ == "__main__":
    main()

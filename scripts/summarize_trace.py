#!/usr/bin/env python3
"""
Print statistics and generate plots for a processed trace JSONL file.

Usage:
    python scripts/summarize_trace.py \
        --input data/processed/burstgpt/burstgpt_10k.jsonl \
        --out-dir results/trace_analysis/

    # Also works with raw BurstGPT CSV (auto-detected by extension):
    python scripts/summarize_trace.py \
        --input tests/fixtures/burstgpt_tiny.csv \
        --out-dir /tmp/trace_test/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.workloads.trace_io_extended import load_extended_jsonl
from llmserveopt.workloads.burstgpt import load_burstgpt_trace
from llmserveopt.core.types import Request


def load_trace(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        requests, report = load_burstgpt_trace(path)
        return requests
    else:
        requests, _ = load_extended_jsonl(path)
        return requests


def print_stats(requests):
    n = len(requests)
    if n == 0:
        print("No requests in trace.")
        return

    arrivals = np.array([r.arrival_time for r in requests])
    prompts = np.array([r.prompt_tokens for r in requests])
    outputs = np.array([r.actual_output_tokens for r in requests])
    predicted = np.array([r.predicted_output_tokens for r in requests])
    priorities = np.array([r.priority for r in requests])
    classes = [r.class_id for r in requests]

    time_range = arrivals[-1] - arrivals[0] if n > 1 else 0.0
    mean_rate = n / time_range if time_range > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Trace Summary")
    print(f"{'='*60}")
    print(f"  N requests       : {n:,}")
    print(f"  Time range       : {time_range:.2f} s")
    print(f"  Mean arv rate    : {mean_rate:.3f} req/s")

    print(f"\n  Prompt tokens:")
    print(f"    mean  : {prompts.mean():.1f}")
    print(f"    median: {np.median(prompts):.1f}")
    print(f"    p95   : {np.percentile(prompts, 95):.1f}")
    print(f"    p99   : {np.percentile(prompts, 99):.1f}")
    print(f"    min   : {prompts.min()}")
    print(f"    max   : {prompts.max()}")

    print(f"\n  Output tokens (actual):")
    print(f"    mean  : {outputs.mean():.1f}")
    print(f"    median: {np.median(outputs):.1f}")
    print(f"    p95   : {np.percentile(outputs, 95):.1f}")
    print(f"    p99   : {np.percentile(outputs, 99):.1f}")
    print(f"    min   : {outputs.min()}")
    print(f"    max   : {outputs.max()}")

    print(f"\n  SLO class distribution:")
    class_counts = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    for cls, cnt in sorted(class_counts.items()):
        print(f"    {cls}: {cnt} ({cnt/n*100:.1f}%)")

    print(f"\n  Priority distribution:")
    unique_priorities, counts = np.unique(priorities, return_counts=True)
    for p, cnt in zip(unique_priorities, counts):
        print(f"    {p:.1f}: {cnt} ({cnt/n*100:.1f}%)")

    print(f"{'='*60}\n")

    return arrivals, prompts, outputs, predicted, classes


def generate_plots(requests, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    arrivals = np.array([r.arrival_time for r in requests])
    prompts = np.array([r.prompt_tokens for r in requests])
    outputs = np.array([r.actual_output_tokens for r in requests])
    predicted = np.array([r.predicted_output_tokens for r in requests])
    classes = [r.class_id for r in requests]

    if len(arrivals) > 1:
        interarrivals = np.diff(arrivals)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(interarrivals, bins=50, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Interarrival time (s)")
        ax.set_ylabel("Count")
        ax.set_title("Interarrival Time Distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "interarrival_hist.png", dpi=100)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prompts, bins=50, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Count")
    ax.set_title("Prompt Token Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_tokens_hist.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(outputs, bins=50, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Output tokens")
    ax.set_ylabel("Count")
    ax.set_title("Output Token Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "output_tokens_hist.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(prompts, outputs, alpha=0.3, s=10)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Output tokens")
    ax.set_title("Prompt vs Output Tokens")
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_vs_output_scatter.png", dpi=100)
    plt.close(fig)

    sorted_prompts = np.sort(prompts)
    cdf = np.arange(1, len(sorted_prompts) + 1) / len(sorted_prompts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_prompts, cdf)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF of Prompt Tokens")
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_cdf.png", dpi=100)
    plt.close(fig)

    sorted_outputs = np.sort(outputs)
    cdf = np.arange(1, len(sorted_outputs) + 1) / len(sorted_outputs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_outputs, cdf)
    ax.set_xlabel("Output tokens")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical CDF of Output Tokens")
    fig.tight_layout()
    fig.savefig(out_dir / "output_cdf.png", dpi=100)
    plt.close(fig)

    class_counts = {}
    for c in classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    if class_counts:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            list(class_counts.values()),
            labels=list(class_counts.keys()),
            autopct="%1.1f%%",
        )
        ax.set_title("SLO Class Distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "slo_class_pie.png", dpi=100)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(outputs, predicted, alpha=0.3, s=10)
    max_val = max(outputs.max(), predicted.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual output tokens")
    ax.set_ylabel("Predicted output tokens")
    ax.set_title("Predicted vs Actual Output Tokens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=100)
    plt.close(fig)

    print(f"Plots saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Summarize a processed trace JSONL file")
    parser.add_argument("--input", required=True, help="Path to JSONL trace file (or BurstGPT CSV)")
    parser.add_argument("--out-dir", default=None, help="Directory for output plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    requests = load_trace(path)
    if not requests:
        print("WARNING: No requests loaded.")
        return

    print_stats(requests)

    if not args.no_plots:
        out_dir = Path(args.out_dir) if args.out_dir else Path("results/trace_analysis") / path.stem
        generate_plots(requests, out_dir)


if __name__ == "__main__":
    main()

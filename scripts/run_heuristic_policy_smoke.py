#!/usr/bin/env python3
"""
Smoke-test: run each example heuristic through a short simulator run and report
weighted_goodput, comparing against FIFO baseline.

Usage:
    python scripts/run_heuristic_policy_smoke.py
    python scripts/run_heuristic_policy_smoke.py --seed 99 --arrival-rate 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.heuristics import build_heuristic_policy
from llmserveopt.heuristics.examples import edf_like, fifo_like, slo_kv_balanced, throughput_oriented
from llmserveopt.policies.registry import make_policy
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.workloads.synthetic import WorkloadConfig, generate_workload
from llmserveopt.core.types import GPUConfig

# Small stressed GPU so policies actually differentiate
DEFAULT_GPU = GPUConfig(
    gpu_id=0,
    max_active_sequences=4,
    max_batch_tokens=512,
    max_kv_tokens=8192,
)


def get_weighted_goodput(results) -> float:
    return float(results.weighted_goodput)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test heuristic policies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    cfg = WorkloadConfig(
        arrival_process="poisson",
        arrival_rate=args.arrival_rate,
        duration=args.duration,
        prompt_mean=256.0,
        output_mean=128.0,
    )
    requests = generate_workload(cfg, seed=args.seed)
    print(f"Generated {len(requests)} requests (rate={args.arrival_rate} req/s, duration={args.duration}s)")
    print()

    results_table = []

    def _run(policy, tag):
        try:
            m = run_policy(policy, requests, [DEFAULT_GPU], seed=args.seed)
            wg = get_weighted_goodput(m)
            results_table.append((tag, wg))
            print(f"  {tag:<38s}  WG={wg:.4f}  completed={m.num_completed}")
        except Exception as e:
            print(f"  {tag:<38s}  ERROR: {e}")
            results_table.append((tag, float("nan")))

    _run(make_policy("fifo"), "fifo [baseline]")

    examples = [
        ("fifo_like", fifo_like()),
        ("edf_like", edf_like()),
        ("slo_kv_balanced", slo_kv_balanced()),
        ("throughput_oriented", throughput_oriented()),
    ]
    for name, doc in examples:
        _run(build_heuristic_policy(doc), f"{name} [heuristic]")

    print()
    print("Summary (sorted by WG desc)")
    print("-" * 55)
    valid = [(n, wg) for n, wg in results_table if wg == wg]
    best_name = max(valid, key=lambda x: x[1])[0] if valid else ""
    for name, wg in sorted(results_table, key=lambda x: -(x[1] if x[1] == x[1] else -1)):
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<38s}  WG={wg:.4f}{marker}")


if __name__ == "__main__":
    main()

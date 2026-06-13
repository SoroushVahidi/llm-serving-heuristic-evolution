#!/usr/bin/env python3
"""
Quick smoke test: run FIFO on a tiny trace and print results.
Exits with code 0 on success, 1 on failure.

Usage:
    python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.core.types import GPUConfig
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.synthetic import make_small_debug_trace


def main():
    print("Running smoke test...")

    requests = make_small_debug_trace(seed=42)
    print(f"  Trace: {len(requests)} requests")

    gpu_configs = [
        GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=64, max_kv_tokens=1024),
        GPUConfig(gpu_id=1, max_active_sequences=8, max_batch_tokens=64, max_kv_tokens=1024),
    ]

    service_model = ServiceModel(step_size=0.001)
    policy = FIFOPolicy()

    metrics = run_policy(
        policy=policy,
        requests=requests,
        gpu_configs=gpu_configs,
        service_model=service_model,
        workload_tag="smoke_test",
        seed=42,
    )

    print(f"\nResults:")
    print(f"  completed        : {metrics.num_completed}")
    print(f"  dropped          : {metrics.num_dropped}")
    print(f"  mean_latency     : {metrics.mean_latency:.4f}s")
    print(f"  p95_latency      : {metrics.p95_latency:.4f}s")
    print(f"  slo_violation_rate: {metrics.slo_violation_rate:.4f}")
    print(f"  request_throughput: {metrics.request_throughput:.2f} req/s")
    print(f"  wall_clock_s     : {metrics.wall_clock_s:.3f}s")

    assert metrics.num_completed > 0, "No requests completed"
    assert metrics.num_dropped >= 0
    assert 0.0 <= metrics.slo_violation_rate <= 1.0

    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

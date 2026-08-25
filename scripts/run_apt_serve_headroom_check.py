#!/usr/bin/env python3
"""run_apt_serve_headroom_check: Runs Phase F headroom sweeps comparing Apt-Serve's
legitimate headroom gains on Target workloads vs Counter workloads against strong baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.compare import compare_policies
from llmserveopt.policies.registry import make_policy
from llmserveopt.policies.apt_serve_faithful import AptServeSchedulerPolicy, AptServeAdapterConfig
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.apt_serve_stress import (
    generate_apt_serve_target_workload,
    generate_apt_serve_counter_workload
)


def run_headroom_check(args) -> int:
    print("=" * 60)
    print("APT-SERVE PHASE F HEADROOM VALIDATION")
    print("=" * 60)

    # 1. Generate Target and Counter workloads with 15 requests (optimized for speed)
    seeds = [2026, 2027, 2028]
    
    workloads = {
        "Target: KV Pressure & Mixed Urgency": {
            seed: generate_apt_serve_target_workload(seed=seed, n_requests=15, long_fraction=0.3)
            for seed in seeds
        },
        "Counter: Low Memory Pressure": {
            seed: generate_apt_serve_counter_workload(seed=seed, n_requests=15, scenario="low_pressure")
            for seed in seeds
        },
        "Counter: Homogeneous relaxed": {
            seed: generate_apt_serve_counter_workload(seed=seed, n_requests=15, scenario="homogeneous")
            for seed in seeds
        }
    }

    # 2. Setup GPUConfig
    gpu_configs = [
        GPUConfig(
            gpu_id=0,
            max_active_sequences=16,
            max_batch_tokens=2048,
            max_kv_tokens=1024 # 64 blocks
        )
    ]

    # Setup larger step_size=0.05 to optimize simulation speed (reduce steps from 10k to 100)
    service_model = ServiceModel(step_size=0.05)

    # 3. Setup Baselines
    baselines = [
        make_policy("fifo"),
        make_policy("edf")
    ]

    # Apt-Serve policy under "test" execution mode
    apt_config = AptServeAdapterConfig(checkout_path="", execution_mode="test")
    apt_serve = AptServeSchedulerPolicy(
        adapter_config=apt_config,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=32,
        hidden_to_kv_memory_ratio=0.5,
        cache_switch_latency=0.005,
        hidden_restore_latency=0.01,
        recomputation_cost_model="hidden_restore"
    )

    all_policies = baselines + [apt_serve]

    results_report = {
        "workloads": [],
        "aggregate_headroom": {}
    }

    # 4. Execute sweeps
    for tag, requests_per_seed in workloads.items():
        print(f"\nEvaluating Regime: {tag}...")
        metrics_list = compare_policies(
            policies=all_policies,
            requests_per_seed=requests_per_seed,
            gpu_configs=gpu_configs,
            service_model=service_model,
            verbose=False
        )

        # Aggregate ANWG per policy across seeds
        anwg_map: Dict[str, List[float]] = {}
        completed_map: Dict[str, List[float]] = {}
        
        for m in metrics_list:
            anwg_map.setdefault(m.policy_name, []).append(m.arrival_normalized_weighted_goodput)
            completed_map.setdefault(m.policy_name, []).append(m.num_completed / (m.num_completed + m.num_dropped))

        # Compute average metrics
        avg_anwg = {p: sum(vals)/len(vals) for p, vals in anwg_map.items()}
        avg_completed = {p: sum(vals)/len(vals) for p, vals in completed_map.items()}

        # Identify best non-Apt-Serve deployable baseline
        non_apt_baselines = [p for p in avg_anwg.keys() if p != "apt_serve_faithful"]
        best_baseline = max(non_apt_baselines, key=lambda p: avg_anwg[p])
        
        apt_anwg = avg_anwg["apt_serve_faithful"]
        base_anwg = avg_anwg[best_baseline]
        gap = apt_anwg - base_anwg

        print(f"  Apt-Serve ANWG: {apt_anwg:.4f} (completion={avg_completed['apt_serve_faithful']:.1%})")
        print(f"  Best Baseline ({best_baseline}) ANWG: {base_anwg:.4f} (completion={avg_completed[best_baseline]:.1%})")
        print(f"  Headroom Gap: {gap:+.4f}")

        results_report["workloads"].append({
            "workload_tag": tag,
            "apt_serve_anwg": apt_anwg,
            "best_baseline_name": best_baseline,
            "best_baseline_anwg": base_anwg,
            "headroom_gap": gap,
            "apt_completion": avg_completed["apt_serve_faithful"],
            "base_completion": avg_completed[best_baseline]
        })

    # Save compact JSON results
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results_report, indent=2))
        print(f"\n[OK] Headroom report saved to {out_path}")

    print("\n" + "=" * 60)
    print("PHASE F SWEEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=None, help="Save report to JSON file")
    args = parser.parse_args()
    sys.exit(run_headroom_check(args))

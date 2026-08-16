#!/usr/bin/env python3
"""Runner script for the Fairness and Starvation Pilot v1 (Family A).

Generates the scenarios, runs the discrete-event simulator for all registered policies
on the grid, and aggregates performance and fairness statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.registry import make_policy_library_v2
from llmserveopt.policy_separation.templates_fairness_starvation import case4_fairness_starvation
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def build_scenarios_from_config(cfg: dict) -> List[any]:
    grid = cfg["sweep_grid"]
    scenarios = []
    for util in grid["target_utilization"]:
        for skew in grid["tenant_weight_skew"]:
            for vol in grid["interactive_volume_fraction"]:
                for seed in grid["seeds"]:
                    scenarios.append(
                        case4_fairness_starvation(
                            target_utilization=util,
                            tenant_weight_skew=skew,
                            interactive_volume_fraction=vol,
                            seed=seed,
                        )
                    )
    return scenarios


def _jains_index(g1: float, g2: float) -> float:
    num = (g1 + g2) ** 2
    denom = 2 * (g1 ** 2 + g2 ** 2)
    return num / denom if denom > 0 else 0.0


def _run_one_task(args: Tuple[str, str, any]) -> dict:
    scenario_id, policy_name, scenario = args
    try:
        policy = make_policy_library_v2(policy_name)
        service_model = ServiceModel()
        sim_config = SimulatorConfig(gpu_configs=list(scenario.gpu_configs))
        sim = Simulator(sim_config)
        sim.load_trace(list(scenario.requests))
        
        sim.run(policy, workload_tag=scenario_id)
        completed = sim._completed
        
        # Segment by tenant/class
        interactive_completed = [cr for cr in completed if cr.request.class_id == "tenant_interactive"]
        bulk_completed = [cr for cr in completed if cr.request.class_id == "tenant_bulk"]
        
        # Calculate violations
        inter_v = sum(1 for cr in interactive_completed if cr.completion_time > cr.request.slo_deadline)
        bulk_v = sum(1 for cr in bulk_completed if cr.completion_time > cr.request.slo_deadline)
        
        inter_total = len(interactive_completed)
        bulk_total = len(bulk_completed)
        
        # Goodput rates
        inter_gp = (inter_total - inter_v) / max(1, inter_total)
        bulk_gp = (bulk_total - bulk_v) / max(1, bulk_total)
        jfi = _jains_index(inter_gp, bulk_gp)
        
        # TTFT (where available)
        ttfts = [cr.ttft for cr in completed if cr.first_token_time >= 0]
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        
        # Re-derive ANWG from core definition
        # goodput = (total_completed - total_violations) / total_loaded
        total_v = inter_v + bulk_v
        anwg = (len(completed) - total_v) / len(scenario.requests)

        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "anwg": anwg,
            "inter_violations": inter_v,
            "inter_total": inter_total,
            "bulk_violations": bulk_v,
            "bulk_total": bulk_total,
            "jains_fairness_index": jfi,
            "mean_ttft": mean_ttft,
            "status": "success",
        }
    except Exception as e:
        import traceback
        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = yaml.safe_all_load(f) if hasattr(yaml, "safe_all_load") else yaml.safe_load(f)
        if isinstance(cfg, Iterator) or isinstance(cfg, list):
            cfg = list(cfg)[0]

    _log(args.run_dir, f"Starting Fairness and Starvation Pilot v1 (dry_run={args.dry_run})")
    
    # Generate scenarios
    scenarios = build_scenarios_from_config(cfg)
    if args.dry_run:
        # Mini slice for local dry-runs
        scenarios = scenarios[:4]
        
    _log(args.run_dir, f"Generated {len(scenarios)} scenarios.")

    # Generate tasks
    policies = list(cfg["policies"])
    tasks = []
    for scen in scenarios:
        for p in policies:
            tasks.append((scen.scenario_id, p, scen))

    _log(args.run_dir, f"Total tasks to run: {len(tasks)}")

    # Execute tasks
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one_task, t): t for t in tasks}
        completed_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed_count += 1
            if completed_count % 50 == 0:
                _log(args.run_dir, f"Completed {completed_count}/{len(tasks)} tasks.")

    elapsed = time.time() - start_time
    _log(args.run_dir, f"Completed execution of {len(results)} tasks in {elapsed:.2f} seconds.")

    # Write results
    success_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]
    
    _log(args.run_dir, f"Successes: {len(success_results)}, Failures: {len(failed_results)}")
    
    # Save raw csv
    out_csv = args.run_dir / "per_policy_results.csv"
    if success_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=success_results[0].keys())
            writer.writeheader()
            writer.writerows(success_results)
            
    # Save failures
    if failed_results:
        with open(args.run_dir / "failures.jsonl", "w") as f:
            for r in failed_results:
                f.write(json.dumps(r) + "\n")

    # Write summary
    summary = {
        "experiment_name": cfg["pilot_metadata"]["experiment_name"],
        "n_scenarios": len(scenarios),
        "n_tasks": len(tasks),
        "n_completed": len(success_results),
        "n_failed": len(failed_results),
        "elapsed_seconds": elapsed,
    }
    with open(args.run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log(args.run_dir, "Completed Pilot Execution successfully!")


if __name__ == "__main__":
    from collections.abc import Iterator
    main()

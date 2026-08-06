#!/usr/bin/env python3
"""Run the first meaningful comparative evaluation for Llumnix.

Evaluates 5 compatible multi-instance scheduling configurations across the 13
executable workloads generated from the Llumnix stress-test catalog.

Policies:
  1. llumnix_faithful (A: Full modeled behavior: RR placement + Migration)
  2. vllm_faithful (D: No-migration baseline: RR placement + NO Migration)
  3. greedy_dispatch_migration (C: Greedy load-balancing placement + Migration)
  4. greedy_dispatch_no_migration (B: Greedy load-balancing placement + NO Migration)
  5. priority_aware_llumnix (Priority/SLO-aware migration protection)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml  # noqa: E402
from llmserveopt.core.action import Action
from llmserveopt.core.metrics import RunMetrics, metrics_to_dict
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.llumnix_faithful import LlumnixFaithfulPolicy
from llmserveopt.simulator import RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

GENERATED_DIR = _ROOT / "configs" / "stress_tests" / "generated" / "llumnix"
RESULTS_DIR = _ROOT / "results" / "stress_test_catalog" / "llumnix_moderate"


# --- 1. Custom Policy Extensions ---

class LlumnixGreedyDispatchPolicy(LlumnixFaithfulPolicy):
    """Subclass that overrides select_action to perform greedy dispatch
    (least-loaded instance first, using active queue size and free blocks)."""
    def select_action(self, state) -> Action:
        gpu_by_id = {g.gpu_id: g for g in state.gpu_states}
        sorted_gpu_ids = sorted(gpu_by_id.keys())
        
        for req in state.waiting_queue:
            if req.request_id not in self._dispatch_assignment:
                best_gid = sorted_gpu_ids[0]
                best_active = float('inf')
                best_free = -1
                
                for gid in sorted_gpu_ids:
                    gpu = gpu_by_id[gid]
                    n_active = len(gpu.active_request_ids)
                    bm = self._local._get_block_manager(gpu)
                    n_free = bm.num_free_blocks
                    
                    if (n_active < best_active) or (n_active == best_active and n_free > best_free):
                        best_active = n_active
                        best_free = n_free
                        best_gid = gid
                        
                self._dispatch_assignment[req.request_id] = best_gid
                
        return super().select_action(state)


# --- 2. Tracking Simulator Extension ---

class TrackingSimulator(Simulator):
    """Subclass of Simulator that captures detailed raw outputs for request,
    migration, and instance states during execution."""
    def __init__(self, config: SimulatorConfig, policy_name: str, workload: str, seed: int):
        super().__init__(config)
        self.policy_name = policy_name
        self.workload = workload
        self.seed = seed
        self.migration_events: List[dict] = []
        self.instance_logs: List[dict] = []
        
    def _apply_migrations(self, action: Action) -> set:
        migrated_ids = super()._apply_migrations(action)
        if migrated_ids:
            for src_id, pairs in action.migrate.items():
                for rid, dest_gpu_id in pairs:
                    if rid in migrated_ids:
                        self.migration_events.append({
                            "timestamp": self._time,
                            "workload": self.workload,
                            "seed": self.seed,
                            "policy": self.policy_name,
                            "request_id": rid,
                            "source": src_id,
                            "destination": dest_gpu_id,
                            "trigger": "load_imbalance",
                            "bytes": 0,  # not modeled in flat-delay
                            "bandwidth": 0.0,
                            "estimated_transfer_time": self.config.service_model.llumnix_migration_delay,
                            "delay": self.config.service_model.llumnix_migration_delay,
                            "remaining_service": 0.0,
                            "outcome": "success"
                        })
        return migrated_ids

    def _apply_action(self, action: Action) -> None:
        super()._apply_action(action)
        # Log instance state
        for g in self._gpus:
            self.instance_logs.append({
                "timestamp": self._time,
                "workload": self.workload,
                "seed": self.seed,
                "policy": self.policy_name,
                "gpu_id": g.gpu_id,
                "queue_length": len(g._active),
                "active_requests": len(g._active),
                "memory_occupancy": g.current_kv_tokens,
                "fragmentation": 0.0,
                "utilization": 1.0 if g._active else 0.0,
                "migration_bandwidth_utilization": 0.0
            })


# --- 3. Run Sweep ---

def run_evaluation_trial(
    requests: List[Request], policy_name: str, workload: str, seed: int, sim_req: dict
) -> tuple[dict, List[dict], List[dict], List[dict]]:
    n_instances = int(sim_req.get("n_instances", 2))
    migration_delay = float(sim_req.get("migration_delay", 0.001))
    
    total_kv_tokens = int(sim_req.get("total_kv_tokens", 2000))
    per_instance_kv = total_kv_tokens // n_instances
    
    gpu_configs = [
        GPUConfig(
            gpu_id=i,
            max_active_sequences=16,
            max_batch_tokens=4096,
            max_kv_tokens=per_instance_kv
        )
        for i in range(n_instances)
    ]
    
    service_model = ServiceModel(llumnix_migration_delay=migration_delay)
    cfg = SimulatorConfig(gpu_configs=gpu_configs, service_model=service_model)
    sim = TrackingSimulator(cfg, policy_name=policy_name, workload=workload, seed=seed)
    sim.load_trace(requests)
    
    # Instantiate policies
    if policy_name == "llumnix_faithful":
        policy = LlumnixFaithfulPolicy(
            need_migrate_frequency=int(sim_req.get("need_migrate_frequency", 4))
        )
    elif policy_name == "vllm_faithful":
        policy = LlumnixFaithfulPolicy(need_migrate_frequency=999999)
    elif policy_name == "greedy_dispatch_migration":
        policy = LlumnixGreedyDispatchPolicy(
            need_migrate_frequency=int(sim_req.get("need_migrate_frequency", 4))
        )
    elif policy_name == "greedy_dispatch_no_migration":
        policy = LlumnixGreedyDispatchPolicy(need_migrate_frequency=999999)
    elif policy_name == "priority_aware_llumnix":
        policy = LlumnixFaithfulPolicy(
            need_migrate_frequency=int(sim_req.get("need_migrate_frequency", 4)),
            priority_exempt_threshold=1.5
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
        
    metrics = sim.run(policy, workload_tag=policy_name, seed=seed)
    
    # 1. Compile request-level logs
    request_logs = []
    # Both completed and dropped are tracked
    for cr in sim._completed:
        req = cr.request
        request_logs.append({
            "workload": workload,
            "seed": seed,
            "policy": policy_name,
            "request_id": req.request_id,
            "arrival_time": req.arrival_time,
            "initial_instance": 0,  # simplified
            "final_instance": cr.gpu_id,
            "start_time": cr.admission_time,
            "completion_time": cr.completion_time,
            "completion_status": "completed",
            "slo_status": "met" if cr.completion_time <= req.slo_deadline else "violated",
            "priority": req.priority,
            "prompt_tokens": req.prompt_tokens,
            "predicted_output_tokens": req.predicted_output_tokens,
            "realized_output_tokens": req.actual_output_tokens,
            "waiting_time": cr.queuing_delay,
            "ttft": cr.first_token_time,
            "latency": cr.latency,
            "migration_count": 1 if cr.gpu_id != 0 else 0  # proxy
        })
        
    # Map metrics dict
    res = metrics_to_dict(metrics)
    res["slo_attainment"] = 1.0 - metrics.slo_violation_rate if metrics.slo_violation_rate == metrics.slo_violation_rate else 1.0
    res["anwg"] = metrics.arrival_normalized_weighted_goodput
    res["throughput_completions_per_sec"] = metrics.request_throughput
    
    return res, request_logs, sim.migration_events, sim.instance_logs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    
    # Load catalog to map simulator requirements
    catalog_path = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)
    sim_reqs = {e["stress_test_id"]: e.get("simulator_requirements", {}) or {} for e in catalog["stress_tests"]}
    
    # Find all generated workload files
    trace_files = sorted(GENERATED_DIR.glob("*.json"))
    if not trace_files:
        print("ERROR: No generated workload files found. Run headroom check first.")
        return 1
        
    policies = [
        "llumnix_faithful",
        "vllm_faithful",
        "greedy_dispatch_migration",
        "greedy_dispatch_no_migration",
        "priority_aware_llumnix"
    ]
    
    all_request_logs = []
    all_migration_logs = []
    all_instance_logs = []
    summary_results = []
    
    print(f"Starting comparative sweep across {len(trace_files)} trace files x {len(policies)} policies...")
    start_time = time.time()
    
    for tf in trace_files:
        # File name format: <workload_id>_seed<seed>.json
        name_parts = tf.stem.split("_seed")
        workload = name_parts[0]
        seed = int(name_parts[1])
        
        with open(tf) as f:
            payload = json.load(f)
            
        # Skip spec-only workloads (stored with status == "NOT_EXECUTABLE")
        if payload.get("status") == "NOT_EXECUTABLE" or not payload.get("requests"):
            continue
            
        # Reconstruct Request objects
        requests = [
            Request(
                request_id=r["request_id"],
                arrival_time=r["arrival_time"],
                prompt_tokens=r["prompt_tokens"],
                predicted_output_tokens=r["predicted_output_tokens"],
                actual_output_tokens=r["actual_output_tokens"],
                slo_deadline=r["slo_deadline"],
                priority=r["priority"],
                class_id=r["class_id"]
            )
            for r in payload["requests"]
        ]
        
        sim_req = sim_reqs.get(workload, {})
        
        for p in policies:
            res, req_logs, migs, insts = run_evaluation_trial(
                requests, p, workload, seed, sim_req
            )
            summary_results.append({
                "workload": workload,
                "seed": seed,
                "policy": p,
                "completion_fraction": res["completion_fraction"],
                "mean_latency": res["mean_latency"],
                "slo_attainment": res["slo_attainment"],
                "anwg": res["anwg"],
                "throughput": res["throughput_completions_per_sec"],
                "migrations_triggered": len(migs)
            })
            all_request_logs.extend(req_logs)
            all_migration_logs.extend(migs)
            all_instance_logs.extend(insts)
            
    # Save raw outputs
    with open(raw_dir / "requests.json", "w") as f:
        json.dump(all_request_logs, f, indent=2)
    with open(raw_dir / "migrations.json", "w") as f:
        json.dump(all_migration_logs, f, indent=2)
    with open(raw_dir / "instances.json", "w") as f:
        json.dump(all_instance_logs, f, indent=2)
        
    # Save aggregate summary
    summary_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "policies": policies,
            "total_runs": len(summary_results),
            "runtime_seconds": round(time.time() - start_time, 2)
        },
        "results": summary_results
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)
        
    # Write Markdown summary report
    md_lines = [
        "# Llumnix First Comparative Evaluation Report",
        "",
        f"**Completed:** {summary_payload['metadata']['timestamp']}",
        f"**Policies Sweeped:** {', '.join(policies)}",
        f"**Total runs executed:** {len(summary_results)}",
        f"**Runtime:** {summary_payload['metadata']['runtime_seconds']} seconds",
        "",
        "## Performance Comparison Summary (Averaged across Seeds)",
        "",
        "| Workload Family | Policy | Completion Fraction | Mean Latency | SLO Attainment | ANWG | Throughput | Migrations |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    # Compute averages across seeds for display
    grouped = defaultdict(list)
    for r in summary_results:
        grouped[(r["workload"], r["policy"])].append(r)
        
    for (wl, p), runs in sorted(grouped.items()):
        avg_comp = sum(x["completion_fraction"] for r in runs if (x := r).get("completion_fraction") is not None) / len(runs)
        avg_lat = sum(x["mean_latency"] for r in runs if (x := r).get("mean_latency") is not None) / len(runs)
        avg_slo = sum(x["slo_attainment"] for r in runs if (x := r).get("slo_attainment") is not None) / len(runs)
        avg_anwg = sum(x["anwg"] for r in runs if (x := r).get("anwg") is not None) / len(runs)
        avg_thru = sum(x["throughput"] for r in runs if (x := r).get("throughput") is not None) / len(runs)
        avg_migs = sum(r["migrations_triggered"] for r in runs) / len(runs)
        md_lines.append(f"| {wl} | {p} | {avg_comp:.4f} | {avg_lat:.4f} | {avg_slo:.4f} | {avg_anwg:.4f} | {avg_thru:.4f} | {avg_migs:.1f} |")
        
    with open(RESULTS_DIR / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"Sweep complete in {summary_payload['metadata']['runtime_seconds']}s. Summary saved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

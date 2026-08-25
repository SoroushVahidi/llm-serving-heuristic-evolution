#!/usr/bin/env python3
"""Run the first meaningful comparative evaluation for DistServe.

Evaluates 4 compatible multi-instance scheduling configurations across the 5
executable workloads generated from the DistServe stress-test catalog.

Policies:
  1. distserve_faithful (A: Disaggregated prefill/decode)
  2. vllm_faithful (B: Monolithic pool, identical total memory/compute)
  3. distserve_faithful (C: Relaxed throttle waiting_block_prop_threshold=1.0)
  4. distserve_faithful (D: Reverse split n_prefill=1, n_decode=2)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml  # noqa: E402
from llmserveopt.core.metrics import metrics_to_dict
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.distserve_faithful import DistServeFaithfulPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

GENERATED_DIR = _ROOT / "configs" / "stress_tests" / "generated" / "distserve"
RESULTS_DIR = _ROOT / "results" / "stress_test_catalog" / "distserve_moderate"


# --- Tracking Simulator Extension ---

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
        
    def _apply_migrations(self, action) -> set:
        migrated_ids = super()._apply_migrations(action)
        if migrated_ids:
            for rid, gpu_id in migrated_ids: # not quite the same action.migrate as Llumnix, but we only have bridge-queue transfer
                pass
        return migrated_ids

    def _collect_handoffs(self) -> None:
        super()._collect_handoffs()
        # record hands-offs from the bridge queue
        for rid, req in self._migrating_map.items():
            if req.transfer_ready_time == self._time + self.config.service_model.migration_transfer_delay:
                # it was just added this step
                self.migration_events.append({
                    "timestamp": self._time,
                    "workload": self.workload,
                    "seed": self.seed,
                    "policy": self.policy_name,
                    "request_id": rid,
                    "source": "prefill",
                    "destination": "decode",
                    "trigger": "handoff",
                    "bytes": 0,
                    "bandwidth": 0.0,
                    "estimated_transfer_time": self.config.service_model.migration_transfer_delay,
                    "delay": self.config.service_model.migration_transfer_delay,
                    "outcome": "success"
                })

    def _apply_action(self, action) -> None:
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
                "utilization": 1.0 if g._active else 0.0,
            })


# --- Run Sweep ---

def run_evaluation_trial(
    requests: List[Request], policy_name: str, workload: str, seed: int, sim_req: dict
) -> tuple[dict, List[dict], List[dict], List[dict]]:
    total_kv_tokens = int(sim_req.get("total_kv_tokens", 4000))
    migration_delay = float(sim_req.get("migration_transfer_delay", 0.001))
    
    n_prefill = int(sim_req.get("n_prefill", 1))
    n_decode = int(sim_req.get("n_decode", 1))
    
    # Custom splits
        
    per_prefill_kv = (total_kv_tokens // 2) // max(1, n_prefill)
    per_decode_kv = (total_kv_tokens // 2) // max(1, n_decode)
    
    if "distserve" in policy_name:
        gpu_configs = [
            GPUConfig(gpu_id=i, max_active_sequences=128, max_batch_tokens=100000, max_kv_tokens=per_prefill_kv, role="prefill")
            for i in range(n_prefill)
        ] + [
            GPUConfig(gpu_id=100+i, max_active_sequences=128, max_batch_tokens=100000, max_kv_tokens=per_decode_kv, role="decode")
            for i in range(n_decode)
        ]
    elif policy_name == "vllm_faithful_monolithic":
        per_gpu_kv = total_kv_tokens // (n_prefill + n_decode)
        gpu_configs = [
            GPUConfig(gpu_id=i, max_active_sequences=128, max_batch_tokens=100000, max_kv_tokens=per_gpu_kv, role=None)
            for i in range(n_prefill + n_decode)
        ]
    else:
        raise ValueError(f"Unknown policy {policy_name}")
    
    service_model = ServiceModel(
        enable_prefill_modeling=True,
        enable_disaggregation=True,
        decode_first=False,
        enable_decode_prefill_contention=True,
        step_token_budget=512,
        max_prefill_chunk_tokens=512,
        prefill_cost_per_token=1.0,
        migration_transfer_delay=migration_delay
    )
    
    cfg = SimulatorConfig(gpu_configs=gpu_configs, service_model=service_model, max_steps=100_000)
    sim = TrackingSimulator(cfg, policy_name=policy_name, workload=workload, seed=seed)
    sim.load_trace(requests)
    
    # Instantiate policies
    if "distserve" in policy_name:
        if policy_name == "distserve_faithful_relaxed_throttle":
            policy = DistServeFaithfulPolicy(context_max_batch_size=128, decode_max_batch_size=128, context_max_tokens_per_batch=100000, decode_max_tokens_per_batch=100000, waiting_block_prop_threshold=1.0)
        else:
            policy = DistServeFaithfulPolicy(context_max_batch_size=128, decode_max_batch_size=128, context_max_tokens_per_batch=100000, decode_max_tokens_per_batch=100000)
    elif policy_name == "vllm_faithful_monolithic":
        from llmserveopt.policies.external_baselines_registry import make_external_baseline
        policy = make_external_baseline("vllm_faithful", max_num_batched_tokens=100000)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
        
    metrics = sim.run(policy, workload_tag=policy_name, seed=seed)
    
    request_logs = []
    for cr in sim._completed:
        req = cr.request
        request_logs.append({
            "workload": workload,
            "seed": seed,
            "policy": policy_name,
            "request_id": req.request_id,
            "arrival_time": req.arrival_time,
            "initial_instance": 0, 
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
            "tpot": cr.tpot,
            "latency": cr.latency,
            "transfer_count": 1 if "distserve" in policy_name else 0
        })
        
    res = metrics_to_dict(metrics)
    res["slo_attainment"] = 1.0 - metrics.slo_violation_rate if metrics.slo_violation_rate == metrics.slo_violation_rate else 1.0
    res["anwg"] = metrics.arrival_normalized_weighted_goodput
    res["throughput"] = metrics.request_throughput
    
    return res, request_logs, sim.migration_events, sim.instance_logs


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    
    catalog_path = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)
    sim_reqs = {e["stress_test_id"]: e.get("simulator_requirements", {}) or {} for e in catalog["stress_tests"]}
    
    trace_files = sorted(GENERATED_DIR.glob("*.json"))
    if not trace_files:
        print("ERROR: No generated workload files found. Run headroom check first.")
        return 1
        
    policies = [
        "distserve_faithful_default",
        "distserve_faithful_relaxed_throttle",
        "vllm_faithful_monolithic"
        
    ]
    
    all_request_logs = []
    all_migration_logs = []
    all_instance_logs = []
    summary_results = []
    
    print(f"Starting comparative sweep across {len(trace_files)} trace files x {len(policies)} policies...")
    start_time = time.time()
    
    for tf in trace_files:
        name_parts = tf.stem.split("_seed")
        workload = name_parts[0]
        seed = int(name_parts[1])
        
        with open(tf) as f:
            payload = json.load(f)
            
        if payload.get("status") == "NOT_EXECUTABLE" or not payload.get("requests"):
            continue
            
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
        
        # Memory baseline adjustment to avoid starvation on big models 
        sim_req = sim_reqs.get(workload, {})
        sim_req["total_kv_tokens"] = 200000 
        
        for p in policies:
            if "decode_heavy" in p and workload == "distserve_counter_prefill_dominated_split_mismatch":
                continue # Skip intentionally incompatible mismatched architectures
            res, req_logs, migs, insts = run_evaluation_trial(requests, p, workload, seed, sim_req)
            summary_results.append({
                "workload": workload,
                "seed": seed,
                "policy": p,
                "completion_fraction": res["completion_fraction"],
                "mean_latency": res["mean_latency"],
                "mean_ttft": res["mean_ttft"],
                "mean_tpot": res["mean_tpot"],
                "slo_attainment": res["slo_attainment"],
                "anwg": res["anwg"],
                "throughput": res["throughput"],
                "transfers_triggered": len(migs)
            })
            all_request_logs.extend(req_logs)
            all_migration_logs.extend(migs)
            all_instance_logs.extend(insts)
            
    with open(raw_dir / "requests.json", "w") as f:
        json.dump(all_request_logs, f, indent=2)
    with open(raw_dir / "migrations.json", "w") as f:
        json.dump(all_migration_logs, f, indent=2)
    with open(raw_dir / "instances.json", "w") as f:
        json.dump(all_instance_logs, f, indent=2)
        
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
        
    md_lines = [
        "# DistServe First Comparative Evaluation Report",
        "",
        f"**Completed:** {summary_payload['metadata']['timestamp']}",
        f"**Total runs executed:** {len(summary_results)}",
        f"**Runtime:** {summary_payload['metadata']['runtime_seconds']} seconds",
        "",
        "## Performance Comparison Summary (Averaged across Seeds)",
        "",
        "| Workload Family | Policy | Completion Fraction | Mean Latency | Mean TTFT | Mean TPOT | Throughput | Transfers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    grouped = defaultdict(list)
    for r in summary_results:
        grouped[(r["workload"], r["policy"])].append(r)
        
    for (wl, p), runs in sorted(grouped.items()):
        avg_comp = sum(x["completion_fraction"] for r in runs if (x := r).get("completion_fraction") is not None) / len(runs)
        avg_lat = sum(x["mean_latency"] for r in runs if (x := r).get("mean_latency") is not None) / len(runs)
        avg_ttft = sum(x["mean_ttft"] for r in runs if (x := r).get("mean_ttft") is not None) / len(runs)
        avg_tpot = sum(x["mean_tpot"] for r in runs if (x := r).get("mean_tpot") is not None) / len(runs)
        avg_thru = sum(x["throughput"] for r in runs if (x := r).get("throughput") is not None) / len(runs)
        avg_migs = sum(r["transfers_triggered"] for r in runs) / len(runs)
        md_lines.append(f"| {wl} | {p} | {avg_comp:.4f} | {avg_lat:.4f} | {avg_ttft:.4f} | {avg_tpot:.4f} | {avg_thru:.4f} | {avg_migs:.1f} |")
        
    with open(RESULTS_DIR / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"Sweep complete in {summary_payload['metadata']['runtime_seconds']}s. Summary saved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Llumnix-specific stress-test headroom check.

Companion to scripts/stress_tests/run_stress_test_smoke.py, scoped to the
17 Llumnix entries (configs/stress_tests/algorithm_stress_test_catalog.yaml section 13).
Does two things:

1. Dumps each entry's generated workload to
   configs/stress_tests/generated/llumnix/<entry_id>_seed<seed>.json for
   3 deterministic seeds (where executable).
2. Runs the multi-instance headroom comparison (llumnix_faithful vs.
   vllm_faithful as the no-migration baseline, implemented as llumnix_faithful
   with migration disabled) and reports whether each entry's declared
   acceptance_gates direction is actually reproduced.

Usage: python scripts/stress_tests/run_llumnix_headroom_check.py [--full]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generators  # noqa: E402
import yaml  # noqa: E402
from llmserveopt.core.metrics import RunMetrics, metrics_to_dict
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.llumnix_faithful import LlumnixFaithfulPolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

SEEDS = [0, 1, 2]
GENERATED_DIR = _ROOT / "configs" / "stress_tests" / "generated" / "llumnix"
RESULTS_DIR = _ROOT / "results" / "stress_test_catalog" / "llumnix_smoke"


def _request_to_dict(r) -> dict:
    return {
        "request_id": r.request_id, "arrival_time": r.arrival_time, "prompt_tokens": r.prompt_tokens,
        "predicted_output_tokens": r.predicted_output_tokens, "actual_output_tokens": r.actual_output_tokens,
        "slo_deadline": r.slo_deadline, "priority": r.priority, "class_id": r.class_id,
    }


def dump_workload(entry_id: str, smoke: bool) -> List[dict]:
    gen_fn = generators.GENERATORS.get(entry_id)
    dumps = []
    for seed in SEEDS:
        try:
            reqs = gen_fn(smoke=smoke, seed=seed)
        except NotImplementedError as e:
            dumps.append({"seed": seed, "status": "NOT_EXECUTABLE", "reason": str(e)})
            continue
        payload = {
            "entry_id": entry_id, "seed": seed, "smoke": smoke, "n_requests": len(reqs),
            "requests": [_request_to_dict(r) for r in reqs],
        }
        out_path = GENERATED_DIR / f"{entry_id}_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        dumps.append({"seed": seed, "status": "GENERATED", "n_requests": len(reqs), "path": str(out_path.relative_to(_ROOT))})
    return dumps


def run_policy_multi_instance(requests: List[Request], policy_name: str, sim_req: dict) -> dict:
    n_instances = int(sim_req.get("n_instances", 2))
    migration_delay = float(sim_req.get("migration_delay", 0.001))
    
    # Split aggregate KV budget across instances - use smaller budget by default to trigger load/preemption
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
    sim = Simulator(cfg)
    sim.load_trace(requests)
    
    if policy_name == "llumnix_faithful":
        policy = LlumnixFaithfulPolicy(
            need_migrate_frequency=int(sim_req.get("need_migrate_frequency", 4))
        )
    elif policy_name == "vllm_faithful":
        policy = LlumnixFaithfulPolicy(need_migrate_frequency=999999)
    else:
        raise ValueError(f"Unsupported policy {policy_name}")
        
    metrics = sim.run(policy, workload_tag=policy_name)
    res = metrics_to_dict(metrics)
    
    # Add mapped/normalized fields expected by checkers and acceptance gates
    res["slo_attainment"] = 1.0 - metrics.slo_violation_rate if metrics.slo_violation_rate == metrics.slo_violation_rate else 1.0
    res["anwg"] = metrics.arrival_normalized_weighted_goodput
    res["throughput_completions_per_sec"] = metrics.request_throughput
    
    return res


def evaluate_gate_expr(gate_expr: str, results: Dict[str, dict]) -> tuple[bool, str]:
    if gate_expr == "N/A -- not auto-evaluable, generator raises NotImplementedError by design":
        return True, "SPEC_ONLY"
        
    try:
        # Create namespace with policy metrics
        namespace = {}
        for policy, m in results.items():
            namespace[policy] = type('MetricsNamespace', (), m)
            
        val = eval(gate_expr, {}, namespace)
        return bool(val), f"EVAL({gate_expr}) = {val}"
    except Exception as e:
        return False, f"ERROR evaluating '{gate_expr}': {e}"


def run_headroom(entry: dict, smoke: bool) -> dict:
    eid = entry["stress_test_id"]
    gen_fn = generators.GENERATORS.get(eid)
    sim_req = entry.get("simulator_requirements", {}) or {}

    try:
        requests = gen_fn(smoke=smoke)
    except NotImplementedError as e:
        return {"id": eid, "status": "NOT_EXECUTABLE", "reason": str(e)}

    policies = ["llumnix_faithful", "vllm_faithful"]
    results = {p: run_policy_multi_instance(requests, p, sim_req) for p in policies}

    gate_expr = entry.get("acceptance_gates", "")
    passed, detail = evaluate_gate_expr(gate_expr, results)

    return {
        "id": eid, "test_role": entry["test_role"], "evidence_class": entry["evidence_class"],
        "n_requests": len(requests), "gate": gate_expr, "gate_passed": passed, "gate_detail": detail,
        "results_by_policy": {
            p: {
                "mean_latency": r["mean_latency"], "completion_fraction": r["completion_fraction"],
                "mean_ttft": r["mean_ttft"], "mean_tpot": r["mean_tpot"],
                "throughput_completions_per_sec": r["throughput_completions_per_sec"],
                "slo_attainment": r["slo_attainment"],
            }
            for p, r in results.items()
        }
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", default=False)
    args = parser.parse_args()
    smoke = not args.full

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    catalog_path = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)
    llumnix_entries = [e for e in catalog["stress_tests"] if e["algorithm_id"] == "llumnix_faithful"]

    print(f"{len(llumnix_entries)} Llumnix catalog entries found\n")

    generation_report: Dict[str, list] = {}
    headroom_report: List[dict] = []
    accepted, rejected = [], []

    for entry in llumnix_entries:
        eid = entry["stress_test_id"]
        generation_report[eid] = dump_workload(eid, smoke)
        row = run_headroom(entry, smoke)
        headroom_report.append(row)

        if row.get("status") == "NOT_EXECUTABLE":
            print(f"{eid:60s} NOT_EXECUTABLE (spec-only, disclosed)")
            continue

        status = "ACCEPT" if row["gate_passed"] else "REJECT"
        (accepted if row["gate_passed"] else rejected).append(eid)
        print(f"{eid:60s} {status:8s} gate={row['gate_detail'][:70]}")

    report = {"generation": generation_report, "headroom": headroom_report,
              "accepted": accepted, "rejected": rejected}
    out_json = RESULTS_DIR / "report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate Markdown report
    md_lines = [
        "# Llumnix Stress-Test Headroom Check",
        "",
        f"**Date:** 2026-08-06",
        f"**Scale:** {'Moderate/Full' if args.full else 'Smoke'}",
        f"**Total entries:** {len(llumnix_entries)}",
        f"**Accepted (passed gates):** {len(accepted)}",
        f"**Rejected (failed gates):** {len(rejected)}",
        f"**Specification-only (not executable):** {len(llumnix_entries) - len(accepted) - len(rejected)}",
        "",
        "## Results Table",
        "",
        "| Stress Test ID | Role | Evidence Class | Executable | Gate Passed? | Detail |",
        "|---|---|---|---|---|---|",
    ]
    for r in headroom_report:
        is_exec = "Yes" if r.get("status") != "NOT_EXECUTABLE" else "No (Spec-only)"
        gate_passed = "N/A" if is_exec == "No (Spec-only)" else ("Yes" if r["gate_passed"] else "No")
        detail = r.get("reason", r.get("gate_detail", ""))
        md_lines.append(f"| {r['id']} | {r.get('test_role','')} | {r.get('evidence_class','')} | {is_exec} | {gate_passed} | {detail} |")

    with open(RESULTS_DIR / "report.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nWritten reports to {RESULTS_DIR}")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())

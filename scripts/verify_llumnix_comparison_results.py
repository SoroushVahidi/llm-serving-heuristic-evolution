#!/usr/bin/env python3
"""Independent verifier for Llumnix comparative evaluation results.

Recomputes performance and behavior metrics from raw request-level and
migration-level outputs, verifying them against the aggregate summary.json.

Includes a self-test corruption checker that deliberate corrupts an aggregate
to verify that the verifier correctly raises a mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = _ROOT / "configs" / "stress_tests" / "generated" / "llumnix"
RESULTS_DIR = _ROOT / "results" / "stress_test_catalog" / "llumnix_moderate"


def load_workload_requests(workload: str, seed: int) -> List[dict]:
    tf = GENERATED_DIR / f"{workload}_seed{seed}.json"
    with open(tf) as f:
        payload = json.load(f)
    return payload.get("requests", [])


def verify_results(test_corruption: bool = False) -> int:
    summary_path = RESULTS_DIR / "summary.json"
    raw_reqs_path = RESULTS_DIR / "raw" / "requests.json"
    raw_migs_path = RESULTS_DIR / "raw" / "migrations.json"
    
    if not summary_path.exists() or not raw_reqs_path.exists():
        print("ERROR: Evaluation outputs not found. Run evaluation first.")
        return 1
        
    with open(summary_path) as f:
        summary_payload = json.load(f)
    with open(raw_reqs_path) as f:
        raw_reqs = json.load(f)
    with open(raw_migs_path) as f:
        raw_migs = json.load(f)
        
    results = summary_payload["results"]
    
    # If testing corruption, deliberately corrupt the first aggregate row
    if test_corruption and results:
        print("TEST CORRUPTION: Deliberately corrupting first result row...")
        results[0]["mean_latency"] = 999.99
        
    # Group raw requests by (workload, policy, seed)
    grouped_raw: Dict[tuple, List[dict]] = {}
    for r in raw_reqs:
        key = (r["workload"], r["policy"], r["seed"])
        if key not in grouped_raw:
            grouped_raw[key] = []
        grouped_raw[key].append(r)
        
    # Group migrations by (workload, policy, seed)
    grouped_migs: Dict[tuple, List[dict]] = {}
    for m in raw_migs:
        key = (m["workload"], m["policy"], m["seed"])
        if key not in grouped_migs:
            grouped_migs[key] = []
        grouped_migs[key].append(m)
        
    mismatches = 0
    checks_performed = 0
    
    # Verify each aggregate row in summary.json
    for r in results:
        workload = r["workload"]
        policy = r["policy"]
        seed = r["seed"]
        key = (workload, policy, seed)
        
        # Load absolute ground-truth trace from generated workload file
        trace_reqs = load_workload_requests(workload, seed)
        n_trace = len(trace_reqs)
        
        completed_reqs = grouped_raw.get(key, [])
        migs = grouped_migs.get(key, [])
        
        # 1. Independent Recomputations
        # Completion fraction
        recomp_comp = len(completed_reqs) / n_trace if n_trace > 0 else 0.0
        
        # Mean latency
        recomp_lat = (
            sum(req["latency"] for req in completed_reqs) / len(completed_reqs)
            if completed_reqs else 0.0
        )
        
        # SLO attainment (completed requests that met deadline / completed requests)
        recomp_slo = (
            sum(1 for req in completed_reqs if req["slo_status"] == "met") / len(completed_reqs)
            if completed_reqs else 1.0
        )
        
        # ANWG: priority-weighted goodput over all arrivals
        # denominator is sum(priority_i) for all arrivals
        total_prio = sum(tr["priority"] for tr in trace_reqs)
        # numerator is sum(priority_i) for met SLO completions
        met_prio = sum(req["priority"] for req in completed_reqs if req["slo_status"] == "met")
        recomp_anwg = met_prio / total_prio if total_prio > 0 else 0.0
        
        # Migrations count
        recomp_migs = len(migs)
        
        # 2. Tolerant Matching Checks
        tol = 1e-4
        
        # Helper check
        def check_field(field_name: str, reported: float, recomputed: float):
            nonlocal mismatches, checks_performed
            checks_performed += 1
            if reported is None or recomputed is None:
                if reported != recomputed:
                    print(f"MISMATCH [{workload}/{policy}/seed{seed}] field {field_name}: reported={reported}, recomputed={recomputed}")
                    mismatches += 1
            elif not math.isclose(reported, recomputed, abs_tol=tol):
                print(f"MISMATCH [{workload}/{policy}/seed{seed}] field {field_name}: reported={reported:.6f}, recomputed={recomputed:.6f}")
                mismatches += 1
                
        check_field("completion_fraction", r["completion_fraction"], recomp_comp)
        check_field("mean_latency", r["mean_latency"], recomp_lat)
        check_field("slo_attainment", r["slo_attainment"], recomp_slo)
        check_field("anwg", r["anwg"], recomp_anwg)
        check_field("migrations", r["migrations_triggered"], recomp_migs)
        
    print(f"\nVerification complete. Performed {checks_performed} matching checks.")
    
    if test_corruption:
        if mismatches > 0:
            print("CORRUPTION TEST SUCCESS: Verifier successfully detected deliberately injected corruption!")
            return 0
        else:
            print("CORRUPTION TEST FAILED: Verifier failed to detect deliberately injected corruption!")
            return 1
            
    if mismatches > 0:
        print(f"Verification FAILED with {mismatches} mismatching values.")
        return 1
    else:
        print("Verification PASSED: Zero mismatches found between raw logs and aggregate summary.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-corruption", action="store_true", default=False,
                        help="Run in corruption-test self-check mode")
    args = parser.parse_args()
    return verify_results(args.test_corruption)


if __name__ == "__main__":
    sys.exit(main())

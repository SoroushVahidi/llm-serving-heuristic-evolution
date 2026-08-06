#!/usr/bin/env python3
"""Independent verifier for DistServe comparative evaluation results."""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = _ROOT / "configs" / "stress_tests" / "generated" / "distserve"
RESULTS_DIR = _ROOT / "results" / "stress_test_catalog" / "distserve_moderate"


def verify_results(test_corruption: bool = False) -> int:
    summary_path = RESULTS_DIR / "summary.json"
    raw_reqs_path = RESULTS_DIR / "raw" / "requests.json"
    raw_migs_path = RESULTS_DIR / "raw" / "migrations.json"
    
    with open(summary_path) as f:
        results = json.load(f)["results"]
    with open(raw_reqs_path) as f:
        raw_reqs = json.load(f)
    with open(raw_migs_path) as f:
        raw_migs = json.load(f)
        
    if test_corruption and results:
        results[0]["mean_latency"] = 999.99
        
    grouped_raw = defaultdict(list)
    for r in raw_reqs:
        grouped_raw[(r["workload"], r["policy"], r["seed"])].append(r)
        
    grouped_migs = defaultdict(list)
    for m in raw_migs:
        grouped_migs[(m["workload"], m["policy"], m["seed"])].append(m)
        
    mismatches = 0
    checks_performed = 0
    tol = 1e-4
    
    def check_field(field_name, reported, recomputed, wl, pol, seed):
        nonlocal mismatches, checks_performed
        checks_performed += 1
        if reported is None or recomputed is None:
            if reported != recomputed:
                mismatches += 1
        elif not math.isclose(reported, recomputed, abs_tol=tol):
            mismatches += 1

    for r in results:
        wl, pol, seed = r["workload"], r["policy"], r["seed"]
        key = (wl, pol, seed)
        
        with open(GENERATED_DIR / f"{wl}_seed{seed}.json") as f:
            trace_reqs = json.load(f)["requests"]
        
        completed_reqs = grouped_raw.get(key, [])
        migs = grouped_migs.get(key, [])
        
        recomp_comp = len(completed_reqs) / len(trace_reqs) if trace_reqs else 0.0
        recomp_lat = sum(req["latency"] for req in completed_reqs) / len(completed_reqs) if completed_reqs else 0.0
        recomp_slo = sum(1 for req in completed_reqs if req["slo_status"] == "met") / len(completed_reqs) if completed_reqs else 1.0
        
        total_prio = sum(tr["priority"] for tr in trace_reqs)
        met_prio = sum(req["priority"] for req in completed_reqs if req["slo_status"] == "met")
        recomp_anwg = met_prio / total_prio if total_prio > 0 else 0.0
        
        recomp_migs = len(migs)
        
        check_field("completion_fraction", r["completion_fraction"], recomp_comp, wl, pol, seed)
        check_field("mean_latency", r["mean_latency"], recomp_lat, wl, pol, seed)
        check_field("slo_attainment", r["slo_attainment"], recomp_slo, wl, pol, seed)
        check_field("anwg", r["anwg"], recomp_anwg, wl, pol, seed)
        check_field("transfers", r["transfers_triggered"], recomp_migs, wl, pol, seed)
        
    print(f"Verification complete. Performed {checks_performed} matching checks.")
    if test_corruption:
        return 0 if mismatches > 0 else 1
    return 1 if mismatches > 0 else 0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test-corruption", action="store_true")
    sys.exit(verify_results(p.parse_args().test_corruption))

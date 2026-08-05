#!/usr/bin/env python3
"""Fairness-validated comparative sweep for the VTC fairness-benchmark repair.

Runs ONLY after scripts/check_vtc_fairness_headroom.py --all passes (verify
before running this). Compares six policies across all six repaired
fairness-extension workload families, at 3 deterministic seeds each:

  - official_vtc (variant A): real, unmodified VTCReqQueue ordering +
    official admission gate, unmodified RECOMMENDED_GPU_CONFIG capacity.
  - matched_admission_fifo (variant B): plain FCFS via the official,
    unmodified ReqQueue base class -- IDENTICAL admission gate to variant A.
  - fairness_isolation_vtc (variant C): VTC ordering + official admission
    gate, capacity rescaled to avoid the units-mismatch confound (see
    baselines/vtc/adapter/variants.py).
  - fifo: this project's native FIFOPolicy (native `_feasible_on_gpu` gate).
  - shortest_prompt_first: throughput-oriented reference.
  - scorpio_style_slo_guard: SLO/admission-oriented reference.

Deliberately excludes any selector/regression policy (out of scope per
this task's explicit instruction).

Usage: python scripts/run_vtc_fairness_comparative_sweep.py [--json OUT.json] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from baselines.vtc.adapter.diagnostics import InstrumentedVTCFairnessPolicy  # noqa: E402
from baselines.vtc.adapter.variants import (  # noqa: E402
    fairness_isolation_vtc,
    matched_admission_fifo,
)
from baselines.vtc.fairness_workloads import ALL_FAIRNESS_FAMILIES, RECOMMENDED_GPU_CONFIG  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policies.scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy  # noqa: E402
from llmserveopt.policies.shortest_prompt_first import ShortestPromptFirstPolicy  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

DEFAULT_SEEDS = [0, 1, 2]


def _jains(values: List[float]) -> float:
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    n = len(values)
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    return (s1 * s1) / (n * s2) if s2 > 0 else float("nan")


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(p * (len(s) - 1))
    return s[idx]


def _starvation_intervals(sim: Simulator, tenants: List[str]) -> Dict[str, float]:
    by_tenant = defaultdict(list)
    for cr in sim._completed:  # noqa: SLF001
        by_tenant[cr.request.class_id].append(cr.completion_time)
    out = {}
    for t in tenants:
        times = sorted(by_tenant.get(t, []))
        if not times:
            out[t] = float("nan")
            continue
        gaps = [times[0]] + [b - a for a, b in zip(times, times[1:])]
        out[t] = max(gaps)
    return out


def _per_tenant_metrics(sim: Simulator, tenants: List[str], checkpoint: float) -> dict:
    by_tenant_ck = defaultdict(list)
    by_tenant_full = defaultdict(list)
    for cr in sim._completed:  # noqa: SLF001
        by_tenant_full[cr.request.class_id].append(cr)
        if cr.completion_time <= checkpoint:
            by_tenant_ck[cr.request.class_id].append(cr)

    checkpoint_service = [
        sum(c.request.prompt_tokens + c.request.actual_output_tokens for c in by_tenant_ck.get(t, []))
        for t in tenants
    ]
    jain = _jains(checkpoint_service)
    max_svc = max(checkpoint_service, default=0)
    min_svc = min(checkpoint_service, default=0)

    per_tenant = {}
    for t in tenants:
        full = by_tenant_full.get(t, [])
        qd = [cr.queuing_delay for cr in full]
        per_tenant[t] = {
            "n_completed": len(full),
            "p95_queuing_delay": _percentile(qd, 0.95),
            "p99_queuing_delay": _percentile(qd, 0.99),
            "slo_violation_rate": (
                sum(1 for cr in full if cr.slo_violated) / len(full) if full else float("nan")
            ),
        }

    return {
        "jains_index_at_checkpoint": jain,
        "max_service_disparity_at_checkpoint": max_svc - min_svc,
        "per_tenant": per_tenant,
        "starvation_intervals": _starvation_intervals(sim, tenants),
    }


POLICY_BUILDERS = {
    "official_vtc": lambda tenants: InstrumentedVTCFairnessPolicy(known_tenants=tenants),
    "matched_admission_fifo": lambda tenants: matched_admission_fifo(),
    "fairness_isolation_vtc": lambda tenants: fairness_isolation_vtc(known_tenants=tenants),
    "fifo": lambda tenants: FIFOPolicy(),
    "shortest_prompt_first": lambda tenants: ShortestPromptFirstPolicy(),
    "scorpio_style_slo_guard": lambda tenants: ScorpioStyleSloGuardPolicy(),
}


def run_one(family_name: str, seed_offset: int, policy_name: str) -> dict:
    fn = ALL_FAIRNESS_FAMILIES[family_name]
    # Every generator's default `seed` kwarg is an integer; offset it
    # deterministically per requested sweep seed without needing to know
    # each function's own base seed value.
    import inspect
    base_seed = inspect.signature(fn).parameters["seed"].default
    requests, tenants = fn(seed=base_seed + seed_offset * 1000)

    checkpoint = 0.6 * max(r.arrival_time for r in requests)
    cfg = SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG])
    sim = Simulator(cfg)
    sim.load_trace(requests)
    policy = POLICY_BUILDERS[policy_name](tenants)
    metrics = sim.run(policy, workload_tag=f"{family_name}_{policy_name}_seed{seed_offset}")

    row = {
        "family": family_name,
        "seed_offset": seed_offset,
        "policy": policy_name,
        "n_requests": len(requests),
        "anwg": metrics.arrival_normalized_weighted_goodput,
        "completion_fraction": metrics.completion_fraction,
        "slo_violation_rate": metrics.slo_violation_rate,
        "request_throughput": metrics.request_throughput,
        "num_completed": metrics.num_completed,
        "num_total": metrics.num_total,
    }
    row.update(_per_tenant_metrics(sim, tenants, checkpoint))

    if isinstance(policy, InstrumentedVTCFairnessPolicy):
        d = policy.decomposition_summary()
        row["reservation_bind_rate"] = d["reservation_bind_rate"]
        row["budget_bind_rate"] = d["budget_bind_rate"]
        row["decision_disagreement_rate"] = d["decision_disagreement_rate"]
    else:
        row["reservation_bind_rate"] = None
        row["budget_bind_rate"] = None
        row["decision_disagreement_rate"] = None

    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--families", type=str, default=None, help="comma-separated, default all")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    families = args.families.split(",") if args.families else list(ALL_FAIRNESS_FAMILIES)

    results = []
    for family_name in families:
        for seed_offset in seeds:
            for policy_name in POLICY_BUILDERS:
                row = run_one(family_name, seed_offset, policy_name)
                results.append(row)

    # Print a compact summary table, averaged across seeds.
    print(f"{'family':28s} {'policy':24s} {'anwg':>7s} {'jain':>7s} {'disparity':>10s} "
          f"{'maxstarve':>10s} {'p95qd':>8s}")
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["family"], r["policy"])].append(r)
    for (family_name, policy_name), rows in grouped.items():
        anwg = statistics.fmean(r["anwg"] for r in rows)
        jain = statistics.fmean(r["jains_index_at_checkpoint"] for r in rows if r["jains_index_at_checkpoint"] == r["jains_index_at_checkpoint"])
        disparity = statistics.fmean(r["max_service_disparity_at_checkpoint"] for r in rows)
        max_starve = statistics.fmean(
            max((v for v in r["starvation_intervals"].values() if v == v), default=float("nan"))
            for r in rows
        )
        p95s = [
            v for r in rows for v in (pt["p95_queuing_delay"] for pt in r["per_tenant"].values())
            if v == v
        ]
        p95 = statistics.fmean(p95s) if p95s else float("nan")
        print(f"{family_name:28s} {policy_name:24s} {anwg:7.3f} {jain:7.3f} {disparity:10.1f} "
              f"{max_starve:10.2f} {p95:8.3f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

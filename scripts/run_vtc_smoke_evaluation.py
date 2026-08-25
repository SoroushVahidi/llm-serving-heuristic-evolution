#!/usr/bin/env python3
"""VTC baseline smoke evaluation.

Per the VTC audit/integration task's step 9: before any full sweep, run a
small controlled comparison among FIFO, VTC (via
baselines/vtc/adapter/simulator_policy.py -- the real, unmodified official
VTCReqQueue), one throughput-oriented policy (shortest_prompt_first), and
one SLO/admission policy (scorpio_style_slo_guard), across the six
VTC-specific fairness-extension workload families in
baselines/vtc/fairness_workloads.py (NOT the accepted canonical suite --
see that module's docstring for why).

Reports, per (workload family, policy): ANWG, completion fraction, SLO
attainment, per-tenant normalized service, max-min fairness ratio, Jain's
fairness index, service disparity, and a starvation proxy (max per-tenant
p95 queuing delay). Intentionally small-scale (single GPU, ~50-220 requests
per family) -- this is a smoke/initial-integration check, not the full
benchmark sweep the task explicitly says not to launch yet.

Usage: python scripts/run_vtc_smoke_evaluation.py [--json OUT.json]
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

from baselines.vtc.adapter.simulator_policy import VTCFairnessPolicy  # noqa: E402
from baselines.vtc.fairness_workloads import ALL_FAIRNESS_FAMILIES  # noqa: E402
from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policies.scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy  # noqa: E402
from llmserveopt.policies.shortest_prompt_first import ShortestPromptFirstPolicy  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

# Deliberately tight relative to the workload families' demand -- fairness
# only becomes visible under real contention; a GPU with unlimited headroom
# makes every policy look identical (everything admits immediately).
GPU = GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=1024, max_kv_tokens=4096)


def jains_index(values: List[float]) -> float:
    values = [v for v in values if v == v]  # drop NaN
    if not values:
        return float("nan")
    n = len(values)
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    if s2 == 0:
        return float("nan")
    return (s1 * s1) / (n * s2)


def per_tenant_metrics(sim: Simulator, tenants: List[str], checkpoint_time: float = None) -> Dict[str, dict]:
    """Per-tenant service accounting. When `checkpoint_time` is given, only
    requests completed BY that wall-clock time count -- a fixed final-drain
    snapshot (completion_fraction usually -> 1.0 given this simulator's
    generous default drain_steps) is fairness-uninformative, since total
    eventual service per tenant is just total demand and barely depends on
    scheduling order at all. A mid-run checkpoint reveals the actual
    scheduling-order-dependent quantity VTC's own paper measures (cumulative
    service received OVER TIME, not at infinite drain)."""
    completed_by_tenant: Dict[str, list] = defaultdict(list)
    for cr in sim._completed:  # noqa: SLF001 -- precedented in this repo's own tests
        if checkpoint_time is not None and cr.completion_time > checkpoint_time:
            continue
        completed_by_tenant[cr.request.class_id].append(cr)

    out = {}
    for tenant in tenants:
        crs = completed_by_tenant.get(tenant, [])
        service = sum(cr.request.prompt_tokens + cr.request.actual_output_tokens for cr in crs)
        queuing_delays = [cr.queuing_delay for cr in crs]
        out[tenant] = {
            "n_completed": len(crs),
            "total_service_tokens": service,
            "mean_queuing_delay": statistics.fmean(queuing_delays) if queuing_delays else float("nan"),
            "p95_queuing_delay": (
                sorted(queuing_delays)[int(0.95 * (len(queuing_delays) - 1))] if queuing_delays else float("nan")
            ),
        }
    return out


def run_one(policy_name: str, policy_factory, requests, tenants) -> dict:
    cfg = SimulatorConfig(gpu_configs=[GPU])
    sim = Simulator(cfg)
    sim.load_trace(requests)
    policy = policy_factory()
    metrics = sim.run(policy, workload_tag=policy_name)

    # Mid-run checkpoint: fraction of the way through the arrival window,
    # the point at which scheduling-order fairness is actually visible
    # (see per_tenant_metrics docstring).
    last_arrival = max(r.arrival_time for r in requests)
    checkpoint_time = 0.6 * last_arrival
    tenant_stats = per_tenant_metrics(sim, tenants, checkpoint_time=checkpoint_time)
    normalized_service = [
        s["total_service_tokens"] / max(1, s["n_completed"]) if s["n_completed"] > 0 else 0.0
        for s in tenant_stats.values()
    ]
    completions = [s["n_completed"] for s in tenant_stats.values()]
    starved_tenants = sum(1 for c in completions if c == 0)
    max_service = max((s["total_service_tokens"] for s in tenant_stats.values()), default=0)
    min_service = min((s["total_service_tokens"] for s in tenant_stats.values()), default=0)

    return {
        "policy": policy_name,
        "anwg": metrics.arrival_normalized_weighted_goodput,
        "completion_fraction": metrics.completion_fraction,
        "slo_violation_rate": metrics.slo_violation_rate,
        "num_completed": metrics.num_completed,
        "num_total": metrics.num_total,
        "jains_fairness_index_normalized_service": jains_index(normalized_service),
        "max_min_fairness_ratio": (min_service / max_service) if max_service > 0 else float("nan"),
        "service_disparity_tokens": max_service - min_service,
        "starved_tenants": starved_tenants,
        "n_tenants": len(tenants),
        "per_tenant": tenant_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    policy_factories = {
        "fifo": lambda: FIFOPolicy(),
        "vtc_fairness_reference": None,  # built per-family below (needs known_tenants)
        "shortest_prompt_first": lambda: ShortestPromptFirstPolicy(),
        "scorpio_style_slo_guard": lambda: ScorpioStyleSloGuardPolicy(),
    }

    results = []
    for family_name, family_fn in ALL_FAIRNESS_FAMILIES.items():
        requests, tenants = family_fn()
        print(f"\n=== {family_name} ({len(requests)} requests, {len(tenants)} tenants) ===")
        for policy_name, factory in policy_factories.items():
            if policy_name == "vtc_fairness_reference":
                factory = lambda tenants=tenants: VTCFairnessPolicy(known_tenants=tenants)
            row = run_one(policy_name, factory, requests, tenants)
            row["workload_family"] = family_name
            results.append(row)
            print(
                f"  {policy_name:26s} anwg={row['anwg']:.3f} "
                f"completion={row['completion_fraction']:.3f} "
                f"jain={row['jains_fairness_index_normalized_service']:.3f} "
                f"maxmin={row['max_min_fairness_ratio']:.3f} "
                f"disparity={row['service_disparity_tokens']:.0f} "
                f"starved={row['starved_tenants']}/{row['n_tenants']}"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

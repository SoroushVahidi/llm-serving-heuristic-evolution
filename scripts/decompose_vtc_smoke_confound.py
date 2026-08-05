#!/usr/bin/env python3
"""Reproduce and quantitatively decompose the VTC smoke-test confound.

Per the fairness-benchmark-repair task's step 2: reproduces the original
6-family smoke comparison and records, for every family: backlog/queue
contention, admission-feasibility rejections split by cause (official
reservation gate vs. batch-token budget vs. none), VTC's own step-by-step
service-order decisions, completion, ANWG, per-tenant service, and
starvation duration -- using baselines/vtc/adapter/diagnostics.py's
InstrumentedVTCFairnessPolicy (verified inert: produces bit-for-bit
identical scheduling decisions to VTCFairnessPolicy; see
tests/test_vtc_fairness_diagnostics.py).

Confirms or refutes, with numbers rather than assertion, the two claims
from docs/audits/vtc_initial_integration_20260805.md:
  1. five families lacked enough sustained contention for ordering to matter;
  2. heterogeneous_token_sizes was dominated by the reservation gate, not
     fairness ordering.

Usage: python scripts/decompose_vtc_smoke_confound.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from baselines.vtc.adapter.diagnostics import InstrumentedVTCFairnessPolicy  # noqa: E402
from baselines.vtc.fairness_workloads import ALL_FAIRNESS_FAMILIES  # noqa: E402
from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

# Same capacity as the original smoke evaluation
# (scripts/run_vtc_smoke_evaluation.py) -- reproducing the SAME confound,
# not a different one.
GPU = GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=1024, max_kv_tokens=4096)


def starvation_intervals(sim: Simulator, tenants) -> dict:
    """Per-tenant longest gap between consecutive completions (or from 0 /
    to end-of-run) -- a direct starvation-duration measurement, not a
    proxy."""
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


def fifo_contention(requests, tenants) -> dict:
    """Run FIFO (a plain, uninstrumented reference) to measure backlog
    contention independent of any VTC-specific mechanism -- how many
    scheduling steps had >=2 distinct tenants simultaneously waiting."""
    cfg = SimulatorConfig(gpu_configs=[GPU])
    sim = Simulator(cfg)
    sim.load_trace(requests)

    contended_steps = 0
    total_steps = 0
    tenant_by_rid = {r.request_id: r.class_id for r in requests}

    class _CountingFIFO(FIFOPolicy):
        def select_action(self, state):
            nonlocal contended_steps, total_steps
            total_steps += 1
            present = {tenant_by_rid[r.request_id] for r in state.waiting_queue}
            if len(present) >= 2:
                contended_steps += 1
            return super().select_action(state)

    policy = _CountingFIFO()
    metrics = sim.run(policy, workload_tag="fifo_contention_probe")
    return {
        "n_steps": total_steps,
        "contended_steps": contended_steps,
        "contention_rate": contended_steps / total_steps if total_steps else float("nan"),
        "completion_fraction": metrics.completion_fraction,
        "anwg": metrics.arrival_normalized_weighted_goodput,
    }


def run_family(name: str, fn) -> dict:
    requests, tenants = fn()

    cfg = SimulatorConfig(gpu_configs=[GPU])
    sim = Simulator(cfg)
    sim.load_trace(requests)
    policy = InstrumentedVTCFairnessPolicy(known_tenants=tenants)
    metrics = sim.run(policy, workload_tag=name)

    decomposition = policy.decomposition_summary()
    starvation = starvation_intervals(sim, tenants)
    fifo_ref = fifo_contention(requests, tenants)

    return {
        "family": name,
        "n_requests": len(requests),
        "n_tenants": len(tenants),
        "vtc_completion_fraction": metrics.completion_fraction,
        "vtc_anwg": metrics.arrival_normalized_weighted_goodput,
        "vtc_num_completed": metrics.num_completed,
        "vtc_num_total": metrics.num_total,
        "fifo_reference": fifo_ref,
        "decomposition": decomposition,
        "starvation_intervals_by_tenant": starvation,
        "max_starvation": max((v for v in starvation.values() if v == v), default=float("nan")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    results = []
    for name, fn in ALL_FAIRNESS_FAMILIES.items():
        row = run_family(name, fn)
        results.append(row)
        d = row["decomposition"]
        print(f"\n=== {name} ({row['n_requests']} requests, {row['n_tenants']} tenants) ===")
        print(f"  VTC completion={row['vtc_completion_fraction']:.3f} anwg={row['vtc_anwg']:.3f}")
        print(f"  FIFO contention_rate={row['fifo_reference']['contention_rate']:.3f} "
              f"(completion={row['fifo_reference']['completion_fraction']:.3f})")
        print(f"  VTC steps: n={d['n_steps']} contended_rate={d['contention_rate']:.3f} "
              f"reservation_bind_rate={d['reservation_bind_rate']:.3f} "
              f"budget_bind_rate={d['budget_bind_rate']:.3f} "
              f"unexplored_backlog_rate={d['unexplored_backlog_rate']:.3f} "
              f"ordering_governed_rate={d['ordering_governed_rate']:.3f}")
        print(f"  max_starvation={row['max_starvation']:.1f}s")

    print("\n=== CLAIM CHECK ===")
    for row in results:
        d = row["decomposition"]
        low_contention = d["contention_rate"] < 0.05 or row["fifo_reference"]["contention_rate"] < 0.05
        reservation_dominated = d["reservation_bind_rate"] > 0.05 or d["unexplored_backlog_rate"] > 0.05
        print(f"  {row['family']:28s} low_contention={low_contention!s:5s} "
              f"reservation_dominated={reservation_dominated!s:5s}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

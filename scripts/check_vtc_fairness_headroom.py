#!/usr/bin/env python3
"""Fairness-headroom gate checker for the VTC fairness-extension workloads.

Per docs/audits/vtc_fairness_benchmark_repair_20260805.md §5 (thresholds
justified there and briefly recapped in each gate's docstring below): a
workload/capacity combination is not worth running through the full
comparative sweep unless it demonstrably (a) produces sustained,
statistically meaningful admission backlog, (b) is not dominated by the
official admission-feasibility gate (the confound
docs/audits/vtc_initial_integration_20260805.md found), and (c) actually
causes VTC's ordering to diverge from FIFO's at contended decision points.
This script checks exactly those conditions -- and, for the three
families with a specific designed discriminative purpose beyond raw
contention, an additional family-specific gate -- and prints PASS/FAIL
with reasons before any full sweep is allowed to proceed.

Deliberately cheap: only `InstrumentedVTCFairnessPolicy` (see
baselines/vtc/adapter/diagnostics.py) and plain `FIFOPolicy` are run here
-- never `matched_admission_fifo`/`fairness_isolation_vtc` (redundant for
a headroom check) and never any selector/regression policy.

Deterministic: every workload generator is seeded; no wall-clock-dependent
or randomized gate logic.

Usage:
    python scripts/check_vtc_fairness_headroom.py --family one_heavy_hitter
    python scripts/check_vtc_fairness_headroom.py --all
    python scripts/check_vtc_fairness_headroom.py --all --json out.json
Exit code: 0 if every checked family passes every applicable gate, else 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from baselines.vtc.adapter.diagnostics import InstrumentedVTCFairnessPolicy  # noqa: E402
from baselines.vtc.fairness_workloads import (  # noqa: E402
    ALL_FAIRNESS_FAMILIES,
    RECOMMENDED_GPU_CONFIG,
    WINDOWED_CONTENTION_FAMILIES,
)
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

# --- Universal gate thresholds (apply to every family) ---------------------
# Justification for each: docs/audits/vtc_fairness_benchmark_repair_20260805.md §5.
MIN_CONTENTION_RATE = 0.15          # measured range across families: 0.234-0.947 (or windowed 0.85-0.88)
MIN_CONTENDED_STEPS = 500           # measured range: 9304-40836
MAX_ADMISSION_GATE_BIND_RATE = 0.05  # measured range: 0.009-0.033; original broken regime: 0.992
MIN_DECISION_DISAGREEMENT_RATE = 0.005  # measured range: 0.010-0.021; original broken regime: ~0.000

# --- Family-specific gate thresholds ---------------------------------------
MAX_DISCRIMINATIVE_FIFO_JAIN = 0.90  # one_heavy_hitter=0.426, heterogeneous_token_sizes=0.703
SLO_DIVERGENCE_RANGE = (0.10, 0.90)  # priority_fairness_conflict: FIFO violation rate must be genuinely mixed
MIN_WINDOWED_CONTENTION_RATE = 0.50  # returning_inactive_tenant overlap windows: measured 0.845-0.878

DISCRIMINATIVE_DISPARITY_FAMILIES = {"one_heavy_hitter", "heterogeneous_token_sizes"}
SLO_DIVERGENCE_FAMILIES = {"priority_fairness_conflict"}


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class FamilyReport:
    family: str
    gates: List[GateResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)


def _jains(values: List[float]) -> float:
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    n = len(values)
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    return (s1 * s1) / (n * s2) if s2 > 0 else float("nan")


def _checkpoint_jain(sim: Simulator, tenants: List[str], checkpoint: float) -> float:
    by_tenant = defaultdict(list)
    for cr in sim._completed:  # noqa: SLF001 -- precedented in this repo's own tests
        if cr.completion_time <= checkpoint:
            by_tenant[cr.request.class_id].append(cr)
    svc = [
        sum(c.request.prompt_tokens + c.request.actual_output_tokens for c in by_tenant.get(t, []))
        for t in tenants
    ]
    return _jains(svc)


def _windowed_contention_rate(requests, tenants, windows) -> float:
    tenant_by_rid = {r.request_id: r.class_id for r in requests}
    cfg = SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG])
    sim = Simulator(cfg)
    sim.load_trace(requests)
    contended = [0]
    total = [0]

    class _Probe(FIFOPolicy):
        def select_action(self, state):
            in_window = any(lo <= state.time <= hi for lo, hi in windows)
            if in_window:
                total[0] += 1
                present = {tenant_by_rid[r.request_id] for r in state.waiting_queue}
                if len(present) >= 2:
                    contended[0] += 1
            return super().select_action(state)

    sim.run(_Probe(), workload_tag="windowed_contention_probe")
    return contended[0] / total[0] if total[0] else float("nan")


def check_family(name: str) -> FamilyReport:
    fn = ALL_FAIRNESS_FAMILIES[name]
    requests, tenants = fn()
    return check_workload(name, requests, tenants)


def check_workload(name: str, requests, tenants) -> FamilyReport:
    """Core gate-evaluation logic, decoupled from the registry lookup so
    tests can pass synthetic (requests, tenants) directly while still
    selecting which family-specific gates apply via `name` (one of
    ALL_FAIRNESS_FAMILIES's keys, used only to decide gate applicability
    -- the workload itself need not match that family's real generator)."""
    report = FamilyReport(family=name)

    cfg = SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG])
    sim = Simulator(cfg)
    sim.load_trace(requests)
    policy = InstrumentedVTCFairnessPolicy(known_tenants=tenants)
    sim.run(policy, workload_tag=name)
    d = policy.decomposition_summary()
    report.metrics.update({f"vtc_{k}": v for k, v in d.items()})

    windowed = name in WINDOWED_CONTENTION_FAMILIES
    if windowed:
        duration = max(r.arrival_time for r in requests)
        windows = [(0.0, 0.25 * duration), (0.83 * duration, duration)]
        w_rate = _windowed_contention_rate(requests, tenants, windows)
        report.metrics["windowed_contention_rate"] = w_rate
        report.gates.append(GateResult(
            "windowed_contention",
            w_rate >= MIN_WINDOWED_CONTENTION_RATE,
            f"windowed_contention_rate={w_rate:.3f} (need >= {MIN_WINDOWED_CONTENTION_RATE}); "
            "returning_inactive_tenant's contention is concentrated in the two overlap "
            "windows by design, so the flat per-run rate is not the right signal here.",
        ))
    else:
        report.gates.append(GateResult(
            "contention",
            d["contention_rate"] >= MIN_CONTENTION_RATE,
            f"contention_rate={d['contention_rate']:.3f} (need >= {MIN_CONTENTION_RATE})",
        ))

    report.gates.append(GateResult(
        "sample_size",
        d["n_contended_steps"] >= MIN_CONTENDED_STEPS,
        f"n_contended_steps={d['n_contended_steps']} (need >= {MIN_CONTENDED_STEPS})",
    ))

    admission_gate_bind_rate = d["reservation_bind_rate"] + d["budget_bind_rate"]
    report.metrics["admission_gate_bind_rate"] = admission_gate_bind_rate
    report.gates.append(GateResult(
        "not_reservation_dominated",
        admission_gate_bind_rate <= MAX_ADMISSION_GATE_BIND_RATE,
        f"admission_gate_bind_rate={admission_gate_bind_rate:.3f} "
        f"(reservation={d['reservation_bind_rate']:.3f} + budget={d['budget_bind_rate']:.3f}; "
        f"need <= {MAX_ADMISSION_GATE_BIND_RATE}) -- this is exactly the confound "
        "docs/audits/vtc_initial_integration_20260805.md found in the original smoke test.",
    ))

    report.gates.append(GateResult(
        "decision_disagreement",
        d["decision_disagreement_rate"] >= MIN_DECISION_DISAGREEMENT_RATE,
        f"decision_disagreement_rate={d['decision_disagreement_rate']:.3f} "
        f"(need >= {MIN_DECISION_DISAGREEMENT_RATE}) -- VTC's min-served pick must "
        "genuinely differ from FIFO's oldest-first pick at a meaningful fraction of "
        "contended decision points, or there is nothing for a comparative sweep to measure.",
    ))

    if name in DISCRIMINATIVE_DISPARITY_FAMILIES:
        checkpoint = 0.6 * max(r.arrival_time for r in requests)
        fifo_sim = Simulator(SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG]))
        fifo_sim.load_trace(requests)
        fifo_sim.run(FIFOPolicy(), workload_tag=f"{name}_fifo_disparity_probe")
        fifo_jain = _checkpoint_jain(fifo_sim, tenants, checkpoint)
        report.metrics["fifo_jain_at_checkpoint"] = fifo_jain
        report.gates.append(GateResult(
            "fifo_shows_real_disparity",
            fifo_jain <= MAX_DISCRIMINATIVE_FIFO_JAIN,
            f"fifo_jain_at_checkpoint={fifo_jain:.3f} (need <= {MAX_DISCRIMINATIVE_FIFO_JAIN}) "
            "-- if FIFO is already nearly perfectly fair, there is no disparity left for "
            "VTC to demonstrably correct.",
        ))

    if name in SLO_DIVERGENCE_FAMILIES:
        fifo_sim = Simulator(SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG]))
        fifo_sim.load_trace(requests)
        fifo_sim.run(FIFOPolicy(), workload_tag=f"{name}_fifo_slo_probe")
        hp_completed = [cr for cr in fifo_sim._completed if cr.request.class_id == "high_priority_tight_slo"]
        viol_rate = (
            sum(1 for cr in hp_completed if cr.slo_violated) / len(hp_completed)
            if hp_completed else float("nan")
        )
        report.metrics["fifo_tight_slo_violation_rate"] = viol_rate
        lo, hi = SLO_DIVERGENCE_RANGE
        report.gates.append(GateResult(
            "slo_divergence_is_meaningful",
            lo < viol_rate < hi if viol_rate == viol_rate else False,
            f"fifo_tight_slo_violation_rate={viol_rate:.3f} (need in ({lo}, {hi}), exclusive) "
            "-- must be genuinely mixed, not floor (SLO never binds) or ceiling "
            "(SLO always violated regardless of policy).",
        ))

    if name == "returning_inactive_tenant":
        duration = max(r.arrival_time for r in requests)
        return_time = 0.85 * duration
        continuous_service_at_return = sum(
            r.prompt_tokens + r.actual_output_tokens
            for r in requests
            if r.class_id == "continuous" and r.arrival_time < return_time
        )
        report.metrics["continuous_demand_before_return"] = continuous_service_at_return
        report.gates.append(GateResult(
            "counter_lift_precondition",
            continuous_service_at_return > 0,
            f"continuous_demand_before_return={continuous_service_at_return} tokens "
            "(need > 0) -- the counter-lift mechanism is only meaningfully exercised if "
            "the OTHER tenant actually accrued real service while `returning` was idle.",
        ))

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default=None, choices=list(ALL_FAIRNESS_FAMILIES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if not args.family and not args.all:
        parser.error("pass --family <name> or --all")

    names = list(ALL_FAIRNESS_FAMILIES) if args.all else [args.family]
    reports = [check_family(n) for n in names]

    all_passed = True
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n=== {r.family}: {status} ===")
        for g in r.gates:
            gs = "PASS" if g.passed else "FAIL"
            print(f"  [{gs}] {g.name}: {g.detail}")
        all_passed = all_passed and r.passed

    print(f"\n{'ALL FAMILIES PASS' if all_passed else 'AT LEAST ONE FAMILY FAILED'}")

    if args.json:
        payload = [
            {
                "family": r.family,
                "passed": r.passed,
                "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in r.gates],
                "metrics": r.metrics,
            }
            for r in reports
        ]
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.json}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

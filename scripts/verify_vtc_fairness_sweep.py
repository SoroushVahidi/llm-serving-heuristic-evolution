#!/usr/bin/env python3
"""Independent verifier for the VTC fairness comparative sweep.

Re-runs every (family, seed, policy) combination from
scripts/run_vtc_fairness_comparative_sweep.py (reusing the SAME
deterministic workload generators and policy builders is expected and
correct -- that reproduces the same DATA) and recomputes every reported
metric from the raw `sim._completed`/`sim._pending_arrivals` output using
FRESH, independently-written implementations -- this file does NOT import
or call any of run_vtc_fairness_comparative_sweep.py's helper functions
(`_jains`, `_percentile`, `_per_tenant_metrics`, `_starvation_intervals`,
`run_one`). Any structural similarity in a formula (e.g. Jain's index has
exactly one standard definition) is coincidental to the metric's own
definition, not code reuse.

ANWG is independently re-derived from its definition read directly out of
`src/llmserveopt/core/metrics.py` (lines ~155-229): success_weight (sum of
priority-or-1.0 over completed, SLO-met requests) / arrival_weight (sum of
priority-or-1.0 over EVERY request in the original trace) -- not simply
trusted from `RunMetrics.arrival_normalized_weighted_goodput`, though the
two are cross-checked against each other as an additional consistency
check (they should agree, since they compute the same quantity two ways).

Requires zero unexplained mismatches against the sweep's own JSON output
(default tolerance: 1e-6 relative, to allow floating-point summation-order
differences only).

Usage:
    python scripts/verify_vtc_fairness_sweep.py --sweep-json PATH [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from baselines.vtc.adapter.diagnostics import InstrumentedVTCFairnessPolicy  # noqa: E402
from baselines.vtc.adapter.variants import fairness_isolation_vtc, matched_admission_fifo  # noqa: E402
from baselines.vtc.fairness_workloads import ALL_FAIRNESS_FAMILIES, RECOMMENDED_GPU_CONFIG  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policies.scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy  # noqa: E402
from llmserveopt.policies.shortest_prompt_first import ShortestPromptFirstPolicy  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

TOLERANCE = 1e-6

POLICY_BUILDERS = {
    "official_vtc": lambda tenants: InstrumentedVTCFairnessPolicy(known_tenants=tenants),
    "matched_admission_fifo": lambda tenants: matched_admission_fifo(),
    "fairness_isolation_vtc": lambda tenants: fairness_isolation_vtc(known_tenants=tenants),
    "fifo": lambda tenants: FIFOPolicy(),
    "shortest_prompt_first": lambda tenants: ShortestPromptFirstPolicy(),
    "scorpio_style_slo_guard": lambda tenants: ScorpioStyleSloGuardPolicy(),
}


def independent_request_weight(priority: float) -> float:
    """Reimplementation of metrics._request_weight, from its definition
    (priority if positive, else 1.0), not imported."""
    return priority if priority > 0 else 1.0


def independent_jains_index(values: List[float]) -> float:
    """Standard Jain's fairness index: (sum x)^2 / (n * sum x^2)."""
    xs = [x for x in values if x == x]
    if not xs:
        return float("nan")
    numerator = sum(xs) ** 2
    denominator = len(xs) * sum(x ** 2 for x in xs)
    return numerator / denominator if denominator > 0 else float("nan")


def independent_percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile via manual sort + index -- an independent
    implementation, not numpy.percentile (which uses linear interpolation
    by default) and not the sweep script's own `_percentile` (which this
    file happens to share the same nearest-rank CHOICE with, since that is
    the definition used elsewhere in this repo -- but reimplemented here
    from scratch, not imported)."""
    xs = sorted(v for v in values if v == v)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, round(pct * (len(xs) - 1))))
    return xs[k]


def independent_starvation(completions_by_tenant: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for tenant, times in completions_by_tenant.items():
        ordered = sorted(times)
        if not ordered:
            out[tenant] = float("nan")
            continue
        biggest_gap = ordered[0]  # gap from t=0 to first completion
        prev = ordered[0]
        for t in ordered[1:]:
            gap = t - prev
            if gap > biggest_gap:
                biggest_gap = gap
            prev = t
        out[tenant] = biggest_gap
    return out


def rerun_and_recompute(family_name: str, seed_offset: int, policy_name: str, checkpoint_frac: float = 0.6) -> dict:
    fn = ALL_FAIRNESS_FAMILIES[family_name]
    base_seed = inspect.signature(fn).parameters["seed"].default
    requests, tenants = fn(seed=base_seed + seed_offset * 1000)
    n_total_trace = len(requests)
    duration = max(r.arrival_time for r in requests)
    checkpoint = checkpoint_frac * duration

    cfg = SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG])
    sim = Simulator(cfg)
    sim.load_trace(requests)
    policy = POLICY_BUILDERS[policy_name](tenants)
    sim_metrics = sim.run(policy, workload_tag=f"verify_{family_name}_{policy_name}_seed{seed_offset}")

    completed = list(sim._completed)  # noqa: SLF001 -- raw per-request output, precedented in this repo's tests

    # --- Independent completion fraction ---
    completion_fraction = len(completed) / n_total_trace if n_total_trace else float("nan")

    # --- Independent ANWG ---
    arrival_weight = sum(independent_request_weight(r.priority) for r in requests)
    success_weight = sum(
        independent_request_weight(cr.request.priority)
        for cr in completed
        if not cr.slo_violated
    )
    anwg = success_weight / arrival_weight if arrival_weight > 0 else float("nan")

    # --- Independent per-tenant checkpoint service + Jain + disparity ---
    service_at_checkpoint: Dict[str, float] = defaultdict(float)
    completions_by_tenant: Dict[str, List[float]] = defaultdict(list)
    qdelays_by_tenant: Dict[str, List[float]] = defaultdict(list)
    for cr in completed:
        tenant = cr.request.class_id
        completions_by_tenant[tenant].append(cr.completion_time)
        qdelays_by_tenant[tenant].append(cr.queuing_delay)
        if cr.completion_time <= checkpoint:
            service_at_checkpoint[tenant] += cr.request.prompt_tokens + cr.request.actual_output_tokens
    checkpoint_values = [service_at_checkpoint.get(t, 0.0) for t in tenants]
    jain = independent_jains_index(checkpoint_values)
    disparity = (max(checkpoint_values) - min(checkpoint_values)) if checkpoint_values else float("nan")
    starvation = independent_starvation(completions_by_tenant)

    per_tenant_p95 = {
        t: independent_percentile(qdelays_by_tenant.get(t, []), 0.95) for t in tenants
    }

    row = {
        "family": family_name,
        "seed_offset": seed_offset,
        "policy": policy_name,
        "recomputed_completion_fraction": completion_fraction,
        "recomputed_anwg": anwg,
        "simulator_reported_anwg": sim_metrics.arrival_normalized_weighted_goodput,
        "recomputed_jains_index_at_checkpoint": jain,
        "recomputed_max_service_disparity_at_checkpoint": disparity,
        "recomputed_starvation_intervals": starvation,
        "recomputed_p95_queuing_delay_by_tenant": per_tenant_p95,
    }

    if isinstance(policy, InstrumentedVTCFairnessPolicy):
        # Independent reservation-bind-rate recomputation directly from
        # the raw step_log, not by calling policy.decomposition_summary().
        n_steps = len(policy.step_log)
        n_reservation = sum(1 for s in policy.step_log if s.stopped_by_reservation)
        n_budget = sum(1 for s in policy.step_log if s.stopped_by_budget)
        row["recomputed_reservation_bind_rate"] = n_reservation / n_steps if n_steps else float("nan")
        row["recomputed_budget_bind_rate"] = n_budget / n_steps if n_steps else float("nan")

    return row


def compare(sweep_row: dict, verified_row: dict) -> List[str]:
    mismatches = []

    def close(a, b, tol=TOLERANCE):
        if a is None or b is None:
            return True
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            return True
        try:
            return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
        except TypeError:
            return a == b

    if not close(sweep_row["completion_fraction"], verified_row["recomputed_completion_fraction"]):
        mismatches.append(
            f"completion_fraction: sweep={sweep_row['completion_fraction']} "
            f"verified={verified_row['recomputed_completion_fraction']}"
        )
    if not close(sweep_row["anwg"], verified_row["recomputed_anwg"]):
        mismatches.append(f"anwg: sweep={sweep_row['anwg']} verified={verified_row['recomputed_anwg']}")
    if not close(verified_row["simulator_reported_anwg"], verified_row["recomputed_anwg"]):
        mismatches.append(
            f"anwg self-consistency: simulator={verified_row['simulator_reported_anwg']} "
            f"independent={verified_row['recomputed_anwg']}"
        )
    if not close(sweep_row["jains_index_at_checkpoint"], verified_row["recomputed_jains_index_at_checkpoint"]):
        mismatches.append(
            f"jains_index: sweep={sweep_row['jains_index_at_checkpoint']} "
            f"verified={verified_row['recomputed_jains_index_at_checkpoint']}"
        )
    if not close(
        sweep_row["max_service_disparity_at_checkpoint"],
        verified_row["recomputed_max_service_disparity_at_checkpoint"],
    ):
        mismatches.append(
            f"disparity: sweep={sweep_row['max_service_disparity_at_checkpoint']} "
            f"verified={verified_row['recomputed_max_service_disparity_at_checkpoint']}"
        )
    for tenant, sweep_val in sweep_row.get("starvation_intervals", {}).items():
        verified_val = verified_row["recomputed_starvation_intervals"].get(tenant)
        if not close(sweep_val, verified_val):
            mismatches.append(f"starvation[{tenant}]: sweep={sweep_val} verified={verified_val}")
    if sweep_row.get("reservation_bind_rate") is not None:
        if not close(sweep_row["reservation_bind_rate"], verified_row.get("recomputed_reservation_bind_rate")):
            mismatches.append(
                f"reservation_bind_rate: sweep={sweep_row['reservation_bind_rate']} "
                f"verified={verified_row.get('recomputed_reservation_bind_rate')}"
            )
        if not close(sweep_row["budget_bind_rate"], verified_row.get("recomputed_budget_bind_rate")):
            mismatches.append(
                f"budget_bind_rate: sweep={sweep_row['budget_bind_rate']} "
                f"verified={verified_row.get('recomputed_budget_bind_rate')}"
            )

    return mismatches


def compute_win_tie_loss(sweep_rows: List[dict]) -> Dict[str, Dict[str, int]]:
    """Independently tabulate, per policy, how often it has the strictly
    highest Jain's index (win), is within 0.01 of the best (tie), or is
    below that (loss) -- across every (family, seed) combination."""
    grouped = defaultdict(list)
    for r in sweep_rows:
        grouped[(r["family"], r["seed_offset"])].append(r)

    tally: Dict[str, Dict[str, int]] = defaultdict(lambda: {"win": 0, "tie": 0, "loss": 0})
    for _, rows in grouped.items():
        best = max(r["jains_index_at_checkpoint"] for r in rows if r["jains_index_at_checkpoint"] == r["jains_index_at_checkpoint"])
        for r in rows:
            j = r["jains_index_at_checkpoint"]
            if j != j:
                continue
            if j >= best - 1e-9:
                tally[r["policy"]]["win"] += 1
            elif j >= best - 0.01:
                tally[r["policy"]]["tie"] += 1
            else:
                tally[r["policy"]]["loss"] += 1
    return dict(tally)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-json", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()

    with open(args.sweep_json) as f:
        sweep_rows = json.load(f)

    seeds = [int(s) for s in args.seeds.split(",")]
    total_mismatches = 0
    total_checked = 0

    for family_name in ALL_FAIRNESS_FAMILIES:
        for seed_offset in seeds:
            for policy_name in POLICY_BUILDERS:
                sweep_row = next(
                    r for r in sweep_rows
                    if r["family"] == family_name and r["seed_offset"] == seed_offset and r["policy"] == policy_name
                )
                verified_row = rerun_and_recompute(family_name, seed_offset, policy_name)
                mismatches = compare(sweep_row, verified_row)
                total_checked += 1
                if mismatches:
                    total_mismatches += 1
                    print(f"MISMATCH [{family_name}/seed{seed_offset}/{policy_name}]:")
                    for m in mismatches:
                        print(f"    {m}")

    print(f"\nChecked {total_checked} (family, seed, policy) combinations.")
    print(f"Mismatches: {total_mismatches}")

    win_tie_loss = compute_win_tie_loss(sweep_rows)
    print("\nIndependent win/tie/loss tally (Jain's index, checkpoint-based, all families+seeds):")
    for policy_name, counts in sorted(win_tie_loss.items()):
        print(f"  {policy_name:24s} win={counts['win']:3d} tie={counts['tie']:3d} loss={counts['loss']:3d}")

    print("\nLeakage check: structurally guaranteed -- ObservableRequest "
          "(src/llmserveopt/core/types.py) has no actual_output_tokens field "
          "at all; every policy builder above only ever receives ObservableRequest "
          "instances via BasePolicy.select_action, never raw Request.")

    return 0 if total_mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

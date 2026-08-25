#!/usr/bin/env python3
"""Runner script for the Fairness and Starvation Pilot v1 (Family A).

Generates the scenarios, runs the discrete-event simulator for all registered
policies on the grid, and aggregates tenant SLO and fairness statistics.

Metric naming (important):
- ``unweighted_slo_success_rate``: fraction of loaded requests that completed
  without SLO violation (unweighted). Job 1182306 historically wrote this
  value under the column name ``anwg``; that historical CSV is preserved
  unchanged and must not be silently rewritten.
- ``arrival_normalized_weighted_goodput``: canonical project ANWG from
  ``RunMetrics`` / ``compute_metrics`` (priority-weighted, arrival-normalized).

Do not conflate the two.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.registry import make_policy_library_v2
from llmserveopt.policy_separation.templates_fairness_starvation import (
    case4_fairness_starvation,
)
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

# Stable schema for future corrected runs (Job 1182306 used historical
# column ``anwg`` for the unweighted SLO-success rate).
RESULT_FIELDNAMES = [
    "scenario_id",
    "policy_name",
    "unweighted_slo_success_rate",
    "arrival_normalized_weighted_goodput",
    "inter_violations",
    "inter_total",
    "bulk_violations",
    "bulk_total",
    "jains_fairness_index",
    "mean_ttft",
    "status",
]


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def build_scenarios_from_config(cfg: dict) -> List[Any]:
    grid = cfg["sweep_grid"]
    scenarios = []
    for util in grid["target_utilization"]:
        for skew in grid["tenant_weight_skew"]:
            for vol in grid["interactive_volume_fraction"]:
                for seed in grid["seeds"]:
                    scenarios.append(
                        case4_fairness_starvation(
                            target_utilization=util,
                            tenant_weight_skew=skew,
                            interactive_volume_fraction=vol,
                            seed=seed,
                        )
                    )
    return scenarios


def jains_index(g1: float, g2: float) -> float:
    """Two-tenant Jain fairness index over goodput rates in [0, 1]."""
    num = (g1 + g2) ** 2
    denom = 2 * (g1 ** 2 + g2 ** 2)
    return num / denom if denom > 0 else 0.0


def _run_one_task(args: Tuple[str, str, Any]) -> dict:
    scenario_id, policy_name, scenario = args
    try:
        policy = make_policy_library_v2(policy_name)
        sim_config = SimulatorConfig(gpu_configs=list(scenario.gpu_configs))
        sim = Simulator(sim_config)
        sim.load_trace(list(scenario.requests))

        metrics = sim.run(policy, workload_tag=scenario_id)
        completed = sim._completed  # noqa: SLF001 -- needed for tenant splits

        interactive_completed = [
            cr for cr in completed if cr.request.class_id == "tenant_interactive"
        ]
        bulk_completed = [
            cr for cr in completed if cr.request.class_id == "tenant_bulk"
        ]

        inter_v = sum(
            1
            for cr in interactive_completed
            if cr.completion_time > cr.request.slo_deadline
        )
        bulk_v = sum(
            1 for cr in bulk_completed if cr.completion_time > cr.request.slo_deadline
        )

        inter_total = len(interactive_completed)
        bulk_total = len(bulk_completed)

        inter_gp = (inter_total - inter_v) / max(1, inter_total)
        bulk_gp = (bulk_total - bulk_v) / max(1, bulk_total)
        jfi = jains_index(inter_gp, bulk_gp)

        ttfts = [cr.ttft for cr in completed if cr.first_token_time >= 0]
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

        total_v = inter_v + bulk_v
        unweighted_slo_success_rate = (len(completed) - total_v) / len(scenario.requests)

        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "unweighted_slo_success_rate": unweighted_slo_success_rate,
            "arrival_normalized_weighted_goodput": float(
                metrics.arrival_normalized_weighted_goodput
            ),
            "inter_violations": inter_v,
            "inter_total": inter_total,
            "bulk_violations": bulk_v,
            "bulk_total": bulk_total,
            "jains_fairness_index": jfi,
            "mean_ttft": mean_ttft,
            "status": "success",
        }
    except Exception as e:
        import traceback

        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, Iterator) or isinstance(cfg, list):
        cfg = list(cfg)[0]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(args.config)

    _log(
        args.run_dir,
        f"Starting Fairness and Starvation Pilot v1 (dry_run={args.dry_run})",
    )

    scenarios = build_scenarios_from_config(cfg)
    if args.dry_run:
        scenarios = scenarios[:4]

    _log(args.run_dir, f"Generated {len(scenarios)} scenarios.")

    policies = list(cfg["policies"])
    tasks = []
    for scen in scenarios:
        for p in policies:
            tasks.append((scen.scenario_id, p, scen))

    _log(args.run_dir, f"Total tasks to run: {len(tasks)}")

    results: List[dict] = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one_task, t): t for t in tasks}
        completed_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed_count += 1
            if completed_count % 50 == 0:
                _log(args.run_dir, f"Completed {completed_count}/{len(tasks)} tasks.")

    elapsed = time.time() - start_time
    _log(
        args.run_dir,
        f"Completed execution of {len(results)} tasks in {elapsed:.2f} seconds.",
    )

    success_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]

    _log(
        args.run_dir,
        f"Successes: {len(success_results)}, Failures: {len(failed_results)}",
    )

    out_csv = args.run_dir / "per_policy_results.csv"
    if success_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(success_results)

    if failed_results:
        with open(args.run_dir / "failures.jsonl", "w") as f:
            for r in failed_results:
                f.write(json.dumps(r) + "\n")

    token_sources = sorted(
        {
            str(s.params.get("token_length_source", "unknown"))
            for s in scenarios
        }
    )
    summary = {
        "experiment_name": cfg["pilot_metadata"]["experiment_name"],
        "n_scenarios": len(scenarios),
        "n_tasks": len(tasks),
        "n_completed": len(success_results),
        "n_failed": len(failed_results),
        "elapsed_seconds": elapsed,
        "result_schema_version": "fairness_starvation_v1_metrics_clarified",
        "token_length_sources_observed": token_sources,
        "metric_notes": {
            "unweighted_slo_success_rate": (
                "fraction of loaded requests completing without SLO violation; "
                "Job 1182306 historically labeled this column 'anwg'"
            ),
            "arrival_normalized_weighted_goodput": (
                "canonical RunMetrics ANWG (priority-weighted, arrival-normalized)"
            ),
        },
    }
    with open(args.run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log(args.run_dir, "Completed Pilot Execution successfully!")


if __name__ == "__main__":
    main()

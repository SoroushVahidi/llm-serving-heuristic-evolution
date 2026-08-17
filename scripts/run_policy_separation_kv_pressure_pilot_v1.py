#!/usr/bin/env python3
"""Family C v1 KV-pressure reserve pairwise-separation pilot runner.

See docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md. This is a
pairwise-separation pilot, NOT a composition falsification: exactly two
policies (`kv_constrained_online`, `least_laxity_first`) are evaluated on
every scenario -- no selector is fit, no child policy is run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.templates_kv_pressure import (  # noqa: E402
    CLASS_BULK,
    CLASS_URGENT,
    assert_policy_visible_fields_clean_kv_v1,
    case_kv_pressure_reserve_contention,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy  # noqa: E402
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


class InstrumentedKVConstrainedOnlinePolicy(KVConstrainedOnlinePolicy):
    """Diagnostic wrapper: counts admission-deferral-due-to-reserve events
    (H2/G3) without changing KVConstrainedOnlinePolicy's own logic. A
    "deferral" is any candidate the gate would have admitted on hard KV
    capacity alone but blocked because it is non-urgent and would push
    utilization past `target_kv_utilization`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_reserve_deferrals = 0

    def _admit_filter(self, req, gpu, admitted, now):
        post_util = (gpu.current_kv_tokens + req.prompt_tokens) / max(gpu.max_kv_tokens, 1)
        from llmserveopt.policies.policy_library_v2_helpers import laxity_seconds
        urgent = laxity_seconds(req, now, self.step_size, self.alpha, self.beta) <= self.urgent_laxity_seconds
        would_defer = post_util > self.target_kv_utilization and not urgent
        if would_defer:
            self.n_reserve_deferrals += 1
        return super()._admit_filter(req, gpu, admitted, now)


POLICIES = {
    "kv_constrained_online": InstrumentedKVConstrainedOnlinePolicy,
    "least_laxity_first": LeastLaxityFirstPolicy,
}

RESULT_FIELDNAMES = [
    "scenario_id", "policy_name", "bulk_pressure", "urgent_arrival_phase",
    "urgent_tightness", "seed", "status",
    "arrival_normalized_weighted_goodput", "unweighted_slo_success_rate",
    "completion_fraction",
    "bulk_n", "bulk_slo_success_rate",
    "urgent_n", "urgent_slo_success_rate",
    "peak_kv_utilization", "steps_over_reserve_threshold", "n_steps",
    "n_reserve_deferrals",
]


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def _class_success(completed, class_id: str) -> Tuple[int, float]:
    rows = [c for c in completed if c.request.class_id == class_id]
    if not rows:
        return 0, 0.0
    ok = sum(1 for c in rows if not c.slo_violated)
    return len(rows), ok / len(rows)


def _run_one(args: Tuple[str, str, Any, int]) -> dict:
    scenario_id, policy_name, scenario, max_kv_tokens = args
    try:
        policy_cls = POLICIES[policy_name]
        policy = policy_cls()
        policy.name = policy_name

        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**scenario.service_model_kwargs),
        ))
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(policy, workload_tag=scenario_id, seed=scenario.seed)
        completed = list(sim._completed)  # noqa: SLF001

        n_req = len(scenario.requests)
        n_violated = sum(1 for c in completed if c.slo_violated)
        unweighted = (len(completed) - n_violated) / max(1, n_req)

        bulk_n, bulk_ok = _class_success(completed, CLASS_BULK)
        urgent_n, urgent_ok = _class_success(completed, CLASS_URGENT)

        kv_hist = sim._gpus[0].step_kv_used  # noqa: SLF001
        peak_util = (max(kv_hist) / max_kv_tokens) if kv_hist else 0.0
        n_over = sum(1 for k in kv_hist if k / max_kv_tokens > 0.82)

        n_deferrals = getattr(policy, "n_reserve_deferrals", 0)

        row = {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "bulk_pressure": scenario.params["bulk_pressure"],
            "urgent_arrival_phase": scenario.params["urgent_arrival_phase"],
            "urgent_tightness": scenario.params["urgent_tightness"],
            "seed": scenario.seed,
            "status": "success",
            "arrival_normalized_weighted_goodput": float(
                metrics.arrival_normalized_weighted_goodput
            ),
            "unweighted_slo_success_rate": float(unweighted),
            "completion_fraction": float(metrics.completion_fraction),
            "bulk_n": bulk_n,
            "bulk_slo_success_rate": bulk_ok,
            "urgent_n": urgent_n,
            "urgent_slo_success_rate": urgent_ok,
            "peak_kv_utilization": float(peak_util),
            "steps_over_reserve_threshold": n_over,
            "n_steps": len(kv_hist),
            "n_reserve_deferrals": n_deferrals,
        }
        return row
    except Exception as e:  # noqa: BLE001
        import traceback
        return {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def build_scenarios(cfg: dict, *, allow_synthetic_tokens: bool, datasets_root) -> List[Any]:
    grid = cfg["sweep_grid"]
    max_kv_tokens = int(cfg.get("max_kv_tokens", 8000))
    max_active_sequences = int(cfg.get("max_active_sequences", 64))
    max_batch_tokens = int(cfg.get("max_batch_tokens", 64))
    scenarios = []
    for bulk_pressure in grid["bulk_pressure"]:
        for phase in grid["urgent_arrival_phase"]:
            for tightness in grid["urgent_tightness"]:
                for seed in grid["seeds"]:
                    s = case_kv_pressure_reserve_contention(
                        bulk_pressure=str(bulk_pressure),
                        urgent_arrival_phase=str(phase),
                        urgent_tightness=str(tightness),
                        seed=int(seed),
                        max_kv_tokens=max_kv_tokens,
                        max_active_sequences=max_active_sequences,
                        max_batch_tokens=max_batch_tokens,
                        allow_synthetic_tokens=allow_synthetic_tokens,
                        datasets_root=datasets_root,
                    )
                    assert_policy_visible_fields_clean_kv_v1(s)
                    scenarios.append(s)
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-synthetic-tokens", action="store_true")
    parser.add_argument("--datasets-root", type=Path, default=None)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _log(args.run_dir, "Starting Family C v1 KV-pressure reserve pairwise-separation pilot.")
    scenarios = build_scenarios(
        cfg, allow_synthetic_tokens=args.allow_synthetic_tokens,
        datasets_root=args.datasets_root,
    )
    max_kv_tokens = int(cfg.get("max_kv_tokens", 8000))
    _log(args.run_dir, f"Generated {len(scenarios)} scenarios.")

    tasks = []
    for s in scenarios:
        for policy_name in POLICIES:
            tasks.append((s.scenario_id, policy_name, s, max_kv_tokens))

    all_results: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            all_results.append(fut.result())

    with open(args.run_dir / "per_policy_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    success = [r for r in all_results if r.get("status") == "success"]
    failed = [r for r in all_results if r.get("status") == "failed"]
    summary = {
        "n_scenarios": len(scenarios),
        "n_tasks": len(all_results),
        "n_completed": len(success),
        "n_failed": len(failed),
        "primary_metric": "arrival_normalized_weighted_goodput",
        "max_kv_tokens": max_kv_tokens,
        "policies": list(POLICIES.keys()),
    }
    with open(args.run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(args.run_dir, f"Completed: {len(success)} success, {len(failed)} failed")


if __name__ == "__main__":
    main()

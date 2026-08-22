#!/usr/bin/env python3
"""Family-A receding-horizon oracle V1 -- induced state-distribution
comparison (design doc SS22 / task section 22).

DESCRIPTIVE ONLY. Compares observable-state trajectory summaries (queue
length, active/batch size, KV utilization) visited by fixed ESTF, fixed WFS,
`family_a_stateful_controller_v1`, and one receding-horizon oracle horizon
(H=20, matching the prior controller's dwell timescale for comparability),
across the same 64 Family-A TRAIN/VAL scenarios. Uses only the simulator's
own already-tracked per-step histories
(`_waiting_queue_history`/`_util_history`/`_batch_history`) -- no new
instrumentation, no reimplementation, no causal claim.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.analysis import family_a_observability_continuation_v1 as family_a_obs
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_receding_horizon_oracle_v1 import (
    COMMON_CONTINUATION_BUDGET,
    FamilyARecedingHorizonOracleV1,
)
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    FamilyAStatefulControllerV1,
    FrozenTreeModeModel,
    STATEFUL_CONTROLLER_FEATURES,
)
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = REPO_ROOT / "experiments/family_a_receding_horizon_oracle_v1"
REPAIRED_EVENTS_PATH = (
    REPO_ROOT
    / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"
)
TREE_RANDOM_STATE = 20260820
REPRESENTATIVE_HORIZON = 20
MAX_PLANNING_CALLS_PER_SCENARIO = 150


def fit_stateful_tree() -> FrozenTreeModeModel:
    events = pd.read_csv(REPAIRED_EVENTS_PATH)
    X = events.loc[:, STATEFUL_CONTROLLER_FEATURES].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = (events["delta_native"].astype(float).to_numpy() > 0.0).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=TREE_RANDOM_STATE)
    tree.fit(X, y)
    return FrozenTreeModeModel.from_sklearn(tree, STATEFUL_CONTROLLER_FEATURES)


def summary_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)), "max": float(np.max(arr)),
    }


def run_and_capture(row: pd.Series, policy_id: str, policy, ) -> Dict[str, Any]:
    scenario = family_a_obs.rebuild_scenario_from_row(row)
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs), service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    if hasattr(policy, "sim_ref"):
        policy.sim_ref = sim
    if hasattr(policy, "reset"):
        policy.reset()
    sim.run(policy, workload_tag=str(row["canonical_scenario_id"]), seed=int(scenario.seed))
    return {
        "canonical_scenario_id": str(row["canonical_scenario_id"]),
        "policy_id": policy_id,
        "queue_length": summary_stats(sim._waiting_queue_history),
        "kv_utilization": summary_stats(sim._util_history),
        "active_batch_size": summary_stats(sim._batch_history),
    }


def main() -> int:
    table = family_a_obs.load_family_a_trainval_scenario_table()
    tree_model = fit_stateful_tree()

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    n = len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        step_size = float(ServiceModel(**family_a_obs.rebuild_scenario_from_row(row).service_model_kwargs).step_size)
        dummy_sim = Simulator(SimulatorConfig(
            gpu_configs=list(family_a_obs.rebuild_scenario_from_row(row).gpu_configs),
            service_model=ServiceModel(**family_a_obs.rebuild_scenario_from_row(row).service_model_kwargs),
        ))
        policies = {
            "estimated_service_time_first": EstimatedServiceTimeFirstPolicy(),
            "weighted_fair_share": WeightedFairSharePolicy(),
            "family_a_stateful_controller_v1": FamilyAStatefulControllerV1(
                mode_model=tree_model, step_size=step_size, min_dwell_steps=20,
                estf_enter_threshold=0.65, wfs_enter_threshold=0.35,
            ),
            f"family_a_receding_horizon_oracle_v1_h{REPRESENTATIVE_HORIZON}": FamilyARecedingHorizonOracleV1(
                sim_ref=dummy_sim, horizon=REPRESENTATIVE_HORIZON, continuation_budget=COMMON_CONTINUATION_BUDGET,
                max_planning_calls_per_scenario=MAX_PLANNING_CALLS_PER_SCENARIO,
            ),
        }
        for pid, policy in policies.items():
            rows.append(run_and_capture(row, pid, policy))
        print(f"[{i + 1}/{n}] {row['canonical_scenario_id']} done", flush=True)

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "family_a_receding_horizon_oracle_v1_state_distribution.json"

    agg: Dict[str, Any] = {}
    for pid, sub in df.groupby("policy_id"):
        agg[pid] = {
            metric: summary_stats([r[metric]["mean"] for _, r in sub.iterrows() if r[metric]["mean"] is not None])
            for metric in ("queue_length", "kv_utilization", "active_batch_size")
        }
    payload = {"per_scenario": rows, "aggregate_of_per_scenario_means": agg, "wall_clock_s": time.time() - t0}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Results written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

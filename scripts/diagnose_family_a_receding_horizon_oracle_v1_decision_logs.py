#!/usr/bin/env python3
"""Diagnostic-only re-run of the FROZEN family_a_receding_horizon_oracle_v1
controller to capture full per-decision logs, which the original scientific
run (scripts/run_family_a_receding_horizon_oracle_v1.py) did not persist
(only the aggregated `diagnostics()` summary was saved per scenario).

This is NOT a new experiment: identical controller code, identical 64
Family-A TRAIN/VAL scenarios, identical horizons {1,5,20}, identical
objective/fallback/continuation-budget. Determinism across repeated runs is
already proven by tests/test_family_a_receding_horizon_oracle_v1.py
(`test_full_run_state_untouched_and_deterministic`), so this re-run is
scientifically equivalent to having logged the original run more verbosely.
No controller/design/objective/horizon change of any kind.

Used only for docs/current/family_a_rollout_value_limit_diagnosis_20260820.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.analysis import family_a_observability_continuation_v1 as family_a_obs
from llmserveopt.policies.family_a_receding_horizon_oracle_v1 import (
    COMMON_CONTINUATION_BUDGET,
    FamilyARecedingHorizonOracleV1,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

OUTPUT_DIR = REPO_ROOT / "experiments/family_a_receding_horizon_oracle_v1"
HORIZONS = (1, 5, 20)
MAX_PLANNING_CALLS_PER_SCENARIO = 150


def build_sim(row: pd.Series):
    scenario = family_a_obs.rebuild_scenario_from_row(row)
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    return sim, scenario


def main() -> int:
    table = family_a_obs.load_family_a_trainval_scenario_table()
    assert not (table["split"].str.lower() == "test").any()
    assert len(table) == 64

    all_rows = []
    t0 = time.perf_counter()
    n = len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        sid = str(row["canonical_scenario_id"])
        for h in HORIZONS:
            sim, scenario = build_sim(row)
            policy = FamilyARecedingHorizonOracleV1(
                sim_ref=sim, horizon=h, continuation_budget=COMMON_CONTINUATION_BUDGET,
                max_planning_calls_per_scenario=MAX_PLANNING_CALLS_PER_SCENARIO,
            )
            policy.reset()
            sim.run(policy, workload_tag=sid, seed=int(scenario.seed))
            for d in policy.decision_log():
                d["canonical_scenario_id"] = sid
                d["split"] = str(row["split"])
                d["horizon"] = h
                all_rows.append(d)
        print(f"[{i + 1}/{n}] {sid} done", flush=True)

    df = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "family_a_receding_horizon_oracle_v1_decision_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"wall_clock_s={time.perf_counter() - t0:.1f}")
    print(f"rows={len(df)}")
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

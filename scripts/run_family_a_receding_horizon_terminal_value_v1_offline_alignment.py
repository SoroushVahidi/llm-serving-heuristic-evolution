#!/usr/bin/env python3
"""Offline alignment gate for the Family-A terminal-value redesign (design
doc SS7, `docs/design/FAMILY_A_TERMINAL_VALUE_V1.md`).

Executes the UNCHANGED old-V1 rollout controller (so the real trajectory
exactly matches the existing `family_a_receding_horizon_oracle_v1_results.json`
run -- no execution-path change during this offline phase), but at every
eligible decision ALSO computes the new terminal value for both branches as
a side channel (never affecting what gets executed here). This answers:
would the new value have preferred a different branch, and does that shift
align better with the already-known scenario-level outcome
(`controller_ANWG - WFS_ANWG`, from the existing V1 results)?

TRAIN/VAL only. No TEST. No controller-mechanics change (eligibility,
fallback, horizons, continuation budget, candidates are all identical to
V1 -- only scoring is computed twice, for comparison, never for execution
here).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.analysis import family_a_observability_continuation_v1 as family_a_obs
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_receding_horizon_oracle_v1 import (
    COMMON_CONTINUATION_BUDGET,
    ESTF_ID,
    WFS_ID,
    _run_chained_branch as _run_chained_branch_old,
)
from llmserveopt.policies.family_a_receding_horizon_terminal_value_v1 import (
    _run_chained_branch_dual,
)
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    actions_disagree,
    restore_gpu_counters,
    snapshot_gpu_counters,
)
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

OUTPUT_DIR = REPO_ROOT / "experiments/family_a_receding_horizon_terminal_value_v1"
HORIZONS = (1, 5, 20)
MAX_PLANNING_CALLS_PER_SCENARIO = 150


class _ShadowDualScoringPolicy(BasePolicy):
    """Diagnostic-only: executes old-V1 preference; also computes the new
    terminal value at every eligible decision as an unused side channel."""

    name = "shadow_dual_scoring_offline_only"

    def __init__(self, sim_ref: Simulator, horizon: int, continuation_budget: int, max_calls: int):
        self.sim_ref = sim_ref
        self.horizon = horizon
        self.continuation_budget = continuation_budget
        self.max_calls = max_calls
        self.estf_policy = EstimatedServiceTimeFirstPolicy()
        self.wfs_policy = WeightedFairSharePolicy()
        self.continuation_policy = WeightedFairSharePolicy()
        self.rows: List[Dict[str, Any]] = []
        self._calls = 0

    def reset(self) -> None:
        self.estf_policy.reset()
        self.wfs_policy.reset()
        self.continuation_policy.reset()

    def select_action(self, state: ObservableState) -> Action:
        if not state.waiting_queue and not state.migrating_queue:
            return self.wfs_policy.select_action(state)

        snapshot = snapshot_gpu_counters(state)
        action_estf = self.estf_policy.select_action(state)
        restore_gpu_counters(state, snapshot)
        action_wfs = self.wfs_policy.select_action(state)
        restore_gpu_counters(state, snapshot)

        if not actions_disagree(action_estf, action_wfs):
            return action_wfs
        if self._calls >= self.max_calls:
            return action_wfs
        self._calls += 1

        old_estf = _run_chained_branch_old(
            self.sim_ref, candidate_policy=self.estf_policy, candidate_policy_id=ESTF_ID,
            candidate_first_action=action_estf, continuation_policy=self.continuation_policy,
            continuation_policy_id=WFS_ID, horizon=self.horizon, continuation_budget=self.continuation_budget,
        )
        old_wfs = _run_chained_branch_old(
            self.sim_ref, candidate_policy=self.wfs_policy, candidate_policy_id=WFS_ID,
            candidate_first_action=action_wfs, continuation_policy=self.continuation_policy,
            continuation_policy_id=WFS_ID, horizon=self.horizon, continuation_budget=self.continuation_budget,
        )
        new_estf = _run_chained_branch_dual(
            self.sim_ref, candidate_policy=self.estf_policy, candidate_policy_id=ESTF_ID,
            candidate_first_action=action_estf, continuation_policy=self.continuation_policy,
            continuation_policy_id=WFS_ID, horizon=self.horizon, continuation_budget=self.continuation_budget,
        )
        new_wfs = _run_chained_branch_dual(
            self.sim_ref, candidate_policy=self.wfs_policy, candidate_policy_id=WFS_ID,
            candidate_first_action=action_wfs, continuation_policy=self.continuation_policy,
            continuation_policy_id=WFS_ID, horizon=self.horizon, continuation_budget=self.continuation_budget,
        )

        old_winner = ESTF_ID if old_estf.window_objective > old_wfs.window_objective else WFS_ID
        new_winner = ESTF_ID if new_estf.new_terminal_value > new_wfs.new_terminal_value else WFS_ID

        self.rows.append({
            "step": int(state.step),
            "old_estf_obj": old_estf.window_objective, "old_wfs_obj": old_wfs.window_objective,
            "new_estf_val": new_estf.new_terminal_value, "new_wfs_val": new_wfs.new_terminal_value,
            "old_winner": old_winner, "new_winner": new_winner,
        })
        # Execute OLD preference -- this phase never changes the real trajectory.
        return action_estf if old_winner == ESTF_ID else action_wfs


def build_sim(row: pd.Series):
    scenario = family_a_obs.rebuild_scenario_from_row(row)
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs), service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    return sim, scenario


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = family_a_obs.load_family_a_trainval_scenario_table()
    assert not (table["split"].str.lower() == "test").any()
    assert len(table) == 64

    all_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(table.iterrows()):
        sid = str(row["canonical_scenario_id"])
        for h in HORIZONS:
            sim, scenario = build_sim(row)
            policy = _ShadowDualScoringPolicy(sim, h, COMMON_CONTINUATION_BUDGET, MAX_PLANNING_CALLS_PER_SCENARIO)
            policy.reset()
            sim.run(policy, workload_tag=sid, seed=int(scenario.seed))
            for r in policy.rows:
                r["canonical_scenario_id"] = sid
                r["split"] = str(row["split"])
                r["horizon"] = h
                all_rows.append(r)
        print(f"[{i + 1}/{len(table)}] {sid} done", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUTPUT_DIR / "offline_alignment_decisions.csv"
    df.to_csv(out_csv, index=False)

    # Join against existing V1 results for the outcome correlation.
    v1_results_csv = REPO_ROOT / "experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_per_scenario_results.csv"
    v1 = pd.read_csv(v1_results_csv)
    piv = v1.pivot_table(index="canonical_scenario_id", columns="policy_id", values="arrival_normalized_weighted_goodput", aggfunc="first")
    wfs_anwg = piv["weighted_fair_share"]

    ids = table[["canonical_scenario_id"]].drop_duplicates().reset_index(drop=True)
    ext = ids["canonical_scenario_id"].str.extract(r"\.(?P<fav>favlong|favshort)\.")
    meta = pd.concat([ids, ext], axis=1).set_index("canonical_scenario_id")

    summary: Dict[str, Any] = {}
    for h in HORIZONS:
        sub = df[df.horizon == h].copy()
        sub = sub.merge(meta, left_on="canonical_scenario_id", right_index=True, how="left")
        old_ctrl_anwg = piv[f"family_a_receding_horizon_oracle_v1_h{h}"]
        outcome = (old_ctrl_anwg - wfs_anwg).rename("outcome")

        def block(name: str, data: pd.DataFrame) -> Dict[str, Any]:
            n = len(data)
            agree = int((data.old_winner == data.new_winner).sum())
            old_estf_share = float((data.old_winner == ESTF_ID).mean()) if n else None
            new_estf_share = float((data.new_winner == ESTF_ID).mean()) if n else None
            new_margin = data.new_estf_val - data.new_wfs_val
            per_scenario = data.assign(new_estf=lambda d: d.new_winner == ESTF_ID).groupby("canonical_scenario_id").agg(
                new_estf_frac=("new_estf", "mean"))
            j = per_scenario.join(outcome).dropna()
            rho, p = (stats.spearmanr(j.new_estf_frac, j.outcome) if len(j) > 3 else (None, None))
            return {
                "n_decisions": n, "agreement_rate": (agree / n) if n else None,
                "old_estf_share": old_estf_share, "new_estf_share": new_estf_share,
                "mean_new_margin": float(new_margin.mean()) if n else None,
                "spearman_new_estf_frac_vs_outcome": rho, "p_value": p, "n_scenarios": len(j),
            }

        summary[f"h{h}"] = {
            "all": block("all", sub),
            "favlong": block("favlong", sub[sub.fav == "favlong"]),
            "favshort": block("favshort", sub[sub.fav == "favshort"]),
        }

    out_json = OUTPUT_DIR / "offline_alignment_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wall_clock_s={time.perf_counter() - t0:.1f}")
    print(f"Results written to {out_json} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

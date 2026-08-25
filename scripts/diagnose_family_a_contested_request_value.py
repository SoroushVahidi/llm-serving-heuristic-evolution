#!/usr/bin/env python3
"""Read-only, diagnostic-only extraction: for the already-frozen 91 Family-A
repaired disagreement events
(`experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv`),
identify the SPECIFIC contested request(s) ESTF and WFS admit differently,
their causal (pre-decision) properties, and their eventual fate under each
of the four already-defined bounded continuation branches
(br_estf_estf/br_wfs_wfs/br_wfs_estf/br_estf_wfs).

This is a deterministic REPLAY of the exact same, already-published 91
events -- same scenarios, same seeds, same frozen stage1/stage2 models, same
disagreement condition, same 3-events-per-scenario cap -- adding
instrumentation only (recording which specific request IDs are admitted by
each side, and following their individual fates through the same bounded
branches the original diagnostic already ran). It produces NO new
scientific claim about controller performance (no ANWG, no GO/NO_GO for any
controller) and touches no controller/simulator/design code. Integrity is
verified by confirming the replayed (scenario_id, step) set matches the
existing frozen events CSV exactly.

TRAIN/VAL only. No TEST. Family A only.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.analysis import family_a_observability_continuation_v1 as fac
from llmserveopt.analysis.decision_criticality_timescale_trainval_v1 import (
    fork_from_live_simulator,
)
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA, deadline_slack, predicted_service_proxy
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import REGIME_A
from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import (
    LiveHierarchicalRouterPolicy,
    build_native_policy_instances,
    build_feature_rows_by_regime,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ESTF_ID = "estimated_service_time_first"
WFS_ID = "weighted_fair_share"
MAX_EXTRA_STEPS = fac.FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS  # 1500, unchanged
MAX_BRANCHES_PER_SCENARIO = fac.FULL_TRAJECTORY_BRANCHES_PER_SCENARIO  # 3, unchanged

OUTPUT_DIR = REPO_ROOT / "experiments/family_a_contested_request_value_diagnosis"
EXISTING_EVENTS_CSV = REPO_ROOT / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"


def _admitted_ids(action: Action) -> List[int]:
    return [rid for ids in action.admit.values() for rid in ids]


def _run_branch_full(sim: Simulator, *, policy: BasePolicy, policy_id: str, first_action: Action) -> List[Any]:
    """Same bounded-rollout semantics as dcm.run_bounded_rollout, but
    returns the full list of newly-CompletedRequest objects instead of just
    a count, so specific contested request IDs can be traced."""
    fork = fork_from_live_simulator(sim, policy=policy, policy_id=policy_id, first_action=first_action)
    steps_run = 1
    while not fork.finished and steps_run < MAX_EXTRA_STEPS:
        fork.advance_one_step()
        steps_run += 1
    return fork.shell._completed[-fork.completed_in_window:] if fork.completed_in_window else []


def _request_row(req, state_time: float) -> Dict[str, Any]:
    weight = req.priority if req.priority > 0 else 1.0
    svc = predicted_service_proxy(req, DEFAULT_ALPHA, DEFAULT_BETA)
    slack = deadline_slack(req, now=state_time, service_proxy=svc)
    return {
        "request_id": req.request_id,
        "priority": req.priority,
        "weight": weight,
        "class_id": req.class_id,
        "prompt_tokens": req.prompt_tokens,
        "predicted_output_tokens": req.predicted_output_tokens,
        "slo_deadline": req.slo_deadline,
        "arrival_time": req.arrival_time,
        "queue_age": state_time - req.arrival_time,
        "predicted_service_proxy": svc,
        "deadline_slack_if_admitted_now": slack,
        "feasible_if_admitted_now": bool(slack >= 0),
    }


def _find_outcome(completed_list: List[Any], request_id: int) -> Dict[str, Any]:
    for c in completed_list:
        if c.request.request_id == request_id:
            weight = c.request.priority if c.request.priority > 0 else 1.0
            return {
                "completed": True, "completion_time": c.completion_time,
                "slo_violated": bool(c.slo_violated),
                "weighted_contribution": weight * (0.0 if c.slo_violated else 1.0),
            }
    return {"completed": False, "completion_time": None, "slo_violated": None, "weighted_contribution": 0.0}


class ContestedRequestExtractor(BasePolicy):
    """Structurally identical control flow to
    `family_a_observability_continuation_v1.FamilyAObservabilityObserver`
    (same disagreement condition, same 3-per-scenario cap, same real
    trajectory driver) -- reused, not reimplemented, except that in addition
    to running the four branches, it records the specific admitted request
    ID sets and each contested request's causal features + traced fate."""

    name = "contested_request_extractor_diagnostic_only"

    def __init__(self, *, sim_ref, inner_router, shadow_policies, canonical_scenario_id, split, max_branches=MAX_BRANCHES_PER_SCENARIO):
        self.sim_ref = sim_ref
        self.inner_router = inner_router
        self.shadow_policies = shadow_policies
        self.canonical_scenario_id = canonical_scenario_id
        self.split = split
        self.max_branches = max_branches
        self.branches_used = 0
        self.event_rows: List[Dict[str, Any]] = []
        self.contested_rows: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.inner_router.reset()
        for p in self.shadow_policies.values():
            p.reset()
        self.branches_used = 0

    def select_action(self, state: ObservableState) -> Action:
        pre = fac.snapshot_gpu_counters(state)
        real_action = self.inner_router.select_action(state)
        row = self.inner_router.trajectory[-1] if self.inner_router.trajectory else None
        post = fac.snapshot_gpu_counters(state)

        if row is not None and row.effective_regime == REGIME_A and self.branches_used < self.max_branches:
            fac.restore_gpu_counters(state, pre)
            action_estf = self.shadow_policies[ESTF_ID].select_action(state)
            fac.restore_gpu_counters(state, pre)
            action_wfs = self.shadow_policies[WFS_ID].select_action(state)
            fac.restore_gpu_counters(state, pre)

            from llmserveopt.policies.family_a_stateful_controller_v1 import actions_disagree
            if actions_disagree(action_estf, action_wfs):
                feature_state = copy.deepcopy(state)
                admit_estf = set(_admitted_ids(action_estf))
                admit_wfs = set(_admitted_ids(action_wfs))
                estf_only = admit_estf - admit_wfs
                wfs_only = admit_wfs - admit_estf
                common = admit_estf & admit_wfs

                req_by_id = {r.request_id: r for r in feature_state.waiting_queue}
                for g in feature_state.gpu_states:
                    for r in g.active_requests_info:
                        req_by_id.setdefault(r.request_id, r)

                branches = {
                    "br_estf_estf": _run_branch_full(self.sim_ref, policy=self.shadow_policies[ESTF_ID], policy_id=ESTF_ID, first_action=copy.deepcopy(action_estf)),
                    "br_wfs_wfs": _run_branch_full(self.sim_ref, policy=self.shadow_policies[WFS_ID], policy_id=WFS_ID, first_action=copy.deepcopy(action_wfs)),
                    "br_wfs_estf": _run_branch_full(self.sim_ref, policy=self.shadow_policies[ESTF_ID], policy_id=ESTF_ID, first_action=copy.deepcopy(action_wfs)),
                    "br_estf_wfs": _run_branch_full(self.sim_ref, policy=self.shadow_policies[WFS_ID], policy_id=WFS_ID, first_action=copy.deepcopy(action_estf)),
                }

                event_id = f"{self.canonical_scenario_id}::{int(state.step)}"
                self.event_rows.append({
                    "event_id": event_id, "canonical_scenario_id": self.canonical_scenario_id,
                    "split": self.split, "step": int(state.step),
                    "n_estf_only": len(estf_only), "n_wfs_only": len(wfs_only), "n_common": len(common),
                    "n_admit_estf": len(admit_estf), "n_admit_wfs": len(admit_wfs),
                })

                for side, ids in (("estf_only", estf_only), ("wfs_only", wfs_only), ("common", common)):
                    for rid in ids:
                        req = req_by_id.get(rid)
                        if req is None:
                            continue
                        base = {"event_id": event_id, "canonical_scenario_id": self.canonical_scenario_id,
                                "split": self.split, "step": int(state.step), "contested_side": side}
                        base.update(_request_row(req, feature_state.time))
                        for br_name, completed_list in branches.items():
                            outcome = _find_outcome(completed_list, rid)
                            for k, v in outcome.items():
                                base[f"{br_name}_{k}"] = v
                        self.contested_rows.append(base)

                self.branches_used += 1

        fac.restore_gpu_counters(state, post)
        return real_action


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = fac.load_family_a_trainval_scenario_table()
    assert not (table["split"].str.lower() == "test").any()
    assert len(table) == 64

    stage1, stage2_selectors = fac.fit_frozen_models()

    all_events: List[Dict[str, Any]] = []
    all_contested: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, (_, row) in enumerate(table.iterrows()):
        canonical_scenario_id = str(row["canonical_scenario_id"])
        scenario = fac.rebuild_scenario_from_row(row)
        feature_rows = build_feature_rows_by_regime(scenario, canonical_scenario_id)
        inner_router = LiveHierarchicalRouterPolicy(
            scenario_id=canonical_scenario_id, stage1=stage1, stage2_selectors=stage2_selectors,
            feature_rows_by_regime=feature_rows, record_trajectory=True,
        )
        sim = Simulator(SimulatorConfig(gpu_configs=list(scenario.gpu_configs), service_model=ServiceModel(**scenario.service_model_kwargs)))
        sim.load_trace(list(scenario.requests))
        shadow_policies = build_native_policy_instances()
        extractor = ContestedRequestExtractor(
            sim_ref=sim, inner_router=inner_router, shadow_policies=shadow_policies,
            canonical_scenario_id=canonical_scenario_id, split=str(row["split"]),
        )
        sim.run(extractor, workload_tag=canonical_scenario_id, seed=int(row["seed"]))
        all_events.extend(extractor.event_rows)
        all_contested.extend(extractor.contested_rows)
        print(f"[{i + 1}/{len(table)}] {canonical_scenario_id} events={len(extractor.event_rows)}", flush=True)

    events_df = pd.DataFrame(all_events)
    contested_df = pd.DataFrame(all_contested)
    events_df.to_csv(OUTPUT_DIR / "contested_events.csv", index=False)
    contested_df.to_csv(OUTPUT_DIR / "contested_requests.csv", index=False)

    # Integrity check: replayed (scenario_id, step) set must equal the
    # existing frozen events CSV exactly.
    existing = pd.read_csv(EXISTING_EVENTS_CSV)[["canonical_scenario_id", "step"]]
    existing_keys = set(zip(existing.canonical_scenario_id, existing.step))
    replayed_keys = set(zip(events_df.canonical_scenario_id, events_df.step)) if len(events_df) else set()
    integrity = {
        "existing_event_count": len(existing_keys),
        "replayed_event_count": len(replayed_keys),
        "keys_match_exactly": existing_keys == replayed_keys,
        "missing_from_replay": len(existing_keys - replayed_keys),
        "extra_in_replay": len(replayed_keys - existing_keys),
    }
    (OUTPUT_DIR / "integrity_check.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    print(f"wall_clock_s={time.perf_counter() - t0:.1f}")
    print(f"Results written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

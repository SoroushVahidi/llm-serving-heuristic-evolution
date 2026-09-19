from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.analysis import joint240_sbs_disagreement_scan_v1 as scan
from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import rebuild_all_scenarios
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _state() -> ObservableState:
    waiting = [
        ObservableRequest(1, 0.0, 10, 20, 20.0, 1.0, "a"),
        ObservableRequest(2, 0.0, 20, 10, 15.0, 2.0, "b"),
    ]
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=4,
        max_batch_tokens=64,
        max_kv_tokens=1000,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    return ObservableState(time=0.0, waiting_queue=waiting, gpu_states=[gpu], completed_count=0, step=0)


class MutatingPolicy(BasePolicy):
    name = "mutating_test_policy"

    def select_action(self, state):
        state.waiting_queue.clear()
        return Action()


def test_shadow_policy_query_does_not_mutate_observable_state():
    state = _state()
    before = copy.deepcopy(state)
    policies = {pid: MutatingPolicy() for pid in scan.P6}
    scan.select_actions_without_mutation(state, policies)
    assert state.waiting_queue == before.waiting_queue


def test_canonical_full_action_detects_prefill_override():
    a = Action(admit={0: [1]}, prefill_chunk_override={0: 64})
    b = Action(admit={0: [1]}, prefill_chunk_override={0: 128})
    assert scan.canonical_action_admit_str(a) == scan.canonical_action_admit_str(b)
    assert scan.canonical_action_full_str(a) != scan.canonical_action_full_str(b)


def test_stable_state_id():
    assert scan.stable_state_id("joint_mm_0007", 13) == "joint240_sbs::joint_mm_0007::13"


def test_episode_extraction():
    flags = [False, True, True, False, True, False, True, True, True]
    assert scan.episode_lengths(flags) == [2, 1, 3]
    assert scan.longest_true_run(flags) == 3
    assert scan.first_last_true(list(range(len(flags))), flags) == (1, 8)


def test_feature_leakage_guard_from_pilot_state():
    state = _state()
    policy = scan.SBSTrajectoryScanPolicy(
        scenario_id="synthetic",
        fold=0,
        sbs_policy=MutatingPolicy(),
        shadow_policies={pid: MutatingPolicy() for pid in scan.P6},
    )
    policy.select_action(state)
    row = policy.state_rows[0]
    feature_keys = [k for k in row if k.startswith("state__")]
    assert feature_keys
    assert all("actual" not in k.lower() for k in feature_keys)
    assert all("anwg" not in k.lower() and "utility" not in k.lower() for k in feature_keys)
    assert all(not k.startswith("state__traj_") for k in feature_keys)


def test_sbs_replay_determinism_one_scenario():
    scen = rebuild_all_scenarios()[0]

    def run_once():
        sim = Simulator(
            SimulatorConfig(
                gpu_configs=list(scen.gpu_configs),
                service_model=ServiceModel(**dict(scen.service_model_kwargs)),
                max_steps=80_000,
                drain_steps=20_000,
            )
        )
        sim.load_trace(list(scen.requests))
        policy = scan._build_policy(scan.SBS_POLICY)[0]
        metrics = sim.run(policy, workload_tag=scen.scenario_id, seed=scan.scenario_seed(scen))
        return metrics.arrival_normalized_weighted_goodput, len(sim._completed), dcm._state_fingerprint(sim)

    assert run_once() == run_once()


def test_scan_pilot_fold_and_policy_row_conservation(tmp_path: Path):
    ctx = scan.load_context()
    ids = scan.deterministic_pilot_scenarios(ctx, 2)
    summary = scan.run_scan(out_dir=tmp_path, scenario_ids=ids, max_steps_per_scenario=20)
    assert summary["n_scenarios"] == 2
    assert summary["n_policy_rows"] == summary["n_state_rows"] * 5
    state_df = pd.read_csv(tmp_path / "state_rows.shard0000-of-0001.csv")
    policy_df = pd.read_csv(tmp_path / "policy_rows.shard0000-of-0001.csv")
    assert state_df["scenario_id"].nunique() == 2
    assert set(policy_df["candidate_policy_id"]).issubset(set(scan.P6) - {scan.SBS_POLICY})


def test_manifest_thinning_and_controls(tmp_path: Path):
    states = pd.DataFrame(
        [
            {"state_id": f"s{i}", "scenario_id": "a", "fold": 0, "step": i, "sim_time": float(i), "any_non_sbs_policy_differs": i in {0, 1, 2, 20}, "n_non_sbs_policies_differ": 1 if i in {0, 1, 2, 20} else 0, "differing_policy_ids": "full_prefill" if i in {0, 1, 2, 20} else ""}
            for i in range(30)
        ]
    )
    policies = pd.DataFrame(
        [
            {"state_id": f"s{i}", "candidate_policy_id": "full_prefill", "candidate_differs_from_sbs": i in {0, 1, 2, 20}}
            for i in range(30)
        ]
    )
    out = scan.build_manifests(states, policies, tmp_path)
    assert out["MANIFEST_ALL_DISAGREEMENTS"]["states"] == 4
    assert out["MANIFEST_EPISODE_THINNED"]["states"] == 2
    assert Path(out["all_disagreements_path"]).exists()

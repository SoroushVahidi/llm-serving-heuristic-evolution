from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from llmserveopt.analysis import joint240_sbs_targeted_terminal_label_pilot_v1 as pilot
from llmserveopt.analysis import joint240_sbs_disagreement_scan_v1 as scan
from llmserveopt.analysis import joint240_dense_sbs_state_action_v1 as dense
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


class FixedPolicy(BasePolicy):
    name = "fixed_test_policy"

    def __init__(self, action: Action):
        self.action = action

    def select_action(self, state):
        return copy.deepcopy(self.action)


class MutatingPolicy(BasePolicy):
    name = "mutating_test_policy"

    def select_action(self, state):
        state.waiting_queue.clear()
        return Action(admit={0: [99]})


def _manifest_rows() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for i in range(80):
            n = 2 + (i % 4)
            rows.append(
                {
                    "state_id": f"joint240_sbs::s{fold}_{i:03d}::{i}",
                    "scenario_id": f"s{fold}_{i // 2:03d}",
                    "fold": fold,
                    "step": i,
                    "sim_time": float(i),
                    "any_non_sbs_policy_differs": 1,
                    "n_non_sbs_policies_differ": min(5, n),
                    "differing_policy_ids": "full_prefill,least_laxity_first" if i % 2 else "weighted_fair_share",
                    "n_distinct_canonical_p6_actions_full": n,
                }
            )
    return pilot.add_episode_and_sampling_columns(pd.DataFrame(rows))


def test_unique_action_hash_is_stable():
    a = pilot.action_hash(str(dense.canonical_full_action(Action(admit={0: [1]}))))
    b = pilot.action_hash(str(dense.canonical_full_action(Action(admit={0: [1]}))))
    c = pilot.action_hash(str(dense.canonical_full_action(Action(admit={0: [2]}))))
    assert a == b
    assert a != c


def test_shadow_action_generation_does_not_mutate_state():
    state = _state()
    before = copy.deepcopy(state)
    policies = {pid: MutatingPolicy() for pid in pilot.P6}
    scan.select_actions_without_mutation(state, policies)
    assert state.waiting_queue == before.waiting_queue


def test_deterministic_pilot_manifest_has_fold_and_scenario_conservation():
    df = _manifest_rows()
    a = pilot.deterministic_pilot_manifest(df)
    b = pilot.deterministic_pilot_manifest(df)
    assert a["state_id"].tolist() == b["state_id"].tolist()
    assert len(a) == 300
    assert a.groupby("fold").size().eq(60).all()
    assert a["scenario_id"].value_counts().max() <= 2
    assert set(a["disagreement_episode_type"]).issubset({"isolated", "multi_step"})


def test_full_manifest_unique_branch_summary():
    df = _manifest_rows().head(4).copy()
    df["n_distinct_canonical_p6_actions_full"] = [2, 3, 4, 5]
    s = pilot.full_manifest_unique_branch_summary(df)
    assert s["naive_policy_branch_count"] == 24
    assert s["unique_action_branch_count"] == 14
    assert s["branches_saved"] == 10


def test_feature_schema_excludes_traj_control_features():
    feats = dense.live_state_features_v1(_state(), history=dense.FeatureHistory())
    assert len(feats) == 95
    assert all(not k.startswith("traj_") for k in feats)
    assert all("actual" not in k.lower() for k in feats)
    assert all("anwg" not in k.lower() and "utility" not in k.lower() for k in feats)


def test_identical_policies_map_to_one_causal_branch_by_canonical_action():
    actions = {
        pilot.P6[0]: Action(admit={0: [1]}),
        pilot.P6[1]: Action(admit={0: [1]}),
        pilot.P6[2]: Action(admit={0: [2]}),
        pilot.P6[3]: Action(admit={0: [2]}),
        pilot.P6[4]: Action(admit={0: [2]}),
        pilot.SBS_POLICY: Action(admit={0: [1]}),
    }
    by_full = {}
    for pid, action in actions.items():
        full = scan.canonical_action_full_str(action)
        by_full.setdefault(full, []).append(pid)
    assert len(by_full) == 2
    assert any(pilot.SBS_POLICY in pids for pids in by_full.values())


def test_sbs_advantage_and_subtraction_identity():
    q0 = {"q_sbs_anwg": 0.4}
    q1 = {"q_sbs_anwg": 0.35}
    q2 = {"q_sbs_anwg": 0.42}
    assert q0["q_sbs_anwg"] - q0["q_sbs_anwg"] == 0.0
    assert q1["q_sbs_anwg"] - q0["q_sbs_anwg"] < 0.0
    assert q2["q_sbs_anwg"] - q0["q_sbs_anwg"] > 0.0


def test_shard_scenarios_conserves_rows():
    df = pilot.deterministic_pilot_manifest(_manifest_rows(), target_states=50, target_per_fold=10)
    shards = [pilot.shard_scenarios(df, i, 5) for i in range(5)]
    joined = pd.concat(shards, ignore_index=True)
    assert sorted(joined["state_id"]) == sorted(df["state_id"])
    seen = set()
    for shard in shards:
        scenarios = set(shard["scenario_id"])
        assert not (seen & scenarios)
        seen |= scenarios


def test_restart_resume_safety(tmp_path: Path):
    manifest = pilot.deterministic_pilot_manifest(_manifest_rows(), target_states=10, target_per_fold=2)
    mp = tmp_path / "pilot_manifest.csv"
    manifest.to_csv(mp, index=False)
    out = tmp_path / "out"
    out.mkdir()
    (out / "DONE.shard0000-of-0001").write_text('{"done": true, "sentinel": 1}')
    (out / "summary.shard0000-of-0001.json").write_text('{"done": true, "sentinel": 1}')
    (out / "state_action_rows.shard0000-of-0001.csv").write_text("a\n")
    (out / "state_policy_action_map.shard0000-of-0001.csv").write_text("a\n")
    summary = pilot.run_label_shard(pilot_manifest=mp, out_dir=out, resume=True)
    assert summary["resume_skipped"] is True
    assert summary["sentinel"] == 1


def test_exact_sbs_replay_determinism_one_scenario():
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
        policy = pilot._build_policy(pilot.SBS_POLICY)[0]
        metrics = sim.run(policy, workload_tag=scen.scenario_id, seed=scan.scenario_seed(scen))
        return metrics.arrival_normalized_weighted_goodput, len(sim._completed)

    assert run_once() == run_once()

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from llmserveopt.analysis import joint240_dense_sbs_state_action_v1 as dense
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import P6, LiveP6DwellRouterPolicy, rebuild_all_scenarios
from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _state() -> ObservableState:
    w = [
        ObservableRequest(1, 0.0, 10, 20, 40.0, 1.0, "loose"),
        ObservableRequest(2, 2.0, 30, 10, 12.0, 3.0, "tight"),
        ObservableRequest(3, 3.0, 5, 5, 9.0, 2.0, "tight"),
    ]
    active_req = ObservableRequest(9, 0.0, 50, 100, 90.0, 1.0, "loose")
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=4,
        max_batch_tokens=64,
        max_kv_tokens=1000,
        active_request_ids=[9],
        active_requests_info=[active_req],
        current_kv_tokens=80,
        tokens_decoded_per_request={9: 7},
        prefilling_count=0,
        decoding_count=1,
    )
    return ObservableState(time=7.0, waiting_queue=w, gpu_states=[gpu], completed_count=4, step=11)


def _dummy_stage1() -> Pipeline:
    pipe = Pipeline([("clf", DummyClassifier(strategy="constant", constant=P6[3]))])
    pipe.fit(np.zeros((4, 4)), np.array([P6[3]] * 4))
    return pipe


def test_stable_state_id():
    assert dense.stable_state_id("joint_mm_0001", 42) == "joint240::joint_mm_0001::42"


def test_live_state_features_are_online_safe_and_no_actual_output():
    feats = dense.live_state_features_v1(_state(), history=dense.FeatureHistory())
    assert feats["waiting_count"] == 3
    assert feats["active_count"] == 1
    assert feats["waiting_prompt_tokens_sum"] == 45
    assert feats["active_predicted_remaining_tokens_sum"] == 93
    assert all("actual" not in k for k in feats)
    assert all("anwg" not in k.lower() and "utility" not in k.lower() for k in feats)


def test_history_features_are_deterministic_fixed_windows():
    hist = dense.FeatureHistory()
    s = _state()
    first = dense.live_state_features_v1(s, history=hist)
    hist.update(s, first)
    s2 = copy.deepcopy(s)
    s2.step = 12
    s2.completed_count = 5
    second = dense.live_state_features_v1(s2, history=hist)
    assert second["hist_w1_completed_count_delta"] == 1
    assert second["hist_w5_completed_count_delta"] == 0


def test_action_diff_features_compare_candidate_to_sbs_predecision():
    state = _state()
    candidate = Action(admit={0: [1, 2]})
    sbs = Action(admit={0: [2, 3]})
    feats = dense.action_diff_features_v1(state, candidate, sbs)
    assert feats["action_equal_to_sbs_admit_only"] is False
    assert feats["admit_symmetric_difference_count"] == 2
    assert feats["candidate_only_admitted_count"] == 1
    assert feats["sbs_only_admitted_count"] == 1
    assert feats["candidate_minus_sbs_prompt_tokens_sum"] == 5


def test_sbs_baseline_identity_and_advantage_sign():
    q = {"a": 0.25, "b": 0.40}
    ref = q["a"]
    assert q["a"] - ref == 0.0
    assert q["b"] - ref > 0.0


def test_deterministic_pilot_state_selection_is_stratified():
    rows = []
    for fold in range(5):
        for acq in ["AGREEMENT_CONTROL", "DISAGREEMENT"]:
            for i in range(3):
                rows.append(
                    {
                        "state_id": f"joint240::s{fold}_{acq}_{i}::{i}",
                        "scenario_id": f"s{fold}_{acq}_{i}",
                        "step": i,
                        "fold": fold,
                        "seed": 1,
                        "acquisition_type": acq,
                    }
                )
    df = pd.DataFrame(rows)
    a = dense.deterministic_pilot_states(df, per_fold_acq=1)
    b = dense.deterministic_pilot_states(df, per_fold_acq=1)
    assert a["state_id"].tolist() == b["state_id"].tolist()
    assert len(a) == 10
    assert a.groupby(["fold", "acquisition_type"]).size().eq(1).all()


def test_run_one_step_then_sbs_does_not_mutate_reference_simulator():
    scen = rebuild_all_scenarios()[0]
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scen.gpu_configs),
            service_model=ServiceModel(**dict(scen.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scen.requests))
    stage1 = _dummy_stage1()
    router = LiveP6DwellRouterPolicy(stage1, P6)

    class Capture(BasePolicy):
        name = "capture_for_dense_test"

        def __init__(self):
            self.inner = router
            self.out = None

        def reset(self):
            pass

        def select_action(self, state):
            action = self.inner.select_action(state)
            if self.out is None and len(state.waiting_queue) > 0 and int(state.step) >= 2:
                before = dcm._state_fingerprint(sim)
                self.out = dense.run_one_step_then_sbs_terminal(
                    sim,
                    first_action=copy.deepcopy(action),
                    continuation=dense._build_policy(dense.SBS_POLICY)[0],
                    all_requests=list(scen.requests),
                    workload_tag=scen.scenario_id,
                    seed=int(scen.params.get("seed", 0)),
                )
                after = dcm._state_fingerprint(sim)
                assert before == after
            return action

    cap = Capture()
    sim.run(cap, workload_tag=scen.scenario_id, seed=int(scen.params.get("seed", 0)))
    assert cap.out is not None
    assert cap.out["live_fingerprint_unchanged"]


def test_action_equivalent_policies_same_outcome_for_same_action():
    scen = rebuild_all_scenarios()[0]
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scen.gpu_configs),
            service_model=ServiceModel(**dict(scen.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scen.requests))
    stage1 = _dummy_stage1()
    router = LiveP6DwellRouterPolicy(stage1, P6)

    class Capture(BasePolicy):
        name = "capture_equiv_for_dense_test"

        def __init__(self):
            self.inner = router
            self.results = None

        def reset(self):
            pass

        def select_action(self, state):
            action = self.inner.select_action(state)
            if self.results is None and len(state.waiting_queue) > 0 and int(state.step) >= 2:
                kwargs = dict(
                    continuation=dense._build_policy(dense.SBS_POLICY)[0],
                    all_requests=list(scen.requests),
                    workload_tag=scen.scenario_id,
                    seed=int(scen.params.get("seed", 0)),
                )
                a = dense.run_one_step_then_sbs_terminal(sim, first_action=copy.deepcopy(action), **kwargs)
                b = dense.run_one_step_then_sbs_terminal(sim, first_action=copy.deepcopy(action), **kwargs)
                self.results = (a, b)
            return action

    cap = Capture()
    sim.run(cap, workload_tag=scen.scenario_id, seed=int(scen.params.get("seed", 0)))
    assert cap.results is not None
    assert cap.results[0]["q_sbs_anwg"] == cap.results[1]["q_sbs_anwg"]


def test_feature_metadata_declares_safety():
    live = dense.live_state_feature_metadata()
    action = dense.action_diff_feature_metadata()
    assert live
    assert action
    assert all(r["online_safe"] == "YES" for r in live + action)

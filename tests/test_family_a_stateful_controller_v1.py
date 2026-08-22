from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    ESTF_MODE,
    WFS_MODE,
    STATEFUL_CONTROLLER_FEATURES,
    FamilyAStatefulControllerV1,
    actions_disagree,
    canonical_action,
    extract_family_a_stateful_features,
    snapshot_gpu_counters,
    validate_feature_names,
)
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy


class ConstantModeModel:
    def __init__(self, p_estf: float) -> None:
        self.p_estf = p_estf

    def predict_estf_probability(self, features):
        return self.p_estf


class SequenceModeModel:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.idx = 0

    def predict_estf_probability(self, features):
        value = self.probabilities[min(self.idx, len(self.probabilities) - 1)]
        self.idx += 1
        return value


def req(request_id: int, class_id: str, prompt: int = 10, output: int = 10) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=0.0,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=10_000.0,
        priority=1.0,
        class_id=class_id,
    )


def disagreement_state(step: int = 100) -> ObservableState:
    active = [req(90, "tight"), req(91, "tight")]
    waiting = [req(1, "tight"), req(2, "loose")]
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=3,
        max_batch_tokens=3,
        max_kv_tokens=10_000,
        active_request_ids=[90, 91],
        active_requests_info=active,
        current_kv_tokens=20,
        tokens_decoded_per_request={90: 1, 91: 1},
    )
    return ObservableState(
        time=step * 0.001,
        waiting_queue=waiting,
        gpu_states=[gpu],
        completed_count=0,
        step=step,
    )


def single_choice_state(step: int = 100) -> ObservableState:
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=2,
        max_batch_tokens=2,
        max_kv_tokens=10_000,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    return ObservableState(
        time=step * 0.001,
        waiting_queue=[req(1, "tight")],
        gpu_states=[gpu],
        completed_count=0,
        step=step,
    )


def test_estf_and_wfs_disagree_in_fixture() -> None:
    state = disagreement_state()
    snapshot = snapshot_gpu_counters(state)
    estf_action = EstimatedServiceTimeFirstPolicy().select_action(state)
    for gpu_id, active_ids, current_kv in snapshot:
        gpu = state.gpu_states[gpu_id]
        gpu.active_request_ids = list(active_ids)
        gpu.current_kv_tokens = current_kv
    wfs_action = WeightedFairSharePolicy().select_action(state)
    assert actions_disagree(estf_action, wfs_action)


def test_initial_mode_and_abstention_fallback_outside_candidate() -> None:
    policy = FamilyAStatefulControllerV1(ConstantModeModel(0.99), min_dwell_steps=0)
    state = single_choice_state()
    action = policy.select_action(state)
    assert policy.mode == WFS_MODE
    assert canonical_action(action) == canonical_action(WeightedFairSharePolicy().select_action(single_choice_state()))
    assert policy.diagnostics()["abstention_count"] == 1


def test_minimum_dwell_prevents_early_switch() -> None:
    policy = FamilyAStatefulControllerV1(ConstantModeModel(0.99), min_dwell_steps=3)
    for step in range(3):
        policy.select_action(disagreement_state(step=step + 10))
        assert policy.mode == WFS_MODE
    policy.select_action(disagreement_state(step=20))
    assert policy.mode == ESTF_MODE


def test_hysteresis_keeps_ambiguous_score_in_current_mode() -> None:
    policy = FamilyAStatefulControllerV1(ConstantModeModel(0.50), min_dwell_steps=0)
    policy.select_action(disagreement_state())
    assert policy.mode == WFS_MODE
    assert policy.diagnostics()["switch_count"] == 0


def test_switches_wfs_to_estf_and_estf_to_wfs() -> None:
    policy = FamilyAStatefulControllerV1(
        SequenceModeModel([0.90, 0.10]),
        min_dwell_steps=0,
    )
    policy.select_action(disagreement_state(step=100))
    assert policy.mode == ESTF_MODE
    policy.select_action(disagreement_state(step=101))
    assert policy.mode == WFS_MODE
    assert policy.diagnostics()["switch_directions"] == {
        "WFS_MODE->ESTF_MODE": 1,
        "ESTF_MODE->WFS_MODE": 1,
    }


def test_dwell_and_hysteresis_prevent_thrashing() -> None:
    policy = FamilyAStatefulControllerV1(
        SequenceModeModel([0.90, 0.10, 0.90, 0.10, 0.90]),
        min_dwell_steps=2,
    )
    modes = []
    for step in range(5):
        policy.select_action(disagreement_state(step=100 + step))
        modes.append(policy.mode)
    assert modes == [WFS_MODE, WFS_MODE, ESTF_MODE, ESTF_MODE, ESTF_MODE]
    assert policy.diagnostics()["switch_count"] == 1


def test_parent_action_equivalence_when_fixed_in_estf_mode() -> None:
    state_parent = disagreement_state()
    state_controller = copy.deepcopy(state_parent)
    parent_action = EstimatedServiceTimeFirstPolicy().select_action(state_parent)
    policy = FamilyAStatefulControllerV1(
        ConstantModeModel(0.99),
        min_dwell_steps=0,
        initial_mode=ESTF_MODE,
        estf_enter_threshold=2.0,
        wfs_enter_threshold=-1.0,
    )
    controller_action = policy.select_action(state_controller)
    assert canonical_action(controller_action) == canonical_action(parent_action)
    assert state_controller.gpu_states[0].active_request_ids == state_parent.gpu_states[0].active_request_ids
    assert state_controller.gpu_states[0].current_kv_tokens == state_parent.gpu_states[0].current_kv_tokens


def test_parent_probe_does_not_leave_extra_state_mutation() -> None:
    state_parent = disagreement_state()
    state_controller = copy.deepcopy(state_parent)
    parent_action = WeightedFairSharePolicy().select_action(state_parent)
    policy = FamilyAStatefulControllerV1(ConstantModeModel(0.50), min_dwell_steps=0)
    controller_action = policy.select_action(state_controller)
    assert canonical_action(controller_action) == canonical_action(parent_action)
    assert state_controller.gpu_states[0].active_request_ids == state_parent.gpu_states[0].active_request_ids
    assert state_controller.gpu_states[0].current_kv_tokens == state_parent.gpu_states[0].current_kv_tokens


def test_reset_restores_wfs_initial_mode_and_diagnostics() -> None:
    policy = FamilyAStatefulControllerV1(ConstantModeModel(0.99), min_dwell_steps=0)
    policy.select_action(disagreement_state())
    assert policy.mode == ESTF_MODE
    policy.reset()
    assert policy.mode == WFS_MODE
    assert policy.diagnostics()["total_decisions"] == 0


def test_feature_vector_is_causal_and_complete() -> None:
    features = extract_family_a_stateful_features(disagreement_state())
    assert tuple(features.keys()) == STATEFUL_CONTROLLER_FEATURES
    assert all(np.isfinite(v) for v in features.values())
    with pytest.raises(ValueError):
        validate_feature_names((*STATEFUL_CONTROLLER_FEATURES, "scenario_id"))


def test_grouped_split_integrity_by_scenario() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "y": [0, 1, 1, 1, 0, 0, 1, 0],
            "x": range(8),
        }
    )
    groups = df["scenario"].to_numpy()
    splitter = GroupKFold(n_splits=4)
    for train_idx, val_idx in splitter.split(df[["x"]], df["y"], groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))

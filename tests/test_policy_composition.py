from __future__ import annotations

import inspect
import math

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.composition import (
    ComponentWiseCompositionPolicy,
    CompositionError,
    ContextualRankEnsemblePolicy,
    ContextualWeightRule,
    RankExpertSpec,
    StaticRankEnsemblePolicy,
    default_module_specs,
    rank_with_named_expert,
)
from llmserveopt.selector.composition_experiment import (
    CompositionExperimentError,
    assert_no_split_group_leakage,
    select_best_fixed_policy_from_development,
    validate_treatment_selection_does_not_use_heldout,
)


def req(
    request_id: int,
    *,
    prompt: int = 64,
    output: int = 64,
    arrival: float = 0.0,
    deadline: float = 20.0,
    priority: float = 1.0,
    class_id: str = "medium",
) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id=class_id,
    )


def gpu(
    *,
    max_seq: int = 4,
    max_kv: int = 4096,
    max_batch: int = 128,
    active: list[int] | None = None,
    kv_used: int = 0,
    prefilling: int = 0,
    decoding: int = 0,
) -> ObservableGPUState:
    return ObservableGPUState(
        gpu_id=0,
        max_active_sequences=max_seq,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=list(active or []),
        active_requests_info=[],
        current_kv_tokens=kv_used,
        tokens_decoded_per_request={},
        prefilling_count=prefilling,
        decoding_count=decoding,
    )


def state(reqs: list[ObservableRequest], *, now: float = 5.0, step: int = 1, g: ObservableGPUState | None = None) -> ObservableState:
    return ObservableState(time=now, waiting_queue=reqs, gpu_states=[g or gpu()], completed_count=0, step=step)


def test_typed_module_interface_marks_unsupported_components():
    specs = default_module_specs()
    by_name = {spec.name: spec for spec in specs}
    assert by_name["AdmissionRule"].supported
    assert by_name["PriorityRule"].supported
    assert by_name["KVCacheRule"].supported
    assert not by_name["CacheReuseRule"].supported


def test_rank_normalization_orders_wsp_short_work_first():
    s = state([
        req(1, prompt=512, output=512),
        req(2, prompt=32, output=32),
        req(3, prompt=128, output=128),
    ])
    out = rank_with_named_expert("weighted_shortest_processing", s)
    assert out.ranked_request_ids == [2, 3, 1]
    assert out.normalized_ranks[2] == pytest.approx(1.0)
    assert out.normalized_ranks[3] == pytest.approx(0.5)
    assert out.normalized_ranks[1] == pytest.approx(0.0)


def test_deterministic_ties_use_arrival_then_request_id():
    s = state([
        req(2, prompt=64, output=64, arrival=0.0),
        req(1, prompt=64, output=64, arrival=0.0),
    ])
    out = rank_with_named_expert("weighted_shortest_processing", s)
    assert out.ranked_request_ids == [1, 2]


def test_missing_expert_outputs_are_reported_not_fabricated():
    s = state([req(1), req(2)])
    out = rank_with_named_expert("not_a_supported_priority_expert", s)
    assert out.normalized_ranks == {}
    assert out.missing_request_ids == [1, 2]


def test_negative_and_nan_weights_are_invalid():
    s = state([req(1)])
    with pytest.raises(CompositionError):
        StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", -1.0)]).select_action(s)
    with pytest.raises(CompositionError):
        StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", math.nan)]).select_action(state([req(1)]))


def test_all_zero_weights_use_fallback_policy():
    policy = StaticRankEnsemblePolicy([
        RankExpertSpec("weighted_shortest_processing", 0.0),
        RankExpertSpec("edf", 0.0),
    ])
    action = policy.select_action(state([req(1, prompt=32), req(2, prompt=64)], g=gpu(max_seq=1)))
    assert action.admit[0] == [1]
    assert policy.decision_logs[-1].fallback_used


def test_top_k_composition_uses_sparse_expert_support():
    policy = StaticRankEnsemblePolicy([
        RankExpertSpec("weighted_shortest_processing", 1.0),
        RankExpertSpec("edf", 0.9),
        RankExpertSpec("shortest_prompt_first", 0.8),
    ], top_k=1)
    policy.select_action(state([req(1), req(2)], g=gpu(max_seq=1)))
    assert list(policy.decision_logs[-1].expert_weights) == ["weighted_shortest_processing"]


def test_fallback_behavior_for_unsupported_only_ensemble():
    policy = StaticRankEnsemblePolicy([RankExpertSpec("unsupported", 1.0)])
    action = policy.select_action(state([req(1, prompt=32), req(2, prompt=64)], g=gpu(max_seq=1)))
    assert action.admit[0] == [1]
    assert policy.decision_logs[-1].fallback_used


def test_feasibility_projection_preserves_gpu_constraints():
    policy = StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)])
    too_large = req(1, prompt=5000, output=1)
    feasible = req(2, prompt=32, output=32)
    action = policy.select_action(state([too_large, feasible], g=gpu(max_seq=1, max_kv=1024)))
    assert action.admit[0] == [2]


def test_contextual_weights_are_causal_deterministic_and_logged():
    policy = ContextualRankEnsemblePolicy(
        [RankExpertSpec("weighted_shortest_processing", 1.0), RankExpertSpec("scorpio_style_slo_guard", 1.0)],
        contextual_rules=[
            ContextualWeightRule("scorpio_style_slo_guard", coefficients={"queue_pressure": 2.0}),
        ],
        min_commitment_steps=2,
    )
    s1 = state([req(i) for i in range(4)], step=10, g=gpu(max_seq=1))
    s2 = state([req(i) for i in range(4)], step=11, g=gpu(max_seq=1))
    policy.select_action(s1)
    weights_1 = dict(policy.decision_logs[-1].expert_weights)
    policy.select_action(s2)
    assert policy.decision_logs[-1].expert_weights == weights_1
    assert policy.decision_logs[-1].weight_entropy > 0.0


def test_component_wise_composition_keeps_scorpio_admission_before_wsp_priority():
    policy = ComponentWiseCompositionPolicy()
    impossible = req(1, deadline=4.0, prompt=512, output=512)
    valid_short = req(2, deadline=20.0, prompt=32, output=32)
    action = policy.select_action(state([impossible, valid_short], now=5.0, g=gpu(max_seq=1)))
    assert action.admit[0] == [2]
    assert not policy.decision_logs[-1].fallback_used


def test_reproducibility_with_fixed_inputs():
    s1 = state([req(1, prompt=256), req(2, prompt=32)], g=gpu(max_seq=1))
    s2 = state([req(1, prompt=256), req(2, prompt=32)], g=gpu(max_seq=1))
    p1 = StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)])
    p2 = StaticRankEnsemblePolicy([RankExpertSpec("weighted_shortest_processing", 1.0)])
    assert p1.select_action(s1).admit == p2.select_action(s2).admit
    assert p1.decision_logs[-1].expert_weights == p2.decision_logs[-1].expert_weights


def test_composition_code_does_not_reference_hidden_request_fields():
    import llmserveopt.policies.composition as composition

    source = inspect.getsource(composition)
    assert "actual_output_tokens" not in source


def test_split_group_leakage_detection():
    clean = [
        {"split": "TRAIN", "split_group_key": "g1"},
        {"split": "VALIDATION", "split_group_key": "g2"},
    ]
    assert_no_split_group_leakage(clean)
    leaked = clean + [{"split": "ID_TEST", "split_group_key": "g1"}]
    with pytest.raises(CompositionExperimentError):
        assert_no_split_group_leakage(leaked)


def test_treatment_selection_forbids_heldout_splits():
    validate_treatment_selection_does_not_use_heldout(["TRAIN", "VALIDATION"])
    with pytest.raises(CompositionExperimentError):
        validate_treatment_selection_does_not_use_heldout(["TRAIN", "FINAL_OOD"])


def test_best_fixed_policy_selected_from_development_only():
    rows = [
        {"split": "TRAIN", "policy_name": "wsp", "metric_arrival_normalized_weighted_goodput": "0.8", "split_group_key": "g1"},
        {"split": "VALIDATION", "policy_name": "wsp", "metric_arrival_normalized_weighted_goodput": "0.7", "split_group_key": "g2"},
        {"split": "TRAIN", "policy_name": "scorpio", "metric_arrival_normalized_weighted_goodput": "0.6", "split_group_key": "g1"},
        {"split": "ID_TEST", "policy_name": "scorpio", "metric_arrival_normalized_weighted_goodput": "1.0", "split_group_key": "g3"},
    ]
    best, means = select_best_fixed_policy_from_development(rows, development_splits=("TRAIN", "VALIDATION"))
    assert best == "wsp"
    assert means["wsp"] == pytest.approx(0.75)

from __future__ import annotations

import inspect

import pytest

from llmserveopt.core.action import Action
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr

from scripts import industry_realism_action_opportunity_phase_a_v1 as phase_a


def test_faithful_window_filter_exact_universe():
    records = phase_a.faithful_records()
    assert len(records) == 60
    counts = {}
    for rec in records:
        assert rec["scenario_evidence_class"] == ptr.FAITHFUL
        assert rec["source_dataset"] in phase_a.WORKLOADS
        counts[rec["source_dataset"]] = counts.get(rec["source_dataset"], 0) + 1
    assert counts == {w: 20 for w in phase_a.WORKLOADS}


def test_augmented_window_rejection(monkeypatch):
    base = phase_a.faithful_records()[0].copy()
    base["scenario_evidence_class"] = ptr.AUGMENTED
    monkeypatch.setattr(ptr, "build_all_scenarios", lambda: [base])
    with pytest.raises(ValueError, match="augmented|unexpected"):
        phase_a.faithful_records()


def test_expected_workload_set_is_frozen():
    assert phase_a.WORKLOADS == ("azure_2023_code", "azure_2023_conv", "burstgpt")
    assert phase_a.P6_POLICIES == (
        "full_prefill",
        "chunked_prefill_small",
        "estimated_service_time_first",
        "weighted_fair_share",
        "least_laxity_first",
        "kv_constrained_online",
    )


def test_canonical_action_equality_is_order_stable():
    a = Action(admit={1: [3, 2], 0: []}, preempt={2: []})
    b = Action(admit={1: [2, 3]}, preempt={})
    assert phase_a.canonical_action(a) == phase_a.canonical_action(b)
    c = Action(admit={1: [2]})
    assert phase_a.canonical_action(a) != phase_a.canonical_action(c)


def test_disagreement_detection_records_zero_and_positive_cases():
    sbs = phase_a.canonical_action(Action(admit={0: [1]}))
    alt_same = phase_a.canonical_action(Action(admit={0: [1]}))
    alt_diff = phase_a.canonical_action(Action(admit={0: [2]}))
    assert alt_same == sbs
    assert alt_diff != sbs


def test_zero_disagreement_aggregation_keeps_denominator():
    rows = [
        {
            "source_dataset": "azure_2023_code",
            "requests": 200,
            "sbs_decision_states": 10,
            "disagreement_states": 0,
            "windows_with_disagreement": 0,
            "no_policy_choice_states": 10,
            "canonical_action_collapse_states": 0,
            "policy_ranking_proxy_difference_states": 0,
            "unique_alternative_canonical_action_hashes": 0,
            "max_active_sequences": 1,
            "mean_active_sequences": 0.5,
            "max_queue_length": 0,
            "mean_queue_length": 0.0,
            "max_kv_utilization": 0.1,
            "mean_kv_utilization": 0.05,
            "active_sequence_capacity_binding_states": 0,
            "active_sequence_capacity_near_binding_states": 0,
            "kv_capacity_binding_or_over_requested_states": 0,
            "kv_capacity_near_binding_states": 0,
            "token_budget_binding_proxy_states": 0,
        }
    ]
    out = phase_a.aggregate_workloads(rows)
    assert out[0]["sbs_decision_states"] == 10
    assert out[0]["disagreement_states"] == 0
    assert out[0]["disagreement_rate"] == 0.0


def test_per_window_aggregation_positive_case():
    rows = [
        {
            "source_dataset": "burstgpt",
            "requests": 200,
            "sbs_decision_states": 10,
            "disagreement_states": 2,
            "windows_with_disagreement": 1,
            "no_policy_choice_states": 8,
            "canonical_action_collapse_states": 3,
            "policy_ranking_proxy_difference_states": 5,
            "unique_alternative_canonical_action_hashes": 2,
            "max_active_sequences": 2,
            "mean_active_sequences": 1.0,
            "max_queue_length": 4,
            "mean_queue_length": 0.5,
            "max_kv_utilization": 0.2,
            "mean_kv_utilization": 0.1,
            "active_sequence_capacity_binding_states": 0,
            "active_sequence_capacity_near_binding_states": 0,
            "kv_capacity_binding_or_over_requested_states": 1,
            "kv_capacity_near_binding_states": 0,
            "token_budget_binding_proxy_states": 0,
        }
    ]
    out = phase_a.aggregate_workloads(rows)
    assert out[0]["disagreement_rate"] == pytest.approx(0.2)
    assert out[0]["windows_with_disagreement"] == 1


def test_pressure_telemetry_has_expected_fields():
    rec = phase_a.faithful_records()[0]
    state = rec["scenario"].requests  # smoke the fixture exists before unit-level helper checks
    assert len(state) == 200
    src = inspect.getsource(phase_a.state_pressure_features)
    for field in (
        "waiting_queue_count",
        "active_sequences",
        "kv_utilization_mean",
        "active_sequence_capacity_binding",
        "kv_capacity_binding_or_over_requested",
        "token_budget_binding_proxy",
    ):
        assert field in src


def test_no_causal_label_or_selector_dependency_in_phase_a_source():
    src = inspect.getsource(phase_a)
    forbidden = [
        "FINAL_CONFIRMATORY_SELECTOR",
        "joblib.load",
        "predict_proba",
        "a_sbs",
        "q_sbs",
        "terminal_label",
        "counterfactual_label",
        "load_labels",
        "realized_gain",
    ]
    lowered = src.lower()
    for token in forbidden:
        assert token.lower() not in lowered


def test_deterministic_manifest_reconstruction():
    a = phase_a.phase_a_records_manifest()
    b = phase_a.phase_a_records_manifest()
    assert a == b
    assert len(a) == 60

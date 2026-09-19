from __future__ import annotations

import inspect

import pytest

from scripts import industry_realism_action_opportunity_phase_a_v1 as phase_a
from scripts import industry_realism_action_opportunity_phase_b_v2 as phase_b


def test_phase_b_uses_same_sixty_faithful_windows():
    phase_a_rows = phase_a.phase_a_records_manifest()
    phase_b_rows = phase_b.faithful_manifest()
    assert phase_b_rows == phase_a_rows
    assert len(phase_b_rows) == 60
    assert {r["scenario_evidence_class"] for r in phase_b_rows} == {phase_a.ptr.FAITHFUL}


def test_historical_phase_b_v1_grid_is_preserved():
    assert phase_b.sha256_file(phase_b.PHASE_B_V1_GRID_PATH) == phase_b.PHASE_B_V1_GRID_SHA256


def test_phase_b_v2_grid_is_exact_and_telemetry_grounded():
    assert phase_b.ARRIVAL_MULTIPLIERS == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert phase_b.ACTIVE_SEQUENCE_CAPS == (512, 64, 32, 16, 8, 4)
    assert phase_b.KV_CAPACITY_TOKENS == (8_000_000, 240_000, 120_000, 60_000, 32_000, 16_000, 8_000)
    telemetry = phase_b.phase_a_telemetry()
    assert telemetry["overall_max_active_sequences_observed"] == 31
    assert telemetry["overall_max_kv_tokens_observed"] == pytest.approx(30416.0)
    assert min(phase_b.KV_CAPACITY_TOKENS) >= 7437


def test_condition_count_and_one_axis_isolation():
    conditions = phase_b.pressure_conditions()
    assert len(conditions) == 18
    assert len(phase_b.condition_records()) == 60 * 18
    for condition in conditions:
        phase_b.validate_one_axis_isolation(condition)


def test_arrival_transform_preserves_request_identity_and_lengths():
    rec = phase_a.faithful_records()[0]
    req = rec["scenario"].requests[5]
    transformed = phase_b.transformed_requests([req], 4.0)[0]
    assert transformed.request_id == req.request_id
    assert transformed.prompt_tokens == req.prompt_tokens
    assert transformed.actual_output_tokens == req.actual_output_tokens
    assert transformed.predicted_output_tokens == req.predicted_output_tokens
    assert transformed.arrival_time == pytest.approx(req.arrival_time / 4.0)
    assert transformed.slo_deadline - transformed.arrival_time == pytest.approx(req.slo_deadline - req.arrival_time)


def test_transformed_scenario_changes_only_declared_axis():
    rec = phase_a.faithful_records()[0]
    for condition in phase_b.pressure_conditions():
        scenario = phase_b.transformed_scenario(rec, condition)
        if condition["axis"] == "arrival_pressure":
            assert scenario.gpu_configs[0].max_active_sequences == 512
            assert scenario.gpu_configs[0].max_kv_tokens == 8_000_000
        elif condition["axis"] == "kv_capacity":
            assert scenario.requests[10].arrival_time == rec["scenario"].requests[10].arrival_time
            assert scenario.gpu_configs[0].max_active_sequences == 512
        elif condition["axis"] == "active_sequence_capacity":
            assert scenario.requests[10].arrival_time == rec["scenario"].requests[10].arrival_time
            assert scenario.gpu_configs[0].max_kv_tokens == 8_000_000


def test_pressure_regime_classification():
    base = {
        "validity_class": "VALID_UNCONSTRAINED",
        "active_sequence_capacity_binding_states": 0,
        "kv_capacity_binding_or_over_requested_states": 0,
        "token_budget_binding_proxy_states": 0,
        "active_sequence_capacity_near_binding_states": 0,
        "kv_capacity_near_binding_states": 0,
        "max_active_pressure": 0.1,
        "max_kv_pressure": 0.1,
    }
    assert phase_b.pressure_validity_from_row(base) == "VALID_UNCONSTRAINED"
    row = {**base, "max_active_pressure": 0.8}
    assert phase_b.pressure_validity_from_row(row) == "VALID_PARTIALLY_CONSTRAINED"
    row = {**base, "kv_capacity_binding_or_over_requested_states": 1}
    assert phase_b.pressure_validity_from_row(row) == "VALID_STRONGLY_CONSTRAINED"
    row = {**base, "validity_class": "INVALID_HORIZON_TRUNCATED"}
    assert phase_b.pressure_validity_from_row(row) == "INVALID_HORIZON_TRUNCATED"


def test_complete_binding_condition_uses_preregistered_validity_class():
    base = {
        "active_sequence_capacity_binding_states": 0,
        "kv_capacity_binding_or_over_requested_states": 2,
        "token_budget_binding_proxy_states": 0,
        "active_sequence_capacity_near_binding_states": 0,
        "kv_capacity_near_binding_states": 0,
        "max_active_pressure": 0.2,
        "max_kv_pressure": 0.4,
    }
    assert phase_b.pressure_validity_from_row(base) == "VALID_STRONGLY_CONSTRAINED"


def test_simulation_invalidity_detects_horizon_and_resource_infeasible():
    class Metrics:
        num_completed = 0
        num_total = 1
        num_dropped = 0

    rec = phase_a.faithful_records()[0]
    scenario = rec["scenario"]
    condition = {
        "max_kv_tokens": 1,
    }
    assert phase_b.simulation_invalidity_class(Metrics(), scenario, condition) == "INVALID_RESOURCE_INFEASIBLE"
    condition = {
        "max_kv_tokens": 8_000_000,
    }
    assert phase_b.simulation_invalidity_class(Metrics(), scenario, condition) == "INVALID_HORIZON_TRUNCATED"


def test_transition_summary_retains_zero_support():
    rows = [
        {
            "source_dataset": "burstgpt",
            "axis": "arrival_pressure",
            "condition_id": "arrival_x1p0",
            "pressure_order": 0,
            "kv_capacity_binding_or_over_requested_states": 0,
            "active_sequence_capacity_binding_states": 0,
            "token_budget_binding_proxy_states": 0,
            "disagreement_states": 0,
            "windows_with_disagreement": 0,
        },
        {
            "source_dataset": "burstgpt",
            "axis": "arrival_pressure",
            "condition_id": "arrival_x2p0",
            "pressure_order": 1,
            "kv_capacity_binding_or_over_requested_states": 0,
            "active_sequence_capacity_binding_states": 1,
            "token_budget_binding_proxy_states": 0,
            "disagreement_states": 3,
            "windows_with_disagreement": 1,
        },
        {
            "source_dataset": "burstgpt",
            "axis": "arrival_pressure",
            "condition_id": "arrival_x4p0",
            "pressure_order": 2,
            "kv_capacity_binding_or_over_requested_states": 0,
            "active_sequence_capacity_binding_states": 1,
            "token_budget_binding_proxy_states": 0,
            "disagreement_states": 5,
            "windows_with_disagreement": 2,
        },
    ]
    out = phase_b.transition_rows(rows)
    assert out == [
        {
            "schema_version": phase_b.SCHEMA_VERSION,
            "source_dataset": "burstgpt",
            "axis": "arrival_pressure",
            "first_binding_point": "arrival_x2p0",
            "first_disagreement_point": "arrival_x2p0",
            "sustained_disagreement_point": "arrival_x4p0",
            "sustained_disagreement_rule": "windows_with_disagreement >= 2",
        }
    ]


def test_source_has_no_label_or_selector_dependency():
    src = inspect.getsource(phase_b).lower()
    forbidden = [
        "final_confirmatory_selector",
        "joblib.load",
        "predict_proba",
        '"a_sbs"',
        '"q_sbs"',
        "terminal_label",
        "realized_gain",
        "fit(",
    ]
    for token in forbidden:
        assert token not in src

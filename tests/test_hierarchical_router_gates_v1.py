"""Focused tests for Hierarchical Regime Router v1 gate evaluator and
mechanical verdict logic (design doc SS R item 15)."""
from __future__ import annotations

import pytest

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import REGIME_A, REGIME_B, REGIME_C
from llmserveopt.policy_separation.hierarchical_router_gates_v1 import (
    VERDICT_GO,
    VERDICT_INCONCLUSIVE,
    VERDICT_NO_GO,
    VERDICT_ROUTING_WORKS_SELECTION_NO_GAIN,
    compute_verdict,
    evaluate_all_gates,
    load_gates_config,
)

ALL_PASS_METRICS = {
    "stage1_input_validity_fraction": 1.0,
    "router_macro_f1": 0.95,
    "catastrophic_misroute_rate": 0.01,
    "stage2_preservation_fraction_by_regime": {REGIME_A: 0.95, REGIME_B: 0.92, REGIME_C: 0.91},
    "mean_delta_anwg": 0.03,
    "bootstrap_ci_lower": 0.01,
    "oracle_gap_closure": 0.80,
    "multi_regime_benefit_count": 3,
    "leakage_instance_count": 0,
    "qualitative_all_clusters_attributable": True,
    "family_c_held_out_delta_anwg": 0.01,
    "blended_microcase_catastrophic_rate": 0.05,
}

ALL_FAIL_METRICS = {
    "stage1_input_validity_fraction": 0.5,
    "router_macro_f1": 0.10,
    "catastrophic_misroute_rate": 0.90,
    "stage2_preservation_fraction_by_regime": {REGIME_A: 0.10, REGIME_B: 0.10, REGIME_C: 0.10},
    "mean_delta_anwg": -0.05,
    "bootstrap_ci_lower": -0.02,
    "oracle_gap_closure": 0.10,
    "multi_regime_benefit_count": 0,
    "leakage_instance_count": 3,
    "qualitative_all_clusters_attributable": False,
    "family_c_held_out_delta_anwg": -0.10,
    "blended_microcase_catastrophic_rate": 0.50,
}


def test_gates_config_loads_and_has_nine_gates():
    config = load_gates_config()
    assert len(config["gates"]) == 9
    assert {g["id"] for g in config["gates"]} == {f"G{i}" for i in range(1, 10)}


def test_critical_gates_are_exactly_g1_g2_g3_g4_g5_g8():
    config = load_gates_config()
    critical = {g["id"] for g in config["gates"] if g["critical"]}
    assert critical == {"G1", "G2", "G3", "G4", "G5", "G8"}
    non_critical = {g["id"] for g in config["gates"] if not g["critical"]}
    assert non_critical == {"G6", "G7", "G9"}


def test_all_pass_synthetic_input_yields_all_gates_passing():
    gates = evaluate_all_gates(ALL_PASS_METRICS)
    for gate_id, result in gates.items():
        assert result.passed is True, f"{gate_id} expected to pass: {result}"


def test_all_fail_synthetic_input_yields_all_gates_failing():
    gates = evaluate_all_gates(ALL_FAIL_METRICS)
    for gate_id, result in gates.items():
        assert result.passed is False, f"{gate_id} expected to fail: {result}"


def test_verdict_go_on_all_pass_metrics():
    gates = evaluate_all_gates(ALL_PASS_METRICS)
    verdict = compute_verdict(gates)
    assert verdict == VERDICT_GO


def test_verdict_no_go_on_all_fail_metrics():
    gates = evaluate_all_gates(ALL_FAIL_METRICS)
    verdict = compute_verdict(gates)
    assert verdict == VERDICT_NO_GO


def test_verdict_no_go_when_g1_fails_even_if_everything_else_passes():
    metrics = dict(ALL_PASS_METRICS)
    metrics["stage1_input_validity_fraction"] = 0.99
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_NO_GO


def test_verdict_no_go_when_g8a_leakage_detected_even_if_everything_else_passes():
    metrics = dict(ALL_PASS_METRICS)
    metrics["leakage_instance_count"] = 1
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_NO_GO


def test_verdict_no_go_when_g2_router_quality_fails():
    metrics = dict(ALL_PASS_METRICS)
    metrics["router_macro_f1"] = 0.5
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_NO_GO


def test_verdict_no_go_when_g3_catastrophic_misrouting_fails():
    metrics = dict(ALL_PASS_METRICS)
    metrics["catastrophic_misroute_rate"] = 0.5
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_NO_GO


def test_verdict_no_go_when_g4_stage2_preservation_fails():
    metrics = dict(ALL_PASS_METRICS)
    metrics["stage2_preservation_fraction_by_regime"] = {REGIME_A: 0.95, REGIME_B: 0.5, REGIME_C: 0.95}
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_NO_GO


def test_verdict_routing_works_selection_no_gain_when_only_g5_fails():
    metrics = dict(ALL_PASS_METRICS)
    metrics["mean_delta_anwg"] = -0.01
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_ROUTING_WORKS_SELECTION_NO_GAIN


def test_verdict_no_go_when_g5_fails_and_g4_also_fails():
    metrics = dict(ALL_PASS_METRICS)
    metrics["mean_delta_anwg"] = -0.01
    metrics["stage2_preservation_fraction_by_regime"] = {REGIME_A: 0.10, REGIME_B: 0.95, REGIME_C: 0.95}
    gates = evaluate_all_gates(metrics)
    assert compute_verdict(gates) == VERDICT_NO_GO


def test_verdict_inconclusive_when_blended_sample_too_small_but_gates_pass():
    gates = evaluate_all_gates(ALL_PASS_METRICS)
    verdict = compute_verdict(gates, blended_microcase_sample_too_small=True)
    assert verdict == VERDICT_INCONCLUSIVE


def test_verdict_inconclusive_when_test_sample_insufficient_for_g5_ci():
    gates = evaluate_all_gates(ALL_PASS_METRICS)
    verdict = compute_verdict(gates, test_sample_insufficient_for_g5_ci=True)
    assert verdict == VERDICT_INCONCLUSIVE


def test_g6_g7_g9_never_force_no_go_or_inconclusive_alone():
    metrics = dict(ALL_PASS_METRICS)
    metrics["oracle_gap_closure"] = 0.01  # fails G6
    metrics["multi_regime_benefit_count"] = 0  # fails G7
    metrics["family_c_held_out_delta_anwg"] = -0.5  # fails G9(a)
    metrics["blended_microcase_catastrophic_rate"] = 0.99  # fails G9(b)
    gates = evaluate_all_gates(metrics)
    assert gates["G6"].passed is False
    assert gates["G7"].passed is False
    assert gates["G9"].passed is False
    assert compute_verdict(gates) == VERDICT_GO


def test_g4_uses_minimum_across_regimes_not_average():
    metrics = dict(ALL_PASS_METRICS)
    # Average would be (0.95+0.95+0.10)/3 = 0.667, still could look "close"
    # but the binding value must be the minimum (0.10), which fails.
    metrics["stage2_preservation_fraction_by_regime"] = {REGIME_A: 0.95, REGIME_B: 0.95, REGIME_C: 0.10}
    gates = evaluate_all_gates(metrics)
    assert gates["G4"].passed is False
    assert gates["G4"].value == pytest.approx(0.10)


def test_g5_requires_both_mean_and_ci_criteria():
    metrics = dict(ALL_PASS_METRICS)
    metrics["bootstrap_ci_lower"] = -0.001  # mean passes but CI lower bound does not exceed 0
    gates = evaluate_all_gates(metrics)
    assert gates["G5"].passed is False


def test_missing_metrics_produce_none_passed_not_a_crash():
    gates = evaluate_all_gates({})
    for gate_id, result in gates.items():
        assert result.passed is None

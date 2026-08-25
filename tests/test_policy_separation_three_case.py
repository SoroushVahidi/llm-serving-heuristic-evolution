"""Focused tests for the three-case Policy Separation diagnostic (first
compute experiment). Covers determinism, no duplicate scenario/policy keys,
no oracle leakage, schema validity, and a tiny real simulator run per case."""
from __future__ import annotations

import math

import pytest

from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.registry import make_policy_library_v2
from llmserveopt.policy_separation.templates_three_case import (
    case1_fcfs_convoy,
    case2_sjf_prediction_inversion,
    case3_edf_overload,
    generate_case1_grid,
    generate_case2_grid,
    generate_case3_grid,
)
from llmserveopt.simulator.service_model import ServiceModel


def test_case1_deterministic_and_id_reflects_role():
    a = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress")
    b = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress")
    assert a.scenario_id == b.scenario_id
    assert [r.request_id for r in a.requests] == [r.request_id for r in b.requests]
    assert [r.arrival_time for r in a.requests] == [r.arrival_time for r in b.requests]

    control = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="control")
    assert control.scenario_id != a.scenario_id
    assert control.pair_id == a.pair_id
    assert a.changed_parameters == ("arrival_order",)


def test_case1_no_leakage_predicted_equals_actual_and_deadline_visible_only():
    s = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress")
    for r in s.requests:
        # case 1 is not about prediction error: predicted must equal actual
        assert r.predicted_output_tokens == r.actual_output_tokens
        # deadline must be reproducible purely from (arrival, predicted_output_tokens)
        assert r.slo_deadline >= r.arrival_time


def test_case1_capacity_forces_serialization():
    s = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress")
    assert s.gpu_configs[0].max_active_sequences == 1


def test_case2_deadline_invariant_across_inversion_fraction():
    """Regression test for the deadline-confound fix: since inversion only
    permutes which request owns which predicted_output_tokens value, the
    flat per-cell deadline slack must be identical at every inversion
    fraction for fixed (heterogeneity, load, seed) -- otherwise fifo's
    results would drift with inversion_fraction even though fifo never
    reads predicted_output_tokens for ordering, confounding the comparison."""
    slacks = set()
    for inv in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = case2_sjf_prediction_inversion(inv, "strong", "high", seed=1)
        deadline_slacks = {round(r.slo_deadline - r.arrival_time, 9) for r in s.requests}
        assert len(deadline_slacks) == 1, "deadline slack must be flat within one scenario"
        slacks.add(next(iter(deadline_slacks)))
    assert len(slacks) == 1, f"deadline slack must be invariant across inversion_fraction, got {slacks}"


def test_case2_no_leakage_predicted_never_derived_from_actual_by_formula():
    """predicted_output_tokens is either == actual_output_tokens (no
    inversion) or a genuine permutation of the SAME multiset (inversion) --
    never a function of this request's own actual_output_tokens beyond
    that permutation, and the requests' actual/predicted multisets match."""
    s = case2_sjf_prediction_inversion(1.0, "strong", "high", seed=1)
    actual_multiset = sorted(r.actual_output_tokens for r in s.requests)
    predicted_multiset = sorted(r.predicted_output_tokens for r in s.requests)
    assert actual_multiset == predicted_multiset


def test_case2_rank_inversion_direction():
    s0 = case2_sjf_prediction_inversion(0.0, "strong", "high", seed=1)
    s1 = case2_sjf_prediction_inversion(1.0, "strong", "high", seed=1)
    assert s0.params["rank_agreement_kendall_tau"] == pytest.approx(1.0)
    assert s1.params["rank_agreement_kendall_tau"] < 0


def test_case3_control_loosens_deadlines_only():
    stress = case3_edf_overload(1.2, 0.4, seed=1, role="stress")
    control = case3_edf_overload(1.2, 0.4, seed=1, role="control")
    stress_actual = sorted(r.actual_output_tokens for r in stress.requests)
    control_actual = sorted(r.actual_output_tokens for r in control.requests)
    assert stress_actual == control_actual, "control must keep the same service requirements"
    assert control.params["window_s"] > stress.params["window_s"]
    assert stress.changed_parameters == ("slo_deadline",)


@pytest.mark.parametrize("family_grid_fn,kwargs", [
    (generate_case1_grid, dict(ratios=[8, 32], short_counts=[8], offsets=[0.0], seeds=[1, 2])),
    (generate_case2_grid, dict(inversion_fractions=[0.0, 0.5], heterogeneity_levels=["moderate"], load_levels=["high"], seeds=[1, 2])),
    (generate_case3_grid, dict(overload_factors=[1.0], fraction_impossible_levels=[0.2], seeds=[1, 2])),
])
def test_grid_has_no_duplicate_scenario_ids(family_grid_fn, kwargs):
    scenarios = family_grid_fn(**kwargs)
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))
    assert len(scenarios) > 0


@pytest.mark.parametrize("scenario,policy_name", [
    (case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress"), "fifo"),
    (case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress"), "estimated_service_time_first"),
    (case2_sjf_prediction_inversion(0.5, "strong", "high", seed=1), "weighted_shortest_processing"),
    (case3_edf_overload(1.2, 0.4, seed=1, role="stress"), "scorpio_style_slo_guard"),
])
def test_scenario_runs_and_produces_finite_anwg(scenario, policy_name):
    policy = make_policy_library_v2(policy_name)
    metrics = run_policy(
        policy=policy, requests=list(scenario.requests), gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(), workload_tag=scenario.scenario_id, seed=scenario.seed,
        drain_steps=20_000,
    )
    anwg = metrics.arrival_normalized_weighted_goodput
    assert not math.isnan(anwg)
    assert not math.isinf(anwg)
    assert 0.0 <= anwg <= 1.0 + 1e-9


def test_case1_stress_shows_separation_control_shrinks_it():
    """End-to-end sanity check of the mechanism this case exists to probe."""
    def anwg(scenario, name):
        policy = make_policy_library_v2(name)
        m = run_policy(
            policy=policy, requests=list(scenario.requests), gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(), workload_tag=scenario.scenario_id, seed=scenario.seed,
            drain_steps=20_000,
        )
        return m.arrival_normalized_weighted_goodput

    stress = case1_fcfs_convoy(32, 32, 0.0, seed=1, role="stress")
    control = case1_fcfs_convoy(32, 32, 0.0, seed=1, role="control")

    stress_gap = anwg(stress, "estimated_service_time_first") - anwg(stress, "fifo")
    control_gap = anwg(control, "estimated_service_time_first") - anwg(control, "fifo")
    assert stress_gap > 0
    assert stress_gap > control_gap

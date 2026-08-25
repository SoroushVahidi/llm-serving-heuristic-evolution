"""Focused tests for the Policy Separation Boundary Refinement v1 diagnostic
(SECOND compute experiment, following job 1170116). Covers: the case1
max_active_sequences extension stays backward-compatible, Study B's
numeric-target-utilization generator is deterministic/leakage-free, grid
generators produce no duplicate scenario ids, and a tiny real simulator run
per study."""
from __future__ import annotations

import math

import pytest

from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.registry import make_policy_library_v2
from llmserveopt.policy_separation.templates_boundary_refinement import (
    case2_prediction_inversion_boundary,
    generate_case1_boundary_grid,
    generate_case2_boundary_grid,
)
from llmserveopt.policy_separation.templates_three_case import (
    CASE1_ACTIVE_SEQUENCES,
    case1_fcfs_convoy,
    generate_case3_grid,
)
from llmserveopt.simulator.service_model import ServiceModel


def test_case1_max_active_sequences_default_matches_original_scenario_id():
    """The mas extension must be fully backward compatible: calling without
    max_active_sequences must produce byte-identical scenario_id/pair_id to
    before, since job 1170116's completed run used this exact function."""
    s_default = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress")
    s_explicit = case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress", max_active_sequences=CASE1_ACTIVE_SEQUENCES)
    assert s_default.scenario_id == s_explicit.scenario_id
    assert s_default.pair_id == s_explicit.pair_id
    assert s_default.gpu_configs[0].max_active_sequences == 1
    assert "mas" not in s_default.scenario_id


def test_case1_max_active_sequences_override_changes_gpu_config_and_id():
    s_mas4 = case1_fcfs_convoy(32, 32, 0.001, seed=1, role="stress", max_active_sequences=4)
    assert s_mas4.gpu_configs[0].max_active_sequences == 4
    assert s_mas4.gpu_configs[0].max_batch_tokens == 4
    assert "mas4" in s_mas4.scenario_id
    assert s_mas4.params["max_active_sequences"] == 4

    s_mas1 = case1_fcfs_convoy(32, 32, 0.001, seed=1, role="stress", max_active_sequences=1)
    assert s_mas1.scenario_id != s_mas4.scenario_id
    assert s_mas1.pair_id != s_mas4.pair_id


def test_case1_boundary_grid_fine_offsets_no_duplicates():
    offsets = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    scenarios = generate_case1_boundary_grid([32], [16, 32], offsets, [1, 2], max_active_sequences_values=[1])
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))
    # 1 ratio * 2 short_counts * 6 offsets * 2 seeds * 2 roles
    assert len(scenarios) == 1 * 2 * 6 * 2 * 2


def test_case1_boundary_grid_mas_cross_no_duplicates_with_main_grid():
    """The mas>1 sub-study in the real config only adds mas=4 (not mas=1),
    specifically to avoid this collision -- verify the collision would
    indeed occur if mas=1 were included twice, and that it does NOT occur
    for the actual (mas=[1] main + mas=[4] sub) construction used by the
    experiment script."""
    main = generate_case1_boundary_grid([32], [32], [0.0, 0.001], [1], max_active_sequences_values=[1])
    sub = generate_case1_boundary_grid([32], [32], [0.0, 0.001], [1], max_active_sequences_values=[4])
    ids = [s.scenario_id for s in main + sub]
    assert len(ids) == len(set(ids))

    dup_sub = generate_case1_boundary_grid([32], [32], [0.0, 0.001], [1], max_active_sequences_values=[1])
    dup_ids = [s.scenario_id for s in main + dup_sub]
    assert len(dup_ids) != len(set(dup_ids)), "sanity check: mas=1 repeated must collide"


def test_case2_boundary_deterministic_and_target_utilization_direct():
    a = case2_prediction_inversion_boundary(0.65, "strong", 0.3, seed=7)
    b = case2_prediction_inversion_boundary(0.65, "strong", 0.3, seed=7)
    assert a.scenario_id == b.scenario_id
    assert [r.arrival_time for r in a.requests] == [r.arrival_time for r in b.requests]
    assert a.params["target_utilization"] == 0.65


def test_case2_boundary_no_leakage_and_role_by_inversion():
    control = case2_prediction_inversion_boundary(0.85, "strong", 0.0, seed=1)
    stress = case2_prediction_inversion_boundary(0.85, "strong", 1.0, seed=1)
    assert control.stress_control_relationship == "control"
    assert stress.stress_control_relationship == "stress"
    actual_multiset = sorted(r.actual_output_tokens for r in stress.requests)
    predicted_multiset = sorted(r.predicted_output_tokens for r in stress.requests)
    assert actual_multiset == predicted_multiset
    assert control.params["rank_agreement_kendall_tau"] == pytest.approx(1.0)
    assert stress.params["rank_agreement_kendall_tau"] < 0
    assert control.params["rank_agreement_spearman"] == pytest.approx(1.0)
    assert stress.params["rank_agreement_spearman"] < 0


def test_case2_boundary_deadline_flat_across_inversion_fraction():
    slacks = set()
    for inv in (0.0, 0.3, 0.6, 1.0):
        s = case2_prediction_inversion_boundary(0.85, "strong", inv, seed=1)
        deadline_slacks = {round(r.slo_deadline - r.arrival_time, 9) for r in s.requests}
        assert len(deadline_slacks) == 1
        slacks.add(next(iter(deadline_slacks)))
    assert len(slacks) == 1


def test_case2_boundary_grid_no_duplicates():
    scenarios = generate_case2_boundary_grid([0.5, 0.85], ["moderate", "strong"], [0.0, 0.5, 1.0], [1, 2])
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))
    assert len(scenarios) == 2 * 2 * 3 * 2


@pytest.mark.parametrize("family_grid_fn,kwargs", [
    (generate_case1_boundary_grid, dict(ratios=[32], short_counts=[16], offsets=[0.0, 0.001], seeds=[1, 2],
                                          max_active_sequences_values=[1, 4])),
    (generate_case2_boundary_grid, dict(target_utilizations=[0.5, 0.9], heterogeneity_levels=["moderate"],
                                          inversion_fractions=[0.0, 0.5], seeds=[1, 2])),
    (generate_case3_grid, dict(overload_factors=[1.0], fraction_impossible_levels=[0.2], seeds=[1, 2])),
])
def test_grid_has_no_duplicate_scenario_ids(family_grid_fn, kwargs):
    scenarios = family_grid_fn(**kwargs)
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))
    assert len(scenarios) > 0


@pytest.mark.parametrize("scenario,policy_name", [
    (case1_fcfs_convoy(32, 8, 0.0, seed=1, role="stress"), "fifo"),
    (case1_fcfs_convoy(32, 32, 0.001, seed=1, role="stress", max_active_sequences=4), "estimated_service_time_first"),
    (case2_prediction_inversion_boundary(0.65, "strong", 0.3, seed=1), "weighted_shortest_processing"),
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


def test_case1_offset_zero_is_the_genuine_choice_regime():
    """End-to-end sanity check of Study A's central claim: at offset=0.0
    under mas=1, ESTF beats FIFO; at any positive offset under mas=1, the
    gap should already have collapsed towards zero (structurally
    uninformative), per job 1170116's finding."""
    def anwg(scenario, name):
        policy = make_policy_library_v2(name)
        m = run_policy(
            policy=policy, requests=list(scenario.requests), gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(), workload_tag=scenario.scenario_id, seed=scenario.seed,
            drain_steps=20_000,
        )
        return m.arrival_normalized_weighted_goodput

    s0 = case1_fcfs_convoy(32, 32, 0.0, seed=1, role="stress")
    s_pos = case1_fcfs_convoy(32, 32, 0.01, seed=1, role="stress")

    gap0 = anwg(s0, "estimated_service_time_first") - anwg(s0, "fifo")
    gap_pos = anwg(s_pos, "estimated_service_time_first") - anwg(s_pos, "fifo")
    assert gap0 > 0
    assert gap0 > gap_pos

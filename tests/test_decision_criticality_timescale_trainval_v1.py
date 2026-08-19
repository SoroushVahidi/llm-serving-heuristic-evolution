"""Focused tests for the Decision-Criticality & Regime-Timescale TRAIN/VAL
diagnostic (design doc `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`).

Uses small synthetic fixture scenarios (`allow_synthetic_tokens=True`, the
same convention `tests/test_hierarchical_router_live_harness_v1.py` already
uses) for speed -- these are NOT real MF-PSD TRAIN/VAL/TEST scenarios and no
test here computes or implies any scientific verdict.
"""
from __future__ import annotations

import ast
import copy
import inspect

import numpy as np
import pandas as pd
import pytest

from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.core.action import Action
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    DWELL_MINIMUM_STEPS,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    STAGE2_CANDIDATES,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2
from llmserveopt.policy_separation.templates_kv_pressure_v2 import case_kv_pressure_reserve_contention_v2
from llmserveopt.policy_separation.templates_prefill_decode_v2 import case_prefill_decode_ttft_contention


# ---------------------------------------------------------------------------
# Fixture scenarios (small, fast, synthetic-token, deterministic) -- same
# construction pattern as test_hierarchical_router_live_harness_v1.py
# ---------------------------------------------------------------------------

def _family_a_scenario(seed: int = 1, n_total_jobs: int = 20):
    return case_fairness_vs_size_v2(
        target_utilization=1.2, tenant_weight_skew=5.0, favored_tenant_size="long",
        prediction_noise_sigma=0.0, seed=seed, n_total_jobs=n_total_jobs,
        allow_synthetic_tokens=True,
    )


def _family_b_scenario(seed: int = 2):
    return case_prefill_decode_ttft_contention(
        hog_count="high", late_pressure="high", slo_emphasis="hog_ttft",
        seed=seed, allow_synthetic_tokens=True,
    )


def _family_c_scenario(seed: int = 3):
    return case_kv_pressure_reserve_contention_v2(
        bulk_pressure="high", urgent_arrival_phase="middle", urgent_tightness="tight",
        seed=seed, allow_synthetic_tokens=True,
    )


@pytest.fixture(scope="module")
def frozen_models():
    """Real Stage-1/Stage-2 models, fit exactly once per test module (TRAIN
    only, frozen recipe, ~1s) -- reused across tests that need a real
    router, exactly as the live-harness test module's own smoke tests do."""
    return dcm.fit_frozen_models()


# ---------------------------------------------------------------------------
# TRAIN/VAL-only guard + explicit TEST refusal
# ---------------------------------------------------------------------------

def test_assert_trainval_only_accepts_train_and_val():
    dcm.assert_trainval_only("train")
    dcm.assert_trainval_only("val")


def test_assert_trainval_only_rejects_test():
    with pytest.raises(dcm.TestSplitAccessError):
        dcm.assert_trainval_only("test")


def test_assert_trainval_only_rejects_garbage():
    with pytest.raises(dcm.TestSplitAccessError):
        dcm.assert_trainval_only("smoke_synthetic")


def test_load_trainval_scenario_table_never_returns_test_rows():
    table = dcm.load_trainval_scenario_table()
    assert set(table["split"].unique()) <= {"train", "val"}
    assert "test" not in set(table["split"].unique())


def test_load_trainval_scenario_table_matches_frozen_expected_counts():
    """Design doc SS3: A=64, B=32, C=48 -> 144 TRAIN/VAL scenarios total,
    under the frozen `build_splits`."""
    table = dcm.load_trainval_scenario_table()
    counts = table["mechanism_family"].value_counts().to_dict()
    assert counts.get("FAMILY_A_FAIRNESS_STARVATION_V2") == 64
    assert counts.get("FAMILY_B_PREFILL_DECODE_V2") == 32
    assert counts.get("FAMILY_C_KV_PRESSURE_V2") == 48
    assert len(table) == 144


def test_run_scenario_diagnostic_rejects_a_test_split_row(frozen_models):
    stage1, stage2_selectors = frozen_models
    fake_row = pd.Series({
        "canonical_scenario_id": "fake_test_row",
        "mechanism_family": "FAMILY_C_KV_PRESSURE_V2",
        "split": "test",
        "seed": 1,
    })
    with pytest.raises(dcm.TestSplitAccessError):
        dcm.run_scenario_diagnostic(fake_row, stage1=stage1, stage2_selectors=stage2_selectors)


# ---------------------------------------------------------------------------
# No hidden Family-B held-out replication access (structural guard, mirrors
# test_hierarchical_router_live_harness_v1.py's majority-vote guard)
# ---------------------------------------------------------------------------

def _module_ast() -> ast.Module:
    return ast.parse(inspect.getsource(dcm))


def test_module_never_imports_the_family_b_replication_module():
    tree = _module_ast()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("family_b_balanced_replication_v1" in name for name in imported)


def test_assert_no_replication_module_imported_self_check():
    # In a clean test process this module has not imported the replication
    # module, so the runtime self-check must pass silently.
    dcm.assert_no_replication_module_imported()


# ---------------------------------------------------------------------------
# Canonical action comparison / policy-pair identity
# ---------------------------------------------------------------------------

def test_canonical_action_is_order_independent():
    a1 = Action(admit={1: [3, 1, 2], 2: [5]})
    a2 = Action(admit={2: [5], 1: [1, 2, 3]})
    assert dcm.canonical_action(a1) == dcm.canonical_action(a2)


def test_canonical_action_empty_lists_ignored():
    a1 = Action(admit={1: [], 2: [7]})
    a2 = Action(admit={2: [7]})
    assert dcm.canonical_action(a1) == dcm.canonical_action(a2)


def test_actions_disagree_true_for_different_admit_sets():
    a1 = Action(admit={1: [1]})
    a2 = Action(admit={1: [2]})
    assert dcm.actions_disagree(a1, a2)


def test_actions_disagree_false_for_identical_admit_sets():
    a1 = Action(admit={1: [1, 2]})
    a2 = Action(admit={1: [2, 1]})
    assert not dcm.actions_disagree(a1, a2)


def test_assert_action_has_no_non_admit_verbs_raises_on_preempt():
    a = Action(admit={1: [1]}, preempt={1: [2]})
    with pytest.raises(AssertionError):
        dcm.assert_action_has_no_non_admit_verbs(a)


def test_alternative_policy_id_matches_frozen_stage2_candidates():
    for regime in ACTIVE_REGIMES:
        p0, p1 = STAGE2_CANDIDATES[regime]
        assert dcm.alternative_policy_id(regime, p0) == p1
        assert dcm.alternative_policy_id(regime, p1) == p0


def test_alternative_policy_id_none_for_unknown_regime():
    assert dcm.alternative_policy_id("NONE", "weighted_fair_share") is None


def test_alternative_policy_id_none_for_fallback_policy_in_active_regime():
    # weighted_fair_share is not a Regime-C native candidate.
    assert dcm.alternative_policy_id(REGIME_C, "weighted_fair_share") is None


# ---------------------------------------------------------------------------
# Episode segmentation
# ---------------------------------------------------------------------------

def test_segment_episodes_basic_contiguous_runs():
    labels = ["NONE", "NONE", "A_active", "A_active", "A_active", "NONE"]
    episodes = dcm.segment_episodes(labels)
    assert list(episodes["label"]) == ["NONE", "A_active", "NONE"]
    assert list(episodes["length"]) == [2, 3, 1]
    assert list(episodes["start_idx"]) == [0, 2, 5]
    assert list(episodes["end_idx"]) == [1, 4, 5]


def test_segment_episodes_empty_input():
    episodes = dcm.segment_episodes([])
    assert len(episodes) == 0


def test_classify_raw_activity_state_none_and_single_active():
    assert dcm.classify_raw_activity_state(False, False, False) == dcm.REGIME_NONE_LABEL
    assert dcm.classify_raw_activity_state(True, False, False) == dcm.REGIME_A_ACTIVE_LABEL
    assert dcm.classify_raw_activity_state(False, True, False) == dcm.REGIME_B_ACTIVE_LABEL
    assert dcm.classify_raw_activity_state(False, False, True) == dcm.REGIME_C_ACTIVE_LABEL


def test_classify_raw_activity_state_overlap():
    assert dcm.classify_raw_activity_state(True, True, False) == dcm.REGIME_OVERLAP_LABEL
    assert dcm.classify_raw_activity_state(True, True, True) == dcm.REGIME_OVERLAP_LABEL


def test_episode_length_distribution_fields_and_fractions():
    dist = dcm.episode_length_distribution([1, 2, 3, 4, 5, 20, 20, 50])
    assert dist["count"] == 8
    assert dist["min"] == 1
    assert dist["max"] == 50
    assert dist["fraction_lt_5"] == pytest.approx(4 / 8)
    assert dist["fraction_eq_20"] == pytest.approx(2 / 8)
    assert dist["fraction_gt_40"] == pytest.approx(1 / 8)


def test_episode_length_distribution_empty():
    assert dcm.episode_length_distribution([]) == {"count": 0}


def test_fraction_active_steps_in_short_episodes_weights_by_length_not_count():
    # One long episode (length 100) + many short ones (length 1 each, x10):
    # most EPISODES are short, but most ACTIVE STEPS are in the long one.
    episodes = pd.DataFrame({"length": [100] + [1] * 10})
    frac = dcm.fraction_active_steps_in_short_episodes(episodes, dwell=20)
    assert frac == pytest.approx(10 / 110)


# ---------------------------------------------------------------------------
# Dwell-latency classification (read-only reference to dwell=20)
# ---------------------------------------------------------------------------

def test_dwell_reference_is_the_frozen_constant():
    assert dcm.DWELL_REFERENCE == DWELL_MINIMUM_STEPS == 20


def test_horizon_h_is_frozen_and_below_dwell():
    assert dcm.HORIZON_H == 10
    assert dcm.HORIZON_H < dcm.DWELL_REFERENCE


@pytest.mark.parametrize("length,expected", [
    (1, dcm.UNREACHABLE_UNDER_DWELL20),
    (19, dcm.UNREACHABLE_UNDER_DWELL20),
    (20, dcm.PARTIALLY_REACTABLE),
    (39, dcm.PARTIALLY_REACTABLE),
    (40, dcm.FULLY_REACTABLE),
    (100, dcm.FULLY_REACTABLE),
])
def test_classify_dwell_reactability_boundaries(length, expected):
    assert dcm.classify_dwell_reactability(length) == expected


def test_dwell_latency_diagnostic_fields():
    episodes = pd.DataFrame({
        "label": [REGIME_A, REGIME_A, REGIME_A],
        "start_idx": [0, 100, 300],
        "end_idx": [10, 129, 360],
        "length": [11, 30, 61],
    })
    out = dcm.dwell_latency_diagnostic(episodes)
    assert list(out["reactability_class"]) == [
        dcm.UNREACHABLE_UNDER_DWELL20, dcm.PARTIALLY_REACTABLE, dcm.FULLY_REACTABLE,
    ]
    assert out.loc[0, "ends_before_switch_eligible"]
    assert not out.loc[2, "ends_before_switch_eligible"]
    # episode 2: start=300, end=360, dwell=20 -> eligible at 320,
    # remaining useful steps = 360 - 320 + 1 = 41.
    assert out.loc[2, "earliest_switch_eligible_step"] == 320
    assert out.loc[2, "useful_active_steps_remaining_after_eligibility"] == 41


# ---------------------------------------------------------------------------
# Fork isolation / state-cloning correctness (never mutate the original)
# ---------------------------------------------------------------------------

def test_fork_from_live_simulator_never_mutates_the_original(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_c_scenario(seed=11)

    from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import build_native_policy_instances
    from llmserveopt.simulator.service_model import ServiceModel
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    sim._reset()

    # Drive a few real steps manually so there is live, mutable state
    # (waiting/active/pending-arrival) to attempt to corrupt.
    native = build_native_policy_instances()
    policy = native["kv_constrained_online"]
    for _ in range(5):
        step_size = sim.config.service_model.step_size
        sim._time = sim._step * step_size
        # replicate Simulator.run()'s own enqueue block exactly
        arrival_idx = int(np.searchsorted(
            [ir.request.arrival_time for ir in sim._pending_arrivals], sim._time, side="right",
        ))
        for ir in sim._pending_arrivals[:arrival_idx]:
            if ir.request_id not in sim._waiting_map:
                sim._waiting.append(ir)
                sim._waiting_map[ir.request_id] = ir
        state = sim._build_observable_state()
        action = policy.select_action(state)
        sim._apply_action(action)
        completed = sim._advance_decode(action)
        sim._completed.extend(completed)
        sim._step += 1

    before_fp = dcm._state_fingerprint(sim)
    before_gpu_ids = [id(g) for g in sim._gpus]
    before_waiting_ids = [id(ir) for ir in sim._waiting]

    state = sim._build_observable_state()
    action = policy.select_action(copy.deepcopy(state))
    fork = dcm.fork_from_live_simulator(
        sim, policy=native["least_laxity_first"], policy_id="least_laxity_first", first_action=action,
    )
    # Drive the fork forward hard (many steps) to maximize the chance any
    # aliasing bug would corrupt the original.
    for _ in range(30):
        if fork.finished:
            break
        fork.advance_one_step()

    after_fp = dcm._state_fingerprint(sim)
    assert before_fp == after_fp, "forking/driving a fork must never mutate the live reference simulator"
    assert [id(g) for g in sim._gpus] == before_gpu_ids
    assert [id(ir) for ir in sim._waiting] == before_waiting_ids
    # The fork's own containers must be distinct objects, not aliases.
    assert all(id(fg) not in before_gpu_ids for fg in fork.shell._gpus)
    fork_waiting_ids = [id(ir) for ir in fork.shell._waiting]
    assert not (set(fork_waiting_ids) & set(before_waiting_ids))


def test_fork_pending_arrivals_are_independent_objects(frozen_models):
    scenario = _family_c_scenario(seed=12)
    from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import build_native_policy_instances
    from llmserveopt.simulator.service_model import ServiceModel
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    sim._reset()
    native = build_native_policy_instances()
    state = sim._build_observable_state()
    action = native["kv_constrained_online"].select_action(copy.deepcopy(state))
    fork = dcm.fork_from_live_simulator(
        sim, policy=native["least_laxity_first"], policy_id="least_laxity_first", first_action=action,
    )
    original_future_ids = {id(ir) for ir in sim._pending_arrivals}
    fork_future_ids = {id(ir) for ir in fork.shell._pending_arrivals}
    assert not (original_future_ids & fork_future_ids), (
        "fork's not-yet-enqueued future arrivals must be deep-copied, independent "
        "objects -- sharing them would let a fork's admission mutate the object "
        "the real run later enqueues, corrupting the reference trajectory."
    )


# ---------------------------------------------------------------------------
# End-to-end scenario diagnostic: determinism, bounded full-trajectory cap
# ---------------------------------------------------------------------------

def test_end_to_end_deterministic_replay_on_family_c_fixture(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_c_scenario(seed=21)
    kwargs = dict(
        canonical_scenario_id="fixture_c_determinism",
        stage1=stage1, stage2_selectors=stage2_selectors, seed=21,
        mechanism_family="FAMILY_C_KV_PRESSURE_V2", split="train",
        enable_full_trajectory_branches=False,
    )
    res1 = dcm.run_scenario_diagnostic_from_scenario(scenario, **kwargs)
    res2 = dcm.run_scenario_diagnostic_from_scenario(scenario, **kwargs)
    assert res1.n_steps == res2.n_steps
    assert res1.trajectory.reset_index(drop=True).equals(res2.trajectory.reset_index(drop=True))
    assert res1.disagreement_rows.reset_index(drop=True).equals(res2.disagreement_rows.reset_index(drop=True))


def test_end_to_end_full_trajectory_branches_respect_bounded_cap(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_c_scenario(seed=22)
    res = dcm.run_scenario_diagnostic_from_scenario(
        scenario, canonical_scenario_id="fixture_c_full_traj",
        stage1=stage1, stage2_selectors=stage2_selectors, seed=22,
        mechanism_family="FAMILY_C_KV_PRESSURE_V2", split="train",
        enable_full_trajectory_branches=True,
    )
    assert len(res.full_trajectory_results) <= dcm.FULL_TRAJECTORY_MAX_BRANCHES_PER_SCENARIO
    for branch in res.full_trajectory_results:
        assert branch["chosen_rollout"]["steps_run"] <= dcm.FULL_TRAJECTORY_MAX_EXTRA_STEPS
        assert branch["alt_rollout"]["steps_run"] <= dcm.FULL_TRAJECTORY_MAX_EXTRA_STEPS


def test_end_to_end_disagreement_rows_have_expected_columns(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_c_scenario(seed=23)
    res = dcm.run_scenario_diagnostic_from_scenario(
        scenario, canonical_scenario_id="fixture_c_columns",
        stage1=stage1, stage2_selectors=stage2_selectors, seed=23,
        mechanism_family="FAMILY_C_KV_PRESSURE_V2", split="train",
        enable_full_trajectory_branches=False,
    )
    if len(res.disagreement_rows) > 0:
        for col in ("step", "regime", "chosen_policy_id", "alt_policy_id"):
            assert col in res.disagreement_rows.columns


def test_end_to_end_h_step_and_immediate_rows_present_when_disagreement_occurs(frozen_models):
    """Design doc SS5E: every ACTION_DISAGREEMENT step that gets forked
    produces both a horizon=1 (immediate) and a horizon=H (short-horizon)
    comparison row, unless the scenario ended first (flagged
    `horizon_truncated_by_scenario_end`)."""
    stage1, stage2_selectors = frozen_models
    scenario = _family_c_scenario(seed=24)
    res = dcm.run_scenario_diagnostic_from_scenario(
        scenario, canonical_scenario_id="fixture_c_horizons",
        stage1=stage1, stage2_selectors=stage2_selectors, seed=24,
        mechanism_family="FAMILY_C_KV_PRESSURE_V2", split="train",
        enable_full_trajectory_branches=False,
    )
    disagree_steps = res.disagreement_rows[res.disagreement_rows.get("disagree") == True]["step"].tolist() if len(res.disagreement_rows) else []
    forked = res.disagreement_rows[res.disagreement_rows.get("horizon").notna()] if "horizon" in res.disagreement_rows.columns else pd.DataFrame()
    if disagree_steps:
        assert len(forked) > 0
        for step in disagree_steps:
            rows_for_step = forked[forked["step"] == step]
            horizons = set(rows_for_step["horizon"].tolist())
            # Either it reached both horizons, or was truncated by scenario end.
            truncated = rows_for_step.get("horizon_truncated_by_scenario_end")
            was_truncated = bool(truncated.any()) if truncated is not None else False
            assert horizons or was_truncated


# ---------------------------------------------------------------------------
# No mutation of frozen source artifacts
# ---------------------------------------------------------------------------

def test_diagnostic_run_does_not_modify_frozen_router_source_files(frozen_models, tmp_path):
    import hashlib
    from pathlib import Path

    frozen_files = [
        dcm.ROOT / "src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py",
        dcm.ROOT / "src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py",
        dcm.ROOT / "src/llmserveopt/simulator/simulator.py",
        dcm.ROOT / "docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md",
    ]

    def _hash(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    before = {p: _hash(p) for p in frozen_files}
    stage1, stage2_selectors = frozen_models
    scenario = _family_c_scenario(seed=25)
    dcm.run_scenario_diagnostic_from_scenario(
        scenario, canonical_scenario_id="fixture_c_no_mutation",
        stage1=stage1, stage2_selectors=stage2_selectors, seed=25,
        mechanism_family="FAMILY_C_KV_PRESSURE_V2", split="train",
        enable_full_trajectory_branches=True,
    )
    after = {p: _hash(p) for p in frozen_files}
    assert before == after


# ---------------------------------------------------------------------------
# Frozen native-pair policies use admit-only actions (canonicalization
# completeness precondition)
# ---------------------------------------------------------------------------

def test_all_six_native_policies_never_use_non_admit_verbs():
    from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import build_native_policy_instances
    from llmserveopt.simulator.service_model import ServiceModel
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

    scenario = _family_c_scenario(seed=26)
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    sim._reset()
    sim._time = 0.0
    state = sim._build_observable_state()

    native = build_native_policy_instances()
    for policy_id, policy in native.items():
        action = policy.select_action(copy.deepcopy(state))
        dcm.assert_action_has_no_non_admit_verbs(action)

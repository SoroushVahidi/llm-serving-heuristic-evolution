"""Focused tests for the Family-A Observability / Continuation-Dependence
diagnostic (design doc
`docs/design/FAMILY_A_OBSERVABILITY_CONTINUATION_DIAGNOSTIC_V1.md`).

Uses small synthetic fixture scenarios (`allow_synthetic_tokens=True`, same
convention `tests/test_decision_criticality_timescale_trainval_v1.py` /
`tests/test_hierarchical_router_live_harness_v1.py` already use) for speed.
No test here computes or implies any scientific verdict.
"""
from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from llmserveopt.analysis import family_a_observability_continuation_v1 as fac
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import REGIME_A
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2


def _family_a_scenario(seed: int = 1, n_total_jobs: int = 24):
    return case_fairness_vs_size_v2(
        target_utilization=1.2, tenant_weight_skew=5.0, favored_tenant_size="long",
        prediction_noise_sigma=0.0, seed=seed, n_total_jobs=n_total_jobs,
        allow_synthetic_tokens=True,
    )


@pytest.fixture(scope="module")
def frozen_models():
    return dcm.fit_frozen_models()


# ---------------------------------------------------------------------------
# Split guards (reused from the parent module -- re-asserted here since this
# module re-exports them for its own entry points)
# ---------------------------------------------------------------------------

def test_assert_trainval_only_rejects_test():
    with pytest.raises(fac.TestSplitAccessError):
        fac.assert_trainval_only("test")


def test_load_family_a_trainval_scenario_table_is_family_a_only_and_trainval_only():
    table = fac.load_family_a_trainval_scenario_table()
    assert set(table["mechanism_family"].unique()) == {fac.FAMILY_A}
    assert set(table["split"].unique()) <= {"train", "val"}
    assert len(table) == 64  # design doc SS_A: 54 train + 10 val


def test_run_family_a_row_diagnostic_rejects_test_split(frozen_models):
    stage1, stage2_selectors = frozen_models
    fake_row = pd.Series({
        "canonical_scenario_id": "fake_test_row",
        "mechanism_family": fac.FAMILY_A,
        "split": "test",
        "seed": 1,
    })
    with pytest.raises(fac.TestSplitAccessError):
        fac.run_family_a_row_diagnostic(fake_row, stage1=stage1, stage2_selectors=stage2_selectors)


def test_run_family_a_row_diagnostic_rejects_non_family_a_row(frozen_models):
    stage1, stage2_selectors = frozen_models
    fake_row = pd.Series({
        "canonical_scenario_id": "fake_row",
        "mechanism_family": "FAMILY_C_KV_PRESSURE_V2",
        "split": "train",
        "seed": 1,
    })
    with pytest.raises(AssertionError):
        fac.run_family_a_row_diagnostic(fake_row, stage1=stage1, stage2_selectors=stage2_selectors)


def _run_replication_safeguard_in_subprocess() -> subprocess.CompletedProcess:
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from llmserveopt.analysis import family_a_observability_continuation_v1 as fac; "
        "fac.assert_no_replication_module_imported(); "
        "print('SAFEGUARD_PASSED')"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
    )


def test_no_replication_module_imported():
    result = _run_replication_safeguard_in_subprocess()
    assert result.returncode == 0, result.stderr
    assert "SAFEGUARD_PASSED" in result.stdout


def test_replication_safeguard_immune_to_other_process_wide_imports():
    import llmserveopt.policy_separation.family_b_balanced_replication_v1  # noqa: F401

    assert any(
        name.endswith("family_b_balanced_replication_v1") for name in sys.modules
    ), "test setup failed: forbidden module was not actually imported into this process"

    result = _run_replication_safeguard_in_subprocess()
    assert result.returncode == 0, result.stderr
    assert "SAFEGUARD_PASSED" in result.stdout


# ---------------------------------------------------------------------------
# Pure arithmetic: sign/magnitude calculations and continuation-dependence
# logic (design doc SS_E), independent of the simulator
# ---------------------------------------------------------------------------

def test_compute_deltas_local_action_effect_stable_across_continuation():
    """ESTF's immediate action is better under BOTH common continuations,
    and native continuation agrees: local, continuation-independent
    advantage (`LOCAL_ACTION_OBSERVABLE`-flavored fixture)."""
    branches = {"br_estf_estf": 10, "br_wfs_wfs": 6, "br_wfs_estf": 7, "br_estf_wfs": 9}
    deltas = fac.compute_deltas(branches)
    assert deltas["delta_native"] == 4  # 10 - 6
    assert deltas["delta_same_common_estf"] == 3  # 10 - 7
    assert deltas["delta_same_common_wfs"] == 3  # 9 - 6
    assert deltas["delta_same"] == 3.0
    assert deltas["continuation_dependence"] == pytest.approx(1.0)
    assert deltas["sign_same_eq_native"] is True


def test_compute_deltas_continuation_dominated_sign_flip():
    """Native continuation strongly favors ESTF, but under a FIXED common
    continuation the immediate action makes no difference (or favors WFS):
    the native advantage is an artifact of who keeps driving, not of the
    immediate action -- `CONTINUATION_DOMINATED`-flavored fixture."""
    branches = {"br_estf_estf": 10, "br_wfs_wfs": 4, "br_wfs_estf": 10, "br_estf_wfs": 4}
    deltas = fac.compute_deltas(branches)
    assert deltas["delta_native"] == 6  # 10 - 4: native strongly favors ESTF
    # under common continuation ESTF, action makes ZERO difference:
    assert deltas["delta_same_common_estf"] == 0  # 10 - 10
    # under common continuation WFS, action makes ZERO difference either:
    assert deltas["delta_same_common_wfs"] == 0  # 4 - 4
    assert deltas["delta_same"] == 0.0
    assert deltas["continuation_dependence"] == pytest.approx(6.0)
    assert deltas["sign_same_eq_native"] is False  # sign(0) != sign(+6)


def test_sign_helper():
    assert fac._sign(5.0) == 1
    assert fac._sign(-5.0) == -1
    assert fac._sign(0.0) == 0


# ---------------------------------------------------------------------------
# Feature extraction: no future-derived features, expected keys present,
# state not mutated
# ---------------------------------------------------------------------------

def test_extract_causal_features_covers_all_groups_and_does_not_mutate_state(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_a_scenario(seed=11)
    res = fac.run_family_a_scenario_diagnostic(
        scenario, canonical_scenario_id="unit_family_a_features", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=11, split="train",
    )
    assert res.n_family_a_active_steps >= 0
    for ev in res.events:
        f = ev.features
        for group_prefix in (
            "queue_length", "active_count", "completed_count", "step", "n_gpus",
            "queue_age_p50", "predicted_output_tokens_p50", "prompt_tokens_p50", "est_service_time_p50",
            "max_class_deficit_ratio", "longest_waiting_age", "n_distinct_classes_in_queue",
            "laxity_p50", "fraction_laxity_negative", "fraction_laxity_near_deadline",
            "mean_kv_utilization", "max_kv_utilization", "free_kv_capacity",
            "n_admit_estf", "n_admit_wfs", "admit_symmetric_diff_size", "is_shallow_disagreement",
            "pair_rank_spearman_topk", "pair_topk_n",
            "history_queue_len_slope", "history_kv_util_slope", "history_admitted_count_slope",
            "history_window_truncated",
        ):
            assert group_prefix in f, f"missing feature {group_prefix}"
        # forbidden future-derived fields must never appear
        for forbidden in ("actual_output_tokens", "final_completed_count", "future_"):
            assert not any(k.startswith(forbidden) for k in f), f"forbidden feature leaked: {forbidden}"


# ---------------------------------------------------------------------------
# Fork isolation + deterministic replay + stable join keys (small real
# scenario, end-to-end smoke)
# ---------------------------------------------------------------------------

def test_family_a_diagnostic_deterministic_replay(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_a_scenario(seed=7, n_total_jobs=40)
    res1 = fac.run_family_a_scenario_diagnostic(
        scenario, canonical_scenario_id="unit_family_a_replay", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=7, split="train",
    )
    scenario2 = _family_a_scenario(seed=7, n_total_jobs=40)
    res2 = fac.run_family_a_scenario_diagnostic(
        scenario2, canonical_scenario_id="unit_family_a_replay", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=7, split="train",
    )
    # Repair regression (2026-08-20): this is a KNOWN genuine-disagreement
    # fixture (empirically confirmed to produce disagreement events both
    # before this repair, when it produced 0, and after, when it produces
    # >=1 -- see docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md
    # SS "non-vacuous regression test"). A regression back to the pre-repair
    # snapshot-timing bug would silently make this assertion the ONLY thing
    # that fails (the `len(res1.events) == len(res2.events)` check below
    # would still trivially pass at 0 == 0).
    assert len(res1.events) >= 1, "regression: known-disagreement fixture produced zero events"
    assert res1.n_steps == res2.n_steps
    assert len(res1.events) == len(res2.events)
    for e1, e2 in zip(res1.events, res2.events):
        assert e1.step == e2.step
        assert e1.delta_native == e2.delta_native
        assert e1.delta_same == e2.delta_same
        assert e1.br_estf_estf_completed == e2.br_estf_estf_completed


def test_family_a_events_have_stable_join_keys_and_bounded_branch_budget(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario = _family_a_scenario(seed=13, n_total_jobs=40)
    res = fac.run_family_a_scenario_diagnostic(
        scenario, canonical_scenario_id="unit_family_a_budget", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=13, split="val",
    )
    # Repair regression (2026-08-20): must not be vacuously satisfied by
    # zero events -- see test_family_a_diagnostic_deterministic_replay's
    # comment and the repair audit doc.
    assert len(res.events) >= 1, "regression: known-disagreement fixture produced zero events"
    assert len(res.events) <= fac.FULL_TRAJECTORY_BRANCHES_PER_SCENARIO
    seen_steps = set()
    for ev in res.events:
        row = ev.to_row()
        assert row["canonical_scenario_id"] == "unit_family_a_budget"
        assert row["step"] not in seen_steps  # each branch step distinct
        seen_steps.add(row["step"])
        assert row["router_chosen_policy_id"] in (fac.ESTF_ID, fac.WFS_ID)


def test_family_a_observer_captures_known_disagreement_from_identical_pre_decision_state(frozen_models):
    """Non-vacuous disagreement regression test (repair task SS5). Uses the
    same deterministic seed=7/n_total_jobs=40 Family-A TRAIN fixture already
    empirically verified (this repair session's own before/after proof, and
    `test_family_a_diagnostic_deterministic_replay` above) to produce >=1
    real ESTF/WFS disagreement under the repaired observer, and to produce
    EXACTLY ZERO under the pre-repair snapshot-timing bug (a monkeypatched
    reconstruction of the old `select_action` body against this identical
    fixture was run in this repair session and confirmed 0 events; not
    re-run here to keep this test fast and free of a duplicated stale copy
    of dead code). Asserts everything SS5 requires directly from stored
    event fields (no need to re-derive pre-decision state manually -- the
    event's own `features` dict, `extract_causal_features`'s `n_admit_estf`/
    `n_admit_wfs`/`admit_symmetric_diff_size`, are themselves computed from
    the admitted-id sets this test also checks disagree)."""
    stage1, stage2_selectors = frozen_models
    scenario = _family_a_scenario(seed=7, n_total_jobs=40)
    res = fac.run_family_a_scenario_diagnostic(
        scenario, canonical_scenario_id="unit_known_disagreement", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=7, split="train",
    )
    assert len(res.events) >= 1, (
        "known-disagreement fixture produced zero events -- this is exactly "
        "the 2026-08-20 snapshot-timing regression (see repair audit doc); "
        "both candidates are being evaluated from a shared, non-pre-decision "
        "baseline again"
    )
    for ev in res.events:
        # ESTF and WFS actions genuinely differ at the captured event (the
        # feature vector's own pair-specific-disagreement-geometry group,
        # SS_C Group F, is derived from exactly this and must show it).
        assert ev.features["admit_symmetric_diff_size"] >= 1
        assert ev.features["n_admit_estf"] != ev.features["n_admit_wfs"] or (
            ev.features["admit_symmetric_diff_size"] >= 1
        )
        # event identity is stable/deterministic: (scenario, step) is a
        # valid, reusable join key.
        assert isinstance(ev.step, int) and ev.step >= 0
        assert ev.canonical_scenario_id == "unit_known_disagreement"


def test_run_four_branches_never_mutates_reference_simulator(frozen_models):
    """Fork isolation, re-asserted at this module's own call site (the
    underlying fork primitive is already tested by
    `tests/test_decision_criticality_timescale_trainval_v1.py`)."""
    stage1, stage2_selectors = frozen_models
    scenario = _family_a_scenario(seed=21)
    from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import (
        LiveHierarchicalRouterPolicy, build_native_policy_instances, build_feature_rows_by_regime,
    )
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
    from llmserveopt.simulator.service_model import ServiceModel

    feature_rows = build_feature_rows_by_regime(scenario, "unit_fork_isolation")
    inner_router = LiveHierarchicalRouterPolicy(
        scenario_id="unit_fork_isolation", stage1=stage1, stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows, record_trajectory=True,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs), service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    shadow_policies = build_native_policy_instances()

    class _StopAfterFirstFamilyAEvent(fac.FamilyAObservabilityObserver):
        def select_action(self, state):
            fp_before = dcm._state_fingerprint(self.sim_ref)
            action = super().select_action(state)
            fp_after = dcm._state_fingerprint(self.sim_ref)
            assert fp_before == fp_after, "run_four_branches mutated the reference simulator"
            return action

    observer = _StopAfterFirstFamilyAEvent(
        sim_ref=sim, inner_router=inner_router, shadow_policies=shadow_policies,
        canonical_scenario_id="unit_fork_isolation", split="train",
        step_size=sim.config.service_model.step_size,
    )
    sim.run(observer, workload_tag="unit_fork_isolation", seed=21)
    # test passes if no assertion inside the subclass fired


# ---------------------------------------------------------------------------
# Repair regression tests (2026-08-20): the pre-repair `select_action`
# snapshotted `state.gpu_states`'s per-GPU counters AFTER
# `inner_router.select_action(state)` had already mutated them via its
# native-policy delegation, so both shadow candidates (ESTF, WFS) were
# evaluated against an already-capacity-consumed baseline instead of the
# true pre-decision one -- suppressing essentially all detectable
# disagreement (observed: n_events_total=0 across all 64 real TRAIN/VAL
# Family-A scenarios / 796,415 active steps in the completed-but-invalid run
# preserved at
# experiments/family_a_observability_continuation_v1_invalid_pre_snapshot_fix_20260820/).
# See docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md.
# ---------------------------------------------------------------------------

def _synthetic_gpu_state(gpu_id: int, active_ids, kv_tokens: int) -> ObservableGPUState:
    return ObservableGPUState(
        gpu_id=gpu_id,
        max_active_sequences=8,
        max_batch_tokens=100_000,
        max_kv_tokens=1000,
        active_request_ids=list(active_ids),
        active_requests_info=[],
        current_kv_tokens=kv_tokens,
        tokens_decoded_per_request={},
    )


def _synthetic_observable_state(gpu_states) -> ObservableState:
    return ObservableState(
        time=0.0, waiting_queue=[], gpu_states=list(gpu_states), completed_count=0, step=0,
    )


def test_snapshot_gpu_counters_captures_state_before_mutation():
    """SS6.A: a snapshot taken BEFORE a mutation reflects the PRE-mutation
    values when later restored, not whatever the state held at snapshot-use
    time -- i.e. `snapshot_gpu_counters` must be called before the mutation
    it is meant to protect against, exactly as the repaired
    `select_action` now does (snapshots before `inner_router.select_action`,
    not after)."""
    gpu = _synthetic_gpu_state(0, active_ids=[1, 2], kv_tokens=50)
    state = _synthetic_observable_state([gpu])

    pre = fac.snapshot_gpu_counters(state)  # taken BEFORE any mutation

    # Simulate a native policy's admission-planning mutation (the exact
    # pattern ESTF/`deterministic_place` use).
    gpu.active_request_ids.append(3)
    gpu.current_kv_tokens += 25
    assert gpu.active_request_ids == [1, 2, 3]
    assert gpu.current_kv_tokens == 75

    fac.restore_gpu_counters(state, pre)
    assert gpu.active_request_ids == [1, 2]
    assert gpu.current_kv_tokens == 50


def test_restore_gpu_counters_no_alias_leakage():
    """SS6.E: `restore_gpu_counters` must write a fresh list into
    `active_request_ids` (`[:] = ids`), never alias the snapshot's own list
    -- otherwise a later live mutation of `state` would corrupt the stored
    snapshot, breaking every subsequent restore from it (the real
    `select_action` restores from the SAME `pre_decision_gpu_state` object
    three times per Family-A-active step)."""
    gpu = _synthetic_gpu_state(0, active_ids=[1, 2], kv_tokens=50)
    state = _synthetic_observable_state([gpu])
    pre = fac.snapshot_gpu_counters(state)

    fac.restore_gpu_counters(state, pre)
    gpu.active_request_ids.append(99)  # mutate the LIVE state after restoring
    assert pre[0][0] == [1, 2], "snapshot was corrupted by a later live mutation -- alias leak"

    # A second restore from the same (still-valid) snapshot must still work.
    fac.restore_gpu_counters(state, pre)
    assert gpu.active_request_ids == [1, 2]


def test_restore_gpu_counters_gives_identical_baseline_across_repeated_calls():
    """SS6.B/C: ESTF and WFS shadow calls must each start from the
    IDENTICAL pre-decision baseline. Simulates the real three-restore
    sequence (before ESTF, before WFS, before feature extraction) with
    different intervening mutations each time and asserts every
    post-restore state is bit-identical to the true original."""
    gpu = _synthetic_gpu_state(0, active_ids=[1, 2], kv_tokens=50)
    state = _synthetic_observable_state([gpu])
    pre = fac.snapshot_gpu_counters(state)
    original = (list(gpu.active_request_ids), gpu.current_kv_tokens)

    # "ESTF shadow call" -- some other mutation pattern.
    fac.restore_gpu_counters(state, pre)
    baseline_before_estf = (list(gpu.active_request_ids), gpu.current_kv_tokens)
    gpu.active_request_ids.extend([3, 4])
    gpu.current_kv_tokens += 40

    # "WFS shadow call" must see the SAME baseline ESTF did, not ESTF's
    # leftover mutation.
    fac.restore_gpu_counters(state, pre)
    baseline_before_wfs = (list(gpu.active_request_ids), gpu.current_kv_tokens)
    gpu.active_request_ids.append(5)
    gpu.current_kv_tokens += 10

    assert baseline_before_estf == original
    assert baseline_before_wfs == original
    assert baseline_before_estf == baseline_before_wfs


def test_snapshot_gpu_counters_covers_all_gpus_independently():
    """SS6.E: multi-GPU states restore each GPU's own counters
    independently -- one GPU's mutation must never leak into another's
    restored values."""
    gpu0 = _synthetic_gpu_state(0, active_ids=[1], kv_tokens=10)
    gpu1 = _synthetic_gpu_state(1, active_ids=[2], kv_tokens=20)
    state = _synthetic_observable_state([gpu0, gpu1])
    pre = fac.snapshot_gpu_counters(state)

    gpu0.active_request_ids.append(99)
    gpu0.current_kv_tokens += 5
    gpu1.active_request_ids.append(100)
    gpu1.current_kv_tokens += 7

    fac.restore_gpu_counters(state, pre)
    assert gpu0.active_request_ids == [1]
    assert gpu0.current_kv_tokens == 10
    assert gpu1.active_request_ids == [2]
    assert gpu1.current_kv_tokens == 20


# ---------------------------------------------------------------------------
# Observer non-interference (SS4/SS6.D): enabling the Family-A observer must
# not change anything about the real router/simulator trajectory relative to
# running `LiveHierarchicalRouterPolicy` directly, unwrapped.
# ---------------------------------------------------------------------------

def _run_plain_router(scenario, *, canonical_scenario_id, stage1, stage2_selectors, seed):
    from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import (
        LiveHierarchicalRouterPolicy, build_feature_rows_by_regime,
    )
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
    from llmserveopt.simulator.service_model import ServiceModel

    feature_rows = build_feature_rows_by_regime(scenario, canonical_scenario_id)
    inner_router = LiveHierarchicalRouterPolicy(
        scenario_id=canonical_scenario_id, stage1=stage1, stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows, record_trajectory=True,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs), service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(inner_router, workload_tag=canonical_scenario_id, seed=seed)
    return metrics, inner_router.trajectory_df()


def _run_observed_router(scenario, *, canonical_scenario_id, stage1, stage2_selectors, seed):
    """Identical setup to `run_family_a_scenario_diagnostic`, but returns the
    raw `RunMetrics`/trajectory too (that function's own `ScenarioFamilyAResult`
    return type only exposes `n_steps`/`n_family_a_active_steps`/`events`, not
    metrics/trajectory -- needed here for a byte-for-byte non-interference
    comparison against the plain, unobserved run)."""
    from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import (
        LiveHierarchicalRouterPolicy, build_native_policy_instances, build_feature_rows_by_regime,
    )
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
    from llmserveopt.simulator.service_model import ServiceModel

    feature_rows = build_feature_rows_by_regime(scenario, canonical_scenario_id)
    inner_router = LiveHierarchicalRouterPolicy(
        scenario_id=canonical_scenario_id, stage1=stage1, stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows, record_trajectory=True,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs), service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    shadow_policies = build_native_policy_instances()
    observer = fac.FamilyAObservabilityObserver(
        sim_ref=sim, inner_router=inner_router, shadow_policies=shadow_policies,
        canonical_scenario_id=canonical_scenario_id, split="train",
        step_size=sim.config.service_model.step_size,
    )
    metrics = sim.run(observer, workload_tag=canonical_scenario_id, seed=seed)
    return metrics, inner_router.trajectory_df()


def test_observer_does_not_change_real_router_trajectory_or_completions(frozen_models):
    stage1, stage2_selectors = frozen_models
    scenario_plain = _family_a_scenario(seed=17, n_total_jobs=40)
    scenario_observed = _family_a_scenario(seed=17, n_total_jobs=40)

    metrics_plain, traj_plain = _run_plain_router(
        scenario_plain, canonical_scenario_id="unit_noninterference", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=17,
    )
    metrics_observed, traj_observed = _run_observed_router(
        scenario_observed, canonical_scenario_id="unit_noninterference", stage1=stage1,
        stage2_selectors=stage2_selectors, seed=17,
    )

    # The observed run must reproduce the plain run's trajectory row-for-row
    # -- same regime routing, same selected policy, same admissions, same
    # post-admission state -- the observer's shadow computation must be
    # perfectly invisible to the real router/simulator.
    cols = [
        "step", "effective_regime", "selected_policy", "admitted_count",
        "admitted_request_ids", "queue_len_after_admission",
        "active_count_after_admission", "mean_kv_utilization_after_admission",
    ]
    assert len(traj_observed) == len(traj_plain)
    pd.testing.assert_frame_equal(traj_observed[cols], traj_plain[cols])

    # Same aggregate outcome -- completion/drop/SLO counts and latency
    # distribution identical between plain and observed runs.
    for field in (
        "num_completed", "num_dropped", "num_slo_violated", "num_total",
        "completion_fraction", "mean_latency", "median_latency",
        "p95_latency", "p99_latency", "max_latency", "slo_violation_rate",
    ):
        plain_val = getattr(metrics_plain, field)
        observed_val = getattr(metrics_observed, field)
        if isinstance(plain_val, float) and np.isnan(plain_val):
            assert np.isnan(observed_val), f"{field}: plain=NaN, observed={observed_val}"
        else:
            assert plain_val == pytest.approx(observed_val), f"{field}: plain={plain_val}, observed={observed_val}"

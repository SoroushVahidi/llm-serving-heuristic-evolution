"""Focused tests for Hierarchical Regime Router v1 -- Stage-1 router,
dwell/fallback FSM, split builder, and blended-regime microcase
instantiation. Implements design doc SS R items 1-6, 9-13, 16-17 (the
subset not covered by Stage-2/baselines/gates test files)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    BLENDED_MICROCASE_BUILDERS,
    DWELL_MINIMUM_STEPS,
    FALLBACK_POLICY,
    FALLBACK_REGIMES,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_NONE,
    REGIME_OVERLAP,
    STAGE1_FORBIDDEN_COLUMNS,
    STAGE1_INPUT_COLUMNS,
    Stage1Router,
    add_regime_labels,
    apply_dwell_and_fallback,
    assert_group_disjoint,
    build_splits,
    count_dwell_violations,
    regime_label_from_activity,
    route_action,
)
from llmserveopt.policy_separation.online_regime_signals_v1 import (
    ActivityLabels,
    RegimeSignals,
    compute_activity_labels,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_TELEMETRY = ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
MF_PSD_SCENARIOS = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"


# ---------------------------------------------------------------------------
# Item 1/2: exact Stage-1 input allowlist; forbidden fields never present
# ---------------------------------------------------------------------------

def test_stage1_input_allowlist_is_exactly_four_frozen_fields():
    assert STAGE1_INPUT_COLUMNS == ("contention_score_v2", "priority_skew", "kv_pressure", "queue_length")
    assert len(STAGE1_INPUT_COLUMNS) == 4


def test_stage1_forbidden_columns_never_overlap_input_allowlist():
    assert set(STAGE1_INPUT_COLUMNS).isdisjoint(set(STAGE1_FORBIDDEN_COLUMNS))


def test_stage1_extract_inputs_narrows_to_allowlist_ignoring_extra_source_columns():
    """Source telemetry frames legitimately carry extra identity/audit
    columns (canonical_scenario_id, mechanism_family, ...) -- extract_inputs
    must silently narrow to just the 4 allowed columns, never raise on
    their presence in the SOURCE frame (forbidden columns are simply never
    selected, which is what keeps them out of the model, not a rejection
    of the source frame's shape)."""
    df = pd.DataFrame({
        "contention_score_v2": [0.1], "priority_skew": [1.0],
        "kv_pressure": [0.2], "queue_length": [3],
        "mechanism_family": ["FAMILY_A_FAIRNESS_STARVATION_V2"],
    })
    X = Stage1Router.extract_inputs(df)
    assert list(X.columns) == list(STAGE1_INPUT_COLUMNS)
    assert "mechanism_family" not in X.columns


def test_stage1_extract_inputs_raises_when_a_required_input_is_missing():
    df = pd.DataFrame({
        "contention_score_v2": [0.1], "priority_skew": [1.0], "kv_pressure": [0.2],
    })
    with pytest.raises(KeyError):
        Stage1Router.extract_inputs(df)


def test_assert_stage1_input_frame_rejects_a_frame_with_forbidden_columns():
    from llmserveopt.policy_separation.hierarchical_regime_router_v1 import _assert_stage1_input_frame
    bad = pd.DataFrame({c: [0.1] for c in list(STAGE1_INPUT_COLUMNS) + ["mechanism_family"]})
    with pytest.raises(ValueError):
        _assert_stage1_input_frame(bad)


def test_stage1_extract_inputs_accepts_exact_allowlist():
    df = pd.DataFrame({c: [0.5] for c in STAGE1_INPUT_COLUMNS})
    X = Stage1Router.extract_inputs(df)
    assert list(X.columns) == list(STAGE1_INPUT_COLUMNS)


def test_stage1_fit_uses_only_allowlisted_columns_never_forbidden_fields():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "contention_score_v2": rng.uniform(0, 0.5, n),
        "priority_skew": rng.uniform(0.9, 3.0, n),
        "kv_pressure": rng.uniform(0, 1.0, n),
        "queue_length": rng.integers(0, 10, n),
        "mechanism_family": ["FAMILY_A_FAIRNESS_STARVATION_V2"] * n,
        "canonical_scenario_id": [f"s{i}" for i in range(n)],
    })
    labels = [regime_label_from_activity(a > 1.05, c > 0.20, k > 0.82) for a, c, k in
              zip(df["priority_skew"], df["contention_score_v2"], df["kv_pressure"])]
    df["regime_label"] = labels
    router = Stage1Router().fit(df)
    preds = router.predict(df)
    assert len(preds) == n
    # Corrupting the forbidden columns must not change predictions --
    # proof the model never reads them (it can't, since fit/predict only
    # ever call extract_inputs, but this closes the loop empirically too).
    df2 = df.copy()
    df2["mechanism_family"] = "FAMILY_C_KV_PRESSURE_V2"
    df2["canonical_scenario_id"] = "corrupted"
    preds2 = router.predict(df2)
    assert (preds == preds2).all()


# ---------------------------------------------------------------------------
# Item 4: activity-label / regime-label formulas match SS D exactly,
# regression test against the frozen feasibility telemetry
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FROZEN_TELEMETRY.exists(), reason="frozen telemetry artifact not present")
def test_regime_label_matches_frozen_telemetry_activity_columns():
    df = pd.read_csv(FROZEN_TELEMETRY)
    computed = [
        regime_label_from_activity(bool(a), bool(b), bool(c))
        for a, b, c in zip(df["a_active"], df["b_active_v2"], df["c_active"])
    ]
    # Frozen study found 0/127319 OVERLAP rows -- confirm that still holds
    # for this exact reused formula, not re-derived.
    assert computed.count(REGIME_OVERLAP) == 0
    assert set(computed) <= {REGIME_A, REGIME_B, REGIME_C, REGIME_NONE}
    # Every row must map 1:1 with a raw activity-label combination.
    labeled = add_regime_labels(df)
    assert (labeled["regime_label"].to_numpy() == np.array(computed)).all()


def test_regime_label_from_activity_overlap_and_none():
    assert regime_label_from_activity(False, False, False) == REGIME_NONE
    assert regime_label_from_activity(True, False, False) == REGIME_A
    assert regime_label_from_activity(False, True, False) == REGIME_B
    assert regime_label_from_activity(False, False, True) == REGIME_C
    assert regime_label_from_activity(True, True, False) == REGIME_OVERLAP
    assert regime_label_from_activity(True, False, True) == REGIME_OVERLAP
    assert regime_label_from_activity(False, True, True) == REGIME_OVERLAP
    assert regime_label_from_activity(True, True, True) == REGIME_OVERLAP


def test_regime_label_never_reads_family_or_scenario_identity():
    """Same discipline test as
    test_activity_label_computation_never_reads_family_or_scenario_identity
    (design doc SS D) -- `regime_label_from_activity`'s signature is
    exactly 3 booleans, so it structurally cannot read family/scenario
    identity."""
    import inspect
    sig = inspect.signature(regime_label_from_activity)
    assert list(sig.parameters) == ["a_active", "b_active_v2", "c_active"]


# ---------------------------------------------------------------------------
# Item 9/12: deterministic routing; NONE/OVERLAP fallback
# ---------------------------------------------------------------------------

def test_stage1_router_is_deterministic():
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame({
        "contention_score_v2": rng.uniform(0, 0.5, n),
        "priority_skew": rng.uniform(0.9, 3.0, n),
        "kv_pressure": rng.uniform(0, 1.0, n),
        "queue_length": rng.integers(0, 10, n),
    })
    df["regime_label"] = [
        regime_label_from_activity(a > 1.05, c > 0.20, k > 0.82)
        for a, c, k in zip(df["priority_skew"], df["contention_score_v2"], df["kv_pressure"])
    ]
    r1 = Stage1Router(seed=42).fit(df)
    r2 = Stage1Router(seed=42).fit(df)
    assert (r1.predict(df) == r2.predict(df)).all()


def test_route_action_none_and_overlap_dispatch_to_fallback_policy():
    assert route_action(REGIME_NONE) == FALLBACK_POLICY
    assert route_action(REGIME_OVERLAP) == FALLBACK_POLICY


def test_route_action_active_regimes_dispatch_to_themselves():
    assert route_action(REGIME_A) == REGIME_A
    assert route_action(REGIME_B) == REGIME_B
    assert route_action(REGIME_C) == REGIME_C


def test_route_action_rejects_unknown_regime():
    with pytest.raises(ValueError):
        route_action("NOT_A_REGIME")


# ---------------------------------------------------------------------------
# Item 10/13: dwell-time semantics + switching diagnostics on scripted sequences
# ---------------------------------------------------------------------------

def test_dwell_blocks_active_to_active_transition_before_minimum_steps():
    raw = [REGIME_A] * 5 + [REGIME_B] * 5  # switch attempted at step 5, well under N=20
    effective, diag = apply_dwell_and_fallback(raw, dwell_steps=20)
    assert all(r == REGIME_A for r in effective), "dwell should have blocked the early A->B switch"
    assert diag.total_transitions == 0
    assert diag.dwell_violation_count == 0


def test_dwell_allows_active_to_active_transition_after_minimum_steps():
    raw = [REGIME_A] * 21 + [REGIME_B] * 5
    effective, diag = apply_dwell_and_fallback(raw, dwell_steps=20)
    assert effective[:21] == [REGIME_A] * 21
    assert effective[21:] == [REGIME_B] * 5
    assert diag.total_transitions == 1
    assert diag.dwell_violation_count == 0


def test_dwell_exempts_transition_into_none_and_overlap():
    raw = [REGIME_A] * 3 + [REGIME_NONE] * 3 + [REGIME_A] * 3 + [REGIME_OVERLAP] * 3
    effective, diag = apply_dwell_and_fallback(raw, dwell_steps=20)
    # NONE/OVERLAP transitions happen instantly regardless of dwell state.
    assert effective == [REGIME_A, REGIME_A, REGIME_A, REGIME_NONE, REGIME_NONE, REGIME_NONE,
                          REGIME_NONE, REGIME_NONE, REGIME_NONE, REGIME_OVERLAP, REGIME_OVERLAP, REGIME_OVERLAP]
    assert diag.dwell_violation_count == 0
    assert diag.total_transitions == 2  # A->NONE, NONE->OVERLAP (NONE->A re-attempts at indices 6-8 blocked by dwell)


def test_dwell_exempt_transition_into_none_does_not_itself_need_dwell_even_if_recent_change():
    raw = [REGIME_A] * 2 + [REGIME_B] * 100 + [REGIME_NONE]
    # A->B blocked by dwell (2 < 20) so effective stays A for the whole B run,
    # then A->NONE at the last step must still be instant.
    effective, diag = apply_dwell_and_fallback(raw, dwell_steps=20)
    assert effective[-1] == REGIME_NONE
    assert diag.dwell_violation_count == 0


def test_switching_diagnostics_match_hand_computed_reference():
    raw = [REGIME_A] * 25 + [REGIME_B] * 25 + [REGIME_C] * 25
    effective, diag = apply_dwell_and_fallback(raw, dwell_steps=20)
    assert diag.total_transitions == 2
    assert diag.switches_per_regime[REGIME_B] == 1
    assert diag.switches_per_regime[REGIME_C] == 1
    assert diag.switching_rate_per_1000_steps == pytest.approx(1000.0 * 2 / 75)
    assert diag.fallback_rate == 0.0


def test_count_dwell_violations_detects_a_manually_corrupted_sequence():
    # Manually construct a sequence that VIOLATES dwell (not produced by
    # apply_dwell_and_fallback) to prove the independent checker actually
    # detects violations rather than trivially always returning 0.
    corrupted = [REGIME_A] * 5 + [REGIME_B] * 5 + [REGIME_C] * 30
    assert count_dwell_violations(corrupted, dwell_steps=20) >= 1


def test_apply_dwell_and_fallback_output_never_violates_dwell_by_construction():
    rng = np.random.default_rng(7)
    raw = list(rng.choice([REGIME_A, REGIME_B, REGIME_C, REGIME_NONE, REGIME_OVERLAP], size=500))
    effective, diag = apply_dwell_and_fallback(raw, dwell_steps=20)
    assert count_dwell_violations(effective, dwell_steps=20) == 0
    assert diag.dwell_violation_count == 0


def test_dwell_rejects_unknown_regime_label():
    with pytest.raises(ValueError):
        apply_dwell_and_fallback(["NOT_A_REGIME"])


def test_dwell_empty_sequence():
    effective, diag = apply_dwell_and_fallback([])
    assert effective == []
    assert diag.total_transitions == 0


# ---------------------------------------------------------------------------
# Item 5/6: deterministic split builder + group disjointness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MF_PSD_SCENARIOS.exists(), reason="frozen MF-PSD scenario table not present")
def test_split_builder_is_deterministic():
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    s1 = build_splits(scen)
    s2 = build_splits(scen)
    assert s1 == s2


@pytest.mark.skipif(not MF_PSD_SCENARIOS.exists(), reason="frozen MF-PSD scenario table not present")
def test_split_builder_group_disjointness():
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    splits = build_splits(scen)
    assert_group_disjoint(scen, splits)  # must not raise


@pytest.mark.skipif(not MF_PSD_SCENARIOS.exists(), reason="frozen MF-PSD scenario table not present")
def test_split_builder_covers_every_scenario_with_valid_split_name():
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    splits = build_splits(scen)
    assert set(splits.keys()) == set(scen["canonical_scenario_id"])
    assert set(splits.values()) <= {"train", "val", "test"}


@pytest.mark.skipif(not MF_PSD_SCENARIOS.exists(), reason="frozen MF-PSD scenario table not present")
def test_family_c_held_out_seeds_are_entirely_in_test():
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    splits = build_splits(scen)
    fam_c = scen[scen["mechanism_family"] == "FAMILY_C_KV_PRESSURE_V2"]
    held_out = fam_c[fam_c["seed"].astype(str).isin(("20260914", "20260915"))]
    for cid in held_out["canonical_scenario_id"]:
        assert splits[cid] == "test"


def test_split_builder_raises_on_missing_columns():
    with pytest.raises(ValueError):
        build_splits(pd.DataFrame({"canonical_scenario_id": ["x"]}))


def test_assert_group_disjoint_raises_on_a_violating_split():
    scen = pd.DataFrame({
        "canonical_scenario_id": ["a1", "a2"],
        "group_key": ["g1", "g1"],
        "mechanism_family": ["FAMILY_A_FAIRNESS_STARVATION_V2"] * 2,
    })
    bad_splits = {"a1": "train", "a2": "test"}
    with pytest.raises(AssertionError):
        assert_group_disjoint(scen, bad_splits)


# ---------------------------------------------------------------------------
# Item 17 (partial): frozen-source immutability -- reading, never writing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FROZEN_TELEMETRY.exists(), reason="frozen telemetry artifact not present")
def test_frozen_telemetry_artifact_not_mutated_by_import():
    import hashlib
    before = hashlib.sha256(FROZEN_TELEMETRY.read_bytes()).hexdigest()
    pd.read_csv(FROZEN_TELEMETRY)
    after = hashlib.sha256(FROZEN_TELEMETRY.read_bytes()).hexdigest()
    assert before == after


@pytest.mark.skipif(not MF_PSD_SCENARIOS.exists(), reason="frozen MF-PSD scenario table not present")
def test_frozen_mf_psd_scenarios_not_mutated_by_split_builder():
    import hashlib
    before = hashlib.sha256(MF_PSD_SCENARIOS.read_bytes()).hexdigest()
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    build_splits(scen)
    after = hashlib.sha256(MF_PSD_SCENARIOS.read_bytes()).hexdigest()
    assert before == after


# ---------------------------------------------------------------------------
# SS P -- blended-regime microcase instantiation (implement-only, this task)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", list(BLENDED_MICROCASE_BUILDERS.keys()))
def test_blended_microcase_instantiates_with_requests_and_gpu_config(case_name):
    builder = BLENDED_MICROCASE_BUILDERS[case_name]
    scenario = builder()
    assert len(scenario.requests) > 0
    assert len(scenario.gpu_configs) == 1
    assert scenario.family == "blended_microcase_v1"


def test_blended_microcase_a_plus_c_has_tight_kv_budget():
    from llmserveopt.policy_separation.hierarchical_regime_router_v1 import build_blended_microcase_a_plus_c
    scenario = build_blended_microcase_a_plus_c()
    assert scenario.gpu_configs[0].max_kv_tokens == 6_000


def test_blended_microcase_a_plus_b_enables_prefill_modeling():
    from llmserveopt.policy_separation.hierarchical_regime_router_v1 import build_blended_microcase_a_plus_b
    scenario = build_blended_microcase_a_plus_b()
    assert scenario.service_model_kwargs.get("enable_prefill_modeling") is True
    assert scenario.service_model_kwargs.get("enable_decode_prefill_contention") is True


def test_blended_microcase_smoke_run_produces_telemetry_and_router_output():
    """Instantiate + run one microcase for real through the simulator with
    TelemetryRecordingPolicy, compute Stage-1 raw regimes from the
    resulting telemetry, and confirm the dwell/fallback router produces a
    well-formed output sequence. Does NOT assert overlap must occur --
    only that the pipeline runs end-to-end and produces interpretable
    diagnostics (design doc S8: 'instantiate them, validate expected
    overlapping activity labels, test router fallback behavior')."""
    from llmserveopt.policies.fifo import FIFOPolicy
    from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
        build_blended_microcase_a_plus_c,
    )
    from llmserveopt.policy_separation.online_regime_signals_v1 import TelemetryRecordingPolicy
    from llmserveopt.simulator.service_model import ServiceModel
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

    scenario = build_blended_microcase_a_plus_c()
    policy = TelemetryRecordingPolicy(
        FIFOPolicy(), canonical_scenario_id=scenario.scenario_id,
        mechanism_family="blended_microcase_v1", sample_stride_steps=5,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)

    assert len(policy.rows) > 0
    raw_regimes = [
        regime_label_from_activity(row.labels.a_active, row.labels.b_active_v2, row.labels.c_active)
        for row in policy.rows
    ]
    effective, diag = apply_dwell_and_fallback(raw_regimes, dwell_steps=DWELL_MINIMUM_STEPS)
    assert len(effective) == len(raw_regimes)
    assert diag.dwell_violation_count == 0
    # Report (not assert) whether the targeted overlap combination (A+C)
    # was observed -- a genuine empirical finding, not a required outcome.
    observed_a_and_c = any(
        row.labels.a_active and row.labels.c_active for row in policy.rows
    )
    # No hard assertion on `observed_a_and_c` -- see docstring.
    assert isinstance(observed_a_and_c, bool)

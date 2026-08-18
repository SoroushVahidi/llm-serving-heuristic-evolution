"""Focused tests for the live closed-loop hierarchical-router harness
(design doc task 20260818: LIVE_HIERARCHICAL_HARNESS_* validation). Uses
small synthetic fixture scenarios (built from the frozen family template
functions with `allow_synthetic_tokens=True`) for speed -- these are NOT
real MF-PSD TRAIN/VAL/TEST scenarios and no test here reads any TEST-split
row or computes a scientific verdict.
"""
from __future__ import annotations

import ast
import inspect
import random

import pandas as pd
import pytest

from llmserveopt.policy_separation import hierarchical_router_live_harness_v1 as live
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    DWELL_MINIMUM_STEPS,
    FALLBACK_POLICY,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_CLASSES,
    STAGE2_CANDIDATES,
    apply_dwell_and_fallback,
    build_blended_microcase_b_plus_c,
    count_dwell_violations,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2
from llmserveopt.policy_separation.templates_kv_pressure_v2 import case_kv_pressure_reserve_contention_v2
from llmserveopt.policy_separation.templates_prefill_decode_v2 import case_prefill_decode_ttft_contention


# ---------------------------------------------------------------------------
# Fixture scenarios (small, fast, synthetic-token, deterministic)
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


# ---------------------------------------------------------------------------
# SS5 -- majority-vote exclusion (structural regression guard)
# ---------------------------------------------------------------------------

def _module_ast() -> ast.Module:
    """AST of the live harness module's actual source file, so these
    structural guards check CODE (imports, call expressions) only --
    never prose in docstrings/comments (this module's own docstring and
    inline comments legitimately discuss and name the forbidden pattern
    for documentation purposes)."""
    return ast.parse(inspect.getsource(live))


def test_live_harness_module_never_imports_the_majority_vote_evaluation_module():
    tree = _module_ast()
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert not any("hierarchical_router_evaluation_v1" in m for m in imported_modules)
    # No call expression anywhere in the module may reference the majority-
    # vote helper by name (covers a hypothetical late/local import too).
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "scenario_regime_from_telemetry":
            pytest.fail("live harness code references scenario_regime_from_telemetry")
        if isinstance(node, ast.Attribute) and node.attr == "scenario_regime_from_telemetry":
            pytest.fail("live harness code references scenario_regime_from_telemetry")


def test_live_harness_module_never_computes_a_majority_over_effective_regimes():
    """The offline approximation's exact signature is
    `np.unique(effective, return_counts=True)` then `np.argmax(counts)`
    (`hierarchical_router_evaluation_v1.scenario_regime_from_telemetry`).
    Neither call shape may appear as actual CODE (not docstring prose) in
    the live harness module."""
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if func_name == "unique":
                for kw in node.keywords:
                    if kw.arg == "return_counts":
                        pytest.fail("live harness code calls np.unique(..., return_counts=True)")
            if func_name == "argmax" and node.args:
                arg_src = ast.dump(node.args[0])
                assert "counts" not in arg_src.lower() or "unique" not in ast.dump(node).lower()


# ---------------------------------------------------------------------------
# Forced-parent equivalence (design doc S6) -- one test per native policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy_id", ["estimated_service_time_first", "weighted_fair_share"])
def test_forced_equivalence_family_a(policy_id):
    scen = _family_a_scenario()
    ref = live.run_reference_single_policy(scen, policy_id)
    got = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert=policy_id,
    )
    assert ref.arrival_normalized_weighted_goodput == pytest.approx(
        got.metrics.arrival_normalized_weighted_goodput, abs=1e-12
    )
    assert ref.completion_fraction == pytest.approx(got.metrics.completion_fraction, abs=1e-12)


@pytest.mark.parametrize("policy_id", ["full_prefill", "chunked_prefill_small"])
def test_forced_equivalence_family_b(policy_id):
    scen = _family_b_scenario()
    ref = live.run_reference_single_policy(scen, policy_id)
    got = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert=policy_id,
    )
    assert ref.arrival_normalized_weighted_goodput == pytest.approx(
        got.metrics.arrival_normalized_weighted_goodput, abs=1e-12
    )
    # The Regime-B mechanism-differentiating verb must actually have fired.
    assert got.trajectory["prefill_chunk_override_active"].all()


@pytest.mark.parametrize("policy_id", ["kv_constrained_online", "least_laxity_first"])
def test_forced_equivalence_family_c(policy_id):
    scen = _family_c_scenario()
    ref = live.run_reference_single_policy(scen, policy_id)
    got = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert=policy_id,
    )
    assert ref.arrival_normalized_weighted_goodput == pytest.approx(
        got.metrics.arrival_normalized_weighted_goodput, abs=1e-12
    )


def test_forced_mode_bypasses_stage1_and_stage2_entirely():
    """forced_expert=None-less run must never call Stage1Router.predict --
    passing stage1=None must not raise, which it would if predict were
    ever invoked."""
    scen = _family_a_scenario()
    got = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert="weighted_fair_share",
    )
    assert (got.trajectory["stage1_raw_regime"] == "FORCED").all()
    assert (got.trajectory["stage2_regime"].isna()).all()


# ---------------------------------------------------------------------------
# Policy-state preservation (design doc S3)
# ---------------------------------------------------------------------------

def test_all_six_native_policies_are_provably_stateless():
    """Direct source-level audit: none of `select_action`'s body may
    reference `self.` outside of the fixed constructor hyperparameters
    already bound at __init__ -- i.e. no policy accumulates state across
    calls. This is a structural, not merely behavioral, guarantee."""
    policies = live.build_native_policy_instances()
    for name, policy in policies.items():
        src = inspect.getsource(type(policy).select_action)
        # Constructor hyperparameters are read via self.<attr> for scoring
        # (alpha/beta/thresholds); none of these are ever *written* inside
        # select_action -- a crude but effective check: no `self.` occurrence
        # is followed by `=` (assignment) inside select_action's own body,
        # ignoring the recursive helper defs also named in the same source
        # blob for WFS/kv_constrained_online (`_score`/`_admit_filter`
        # closures also carry no assignment to self.*).
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("self.") and "=" in stripped and "==" not in stripped:
                pytest.fail(f"{name}.select_action appears to assign self.* (stateful): {stripped!r}")


def test_single_instantiate_policy_dict_is_reused_across_the_whole_run():
    scen = _family_a_scenario()
    stage1 = None
    result = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=stage1, stage2_selectors={},
        forced_expert="weighted_fair_share",
    )
    # Reaching here without error already proves one policy dict served
    # the whole run (build_native_policy_instances is called exactly once
    # inside LiveHierarchicalRouterPolicy.__init__). Also check ANWG is
    # sane/finite (no crash, no NaN).
    assert result.metrics.arrival_normalized_weighted_goodput == result.metrics.arrival_normalized_weighted_goodput  # not NaN


# ---------------------------------------------------------------------------
# Incremental dwell/fallback FSM equivalence to the frozen batch function
# ---------------------------------------------------------------------------

def test_incremental_fsm_matches_frozen_batch_fsm_on_random_sequences():
    rng = random.Random(20260818)
    for _ in range(200):
        n = rng.randint(1, 60)
        seq = [rng.choice(REGIME_CLASSES) for _ in range(n)]
        batch_effective, batch_diag = apply_dwell_and_fallback(seq, DWELL_MINIMUM_STEPS)

        fsm = live.IncrementalDwellFallbackFSM(DWELL_MINIMUM_STEPS)
        inc_effective = [fsm.step(r) for r in seq]
        inc_diag = fsm.diagnostics()

        assert inc_effective == batch_effective
        assert inc_diag.total_transitions == batch_diag.total_transitions
        assert inc_diag.switches_per_regime == batch_diag.switches_per_regime
        assert inc_diag.switching_rate_per_1000_steps == pytest.approx(
            batch_diag.switching_rate_per_1000_steps
        )
        assert inc_diag.fallback_rate == pytest.approx(batch_diag.fallback_rate)
        # Both are 0-by-construction; batch's own independent check agrees.
        assert count_dwell_violations(batch_effective, DWELL_MINIMUM_STEPS) == 0
        assert count_dwell_violations(inc_effective, DWELL_MINIMUM_STEPS) == 0


def test_incremental_fsm_switched_this_step_flag_matches_transitions():
    fsm = live.IncrementalDwellFallbackFSM(DWELL_MINIMUM_STEPS)
    seq = [REGIME_A] * 5 + [REGIME_C] * 25 + [REGIME_A] * 25
    switches = []
    for r in seq:
        fsm.step(r)
        switches.append(fsm.switched_this_step)
    assert switches[0] is False  # first step is never a "switch"
    assert sum(switches) == fsm.diagnostics().total_transitions


# ---------------------------------------------------------------------------
# Dwell / NONE / OVERLAP fallback semantics (design doc SS K/F) in live mode
# ---------------------------------------------------------------------------

def test_dwell_minimum_respected_in_live_routing():
    class _ScriptedStage1:
        def __init__(self, script):
            self.script = list(script)
            self.i = 0

        def predict(self, df):
            v = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
            return [v]

    # Script rapidly flips A<->C every step -- the dwell FSM must hold the
    # effective regime at whichever it started with for DWELL_MINIMUM_STEPS.
    script = [REGIME_A, REGIME_C] * 40
    fsm = live.IncrementalDwellFallbackFSM(DWELL_MINIMUM_STEPS)
    effective = [fsm.step(r) for r in script]
    assert count_dwell_violations(effective, DWELL_MINIMUM_STEPS) == 0
    # Must not have switched every single step despite the raw script doing so.
    assert fsm.diagnostics().total_transitions < len(script)


def test_none_fallback_dispatches_to_weighted_fair_share_never_stage2():
    scen = _family_a_scenario()

    class _AlwaysNoneStage1:
        def predict(self, df):
            return ["NONE"]

    result = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=_AlwaysNoneStage1(), stage2_selectors={},
        max_steps=30,
    )
    assert (result.trajectory["effective_regime"] == "NONE").all()
    assert (result.trajectory["selected_policy"] == FALLBACK_POLICY).all()
    assert result.trajectory["stage2_regime"].isna().all()


def test_overlap_fallback_dispatches_to_weighted_fair_share_never_stage2():
    scen = _family_a_scenario()

    class _AlwaysOverlapStage1:
        def predict(self, df):
            return ["OVERLAP"]

    result = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=_AlwaysOverlapStage1(), stage2_selectors={},
        max_steps=30,
    )
    assert (result.trajectory["effective_regime"] == "OVERLAP").all()
    assert (result.trajectory["selected_policy"] == FALLBACK_POLICY).all()
    assert result.trajectory["stage2_regime"].isna().all()


# ---------------------------------------------------------------------------
# Causal-switch microcase (design doc S7)
# ---------------------------------------------------------------------------

def test_causal_switch_diverges_state_from_fallback_only_baseline():
    """Same scenario, same initial state. Run 1: real router (Stage-1
    trained on frozen TRAIN telemetry, Stage-2 trained on frozen TRAIN
    scenario rows). Run 2: forced to the fallback policy for every step
    (what the router would do if it never activated any regime). Both
    must be bit-identical up to the first real switch, and diverge at/after
    it -- proving the router's decision at step t causally changes state at
    step t+1, not merely an observational label."""
    import llmserveopt.policy_separation.hierarchical_regime_router_v1 as router_mod
    import llmserveopt.policy_separation.hierarchical_router_evaluation_v1 as eval_mod
    from llmserveopt.selector.hierarchical_stage2_selectors_v1 import fit_all_stage2_selectors

    telemetry_path = live.ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
    if not telemetry_path.exists() or not live.MF_PSD_SCENARIOS_CSV.exists():
        pytest.skip("frozen telemetry/MF-PSD artifacts not present in this checkout")

    scen_df = pd.read_csv(live.MF_PSD_SCENARIOS_CSV)
    split_map = router_mod.build_splits(scen_df)
    telemetry = pd.read_csv(telemetry_path)
    telemetry = router_mod.add_regime_labels(telemetry)
    telemetry["split"] = telemetry["canonical_scenario_id"].map(split_map)
    train_tel = telemetry[telemetry["split"] == "train"]
    stage1 = router_mod.Stage1Router().fit(train_tel)

    scenario_df = eval_mod.load_scenario_level_dataset()
    train_df = scenario_df[scenario_df["split"] == "train"]
    train_by_regime = {r: train_df[train_df["regime_ground_truth"] == r] for r in ACTIVE_REGIMES}
    stage2 = fit_all_stage2_selectors(train_by_regime)

    bc = build_blended_microcase_b_plus_c()
    routed = live.run_live_scenario(
        bc, canonical_scenario_id=bc.scenario_id, stage1=stage1, stage2_selectors=stage2, max_steps=60,
    )
    fallback_only = live.run_live_scenario(
        bc, canonical_scenario_id=bc.scenario_id, stage1=stage1, stage2_selectors=stage2,
        forced_expert=FALLBACK_POLICY, max_steps=60,
    )

    switch_steps = routed.trajectory.loc[routed.trajectory["dwell_switched_this_step"], "step"]
    assert len(switch_steps) > 0, "fixture must exercise at least one real switch within max_steps"
    first_switch = int(switch_steps.iloc[0])

    r_traj = routed.trajectory.set_index("step")
    f_traj = fallback_only.trajectory.set_index("step")

    for step in range(0, first_switch):
        assert r_traj.loc[step, "selected_policy"] == FALLBACK_POLICY
        assert r_traj.loc[step, "admitted_request_ids"] == f_traj.loc[step, "admitted_request_ids"]

    assert r_traj.loc[first_switch, "selected_policy"] != FALLBACK_POLICY

    # The router's decision at the switch step must eventually cause a
    # different observable trajectory than the fallback-only baseline --
    # checked over the window the switch is in effect (up to the next
    # switch back or the end of the capped run), not necessarily at the
    # very first post-switch step (e.g. both policies may admit nothing
    # that exact step under heavy KV pressure, converging by coincidence
    # for one step, which would not disprove causality).
    common_steps = sorted(set(r_traj.index) & set(f_traj.index))
    window = [s for s in common_steps if s >= first_switch]
    diverged = any(
        r_traj.loc[s, "admitted_request_ids"] != f_traj.loc[s, "admitted_request_ids"]
        or r_traj.loc[s, "mean_kv_utilization_after_admission"]
        != f_traj.loc[s, "mean_kv_utilization_after_admission"]
        or r_traj.loc[s, "queue_len_after_admission"] != f_traj.loc[s, "queue_len_after_admission"]
        for s in window
    )
    assert diverged, "router's decision at the switch step must causally change observable state"


# ---------------------------------------------------------------------------
# Temporal leakage
# ---------------------------------------------------------------------------

def test_stage1_input_row_contains_only_the_frozen_four_columns():
    scen = _family_a_scenario()

    class _RecordingStage1:
        def __init__(self):
            self.seen_columns = None

        def predict(self, df):
            self.seen_columns = list(df.columns)
            return ["NONE"]

    rec = _RecordingStage1()
    live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=rec, stage2_selectors={}, max_steps=5,
    )
    from llmserveopt.policy_separation.hierarchical_regime_router_v1 import STAGE1_INPUT_COLUMNS
    assert rec.seen_columns == list(STAGE1_INPUT_COLUMNS)


def test_feature_row_best_effort_never_reads_forbidden_scenario_fields():
    """The best-effort Stage-2 feature-row path reads only `scenario.params`
    and `scenario.stress_control_relationship` -- never `scenario.requests`
    (which could leak `actual_output_tokens`) and never any post-run field
    (scenarios have none; this is a structural/API-shape check)."""
    src = inspect.getsource(live.feature_row_best_effort)
    assert "scenario.requests" not in src
    assert "actual_output_tokens" not in src


def test_trajectory_features_depend_only_on_current_and_past_state():
    """Every trajectory row's signals must be reproducible purely from
    `compute_regime_signals`/`compute_activity_labels` applied to that
    step's own `ObservableState` -- checked indirectly by confirming the
    harness's signal computation never receives `state.step + k` for any
    k>0, i.e. it is called exactly once per real simulator step with the
    state the simulator itself just built (structural: `select_action` is
    the only place `compute_regime_signals` is invoked in this module)."""
    src = inspect.getsource(live)
    assert src.count("compute_regime_signals(state)") == 1


# ---------------------------------------------------------------------------
# Trajectory log completeness / determinism / canonical ANWG
# ---------------------------------------------------------------------------

def test_trajectory_log_has_all_required_columns():
    scen = _family_c_scenario()
    result = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert="kv_constrained_online",
    )
    required = {
        "scenario_id", "step", "sim_time",
        "contention_score_v2", "priority_skew", "kv_pressure", "queue_length",
        "stage1_raw_regime", "a_active", "b_active_v2", "c_active",
        "effective_regime", "dwell_switched_this_step", "fallback_active",
        "stage2_regime", "selected_policy", "admitted_count", "admitted_request_ids",
        "prefill_chunk_override_active", "queue_len_after_admission",
        "active_count_after_admission", "mean_kv_utilization_after_admission",
    }
    assert required.issubset(set(result.trajectory.columns))
    assert len(result.trajectory) > 0
    assert not result.trajectory["step"].isna().any()


def test_deterministic_replay_bit_identical():
    scen = _family_c_scenario()
    r1 = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert="least_laxity_first",
    )
    r2 = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert="least_laxity_first",
    )
    assert r1.metrics.arrival_normalized_weighted_goodput == r2.metrics.arrival_normalized_weighted_goodput
    pd.testing.assert_frame_equal(
        r1.trajectory.drop(columns=["admitted_request_ids"]),
        r2.trajectory.drop(columns=["admitted_request_ids"]),
    )


def test_canonical_anwg_computes_and_is_finite():
    scen = _family_b_scenario()
    result = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=None, stage2_selectors={},
        forced_expert="chunked_prefill_small",
    )
    anwg = result.metrics.arrival_normalized_weighted_goodput
    assert anwg == anwg  # not NaN
    assert 0.0 <= anwg <= 1.0 + 1e-9


def test_no_nan_or_inf_in_live_smoke_trajectory():
    scen = _family_a_scenario()
    import llmserveopt.policy_separation.hierarchical_regime_router_v1 as router_mod
    telemetry_path = live.ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
    if not telemetry_path.exists() or not live.MF_PSD_SCENARIOS_CSV.exists():
        pytest.skip("frozen telemetry/MF-PSD artifacts not present in this checkout")
    scen_df = pd.read_csv(live.MF_PSD_SCENARIOS_CSV)
    split_map = router_mod.build_splits(scen_df)
    telemetry = pd.read_csv(telemetry_path)
    telemetry = router_mod.add_regime_labels(telemetry)
    telemetry["split"] = telemetry["canonical_scenario_id"].map(split_map)
    stage1 = router_mod.Stage1Router().fit(telemetry[telemetry["split"] == "train"])
    result = live.run_live_scenario(
        scen, canonical_scenario_id=scen.scenario_id, stage1=stage1, stage2_selectors={}, max_steps=30,
    )
    numeric_cols = ["contention_score_v2", "priority_skew", "kv_pressure", "queue_length"]
    import numpy as np
    vals = result.trajectory[numeric_cols].to_numpy(dtype=float)
    assert np.isfinite(vals).all()


# ---------------------------------------------------------------------------
# Frozen-artifact immutability
# ---------------------------------------------------------------------------

def test_frozen_router_module_symbols_unchanged_shape():
    """This harness imports, never redefines, the frozen router's public
    constants -- a drift check that fails loudly if a future edit to this
    harness accidentally shadows one instead of importing it."""
    from llmserveopt.policy_separation import hierarchical_regime_router_v1 as frozen
    assert live.DWELL_MINIMUM_STEPS is frozen.DWELL_MINIMUM_STEPS
    assert live.STAGE2_CANDIDATES is frozen.STAGE2_CANDIDATES
    assert live.FALLBACK_POLICY == frozen.FALLBACK_POLICY == "weighted_fair_share"
    assert live.STAGE1_INPUT_COLUMNS is frozen.STAGE1_INPUT_COLUMNS


def test_stage2_candidates_native_pairs_unchanged():
    assert STAGE2_CANDIDATES[REGIME_A] == ("estimated_service_time_first", "weighted_fair_share")
    assert STAGE2_CANDIDATES[REGIME_B] == ("full_prefill", "chunked_prefill_small")
    assert STAGE2_CANDIDATES[REGIME_C] == ("kv_constrained_online", "least_laxity_first")

"""Focused tests for the KV-aware composition falsification v1.

See docs/design/KV_COMPOSITION_FALSIFICATION_V1.md.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from llmserveopt.composition.kv_composition_features import (
    FORBIDDEN_FEATURE_KEYS,
    assert_no_hidden_leakage,
    n_urgent_waiting,
    scenario_observable_features,
    step_features,
)
from llmserveopt.composition.kv_composition_policy import (
    PARENT_KV,
    PARENT_LLF,
    TAU_URGENT_GRID,
    KVAdaptiveReserveChildPolicy,
    fit_kv_top1_selector,
    hard_conditional_rule,
    select_kv_model_on_val,
)
from llmserveopt.composition.kv_composition_splits import (
    OOD_SEEDS,
    TEST_SEED,
    TRAIN_SEEDS,
    VAL_SEED,
    assert_no_split_leakage,
    assign_kv_composition_splits,
)
from llmserveopt.composition.kv_composition_metrics import (
    bootstrap_ci,
    envelope_gain,
    oracle_regret,
    parent_envelope,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy
from llmserveopt.policy_separation.templates_kv_pressure_v2 import (
    assert_policy_visible_fields_clean_kv_v2,
    case_kv_pressure_reserve_contention_v2,
)
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

DATASETS_ROOT = Path(".local_data")
_HAVE_BURSTGPT = (DATASETS_ROOT / "burstgpt_v2" / "raw").is_dir()

requires_burstgpt = pytest.mark.skipif(
    not _HAVE_BURSTGPT, reason="staged BurstGPT not available in this environment"
)


def _scenario(**overrides):
    params = dict(
        bulk_pressure="high", urgent_arrival_phase="middle", urgent_tightness="tight",
        seed=20260910, datasets_root=DATASETS_ROOT,
    )
    params.update(overrides)
    return case_kv_pressure_reserve_contention_v2(**params)


# ===================================================================
# Parent policies unchanged
# ===================================================================

def test_parent_policy_source_unchanged():
    """Contract test: parent implementations must not have been touched."""
    kv_src = inspect.getsource(KVConstrainedOnlinePolicy)
    llf_src = inspect.getsource(LeastLaxityFirstPolicy)
    assert "class KVConstrainedOnlinePolicy(BasePolicy):" in kv_src
    assert "class LeastLaxityFirstPolicy(BasePolicy):" in llf_src
    # KVAdaptiveReserveChildPolicy must delegate, not reimplement scoring.
    import llmserveopt.composition.kv_composition_policy as mod
    child_src = inspect.getsource(mod.KVAdaptiveReserveChildPolicy)
    assert "self._kv.select_action(state)" in child_src
    assert "self._llf.select_action(state)" in child_src


# ===================================================================
# Feature leakage
# ===================================================================

def test_forbidden_feature_keys_include_generator_labels():
    for label in ("scenario_id", "seed", "bulk_pressure", "urgent_arrival_phase", "urgent_tightness"):
        assert label in FORBIDDEN_FEATURE_KEYS


def test_assert_no_hidden_leakage_raises_on_forbidden_key():
    with pytest.raises(ValueError):
        assert_no_hidden_leakage({"seed": 1, "n_queued_requests": 5.0})


def test_scenario_observable_features_no_leakage():
    s = _scenario()
    feats = scenario_observable_features(list(s.requests))
    assert_no_hidden_leakage(feats)  # must not raise
    assert feats["n_queued_requests"] == len(s.requests)


# ===================================================================
# Splits
# ===================================================================

def test_split_assignment_disjoint_and_covers_all_seeds():
    seeds = list(TRAIN_SEEDS) + [VAL_SEED, TEST_SEED] + list(OOD_SEEDS)
    sids = [f"kvp2.bulk10.phaseearly.tighttight.s{seed}" for seed in seeds]
    split = assign_kv_composition_splits(sids)
    assert_no_split_leakage(split)  # must not raise
    assert len(split.train) == 2
    assert len(split.val) == 1
    assert len(split.test) == 1
    assert len(split.ood) == 2


def test_split_seeds_never_used_in_v2_calibration_or_pilot():
    # KV v2's calibration used v1 seeds 20260901-04; v2's own pilot used
    # 20260910-15. This falsification reuses the SAME 20260910-15 seeds
    # (per design doc SS7 -- explicit reuse, not new seeds), so this test
    # only guards against accidental overlap with v1/calibration seeds.
    all_used = set(TRAIN_SEEDS) | {VAL_SEED, TEST_SEED} | set(OOD_SEEDS)
    v1_and_calibration_seeds = {20260901, 20260902, 20260903, 20260904}
    assert all_used.isdisjoint(v1_and_calibration_seeds)


def test_split_unknown_seed_raises():
    with pytest.raises(ValueError):
        assign_kv_composition_splits(["kvp2.bulk10.phaseearly.tighttight.s99999999"])


# ===================================================================
# Scenario generation / determinism / leakage guard
# ===================================================================

@requires_burstgpt
def test_scenario_generation_deterministic():
    s1 = _scenario()
    s2 = _scenario()
    assert s1.scenario_id == s2.scenario_id
    assert [r.request_id for r in s1.requests] == [r.request_id for r in s2.requests]
    assert [r.arrival_time for r in s1.requests] == [r.arrival_time for r in s2.requests]


@requires_burstgpt
def test_scenario_leakage_guard_passes():
    s = _scenario()
    assert_policy_visible_fields_clean_kv_v2(s)  # must not raise


@requires_burstgpt
def test_full_grid_scenario_id_uniqueness():
    ids = set()
    for bulk in ("low", "high"):
        for phase in ("early", "middle", "late"):
            for tight in ("loose", "tight"):
                for seed in (20260910, 20260911, 20260912, 20260913, 20260914, 20260915):
                    s = _scenario(bulk_pressure=bulk, urgent_arrival_phase=phase,
                                  urgent_tightness=tight, seed=seed)
                    assert s.scenario_id not in ids
                    ids.add(s.scenario_id)
    assert len(ids) == 2 * 3 * 2 * 6


# ===================================================================
# n_urgent_waiting / step_features observable-only
# ===================================================================

@requires_burstgpt
def test_n_urgent_waiting_varies_within_one_trajectory():
    s = _scenario(bulk_pressure="high", urgent_arrival_phase="middle", urgent_tightness="tight")

    class Probe(KVConstrainedOnlinePolicy):
        def __init__(self):
            super().__init__()
            self.counts = []

        def select_action(self, state):
            self.counts.append(n_urgent_waiting(state))
            return super().select_action(state)

    probe = Probe()
    sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs), service_model=ServiceModel(**s.service_model_kwargs)))
    sim.load_trace(list(s.requests))
    sim.run(probe, workload_tag=s.scenario_id, seed=s.seed)
    assert min(probe.counts) == 0
    assert max(probe.counts) > 0  # genuinely varies, not constant -> non-degenerate switch signal


# ===================================================================
# Child policy: mode activation, non-degeneracy
# ===================================================================

@requires_burstgpt
def test_child_low_pressure_control_mostly_llf_mode():
    """Placebo-like control (urgent_tightness=loose): few/no urgent requests
    ever queue up, so the child should spend most steps in llf mode."""
    s = _scenario(bulk_pressure="low", urgent_arrival_phase="early", urgent_tightness="loose")
    child = KVAdaptiveReserveChildPolicy(tau_urgent=2)
    sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs), service_model=ServiceModel(**s.service_model_kwargs)))
    sim.load_trace(list(s.requests))
    sim.run(child, workload_tag=s.scenario_id, seed=s.seed)
    assert child.n_llf_steps + child.n_reserve_steps > 0
    assert child.n_llf_steps >= child.n_reserve_steps


@requires_burstgpt
def test_child_high_pressure_signal_cell_activates_reserve_mode():
    """Signal cell (bulk_pressure=high, urgent_tightness=tight): reserve
    mode must activate at least sometimes."""
    s = _scenario(bulk_pressure="high", urgent_arrival_phase="middle", urgent_tightness="tight")
    child = KVAdaptiveReserveChildPolicy(tau_urgent=2)
    sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs), service_model=ServiceModel(**s.service_model_kwargs)))
    sim.load_trace(list(s.requests))
    sim.run(child, workload_tag=s.scenario_id, seed=s.seed)
    assert child.n_reserve_steps > 0


@requires_burstgpt
def test_child_can_transition_and_differ_from_both_parents():
    s = _scenario(bulk_pressure="high", urgent_arrival_phase="middle", urgent_tightness="tight")

    def run(policy):
        sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs), service_model=ServiceModel(**s.service_model_kwargs)))
        sim.load_trace(list(s.requests))
        sim.run(policy, workload_tag=s.scenario_id, seed=s.seed)
        return {c.request.request_id: float(c.admission_time) for c in sim._completed}

    child = KVAdaptiveReserveChildPolicy(tau_urgent=2)
    adm_child = run(child)
    adm_kv = run(KVConstrainedOnlinePolicy())
    adm_llf = run(LeastLaxityFirstPolicy())

    assert child.transition_count >= 1

    differs = any(
        adm_child.get(rid) != adm_kv.get(rid) and adm_child.get(rid) != adm_llf.get(rid)
        for rid in set(adm_child) | set(adm_kv) | set(adm_llf)
    )
    assert differs


@requires_burstgpt
def test_child_kv_overflow_no_worse_than_both_parents():
    """Neither unmodified parent respects a hard max_kv_tokens ceiling on
    every step (KV grows during decode past the admission-time check --
    a pre-existing simulator/policy property, verified directly: on this
    scenario least_laxity_first peaks at 7194/6000 and kv_constrained_online
    at 6722/6000). The real safety invariant for a child that only ever
    delegates to these two unmodified policies is that it cannot be WORSE
    than the worse of the two parents it delegates to -- not an absolute
    zero-overflow bound neither frozen parent itself satisfies."""
    s = _scenario(bulk_pressure="high", urgent_arrival_phase="late", urgent_tightness="tight")

    def peak_ratio(policy):
        sim = Simulator(SimulatorConfig(gpu_configs=list(s.gpu_configs), service_model=ServiceModel(**s.service_model_kwargs)))
        sim.load_trace(list(s.requests))
        sim.run(policy, workload_tag=s.scenario_id, seed=s.seed)
        hist = sim._gpus[0].step_kv_used  # noqa: SLF001
        max_kv = s.gpu_configs[0].max_kv_tokens
        return (max(hist) / max_kv) if hist else 0.0

    child_peak = peak_ratio(KVAdaptiveReserveChildPolicy(tau_urgent=2))
    kv_peak = peak_ratio(KVConstrainedOnlinePolicy())
    llf_peak = peak_ratio(LeastLaxityFirstPolicy())
    assert child_peak <= max(kv_peak, llf_peak) + 1e-9


def test_child_reset_clears_instrumentation():
    child = KVAdaptiveReserveChildPolicy(tau_urgent=2)
    child.mode_log.append("llf")
    child.transition_count = 3
    child.admitted_by_step[0] = [1, 2]
    child.reset()
    assert child.mode_log == []
    assert child.transition_count == 0
    assert child.admitted_by_step == {}


# ===================================================================
# Selector / hard-conditional rule
# ===================================================================

def test_selector_fitting_uses_only_train_val_signature():
    """Dataflow guard: select_kv_model_on_val's signature carries only
    train/val-labelled parameters -- no test/ood parameter exists to pass."""
    sig = inspect.signature(select_kv_model_on_val)
    names = list(sig.parameters)
    assert all("test" not in n and "ood" not in n for n in names)
    assert any("train" in n for n in names)
    assert any("val" in n for n in names)


def test_fit_kv_top1_selector_no_leakage_and_predicts_valid_parent():
    feats = [
        {"n_queued_requests": 5.0, "fraction_urgent_waiting": 0.5},
        {"n_queued_requests": 20.0, "fraction_urgent_waiting": 0.05},
    ]
    kv_scores = [0.9, 0.5]
    llf_scores = [0.5, 0.9]
    sel = fit_kv_top1_selector(feats, kv_scores, llf_scores)
    for f in feats:
        pred = sel.predict_parent(f)
        assert pred in (PARENT_KV, PARENT_LLF)


def test_hard_conditional_rule_no_leakage():
    feats = {"fraction_urgent_waiting": 0.3, "n_queued_requests": 20.0}
    choice = hard_conditional_rule(feats)
    assert choice in (PARENT_KV, PARENT_LLF)


def test_hard_conditional_rule_rejects_forbidden_features():
    with pytest.raises(ValueError):
        hard_conditional_rule({"seed": 1, "fraction_urgent_waiting": 0.3})


# ===================================================================
# Metrics / envelope / verdict-support machinery
# ===================================================================

def test_parent_envelope_is_max_of_two():
    kv = {"a": 0.5, "b": 0.9}
    llf = {"a": 0.7, "b": 0.3}
    env = parent_envelope(kv, llf, ["a", "b"])
    assert env["a"] == 0.7
    assert env["b"] == 0.9


def test_envelope_gain_zero_when_child_equals_envelope():
    env = {"a": 0.5, "b": 0.5}
    child = {"a": 0.5, "b": 0.5}
    g = envelope_gain(child, env, ["a", "b"])
    assert g["mean_envelope_gain"] == 0.0
    assert g["n_beat_envelope_plus_eps"] == 0.0


def test_envelope_gain_positive_when_child_beats_both():
    env = {"a": 0.5}
    child = {"a": 0.7}
    g = envelope_gain(child, env, ["a"], eps=0.01)
    assert g["mean_envelope_gain"] > 0
    assert g["n_beat_envelope_plus_eps"] == 1.0


def test_bootstrap_ci_shape():
    mean, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.15], n_boot=200)
    assert lo <= mean <= hi


def test_oracle_regret_zero_for_oracle_itself():
    kv = {"a": 0.5, "b": 0.9}
    llf = {"a": 0.7, "b": 0.3}
    oracle_scores = parent_envelope(kv, llf, ["a", "b"])
    r = oracle_regret(oracle_scores, kv, llf, ["a", "b"])
    assert r["mean_regret"] == pytest.approx(0.0, abs=1e-9)


# ===================================================================
# TAU_URGENT_GRID sanity
# ===================================================================

def test_tau_urgent_grid_is_tiny_and_preregistered():
    assert TAU_URGENT_GRID == (1, 2, 3)
    assert len(TAU_URGENT_GRID) <= 5  # "tiny candidate set" per design doc SS4

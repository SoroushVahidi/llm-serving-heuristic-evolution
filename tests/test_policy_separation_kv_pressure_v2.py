"""Focused tests for Family C v2 KV-pressure reserve refinement.

See docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md. Covers only
what changed from v1 (population size, BULK_SLACK_S fix, third phase level,
held-out seed wiring) -- see test_policy_separation_kv_pressure_v1.py for
mechanism-level coverage shared with v1 (that file is unmodified/frozen).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.policy_separation.templates_kv_pressure_v2 import (
    N_BULK_V2,
    N_URGENT_V2,
    URGENT_ARRIVAL_PHASE_FRACTION,
    assert_policy_visible_fields_clean_kv_v2,
    case_kv_pressure_reserve_contention_v2,
)
from llmserveopt.policy_separation.templates_kv_pressure import CLASS_BULK, CLASS_URGENT
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy
from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.simulator.service_model import ServiceModel

from run_policy_separation_kv_pressure_pilot_v1 import (  # noqa: E402
    InstrumentedKVConstrainedOnlinePolicy,
    build_scenarios,
)


def _make_v2(**overrides):
    args = dict(
        bulk_pressure="high",
        urgent_arrival_phase="middle",
        urgent_tightness="tight",
        seed=20260910,
        allow_synthetic_tokens=True,
    )
    args.update(overrides)
    return case_kv_pressure_reserve_contention_v2(**args)


class TestParentPoliciesUnmodified:
    """v2 must not alter kv_constrained_online / least_laxity_first at all."""

    def test_kv_constrained_online_contract(self):
        p = KVConstrainedOnlinePolicy()
        assert p.target_kv_utilization == 0.82
        assert p.urgent_laxity_seconds == 0.25
        assert p.alpha == DEFAULT_ALPHA
        assert p.beta == DEFAULT_BETA

    def test_least_laxity_first_contract(self):
        p = LeastLaxityFirstPolicy()
        assert p.alpha == DEFAULT_ALPHA
        assert p.beta == DEFAULT_BETA
        assert not hasattr(p, "target_kv_utilization")
        assert not hasattr(p, "_admit_filter")


class TestV2ScenarioGeneration:
    def test_population_larger_than_v1(self):
        scen = _make_v2(bulk_pressure="high")
        assert N_BULK_V2["high"] > 14  # v1's N_BULK["high"]
        assert N_URGENT_V2 > 6  # v1's N_URGENT
        bulk = [r for r in scen.requests if r.class_id == CLASS_BULK]
        urgent = [r for r in scen.requests if r.class_id == CLASS_URGENT]
        assert len(bulk) == N_BULK_V2["high"]
        assert len(urgent) == N_URGENT_V2

    def test_three_phase_levels(self):
        assert set(URGENT_ARRIVAL_PHASE_FRACTION) == {"early", "middle", "late"}
        assert URGENT_ARRIVAL_PHASE_FRACTION["early"] < URGENT_ARRIVAL_PHASE_FRACTION["middle"]
        assert URGENT_ARRIVAL_PHASE_FRACTION["middle"] < URGENT_ARRIVAL_PHASE_FRACTION["late"]

    def test_middle_phase_arrival_between_early_and_late(self):
        early = _make_v2(urgent_arrival_phase="early")
        middle = _make_v2(urgent_arrival_phase="middle")
        late = _make_v2(urgent_arrival_phase="late")
        e = min(r.arrival_time for r in early.requests if r.class_id == CLASS_URGENT)
        m = min(r.arrival_time for r in middle.requests if r.class_id == CLASS_URGENT)
        l = min(r.arrival_time for r in late.requests if r.class_id == CLASS_URGENT)
        assert e < m < l

    def test_invalid_middle_variant_rejected(self):
        with pytest.raises(ValueError):
            _make_v2(urgent_arrival_phase="midway")

    def test_bulk_tenants_are_mostly_not_urgent(self):
        """Regression test for the v1->v2 accidental-urgency partial fix
        (design doc v2 SS4 calibration log): with real BurstGPT-anchored
        (heavy-tailed) prompt sampling, individual scenarios/seeds can still
        show a nontrivial fraction of bulk tenants classified 'urgent' by
        KVConstrainedOnlinePolicy's own laxity threshold (this is
        documented, not hidden -- see the design doc's honest caveat that
        BULK_SLACK_S=1.65 does not eliminate this confound, only reduces it
        relative to v1's 1.5). This test checks the AGGREGATE rate across a
        representative multi-seed, multi-cell sample stays a minority, not
        that any single scenario is exactly zero (that would be flaky under
        real-data sampling variance and was not what v2 fixed)."""
        n_urgent = 0
        n_total = 0
        for bulk_pressure in ("low", "high"):
            for seed in (20260910, 20260911, 20260912, 20260913):
                scen = _make_v2(bulk_pressure=bulk_pressure, seed=seed)
                bulk = [r for r in scen.requests if r.class_id == CLASS_BULK]
                for r in bulk:
                    laxity = (
                        r.slo_deadline - r.arrival_time
                        - (DEFAULT_ALPHA * r.prompt_tokens + DEFAULT_BETA * r.predicted_output_tokens) * 0.001
                    )
                    n_total += 1
                    if laxity <= 0.25:
                        n_urgent += 1
        frac_urgent = n_urgent / n_total
        assert frac_urgent < 0.5, (
            f"{frac_urgent:.2f} of bulk tenants classify as 'urgent' by the policy's own "
            "threshold across this sample -- the majority-confound v1 had would not be fixed"
        )

    def test_bulk_prompt_tokens_always_admittable(self):
        for bulk_pressure in ("low", "high"):
            for seed in (20260910, 20260911, 20260912, 20260913, 20260914, 20260915):
                scen = _make_v2(bulk_pressure=bulk_pressure, seed=seed)
                for r in scen.requests:
                    if r.class_id == CLASS_BULK:
                        assert r.prompt_tokens < 6000

    def test_reproducibility(self):
        s1 = _make_v2(seed=42)
        s2 = _make_v2(seed=42)
        assert s1.scenario_id == s2.scenario_id
        for r1, r2 in zip(s1.requests, s2.requests):
            assert r1.arrival_time == r2.arrival_time
            assert r1.prompt_tokens == r2.prompt_tokens

    def test_generator_version_tagged(self):
        scen = _make_v2()
        assert scen.generator_version == "kv_pressure_v2"
        assert scen.family == "family_c_kv_pressure_v2"


class TestLeakageGuardV2:
    def test_clean_scenario_passes(self):
        assert_policy_visible_fields_clean_kv_v2(_make_v2())

    def test_illegal_class_id_fails(self):
        scen = _make_v2()
        object.__setattr__(scen.requests[0], "class_id", "tenant_bulk_phasemiddle")
        with pytest.raises(AssertionError):
            assert_policy_visible_fields_clean_kv_v2(scen)


class TestFullGridUniqueness:
    def test_all_72_scenario_ids_unique(self):
        ids = set()
        for bulk_pressure in ("low", "high"):
            for phase in ("early", "middle", "late"):
                for tight in ("loose", "tight"):
                    for seed in (20260910, 20260911, 20260912, 20260913, 20260914, 20260915):
                        scen = case_kv_pressure_reserve_contention_v2(
                            bulk_pressure=bulk_pressure, urgent_arrival_phase=phase,
                            urgent_tightness=tight, seed=seed, allow_synthetic_tokens=True,
                        )
                        ids.add(scen.scenario_id)
        assert len(ids) == 2 * 3 * 2 * 6


class TestHeldOutSplitIntegrity:
    def test_build_scenarios_v2_via_runner(self):
        cfg = {
            "sweep_grid": {
                "bulk_pressure": ["high"],
                "urgent_arrival_phase": ["middle"],
                "urgent_tightness": ["tight"],
                "seeds": [20260910, 20260914],
            },
            "held_out_seeds": [20260914],
            "max_kv_tokens": 6000,
        }
        scenarios = build_scenarios(
            cfg, template_version="v2", allow_synthetic_tokens=True, datasets_root=None,
        )
        assert len(scenarios) == 2
        seeds = {s.seed for s in scenarios}
        assert seeds == {20260910, 20260914}

    def test_held_out_seeds_rejected_for_v1(self):
        cfg = {
            "sweep_grid": {
                "bulk_pressure": ["low"], "urgent_arrival_phase": ["early"],
                "urgent_tightness": ["loose"], "seeds": [20260901],
            },
            "held_out_seeds": [20260901],
        }
        with pytest.raises(ValueError):
            build_scenarios(cfg, template_version="v1", allow_synthetic_tokens=True, datasets_root=None)

    def test_invalid_template_version_rejected(self):
        cfg = {"sweep_grid": {"bulk_pressure": ["low"], "urgent_arrival_phase": ["early"],
                              "urgent_tightness": ["loose"], "seeds": [1]}}
        with pytest.raises(ValueError):
            build_scenarios(cfg, template_version="v3", allow_synthetic_tokens=True, datasets_root=None)


class TestEndpointSanity:
    def test_both_policies_run_and_anwg_present(self):
        scen = _make_v2()
        for pol_cls in (KVConstrainedOnlinePolicy, LeastLaxityFirstPolicy):
            sim = Simulator(SimulatorConfig(
                gpu_configs=list(scen.gpu_configs), service_model=ServiceModel(**scen.service_model_kwargs),
            ))
            sim.load_trace(list(scen.requests))
            policy = pol_cls()
            policy.name = pol_cls.name
            m = sim.run(policy, workload_tag=policy.name, seed=scen.seed)
            assert np.isfinite(m.arrival_normalized_weighted_goodput)

    def test_reserve_activates_under_high_pressure_tight(self):
        scen = _make_v2(bulk_pressure="high", urgent_tightness="tight", urgent_arrival_phase="middle")
        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scen.gpu_configs), service_model=ServiceModel(**scen.service_model_kwargs),
        ))
        sim.load_trace(list(scen.requests))
        policy = InstrumentedKVConstrainedOnlinePolicy()
        policy.name = "kv_constrained_online"
        sim.run(policy, workload_tag="test", seed=scen.seed)
        assert policy.n_reserve_deferrals > 0

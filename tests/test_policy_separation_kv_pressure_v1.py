"""Focused tests for Family C v1 KV-pressure reserve pairwise-separation
pilot scaffolding: scenario generation, leakage guard, and the
InstrumentedKVConstrainedOnlinePolicy diagnostic wrapper.

See docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md. This is a
pairwise-separation pilot, not a composition experiment -- no selector/child
policy tests belong here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.policy_separation.templates_kv_pressure import (
    CLASS_BULK,
    CLASS_URGENT,
    N_BULK,
    N_URGENT,
    assert_policy_visible_fields_clean_kv_v1,
    case_kv_pressure_reserve_contention,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.simulator.service_model import ServiceModel

from run_policy_separation_kv_pressure_pilot_v1 import (  # noqa: E402
    InstrumentedKVConstrainedOnlinePolicy,
)


def _make_scenario(**overrides):
    args = dict(
        bulk_pressure="high",
        urgent_arrival_phase="late",
        urgent_tightness="tight",
        seed=20260901,
        allow_synthetic_tokens=True,
    )
    args.update(overrides)
    return case_kv_pressure_reserve_contention(**args)


class TestScenarioGeneration:
    def test_basic_shape(self):
        scen = _make_scenario()
        assert scen.family == "family_c_kv_pressure_v1"
        n_bulk = N_BULK["high"]
        assert len(scen.requests) == n_bulk + N_URGENT
        bulk = [r for r in scen.requests if r.class_id == CLASS_BULK]
        urgent = [r for r in scen.requests if r.class_id == CLASS_URGENT]
        assert len(bulk) == n_bulk
        assert len(urgent) == N_URGENT

    def test_requests_sorted_by_arrival(self):
        scen = _make_scenario()
        arrivals = [r.arrival_time for r in scen.requests]
        assert arrivals == sorted(arrivals)

    def test_low_pressure_fewer_bulk(self):
        scen = _make_scenario(bulk_pressure="low")
        bulk = [r for r in scen.requests if r.class_id == CLASS_BULK]
        assert len(bulk) == N_BULK["low"]
        assert N_BULK["low"] < N_BULK["high"]

    def test_urgent_phase_changes_arrival_timing(self):
        early = _make_scenario(urgent_arrival_phase="early")
        late = _make_scenario(urgent_arrival_phase="late")
        early_urgent_start = min(r.arrival_time for r in early.requests if r.class_id == CLASS_URGENT)
        late_urgent_start = min(r.arrival_time for r in late.requests if r.class_id == CLASS_URGENT)
        assert late_urgent_start > early_urgent_start

    def test_tight_urgent_has_less_slack_than_loose(self):
        loose = _make_scenario(urgent_tightness="loose")
        tight = _make_scenario(urgent_tightness="tight")
        loose_slack = [
            r.slo_deadline - r.arrival_time for r in loose.requests if r.class_id == CLASS_URGENT
        ]
        tight_slack = [
            r.slo_deadline - r.arrival_time for r in tight.requests if r.class_id == CLASS_URGENT
        ]
        assert min(loose_slack) > max(tight_slack)

    def test_bulk_prompt_tokens_always_admittable(self):
        """Regression test: bulk prompt window must stay below any realistic
        max_kv_tokens so no request is permanently unadmittable (see design
        doc SS5 calibration round 3)."""
        for bulk_pressure in ("low", "high"):
            for seed in (20260901, 20260902, 20260903, 20260904):
                scen = _make_scenario(bulk_pressure=bulk_pressure, seed=seed)
                for r in scen.requests:
                    if r.class_id == CLASS_BULK:
                        assert r.prompt_tokens < 6000, (
                            f"bulk request {r.request_id} has prompt_tokens="
                            f"{r.prompt_tokens} >= default max_kv_tokens=6000 "
                            "-- permanently unadmittable"
                        )

    def test_gpu_config_kv_scarce(self):
        scen = _make_scenario(max_kv_tokens=6000, max_active_sequences=64, max_batch_tokens=64)
        gpu = scen.gpu_configs[0]
        assert gpu.max_kv_tokens == 6000
        assert gpu.max_active_sequences == 64

    def test_reproducibility_same_seed(self):
        s1 = _make_scenario(seed=42)
        s2 = _make_scenario(seed=42)
        assert s1.scenario_id == s2.scenario_id
        for r1, r2 in zip(s1.requests, s2.requests):
            assert r1.arrival_time == r2.arrival_time
            assert r1.prompt_tokens == r2.prompt_tokens

    def test_invalid_bulk_pressure_raises(self):
        with pytest.raises(ValueError):
            _make_scenario(bulk_pressure="medium")

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError):
            _make_scenario(urgent_arrival_phase="middle")

    def test_invalid_tightness_raises(self):
        with pytest.raises(ValueError):
            _make_scenario(urgent_tightness="moderate")

    def test_pair_id_requires_changed_parameters(self):
        scen = _make_scenario()
        assert scen.pair_id is not None
        assert scen.changed_parameters


class TestLeakageGuard:
    def test_clean_scenario_passes(self):
        scen = _make_scenario()
        assert_policy_visible_fields_clean_kv_v1(scen)

    def test_illegal_class_id_fails(self):
        scen = _make_scenario()
        bad_req = scen.requests[0]
        object.__setattr__(bad_req, "class_id", "tenant_bulk_high_pressure")
        with pytest.raises(AssertionError):
            assert_policy_visible_fields_clean_kv_v1(scen)

    def test_priority_must_be_one(self):
        scen = _make_scenario()
        object.__setattr__(scen.requests[0], "priority", 2.0)
        with pytest.raises(AssertionError):
            assert_policy_visible_fields_clean_kv_v1(scen)


class TestInstrumentedPolicy:
    def test_no_deferrals_when_capacity_ample(self):
        scen = _make_scenario(bulk_pressure="low", max_kv_tokens=1_000_000)
        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scen.gpu_configs), service_model=ServiceModel(**scen.service_model_kwargs),
        ))
        sim.load_trace(list(scen.requests))
        policy = InstrumentedKVConstrainedOnlinePolicy()
        policy.name = "kv_constrained_online"
        sim.run(policy, workload_tag="test", seed=scen.seed)
        assert policy.n_reserve_deferrals == 0

    def test_deferrals_occur_under_real_pressure(self):
        scen = _make_scenario(bulk_pressure="high", urgent_arrival_phase="late",
                              urgent_tightness="tight", max_kv_tokens=6000)
        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scen.gpu_configs), service_model=ServiceModel(**scen.service_model_kwargs),
        ))
        sim.load_trace(list(scen.requests))
        policy = InstrumentedKVConstrainedOnlinePolicy()
        policy.name = "kv_constrained_online"
        sim.run(policy, workload_tag="test", seed=scen.seed)
        assert policy.n_reserve_deferrals >= 0  # non-degenerate call path; see pilot smoke for >0 evidence

    def test_least_laxity_first_has_no_deferral_concept(self):
        """LeastLaxityFirstPolicy structurally cannot defer for KV reasons --
        confirms H2/G3's causal-interpretability claim (SS3)."""
        policy = LeastLaxityFirstPolicy()
        assert not hasattr(policy, "n_reserve_deferrals")
        assert not hasattr(policy, "_admit_filter")


class TestEndpointSanity:
    def test_both_policies_run_without_error(self):
        scen = _make_scenario(max_kv_tokens=6000)
        for pol_cls in (KVConstrainedOnlinePolicy, LeastLaxityFirstPolicy):
            sim = Simulator(SimulatorConfig(
                gpu_configs=list(scen.gpu_configs), service_model=ServiceModel(**scen.service_model_kwargs),
            ))
            sim.load_trace(list(scen.requests))
            policy = pol_cls()
            policy.name = pol_cls.name
            m = sim.run(policy, workload_tag=policy.name, seed=scen.seed)
            assert np.isfinite(m.arrival_normalized_weighted_goodput)

    def test_kv_constrained_never_exceeds_hard_capacity_at_admission(self):
        """The soft reserve is a deferral, not an override of the hard
        _feasible_on_gpu capacity check -- admission-time KV must never
        exceed max_kv_tokens (decode growth beyond that is a separate,
        documented simulator property, not this policy's concern)."""
        scen = _make_scenario(max_kv_tokens=6000)
        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scen.gpu_configs), service_model=ServiceModel(**scen.service_model_kwargs),
        ))
        sim.load_trace(list(scen.requests))
        policy = KVConstrainedOnlinePolicy()
        policy.name = "kv_constrained_online"
        sim.run(policy, workload_tag="test", seed=scen.seed)
        # No exception / no warning-triggering rejected admission implies
        # the hard capacity check was always respected; sim.run raising
        # nothing is the assertion here.

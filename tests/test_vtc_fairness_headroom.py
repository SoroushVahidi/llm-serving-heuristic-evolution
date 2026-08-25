"""Tests for scripts/check_vtc_fairness_headroom.py and the diagnostic
instrumentation it depends on (baselines/vtc/adapter/diagnostics.py).

See docs/audits/vtc_fairness_benchmark_repair_20260805.md for the full
gate-threshold justification these tests lock in.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from baselines.vtc.adapter.diagnostics import InstrumentedVTCFairnessPolicy
from baselines.vtc.adapter.official_loader import verify_official_clone
from baselines.vtc.adapter.simulator_policy import VTCFairnessPolicy
from baselines.vtc.fairness_workloads import ALL_FAIRNESS_FAMILIES, RECOMMENDED_GPU_CONFIG
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

# scripts/ is not a package -- load the module directly by path, matching
# this project's convention for testing standalone scripts.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_vtc_fairness_headroom.py"
_spec = importlib.util.spec_from_file_location("check_vtc_fairness_headroom", _SCRIPT_PATH)
headroom = importlib.util.module_from_spec(_spec)
sys.modules["check_vtc_fairness_headroom"] = headroom
_spec.loader.exec_module(headroom)


def _clone_present() -> bool:
    try:
        verify_official_clone()
        return True
    except Exception:
        return False


requires_clone = pytest.mark.skipif(
    not _clone_present(),
    reason="Pinned Ying1123/VTC-artifact clone not found; see PROVENANCE.md to clone it.",
)


def _uniform_requests(n_tenants: int, n_per_tenant: int, prompt: int = 50, out: int = 20,
                       all_same_arrival: bool = True) -> tuple:
    """A minimal, fast synthetic workload for isolated gate testing."""
    tenants = [f"t{i}" for i in range(n_tenants)]
    reqs = []
    rid = 0
    for t in tenants:
        for i in range(n_per_tenant):
            arrival = 0.0 if all_same_arrival else float(rid) * 0.5
            reqs.append(Request(
                request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
                predicted_output_tokens=out, actual_output_tokens=out,
                slo_deadline=arrival + 1000.0, priority=1.0, class_id=t,
            ))
            rid += 1
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs, tenants


@requires_clone
class TestInsufficientContention:
    def test_single_tenant_never_contends(self):
        """One tenant, alone, can never have >=2 tenants backlogged --
        the contention gate must fail cleanly, not silently pass."""
        reqs, tenants = _uniform_requests(n_tenants=1, n_per_tenant=50, all_same_arrival=False)
        report = headroom.check_workload("balanced_tenants", reqs, tenants)
        contention_gate = next(g for g in report.gates if g.name == "contention")
        assert contention_gate.passed is False
        assert not report.passed

    def test_widely_spaced_arrivals_produce_low_contention(self):
        """Requests spaced far enough apart that each is served and
        drained before the next tenant's request arrives -- no real
        backlog ever forms, even with multiple tenants present."""
        tenants = ["a", "b"]
        reqs = []
        for i in range(20):
            t = tenants[i % 2]
            reqs.append(Request(
                request_id=i, arrival_time=float(i) * 50.0, prompt_tokens=10,
                predicted_output_tokens=5, actual_output_tokens=5,
                slo_deadline=float(i) * 50.0 + 1000.0, priority=1.0, class_id=t,
            ))
        report = headroom.check_workload("balanced_tenants", reqs, tenants)
        contention_gate = next(g for g in report.gates if g.name == "contention")
        assert contention_gate.passed is False


@requires_clone
class TestReservationDomination:
    def test_tiny_batch_token_budget_triggers_confound(self):
        """Reproduces the ORIGINAL smoke-test confound directly: a
        batch_max_tokens value far below typical prompt size makes the
        official admission gate reject almost everything, which the
        not_reservation_dominated gate must catch."""
        reqs, tenants = _uniform_requests(n_tenants=4, n_per_tenant=30, prompt=500, out=100,
                                           all_same_arrival=False)
        tiny_gpu = GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=64, max_kv_tokens=4096)
        cfg = SimulatorConfig(gpu_configs=[tiny_gpu])
        sim = Simulator(cfg)
        sim.load_trace(reqs)
        policy = InstrumentedVTCFairnessPolicy(known_tenants=tenants)
        sim.run(policy, workload_tag="reservation_domination_probe")
        d = policy.decomposition_summary()
        bind_rate = d["reservation_bind_rate"] + d["budget_bind_rate"]
        assert bind_rate > headroom.MAX_ADMISSION_GATE_BIND_RATE


class TestValidFairnessHeadroom:
    @requires_clone
    @pytest.mark.parametrize("family_name", list(ALL_FAIRNESS_FAMILIES))
    def test_every_repaired_family_passes_its_gates(self, family_name):
        report = headroom.check_family(family_name)
        failures = [(g.name, g.detail) for g in report.gates if not g.passed]
        assert report.passed, f"{family_name} failed gates: {failures}"

    @requires_clone
    def test_discriminative_families_show_real_fifo_disparity(self):
        for name in headroom.DISCRIMINATIVE_DISPARITY_FAMILIES:
            report = headroom.check_family(name)
            assert report.metrics["fifo_jain_at_checkpoint"] <= headroom.MAX_DISCRIMINATIVE_FIFO_JAIN

    @requires_clone
    def test_slo_divergence_family_is_genuinely_mixed(self):
        for name in headroom.SLO_DIVERGENCE_FAMILIES:
            report = headroom.check_family(name)
            lo, hi = headroom.SLO_DIVERGENCE_RANGE
            assert lo < report.metrics["fifo_tight_slo_violation_rate"] < hi


@requires_clone
class TestReturningTenantBehavior:
    def test_counter_lift_precondition_holds_for_real_family(self):
        report = headroom.check_family("returning_inactive_tenant")
        gate = next(g for g in report.gates if g.name == "counter_lift_precondition")
        assert gate.passed
        assert report.metrics["continuous_demand_before_return"] > 0

    def test_counter_lift_precondition_fails_when_other_tenant_never_active_before_return(self):
        """If the OTHER tenant has no demand before the return event, the
        counter-lift mechanism has nothing to lift against -- the gate
        must catch this rather than silently reporting PASS."""
        tenants = ["returning", "continuous"]
        reqs = [
            Request(request_id=0, arrival_time=0.0, prompt_tokens=20, predicted_output_tokens=10,
                    actual_output_tokens=10, slo_deadline=1000.0, priority=1.0, class_id="returning"),
            # "continuous" only arrives AFTER the (synthetic) return time used by the
            # gate (0.85 * duration) -- so continuous_demand_before_return == 0.
            Request(request_id=1, arrival_time=99.0, prompt_tokens=20, predicted_output_tokens=10,
                    actual_output_tokens=10, slo_deadline=1099.0, priority=1.0, class_id="continuous"),
        ]
        report = headroom.check_workload("returning_inactive_tenant", reqs, tenants)
        gate = next(g for g in report.gates if g.name == "counter_lift_precondition")
        assert gate.passed is False
        assert report.metrics["continuous_demand_before_return"] == 0


@requires_clone
class TestDeterministicOutput:
    def test_same_family_produces_identical_report_twice(self):
        r1 = headroom.check_family("one_heavy_hitter")
        r2 = headroom.check_family("one_heavy_hitter")
        assert r1.metrics == r2.metrics
        assert [(g.name, g.passed) for g in r1.gates] == [(g.name, g.passed) for g in r2.gates]

    def test_main_all_families_exits_zero(self, capsys):
        argv_backup = sys.argv
        try:
            sys.argv = ["check_vtc_fairness_headroom.py", "--all"]
            code = headroom.main()
        finally:
            sys.argv = argv_backup
        assert code == 0
        out = capsys.readouterr().out
        assert "ALL FAMILIES PASS" in out


@requires_clone
class TestDiagnosticsInstrumentationIsInert:
    """The InstrumentedVTCFairnessPolicy wraps `_can_add_new_req` purely
    to record decisions -- it must never change what those decisions ARE.
    See baselines/vtc/adapter/diagnostics.py's module docstring."""

    def test_instrumented_and_plain_policy_produce_identical_outcomes(self):
        reqs, tenants = ALL_FAIRNESS_FAMILIES["one_heavy_hitter"]()
        results = {}
        for cls in (VTCFairnessPolicy, InstrumentedVTCFairnessPolicy):
            cfg = SimulatorConfig(gpu_configs=[RECOMMENDED_GPU_CONFIG])
            sim = Simulator(cfg)
            sim.load_trace(reqs)
            policy = cls(known_tenants=tenants)
            m = sim.run(policy, workload_tag=cls.__name__)
            results[cls.__name__] = (m.num_completed, m.completion_fraction, policy.served_snapshot())
        assert results["VTCFairnessPolicy"] == results["InstrumentedVTCFairnessPolicy"]

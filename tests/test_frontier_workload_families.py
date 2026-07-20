"""Tests for the 5 new synthetic contention-frontier workload families
(see docs/selector_v2_slo_calibrated_frontier_search.md section 5) and
their integration with the policy-independent SLO calibration.

Family F (real-trace stress) is not generated in this module -- it
reuses `scenario_redesign.local_real_trace_stress_specs` directly, whose
own provenance-preservation is exercised by the last test class here.
"""
from __future__ import annotations

import random

import pytest

from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.weighted_shortest_processing import WeightedShortestProcessingPolicy
from llmserveopt.selector.dataset_v2.frontier_workload_families import FAMILY_GENERATORS
from llmserveopt.selector.dataset_v2.scenario_redesign import local_real_trace_stress_specs
from llmserveopt.selector.dataset_v2.slo_calibration import calibrate_window_e2e
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = None  # set in fixture


@pytest.fixture(autouse=True)
def _root():
    global ROOT
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent


class TestFamilyGeneratorsStructural:

    @pytest.mark.parametrize("name", list(FAMILY_GENERATORS.keys()))
    def test_generator_produces_valid_requests(self, name):
        gen = FAMILY_GENERATORS[name]
        rng = random.Random(1)
        for _ in range(20):
            window = gen(rng)
            assert len(window["requests"]) >= 2
            ids = [r.request_id for r in window["requests"]]
            assert len(ids) == len(set(ids)), "duplicate request_id"
            assert all(r.prompt_tokens > 0 for r in window["requests"])
            assert all(r.arrival_time >= 0.0 for r in window["requests"])
            assert window["budget"] > 0
            assert window["chunk"] > 0

    def test_generators_are_deterministic_given_same_rng_state(self):
        for gen in FAMILY_GENERATORS.values():
            w1 = gen(random.Random(42))
            w2 = gen(random.Random(42))
            assert [r.prompt_tokens for r in w1["requests"]] == [r.prompt_tokens for r in w2["requests"]]
            assert w1["budget"] == w2["budget"]

    def test_family_a_all_arrivals_simultaneous(self):
        rng = random.Random(7)
        gen = FAMILY_GENERATORS["A_same_arrival_heterogeneous_cluster"]
        for _ in range(10):
            w = gen(rng)
            arrivals = {r.arrival_time for r in w["requests"]}
            assert arrivals == {0.0}

    def test_family_b_arrivals_are_spread_but_close(self):
        rng = random.Random(7)
        gen = FAMILY_GENERATORS["B_closely_spaced_heterogeneous_cluster"]
        w = gen(rng)
        arrivals = sorted(r.arrival_time for r in w["requests"])
        assert arrivals[0] == 0.0
        # up to 8 requests at up to 3 step_sizes apart -- a handful of steps, not a large gap
        assert arrivals[-1] < 0.03

    def test_family_c_sizes_straddle_chunk_boundary(self):
        rng = random.Random(7)
        gen = FAMILY_GENERATORS["C_admission_reorder_boundary"]
        w = gen(rng)
        chunk = w["chunk"]
        multiples = [r.prompt_tokens / chunk for r in w["requests"]]
        assert any(m < 1.5 for m in multiples)  # at least one ~1-chunk request

    def test_family_e_kv_tokens_reflect_headroom_over_total_prompt(self):
        rng = random.Random(7)
        gen = FAMILY_GENERATORS["E_kv_pressure_admission_order"]
        for _ in range(10):
            w = gen(rng)
            total_prompt = sum(r.prompt_tokens for r in w["requests"])
            assert w["max_kv_tokens"] >= total_prompt  # headroom is always >= 1.05x by construction
            assert w["max_kv_tokens"] < total_prompt * 2.0  # never absurdly generous


class TestCalibrationIntegration:
    """Regression lock on the core finding: calibrated SLOs (unlike the
    prior fixed slo_deadline=1000.0 placeholder) can make these families
    ANWG-discriminative between policies."""

    def test_calibrated_family_a_window_can_discriminate_policies(self):
        rng = random.Random(20260720)
        gen = FAMILY_GENERATORS["A_same_arrival_heterogeneous_cluster"]
        found_divergence = False
        for i in range(30):
            w = gen(rng)
            sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=w["budget"],
                               max_prefill_chunk_tokens=w["chunk"])
            calibrated = calibrate_window_e2e(w["requests"], sm, multiplier=2.0)
            gpu_configs = [GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000,
                                       max_kv_tokens=w["max_kv_tokens"])]
            anwgs = {}
            for pname, policy in [("fifo", FIFOPolicy()),
                                    ("weighted_shortest_processing", WeightedShortestProcessingPolicy())]:
                sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=2000))
                sim.load_trace(list(calibrated))
                policy.reset()
                m = sim.run(policy=policy, workload_tag="test", seed=i)
                if m.arrival_normalized_weighted_goodput is not None:
                    anwgs[pname] = m.arrival_normalized_weighted_goodput
            if len(anwgs) == 2 and abs(anwgs["fifo"] - anwgs["weighted_shortest_processing"]) > 0.002:
                found_divergence = True
                break
        assert found_divergence, "expected at least one of 30 calibrated windows to show ANWG divergence"

    def test_uncalibrated_placeholder_deadline_would_have_masked_this(self):
        """Sanity check that the divergence in the test above is actually
        attributable to calibration, not the workload shape alone: re-run
        one window at the OLD fixed-1000.0 placeholder and confirm ANWG
        saturates back to a near-tie (the exact failure mode this task
        fixes)."""
        rng = random.Random(20260720)
        w = FAMILY_GENERATORS["A_same_arrival_heterogeneous_cluster"](rng)
        from dataclasses import replace
        placeholder_requests = [replace(r, slo_deadline=r.arrival_time + 1000.0) for r in w["requests"]]
        sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=w["budget"],
                           max_prefill_chunk_tokens=w["chunk"])
        gpu_configs = [GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000,
                                   max_kv_tokens=w["max_kv_tokens"])]
        anwgs = {}
        for pname, policy in [("fifo", FIFOPolicy()),
                                ("weighted_shortest_processing", WeightedShortestProcessingPolicy())]:
            sim = Simulator(SimulatorConfig(gpu_configs=gpu_configs, service_model=sm, drain_steps=2000))
            sim.load_trace(list(placeholder_requests))
            policy.reset()
            m = sim.run(policy=policy, workload_tag="test", seed=0)
            anwgs[pname] = m.arrival_normalized_weighted_goodput
        assert abs(anwgs["fifo"] - anwgs["weighted_shortest_processing"]) <= 0.002


class TestRealTraceProvenance:

    def test_local_real_trace_stress_specs_preserve_ancestor_and_source(self):
        specs = local_real_trace_stress_specs(ROOT, max_requests=48)
        if not specs:
            pytest.skip("real-trace data files not present in this checkout")
        for spec in specs[:4]:
            assert spec.source_trace != "synthetic"
            assert spec.request_plan_ancestor_id is not None
            assert spec.request_plan_ancestor_id.startswith("real_trace__")
            reqs = spec.build(seed=0)
            assert len(reqs) > 0

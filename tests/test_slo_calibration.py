"""Tests for the policy-independent SLO calibration module (see
docs/selector_v2_slo_calibrated_frontier_search.md).
"""
from __future__ import annotations

import inspect

from llmserveopt.core.types import Request
from llmserveopt.selector.dataset_v2 import slo_calibration
from llmserveopt.selector.dataset_v2.slo_calibration import (
    CALIBRATION_MULTIPLIER_GRID,
    calibrate_dual_slo,
    calibrate_e2e_deadline,
    calibrate_window_e2e,
    reference_latency,
)
from llmserveopt.simulator.service_model import ServiceModel


def _req(rid=0, arrival=0.0, prompt=4000, output=40):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=1.0, priority=1.0, class_id="test",
    )


class TestNoPolicyLabelLeakage:
    """The calibration module must be computable BEFORE any policy runs
    -- enforced structurally by never importing anything policy- or
    metrics-shaped."""

    def test_module_never_imports_policies_or_metrics(self):
        import ast
        tree = ast.parse(inspect.getsource(slo_calibration))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_names.add(node.module or "")
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
        for forbidden in ("policies", "RunMetrics", "run_policy", "PolicyOutcomeVector",
                            "CompletedRequest", "Simulator", "evaluation"):
            assert not any(forbidden in name for name in imported_names), (
                f"slo_calibration.py imports something matching {forbidden!r} -- policy/outcome leakage"
            )

    def test_reference_latency_signature_has_no_policy_or_metrics_params(self):
        for fn in (reference_latency, calibrate_e2e_deadline, calibrate_dual_slo, calibrate_window_e2e):
            params = set(inspect.signature(fn).parameters)
            assert not params & {"policy", "metrics", "outcome", "run_metrics"}


class TestDeterminism:

    def test_reference_latency_is_pure_and_deterministic(self):
        sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=512,
                           max_prefill_chunk_tokens=512, step_size=0.001)
        r = _req()
        a = reference_latency(r, sm)
        b = reference_latency(r, sm)
        assert a == b

    def test_prefill_modeling_disabled_gives_zero_prefill_reference(self):
        sm = ServiceModel(enable_prefill_modeling=False, step_size=0.001)
        r = _req(prompt=4000, output=10)
        ref = reference_latency(r, sm)
        assert ref.reference_prefill_s == 0.0
        assert ref.reference_e2e_s == 10 * 0.001

    def test_prefill_modeling_enabled_scales_with_chunk_count(self):
        sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=512,
                           max_prefill_chunk_tokens=512, step_size=0.001)
        short = reference_latency(_req(prompt=1, output=1), sm)
        long = reference_latency(_req(prompt=4000, output=1), sm)
        assert long.reference_prefill_s > short.reference_prefill_s
        # ceil(4000/512) == 8 steps of 0.001s each
        assert long.reference_prefill_s == 8 * 0.001


class TestCalibratedDeadline:

    def test_deadline_scales_linearly_with_multiplier(self):
        sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=512,
                           max_prefill_chunk_tokens=512, step_size=0.001)
        r = _req(arrival=1.0)
        d1 = calibrate_e2e_deadline(r, sm, 1.0) - r.arrival_time
        d2 = calibrate_e2e_deadline(r, sm, 2.0) - r.arrival_time
        assert abs(d2 - 2 * d1) < 1e-12

    def test_calibrate_window_e2e_does_not_mutate_input(self):
        sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=512,
                           max_prefill_chunk_tokens=512, step_size=0.001)
        reqs = [_req(rid=0), _req(rid=1, arrival=0.01)]
        original_deadlines = [r.slo_deadline for r in reqs]
        calibrated = calibrate_window_e2e(reqs, sm, 1.5)
        assert [r.slo_deadline for r in reqs] == original_deadlines
        assert all(c.slo_deadline != o for c, o in zip(calibrated, original_deadlines))

    def test_calibrate_dual_slo_ttft_le_e2e(self):
        sm = ServiceModel(enable_prefill_modeling=True, step_token_budget=512,
                           max_prefill_chunk_tokens=512, step_size=0.001)
        r = _req(prompt=4000, output=40)
        ttft_slo, tpot_slo = calibrate_dual_slo(r, sm, 1.0)
        e2e = calibrate_e2e_deadline(r, sm, 1.0) - r.arrival_time
        assert ttft_slo <= e2e
        assert tpot_slo > 0


class TestMultiplierGrid:

    def test_grid_is_sorted_and_bounded(self):
        assert list(CALIBRATION_MULTIPLIER_GRID) == sorted(CALIBRATION_MULTIPLIER_GRID)
        assert CALIBRATION_MULTIPLIER_GRID[0] < 1.0 < CALIBRATION_MULTIPLIER_GRID[-1]

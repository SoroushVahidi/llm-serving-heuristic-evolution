"""Regression pins for verified HeuristicPolicy / calibrated-model gaps.

These tests document *current* behavior intentionally retained for historical
reproducibility. See docs/current/KNOWN_SIMULATOR_HEURISTIC_GAPS.md.
Do not "fix" the pinned behaviors here without an explicit semantics bump.
"""
from __future__ import annotations

import inspect

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.heuristics import build_heuristic_policy
from llmserveopt.heuristics.examples import fifo_like
from llmserveopt.heuristics.policy import HeuristicPolicy, _build_batch_vars, _build_req_vars
from llmserveopt.simulator import gpu as gpu_mod


def _req(req_id: int, *, arrival: float = 0.0, deadline_abs: float = 10.0) -> ObservableRequest:
    return ObservableRequest(
        request_id=req_id,
        arrival_time=arrival,
        prompt_tokens=64,
        predicted_output_tokens=32,
        slo_deadline=deadline_abs,
        priority=1.0,
        class_id="standard",
    )


def test_batch_deadline_features_currently_use_absolute_deadlines():
    """Pin the incomplete absolute-deadline binding (gap #2)."""
    now = 5.0
    admitted = [_req(1, deadline_abs=10.0), _req(2, deadline_abs=12.0)]
    batch = _build_batch_vars(admitted, {})

    # Correct remaining slack would be min(10-5, 12-5) = 5.0; current code
    # returns min absolute deadline = 10.0.
    assert batch["batch.min_deadline_slack"] == pytest.approx(10.0)
    intended_min_slack = min(r.slo_deadline - now for r in admitted)
    assert intended_min_slack == pytest.approx(5.0)
    assert batch["batch.min_deadline_slack"] != pytest.approx(intended_min_slack)

    # Absolute deadline < 1.0 is false for realistic deadlines, so risk is 0.
    assert batch["batch.deadline_risk"] == pytest.approx(0.0)
    intended_risk = sum(1 for r in admitted if (r.slo_deadline - now) < 1.0) / len(admitted)
    assert intended_risk == pytest.approx(0.0)  # both still have slack >= 5

    # Contrast: req.* slack is already time-relative.
    req_vars = _build_req_vars(admitted[0], now)
    assert req_vars["req.deadline_slack"] == pytest.approx(5.0)


def test_request_scores_do_not_rescore_when_batch_grows():
    """Pin empty-batch scoring for request_score (gap #3)."""
    heuristic = {
        "name": "batch_size_score",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"var": "batch.size"}},
    }
    policy = build_heuristic_policy(heuristic, max_candidates=8)
    state = ObservableState(
        time=1.0,
        waiting_queue=[_req(i, deadline_abs=20.0 + i) for i in range(3)],
        gpu_states=[
            ObservableGPUState(
                gpu_id=0,
                max_active_sequences=8,
                max_batch_tokens=512,
                max_kv_tokens=8192,
                active_request_ids=[],
                active_requests_info=[],
                current_kv_tokens=0,
                tokens_decoded_per_request={},
            )
        ],
        completed_count=0,
        step=1,
    )
    # All candidates are scored against empty batch.size == 0, so scores tie
    # and admission follows arrival_order (request_id order via arrival).
    action = policy.select_action(state)
    admitted = action.admit.get(0, [])
    assert admitted == [0, 1, 2]


def test_record_completion_is_not_invoked_anywhere_in_simulator_package():
    """Pin gap #4: simulator never wires HeuristicPolicy.record_completion."""
    src = inspect.getsource(gpu_mod)
    assert "record_completion" not in src

    # Method exists and works when called manually.
    policy = build_heuristic_policy(fifo_like())
    assert isinstance(policy, HeuristicPolicy)
    policy.record_completion(True)
    policy.record_completion(False)
    assert list(policy._recent_violations) == [True, False]


def test_gpu_step_source_does_not_reference_compute_decode_step_time():
    """Pin gap #1: calibrated decode wall-clock helper is not on the DES path."""
    src = inspect.getsource(gpu_mod)
    assert "compute_decode_step_time" not in src
    assert "decode_time(" not in src


def test_build_batch_vars_signature_has_no_now_parameter():
    """Document that the unfinished 'subtract time later' path was never wired."""
    params = inspect.signature(_build_batch_vars).parameters
    assert "now" not in params
    assert "admitted" in params

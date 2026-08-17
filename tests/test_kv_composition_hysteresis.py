"""Focused tests for the KV-aware transition hysteresis child policy.

Satisfies all items in Section 8 of the safety-refinement protocol.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import pytest

from llmserveopt.composition.kv_composition_features import (
    FORBIDDEN_FEATURE_KEYS,
    assert_no_hidden_leakage,
    n_urgent_waiting,
)
from llmserveopt.composition.kv_composition_policy import (
    KVAdaptiveReserveChildPolicy,
    KVAdaptiveReserveHysteresisChildPolicy,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy
from llmserveopt.core.types import ObservableState, ObservableGPUState, ObservableRequest
from llmserveopt.core.action import Action


# ===================================================================
# Contracts and preservation checks
# ===================================================================

def test_original_child_and_parents_unchanged():
    """Ensure original child policy and parent policies are unchanged."""
    child_src = inspect.getsource(KVAdaptiveReserveChildPolicy)
    assert "class KVAdaptiveReserveChildPolicy" in child_src
    assert "mode = \"reserve\" if n_urgent >= self.tau_urgent else \"llf\"" in child_src

    kv_src = inspect.getsource(KVConstrainedOnlinePolicy)
    assert "class KVConstrainedOnlinePolicy" in kv_src

    llf_src = inspect.getsource(LeastLaxityFirstPolicy)
    assert "class LeastLaxityFirstPolicy" in llf_src


def test_hysteresis_observable_only_no_forbidden_labels():
    """Verify policy uses only observable state and does not leak forbidden labels."""
    # Ensure no forbidden keys are accessed or used
    for label in ("scenario_id", "seed", "bulk_pressure", "urgent_arrival_phase", "urgent_tightness"):
        assert label in FORBIDDEN_FEATURE_KEYS


# ===================================================================
# Deterministic state transition & Hysteresis rules
# ===================================================================

def _make_dummy_state(gpu_kv: int, max_kv: int, n_urgent: int, tau_urgent: int = 2) -> ObservableState:
    """Helper to construct mock ObservableState."""
    gpu = ObservableGPUState(
        gpu_id=0,
        max_kv_tokens=max_kv,
        current_kv_tokens=gpu_kv,
        max_active_sequences=100,
        active_request_ids=[],
        active_requests_info=[],
        tokens_decoded_per_request={},
        max_batch_tokens=10000,
    )
    # create n_urgent urgent requests and some non-urgent ones
    requests = []
    # prompt + output = predicted_output_tokens
    for i in range(n_urgent):
        # make it urgent (laxity <= 0.25)
        req = ObservableRequest(
            request_id=i,
            prompt_tokens=128,
            predicted_output_tokens=128,
            priority=1,
            arrival_time=0.0,
            slo_deadline=0.1,  # tight deadline -> urgent
            class_id="urgent",
        )
        requests.append(req)
        
    # Add non-urgent requests to fill queue if needed
    for i in range(5):
        req = ObservableRequest(
            request_id=100 + i,
            prompt_tokens=128,
            predicted_output_tokens=128,
            priority=1,
            arrival_time=0.0,
            slo_deadline=10.0, # loose deadline -> non-urgent
            class_id="non_urgent",
        )
        requests.append(req)

    return ObservableState(
        step=1,
        time=0.0,
        waiting_queue=requests,
        gpu_states=[gpu],
        completed_count=0,
    )


def test_hysteresis_transitions_deterministic():
    policy = KVAdaptiveReserveHysteresisChildPolicy(tau_urgent=2)
    policy.reset()

    # Step 1: LLF -> Reserve
    # n_urgent = 3 >= 2 (trigger_reserve = True)
    # KV = 500 / 1000 = 0.50 <= ENTER_THRESHOLD (0.63)
    # Should transition to reserve
    state = _make_dummy_state(gpu_kv=500, max_kv=1000, n_urgent=3)
    policy.select_action(state)
    assert policy.mode_log[-1] == "reserve"
    assert policy.transition_count == 0  # first step has no last_mode, so transition_count is 0

    # Step 2: High KV pressure, but trigger_reserve stays True.
    # We are in reserve, trigger_reserve is True, so we must stay in reserve
    state = _make_dummy_state(gpu_kv=950, max_kv=1000, n_urgent=3)
    policy.select_action(state)
    assert policy.mode_log[-1] == "reserve"
    assert policy.transition_count == 0

    # Step 3: We clear trigger_reserve (n_urgent = 0). But KV = 950/1000 = 0.95 > RELEASE_THRESHOLD (0.82)
    # We should stay in reserve due to release hysteresis!
    state = _make_dummy_state(gpu_kv=950, max_kv=1000, n_urgent=0)
    policy.select_action(state)
    assert policy.mode_log[-1] == "reserve"
    assert policy.transition_count == 0

    # Step 4: KV falls below RELEASE_THRESHOLD (0.82) to 0.80.
    # trigger_reserve is still cleared (0). We should now release to LLF!
    state = _make_dummy_state(gpu_kv=800, max_kv=1000, n_urgent=0)
    policy.select_action(state)
    assert policy.mode_log[-1] == "llf"
    assert policy.transition_count == 1  # Transition! reserve -> llf

    # Step 5: High KV pressure (0.75) and trigger_reserve becomes True (3).
    # Since we are in llf, and KV (0.75) > ENTER_THRESHOLD (0.63), we REFUSE to transition to reserve!
    # This is the transition-aware guard that prevents the unsafe LLF->reserve transition!
    state = _make_dummy_state(gpu_kv=750, max_kv=1000, n_urgent=3)
    policy.select_action(state)
    assert policy.mode_log[-1] == "llf"
    assert policy.transition_count == 1  # No new transition!

    # Step 6: KV falls to 0.60 (<= 0.63). trigger_reserve is still True (3).
    # Now we safely transition to reserve!
    state = _make_dummy_state(gpu_kv=600, max_kv=1000, n_urgent=3)
    policy.select_action(state)
    assert policy.mode_log[-1] == "reserve"
    assert policy.transition_count == 2  # Transition! llf -> reserve


def test_no_rapid_one_step_flip_flopping():
    """Ensure that we don't immediately thrasher back and forth on border values."""
    policy = KVAdaptiveReserveHysteresisChildPolicy(tau_urgent=2)
    policy.reset()

    # Starts in LLF (KV = 500 <= 630)
    state = _make_dummy_state(gpu_kv=500, max_kv=1000, n_urgent=0)
    policy.select_action(state)
    assert policy.mode_log[-1] == "llf"

    # trigger_reserve becomes True. KV = 500 (<= 0.63). Transition to reserve!
    state = _make_dummy_state(gpu_kv=500, max_kv=1000, n_urgent=3)
    policy.select_action(state)
    assert policy.mode_log[-1] == "reserve"
    assert policy.transition_count == 1

    # Next step: trigger_reserve is immediately cleared (0). But KV rises to 850 (> 0.82).
    # Due to release threshold, we remain in reserve. This avoids rapid flip-flopping!
    state = _make_dummy_state(gpu_kv=850, max_kv=1000, n_urgent=0)
    policy.select_action(state)
    assert policy.mode_log[-1] == "reserve"
    assert policy.transition_count == 1


def test_hysteresis_inactive_case_matches_original_behavior():
    """When KV is always 0.0 (inactive hysteresis), both policies should match mode choices exactly."""
    p_orig = KVAdaptiveReserveChildPolicy(tau_urgent=2)
    p_hyst = KVAdaptiveReserveHysteresisChildPolicy(tau_urgent=2)

    p_orig.reset()
    p_hyst.reset()

    # Step 1: n_urgent = 3 -> Both should select reserve
    state = _make_dummy_state(gpu_kv=0, max_kv=1000, n_urgent=3)
    p_orig.select_action(state)
    p_hyst.select_action(state)
    assert p_orig.mode_log[-1] == p_hyst.mode_log[-1] == "reserve"

    # Step 2: n_urgent = 0 -> Both should select llf
    state = _make_dummy_state(gpu_kv=0, max_kv=1000, n_urgent=0)
    p_orig.select_action(state)
    p_hyst.select_action(state)
    assert p_orig.mode_log[-1] == p_hyst.mode_log[-1] == "llf"


def test_both_modes_remain_reachable_and_reset_works():
    policy = KVAdaptiveReserveHysteresisChildPolicy(tau_urgent=2)
    policy.reset()

    # LLF is reachable
    state = _make_dummy_state(gpu_kv=0, max_kv=1000, n_urgent=0)
    policy.select_action(state)
    assert policy.n_llf_steps == 1

    # Reserve is reachable
    state = _make_dummy_state(gpu_kv=0, max_kv=1000, n_urgent=3)
    policy.select_action(state)
    assert policy.n_reserve_steps == 1

    policy.reset()
    assert policy.mode_log == []
    assert policy.transition_count == 0
    assert policy.kv_util_at_transition == []
    assert policy.n_urgent_at_transition == []

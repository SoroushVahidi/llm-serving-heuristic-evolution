from __future__ import annotations

import inspect

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.adaptive_chunked_prefill import AdaptiveChunkedPrefillPolicy
from llmserveopt.policies.aging_priority import AgingPriorityPolicy
from llmserveopt.policies.flow_control_stability import FlowControlStabilityPolicy
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy
from llmserveopt.policies.registry import (
    BASELINE_NAMES,
    POLICY_LIBRARY_V2_NAMES,
    POLICY_LIBRARY_V2_NEW_NAMES,
    make_policy_library_v2,
)
from llmserveopt.policies.slai_style_phase_aware import SlaiStylePhaseAwarePolicy
from llmserveopt.policies.sola_style_state_aware import SolaStyleStateAwarePolicy
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy


def req(
    request_id: int,
    *,
    prompt: int = 64,
    output: int = 64,
    arrival: float = 0.0,
    deadline: float = 20.0,
    priority: float = 1.0,
    class_id: str = "medium",
) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id=class_id,
    )


def gpu(
    *,
    max_seq: int = 4,
    max_kv: int = 4096,
    max_batch: int = 128,
    active: list[int] | None = None,
    kv_used: int = 0,
    prefilling: int = 0,
    decoding: int = 0,
    active_info: list[ObservableRequest] | None = None,
) -> ObservableGPUState:
    return ObservableGPUState(
        gpu_id=0,
        max_active_sequences=max_seq,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=list(active or []),
        active_requests_info=list(active_info or []),
        current_kv_tokens=kv_used,
        tokens_decoded_per_request={},
        prefilling_count=prefilling,
        decoding_count=decoding,
    )


def state(reqs: list[ObservableRequest], *, now: float = 5.0, g: ObservableGPUState | None = None) -> ObservableState:
    return ObservableState(time=now, waiting_queue=reqs, gpu_states=[g or gpu()], completed_count=0, step=1)


@pytest.mark.parametrize(
    "name,cls",
    [
        ("sola_style_state_aware", SolaStyleStateAwarePolicy),
        ("slai_style_phase_aware", SlaiStylePhaseAwarePolicy),
        ("flow_control_stability", FlowControlStabilityPolicy),
        ("kv_constrained_online", KVConstrainedOnlinePolicy),
        ("adaptive_chunked_prefill", AdaptiveChunkedPrefillPolicy),
        ("aging_priority", AgingPriorityPolicy),
        ("weighted_fair_share", WeightedFairSharePolicy),
    ],
)
def test_policy_library_v2_registration(name, cls):
    assert name in POLICY_LIBRARY_V2_NEW_NAMES
    assert name in POLICY_LIBRARY_V2_NAMES
    assert name not in BASELINE_NAMES
    assert isinstance(make_policy_library_v2(name), cls)


def test_policy_library_v2_count_extends_historical_without_replacing_it():
    assert len(BASELINE_NAMES) == 20
    assert len(POLICY_LIBRARY_V2_NEW_NAMES) == 7
    assert len(POLICY_LIBRARY_V2_NAMES) == 27


@pytest.mark.parametrize("name", POLICY_LIBRARY_V2_NEW_NAMES)
def test_policy_library_v2_deterministic_and_causal(name):
    policy = make_policy_library_v2(name)
    s1 = state([req(1, prompt=256, output=32, deadline=8.0), req(2, prompt=32, output=256, deadline=9.0)])
    s2 = state([req(1, prompt=256, output=32, deadline=8.0), req(2, prompt=32, output=256, deadline=9.0)])
    assert policy.select_action(s1).admit == policy.select_action(s2).admit
    source = inspect.getsource(policy.__class__)
    assert "actual_output_tokens" not in source
    assert "oracle" not in source.lower()


def test_kv_constrained_online_preserves_kv_reserve_for_nonurgent_requests():
    policy = KVConstrainedOnlinePolicy(target_kv_utilization=0.80, urgent_laxity_seconds=0.02)
    g = gpu(max_kv=1000, kv_used=760, max_seq=4)
    nonurgent = req(1, prompt=120, output=64, deadline=50.0)
    urgent = req(2, prompt=120, output=64, deadline=5.05)
    action = policy.select_action(state([nonurgent, urgent], now=5.0, g=g))
    admitted = action.admit[0]
    assert 2 in admitted
    assert 1 not in admitted


def test_adaptive_chunked_prefill_limits_long_prompt_concurrency_under_pressure():
    policy = AdaptiveChunkedPrefillPolicy(long_prompt_threshold=1000, pressure_threshold=0.20)
    g = gpu(max_seq=4, max_kv=10000, active=[99, 100], kv_used=5000)
    long_a = req(1, prompt=2000, output=32)
    long_b = req(2, prompt=2200, output=32)
    short = req(3, prompt=64, output=64)
    action = policy.select_action(state([long_a, long_b, short], g=g))
    admitted = action.admit[0]
    assert 3 in admitted
    assert len([rid for rid in admitted if rid in {1, 2}]) <= 1


def test_aging_priority_can_overtake_shorter_newer_request():
    policy = AgingPriorityPolicy(aging_rate=1.0)
    old = req(1, prompt=512, output=512, arrival=0.0)
    new = req(2, prompt=64, output=64, arrival=9.9)
    action = policy.select_action(state([new, old], now=10.0, g=gpu(max_seq=1)))
    assert action.admit[0] == [1]


def test_slai_phase_aware_penalizes_decode_heavy_work_under_decode_pressure():
    policy = SlaiStylePhaseAwarePolicy()
    decode_heavy = req(1, prompt=32, output=512)
    prefill_heavy = req(2, prompt=512, output=32)
    g = gpu(max_seq=4, max_kv=4096, active=[10, 11, 12], decoding=3)
    action = policy.select_action(state([decode_heavy, prefill_heavy], g=g))
    assert action.admit[0][0] == 2


def test_flow_control_stability_throttles_under_overload():
    policy = FlowControlStabilityPolicy(budget_max=1.0, budget_refill=0.0, overload_threshold=0.05)
    reqs = [req(i, arrival=4.9 + i * 0.001) for i in range(10)]
    action = policy.select_action(state(reqs, now=5.0, g=gpu(max_seq=8)))
    assert sum(len(v) for v in action.admit.values()) <= 1


def test_weighted_fair_share_uses_class_deficit():
    policy = WeightedFairSharePolicy()
    active_tight = [req(90, class_id="tight"), req(91, class_id="tight")]
    g = gpu(max_seq=4, active=[90, 91], active_info=active_tight)
    tight = req(1, class_id="tight")
    loose = req(2, class_id="loose")
    action = policy.select_action(state([tight, loose], g=g))
    assert action.admit[0][0] == 2

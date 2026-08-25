"""Tests for the CC2 canonical primitive interface (primitives.py).

Covers registry typed-metadata contracts, family separation, parameter
bounds/validation, determinism, and unsupported-state error handling.
Equivalence between reconstructed policies and their originals is tested
separately in test_primitive_reconstructed_policies.py.
"""
from __future__ import annotations

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies import primitives as prim


def req(request_id, *, prompt=64, output=64, arrival=0.0, deadline=20.0, priority=1.0, class_id="medium"):
    return ObservableRequest(
        request_id=request_id, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline, priority=priority, class_id=class_id,
    )


def gpu(*, gpu_id=0, max_seq=4, max_kv=4096, max_batch=128, active=None, kv_used=0, prefilling=0, decoding=0):
    return ObservableGPUState(
        gpu_id=gpu_id, max_active_sequences=max_seq, max_batch_tokens=max_batch, max_kv_tokens=max_kv,
        active_request_ids=list(active or []), active_requests_info=[], current_kv_tokens=kv_used,
        tokens_decoded_per_request={}, prefilling_count=prefilling, decoding_count=decoding,
    )


def state(reqs, *, now=5.0, step=1, gpus=None):
    return ObservableState(time=now, waiting_queue=reqs, gpu_states=gpus or [gpu()], completed_count=0, step=step)


# ---------------------------------------------------------------------------
# Registry contracts
# ---------------------------------------------------------------------------


def test_registry_has_all_required_families_represented():
    families = {spec.family for spec in prim.list_primitives()}
    assert families == set(prim.PrimitiveFamily)


def test_registry_names_are_unique():
    names = [spec.name for spec in prim.list_primitives()]
    assert len(names) == len(set(names))


def test_required_primitive_families_from_roadmap_are_present():
    required = {
        "deadline_urgency", "laxity", "prompt_length", "predicted_output_length",
        "estimated_service_time", "priority", "queue_age", "kv_pressure",
        "projected_gpu_load", "admission_risk", "prefill_pressure",
        "fairness_starvation_bonus",
    }
    registered = {spec.name for spec in prim.list_primitives()}
    assert required <= registered


def test_get_primitive_spec_unknown_name_raises():
    with pytest.raises(prim.PrimitiveError):
        prim.get_primitive_spec("does_not_exist")


def test_register_duplicate_name_raises():
    spec = prim.get_primitive_spec("priority")
    with pytest.raises(prim.PrimitiveError):
        prim.register_primitive(spec)


def test_every_primitive_has_a_nonempty_doc():
    for spec in prim.list_primitives():
        assert spec.doc and len(spec.doc) > 20


def test_every_primitive_is_causal_and_deterministic():
    for spec in prim.list_primitives():
        assert spec.causal is True
        assert spec.deterministic is True


def test_list_primitives_filters_by_family():
    ranking = prim.list_primitives(prim.PrimitiveFamily.RANKING)
    assert ranking
    assert all(s.family == prim.PrimitiveFamily.RANKING for s in ranking)
    admission = prim.list_primitives(prim.PrimitiveFamily.ADMISSION)
    assert admission
    assert all(s.family == prim.PrimitiveFamily.ADMISSION for s in admission)


# ---------------------------------------------------------------------------
# Parameter bounds and explicit unsupported-state errors
# ---------------------------------------------------------------------------


def test_param_bound_rejects_out_of_range_value():
    s = state([req(1)])
    with pytest.raises(prim.PrimitiveError):
        prim.LAXITY.value(req(1), s, alpha=-1.0)


def test_param_bound_rejects_nan():
    s = state([req(1)])
    with pytest.raises(prim.PrimitiveError):
        prim.LAXITY.value(req(1), s, alpha=float("nan"))


def test_unknown_parameter_name_raises():
    s = state([req(1)])
    with pytest.raises(prim.PrimitiveError):
        prim.PRIORITY.value(req(1), s, not_a_real_param=1.0)


def test_param_bound_accepts_default_when_omitted():
    s = state([req(1, prompt=100, output=50)])
    value = prim.ESTIMATED_SERVICE_TIME.value(req(1, prompt=100, output=50), s)
    assert value == pytest.approx(0.5 * 100 + 1.0 * 50)


def test_feasible_on_gpu_raises_on_non_positive_capacity():
    bad_gpu = gpu(max_seq=0)
    with pytest.raises(prim.PrimitiveError):
        prim.feasible_on_gpu(bad_gpu, req(1))


def test_feasible_on_gpu_matches_capacity_arithmetic():
    g = gpu(max_seq=1, max_kv=100, max_batch=1)
    assert prim.feasible_on_gpu(g, req(1, prompt=50)) is True
    assert prim.feasible_on_gpu(g, req(1, prompt=200)) is False


# ---------------------------------------------------------------------------
# Ranking family
# ---------------------------------------------------------------------------


def test_deadline_urgency_orders_earliest_deadline_first():
    reqs = [req(1, deadline=50.0), req(2, deadline=5.0), req(3, deadline=20.0)]
    s = state(reqs)
    ranked = prim.rank_requests(s, [(prim.DEADLINE_URGENCY, {}), (prim.REQUEST_ID_TIEBREAK, {})])
    assert [r.request_id for r in ranked] == [2, 3, 1]


def test_queue_age_ordering_equals_arrival_ordering_within_one_snapshot():
    reqs = [req(1, arrival=3.0), req(2, arrival=1.0), req(3, arrival=2.0)]
    s = state(reqs, now=10.0)
    ranked = prim.rank_requests(s, [(prim.QUEUE_AGE, {}), (prim.REQUEST_ID_TIEBREAK, {})])
    assert [r.request_id for r in ranked] == [2, 3, 1]


def test_queue_age_clamped_to_zero_for_future_arrival():
    r = req(1, arrival=100.0)
    s = state([r], now=0.0)
    assert prim.QUEUE_AGE.value(r, s) == 0.0


def test_priority_higher_preferred_first():
    reqs = [req(1, priority=1.0), req(2, priority=5.0), req(3, priority=2.0)]
    s = state(reqs)
    ranked = prim.rank_requests(s, [(prim.PRIORITY, {}), (prim.REQUEST_ID_TIEBREAK, {})])
    assert [r.request_id for r in ranked] == [2, 3, 1]


def test_wsp_score_matches_scoring_module_formula():
    from llmserveopt.policies.scoring import weighted_shortest_processing_score
    r = req(1, prompt=200, output=50, priority=2.0)
    s = state([r])
    expected = weighted_shortest_processing_score(r)
    assert prim.WEIGHTED_SHORTEST_PROCESSING_SCORE.value(r, s) == pytest.approx(expected)


def test_laxity_urgency_is_derived_from_laxity_inverse():
    r = req(1, deadline=10.0, prompt=100, output=100)
    s = state([r], now=1.0)
    lax = prim.LAXITY.value(r, s)
    urgency = prim.LAXITY_URGENCY.value(r, s)
    assert urgency == pytest.approx(1.0 / max(lax, 1e-9))


def test_build_ranking_key_requires_at_least_one_component():
    s = state([req(1)])
    with pytest.raises(prim.PrimitiveError):
        prim.build_ranking_key([], s)


def test_request_id_tiebreak_breaks_exact_ties():
    reqs = [req(2, prompt=64, output=64), req(1, prompt=64, output=64)]
    s = state(reqs)
    ranked = prim.rank_requests(s, [(prim.PROMPT_LENGTH, {}), (prim.REQUEST_ID_TIEBREAK, {})])
    assert [r.request_id for r in ranked] == [1, 2]


# ---------------------------------------------------------------------------
# Admission family
# ---------------------------------------------------------------------------


def test_laxity_gate_filters_requests_that_already_missed_slack():
    urgent = req(1, deadline=-100.0, prompt=1, output=1)
    fine = req(2, deadline=1000.0, prompt=1, output=1)
    s = state([urgent, fine], now=0.0)
    assert prim.LAXITY_GATE.passes(urgent, s, laxity_threshold=0.0) is False
    assert prim.LAXITY_GATE.passes(fine, s, laxity_threshold=0.0) is True


def test_laxity_gate_infinite_threshold_admits_everything():
    r = req(1, deadline=-100.0)
    s = state([r], now=0.0)
    assert prim.LAXITY_GATE.passes(r, s) is True


def test_ttft_slack_gate_uses_prefill_only_proxy():
    r = req(1, deadline=1.0, prompt=1_000_000, output=1)
    s = state([r], now=0.0)
    assert prim.TTFT_SLACK_GATE.passes(r, s, ttft_slack_threshold=0.0) is False


def test_admission_risk_agrees_with_laxity_gate_at_threshold_half():
    r = req(1, deadline=5.0, prompt=10, output=10)
    s = state([r], now=0.0)
    risk = prim.ADMISSION_RISK.value(r, s)
    gate_pass = prim.LAXITY_GATE.passes(r, s, laxity_threshold=0.0)
    assert (risk <= 0.5) == gate_pass


def test_admission_risk_is_bounded_in_unit_interval():
    for deadline in (-1000.0, 0.0, 5.0, 1000.0):
        r = req(1, deadline=deadline)
        s = state([r], now=0.0)
        risk = prim.ADMISSION_RISK.value(r, s)
        assert 0.0 <= risk <= 1.0


# ---------------------------------------------------------------------------
# Resource-guard family
# ---------------------------------------------------------------------------


def test_system_kv_pressure_is_max_over_gpus():
    g1 = gpu(gpu_id=0, max_kv=100, kv_used=10)
    g2 = gpu(gpu_id=1, max_kv=100, kv_used=90)
    s = state([], gpus=[g1, g2])
    assert prim.system_kv_pressure(s) == pytest.approx(0.9)


def test_system_kv_pressure_empty_gpu_list_is_zero():
    s = state([], gpus=[])
    assert prim.system_kv_pressure(s) == 0.0


def test_decode_pressure_matches_scorpio_formula():
    g = gpu(max_seq=4, decoding=3)
    s = state([], gpus=[g])
    assert prim.decode_pressure(s) == pytest.approx(0.75)


def test_prefill_pressure_matches_composition_formula():
    g = gpu(max_seq=4, prefilling=1)
    s = state([], gpus=[g])
    assert prim.prefill_pressure(s) == pytest.approx(0.25)


def test_system_overload_guard_true_when_any_threshold_exceeded():
    assert prim.system_overload_guard(
        kv_pressure=0.9, decode_pressure=0.0, queue_pressure=0.0, mean_laxity=10.0,
        kv_threshold=0.65, decode_threshold=0.70, queue_threshold=3.0,
    ) is True
    assert prim.system_overload_guard(
        kv_pressure=0.0, decode_pressure=0.0, queue_pressure=0.0, mean_laxity=10.0,
        kv_threshold=0.65, decode_threshold=0.70, queue_threshold=3.0,
    ) is False


def test_system_overload_guard_true_when_mean_laxity_negative():
    assert prim.system_overload_guard(
        kv_pressure=0.0, decode_pressure=0.0, queue_pressure=0.0, mean_laxity=-0.01,
        kv_threshold=0.65, decode_threshold=0.70, queue_threshold=3.0,
    ) is True


# ---------------------------------------------------------------------------
# Placement family
# ---------------------------------------------------------------------------


def test_tightest_kv_fit_prefers_smallest_remaining_capacity():
    g_loose = gpu(gpu_id=0, max_kv=1000, kv_used=0)
    g_tight = gpu(gpu_id=1, max_kv=1000, kv_used=900)
    r = req(1, prompt=10)
    key_loose = prim.TIGHTEST_KV_FIT.key(g_loose, r)
    key_tight = prim.TIGHTEST_KV_FIT.key(g_tight, r)
    assert key_tight < key_loose


def test_projected_gpu_load_matches_gpu_pressure_helper():
    from llmserveopt.policies.policy_library_v2_helpers import gpu_pressure
    g = gpu(max_seq=4, max_kv=1000, kv_used=100, prefilling=1)
    assert prim.PROJECTED_GPU_LOAD.key(g, req(1))[0] == pytest.approx(gpu_pressure(g))


def test_place_round_robin_respects_feasibility():
    g0 = gpu(gpu_id=0, max_seq=1)
    g1 = gpu(gpu_id=1, max_seq=1)
    r1, r2, r3 = req(1), req(2), req(3)
    s = state([r1, r2, r3], gpus=[g0, g1])
    action = prim.place_round_robin(s, [r1, r2, r3])
    admitted = action.all_admitted_ids()
    assert admitted == {1, 2}  # third request has no feasible GPU this step


def test_place_round_robin_no_feasible_gpu_leaves_request_unplaced():
    r1, r2 = req(1), req(2)
    g_full = gpu(gpu_id=0, max_seq=1, active=[99])
    g_open = gpu(gpu_id=1, max_seq=1)
    s = state([r1, r2], gpus=[g_full, g_open])
    action = prim.place_round_robin(s, [r1, r2], advance_index_on_failure=False)
    assert action.all_admitted_ids() == {1}


def test_place_greedy_key_admits_nothing_when_no_gpu_feasible():
    g = gpu(max_seq=1, active=[99])
    s = state([req(1)], gpus=[g])
    action = prim.place_greedy_key(s, [req(1)], prim.TIGHTEST_KV_FIT)
    assert action.all_admitted_ids() == set()


# ---------------------------------------------------------------------------
# Batching family
# ---------------------------------------------------------------------------


def test_token_budget_remaining_matches_feasibility_module():
    from llmserveopt.policies.feasibility import token_budget_remaining as tbr
    g = gpu(max_batch=10, active=[1, 2, 3])
    assert prim.token_budget_remaining(g) == tbr(g)


def test_admission_credit_budget_refill_caps_at_max():
    budget = prim.AdmissionCreditBudget(refill_per_step=10.0, max_budget=4.0, cost_per_admit=1.0)
    budget.consume(4)
    assert budget.remaining() == 0.0
    budget.refill()
    assert budget.remaining() == 4.0  # capped, not 10.0


def test_admission_credit_budget_max_admits_is_at_least_one():
    budget = prim.AdmissionCreditBudget(refill_per_step=0.0, max_budget=4.0, cost_per_admit=1.0)
    budget.consume(100)
    assert budget.remaining() == 0.0
    assert budget.max_admits() == 1


def test_admission_credit_budget_rejects_negative_params():
    with pytest.raises(prim.PrimitiveError):
        prim.AdmissionCreditBudget(refill_per_step=-1.0)

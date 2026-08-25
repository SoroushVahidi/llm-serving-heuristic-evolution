"""CC3: compositional DSL/verifier/compiler tests.

Covers the 8 required CC3 constructs (named primitive references, weighted
sums, sparse top-k mixtures, conditional branches, admission gates,
placement scores, externally supplied parameters, deterministic
tie-breaking) plus safe fallback, canonical hashing, execution-cost limits,
legacy compatibility, verifier error codes, and compiler/runtime
determinism. Primitive-interface-only tests live in test_primitive_interface.py
and test_primitive_reconstructed_policies.py (CC2); this file is CC3-only.
"""
from __future__ import annotations

import copy

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.heuristics import primitive_bridge as bridge
from llmserveopt.heuristics.compiler import CompilationError, compile_heuristic
from llmserveopt.heuristics.dsl_schema import heuristic_hash
from llmserveopt.heuristics.examples import edf_like, fifo_like, slo_kv_balanced, throughput_oriented
from llmserveopt.heuristics.policy import build_heuristic_policy
from llmserveopt.heuristics.primitive_composition_examples import (
    ALL_PRIMITIVE_COMPOSITION_EXAMPLES,
    admission_gate_with_fallback,
    bounded_external_parameter_example,
    conditional_kv_pressure_branch,
    edf_primitive,
    placement_score_composition,
    sparse_topk_ranking_mixture,
    weighted_deadline_length_ranking,
)
from llmserveopt.heuristics.verifier import verify_heuristic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_error(result, code):
    return any(c == code for c, _ in result.errors)


def req(rid, *, prompt=64, output=64, arrival=0.0, deadline=20.0, priority=1.0):
    return ObservableRequest(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline, priority=priority, class_id="medium",
    )


def gpu(gid, *, max_seq=4, max_kv=4096, max_batch=128, active=None, kv=0, prefilling=0, decoding=0):
    return ObservableGPUState(
        gpu_id=gid, max_active_sequences=max_seq, max_batch_tokens=max_batch, max_kv_tokens=max_kv,
        active_request_ids=list(active or []), active_requests_info=[], current_kv_tokens=kv,
        tokens_decoded_per_request={}, prefilling_count=prefilling, decoding_count=decoding,
    )


def state(reqs, *, now=0.0, gpus=None, step=1):
    return ObservableState(time=now, waiting_queue=reqs, gpu_states=gpus or [gpu(0)], completed_count=0, step=step)


# ---------------------------------------------------------------------------
# 1. Schema parsing / primitive resolution
# ---------------------------------------------------------------------------


def test_all_seven_construct_examples_verify_and_compile():
    for fn in ALL_PRIMITIVE_COMPOSITION_EXAMPLES:
        doc = fn()
        r = verify_heuristic(doc)
        assert r.valid, (fn.__name__, r.errors)
        compile_heuristic(doc)  # must not raise


def test_unknown_primitive_rejected():
    doc = {
        "name": "bad_primitive",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"primitive": "does_not_exist"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PRIMITIVE_UNKNOWN")


def test_primitive_wrong_shape_rejected():
    # projected_gpu_load is a PLACEMENT key, not a value/gate primitive.
    doc = {
        "name": "wrong_shape",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"primitive": "projected_gpu_load"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PRIMITIVE_WRONG_SHAPE")


def test_gate_primitive_used_as_value_rejected():
    doc = {
        "name": "gate_as_value",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"primitive": "laxity_gate"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PRIMITIVE_WRONG_SHAPE")


def test_stateful_primitive_inline_rejected():
    doc = {
        "name": "stateful_inline",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"primitive": "admission_credit_budget"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "STATEFUL_PRIMITIVE_MISPLACED")


def test_primitive_param_out_of_bounds_rejected():
    doc = {
        "name": "bad_param",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"primitive": "laxity", "params": {"alpha": -5.0}},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PRIMITIVE_PARAM_INVALID")


def test_reserved_var_name_rejected():
    doc = {
        "name": "reserved_var",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"var": "__prim__::sneaky"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "RESERVED_VAR_NAME")


# ---------------------------------------------------------------------------
# 2. Weighted sums / mixture family compatibility / normalization
# ---------------------------------------------------------------------------


def test_weighted_sum_empty_mixture_rejected():
    doc = {
        "name": "empty_mix",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"op": "weighted_sum", "terms": []}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "MIXTURE_EMPTY")


def test_weighted_sum_incompatible_families_rejected():
    doc = {
        "name": "incompatible_mix",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {
                "op": "weighted_sum",
                "terms": [[{"primitive": "projected_gpu_load"}, 0.5], [{"primitive": "priority"}, 0.5]],
            }
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "MIXTURE_FAMILY_INCOMPATIBLE")


def test_weighted_sum_is_unnormalized_literal_coefficients():
    """Documented CC3 normalization rule: weighted_sum weights are used as
    literal, deterministic coefficients -- no implicit renormalization."""
    doc = {
        "name": "literal_weights",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {
                "op": "weighted_sum",
                "terms": [[{"const": 2.0}, 3.0], [{"const": 1.0}, 4.0]],
            }
        },
    }
    compiled = compile_heuristic(doc)
    score = compiled.score_request({}, {}, {})
    assert score == pytest.approx(2.0 * 3.0 + 1.0 * 4.0)


# ---------------------------------------------------------------------------
# 3. Sparse top-k mixtures
# ---------------------------------------------------------------------------


def test_topk_invalid_k_rejected():
    doc = {
        "name": "bad_k",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {
                "op": "topk_mixture",
                "k": 5,
                "terms": [[{"const": 1.0}, 1.0], [{"const": 2.0}, 1.0]],
            }
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "TOPK_INVALID_K")


def test_topk_mixture_selects_largest_abs_weight_terms_deterministically():
    doc = {
        "name": "topk_select",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {
                "op": "topk_mixture",
                "k": 2,
                "terms": [
                    [{"const": 10.0}, 1.0],
                    [{"const": 100.0}, 5.0],
                    [{"const": 1000.0}, 0.1],
                ],
            }
        },
    }
    compiled = compile_heuristic(doc)
    score = compiled.score_request({}, {}, {})
    # Largest |weight| terms are (100,5.0) and (10,1.0); (1000,0.1) excluded.
    assert score == pytest.approx(100.0 * 5.0 + 10.0 * 1.0)


def test_topk_mixture_tie_breaks_by_ascending_term_index():
    doc = {
        "name": "topk_tie",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {
                "op": "topk_mixture",
                "k": 1,
                "terms": [[{"const": 7.0}, 2.0], [{"const": 9.0}, 2.0]],
            }
        },
    }
    compiled = compile_heuristic(doc)
    score = compiled.score_request({}, {}, {})
    assert score == pytest.approx(7.0 * 2.0)  # first term wins the tie


# ---------------------------------------------------------------------------
# 4. Conditional branches
# ---------------------------------------------------------------------------


def test_conditional_kv_pressure_branch_switches_regime():
    doc = conditional_kv_pressure_branch()
    compiled = compile_heuristic(doc)
    r1 = req(1)
    low_state = state([r1], gpus=[gpu(0, kv=100)])   # low KV pressure -> default (laxity_urgency)
    high_state = state([r1], gpus=[gpu(0, kv=4000)])  # high KV pressure -> regime_0 (neg prompt_length)

    pol = build_heuristic_policy(doc)
    pol.select_action(low_state)
    assert pol.last_trace["active_regime"] == "default"
    pol.select_action(high_state)
    assert pol.last_trace["active_regime"] == "regime_0"


def test_bool_and_or_not_ops():
    from llmserveopt.heuristics.expressions import evaluate_expression

    ctx = {"a": 1.0, "b": -1.0}
    assert evaluate_expression({"op": "bool_and", "args": [{"var": "a"}, {"var": "a"}]}, ctx) == 1.0
    assert evaluate_expression({"op": "bool_and", "args": [{"var": "a"}, {"var": "b"}]}, ctx) == 0.0
    assert evaluate_expression({"op": "bool_or", "args": [{"var": "a"}, {"var": "b"}]}, ctx) == 1.0
    assert evaluate_expression({"op": "bool_or", "args": [{"var": "b"}, {"var": "b"}]}, ctx) == 0.0
    assert evaluate_expression({"op": "bool_not", "args": [{"var": "b"}]}, ctx) == 1.0


# ---------------------------------------------------------------------------
# 5. Admission gates + safe fallback
# ---------------------------------------------------------------------------


def test_admission_gate_requires_on_no_admits():
    doc = admission_gate_with_fallback()
    del doc["on_no_admits"]
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "ON_NO_ADMITS_MISSING")


def test_admission_gate_invalid_on_no_admits_value_rejected():
    doc = admission_gate_with_fallback()
    doc["on_no_admits"] = "do_something_weird"
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "ON_NO_ADMITS_INVALID")


def test_legacy_admission_condition_without_primitive_gate_does_not_require_on_no_admits():
    """Backward compatibility: pre-CC3 admission_condition (plain var/op, no
    primitive_gate) must NOT trigger ON_NO_ADMITS_MISSING."""
    doc = {
        "name": "legacy_admission",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"const": 1.0},
            "admission_condition": {"op": "sub", "args": [{"var": "sys.kv_utilization"}, {"const": 0.9}]},
        },
    }
    r = verify_heuristic(doc)
    assert r.valid, r.errors


def test_safe_fallback_activates_when_gate_rejects_everyone():
    doc = admission_gate_with_fallback()
    pol = build_heuristic_policy(doc)
    # Modest token counts (comfortably GPU-feasible) but a deadline already
    # in the past relative to estimated service time, so laxity < 0.
    tight = req(1, prompt=64, output=64, deadline=0.001)
    s = state([tight])
    action = pol.select_action(s)
    # fifo_like fallback has no admission_condition, so the request is admitted via the fallback path.
    assert tight.request_id in action.admit[0]


def test_fallback_invalid_policy_name_rejected():
    doc = {
        "name": "bad_fallback",
        "tie_breaker": "arrival_order",
        "fallback": {"policy": "not_a_real_policy"},
        "default": {"request_score": {"const": 1.0}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "FALLBACK_INVALID")


def test_fallback_defaults_to_fifo_like_when_absent():
    doc = {
        "name": "no_fallback_declared",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"const": 1.0}},
    }
    compiled = compile_heuristic(doc)
    assert compiled.fallback_name == "fifo_like"
    assert compiled.fallback is not None


def test_canonical_fallback_policies_have_no_further_fallback():
    compiled = compile_heuristic(fifo_like())
    assert compiled.fallback is None
    assert compiled.fallback_name is None


# ---------------------------------------------------------------------------
# 6. Placement scores
# ---------------------------------------------------------------------------


def test_placement_empty_keys_rejected():
    doc = placement_score_composition()
    doc["placement"]["keys"] = []
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PLACEMENT_EMPTY")


def test_placement_unknown_key_rejected():
    doc = placement_score_composition()
    doc["placement"]["keys"] = [{"name": "not_a_placement_primitive"}]
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PLACEMENT_KEY_UNKNOWN")


def test_placement_prefers_least_loaded_gpu():
    doc = placement_score_composition()
    pol = build_heuristic_policy(doc)
    r1 = req(1)
    s = state([r1], gpus=[gpu(0, kv=4000), gpu(1, kv=0)])
    action = pol.select_action(s)
    assert action.admit[1] == [1]
    assert action.admit[0] == []


def test_placement_key_always_ends_in_gpu_id_tie_break():
    key_fn = bridge.build_composite_placement_key([("projected_gpu_load", {})])
    g = gpu(7, kv=0)
    r = req(1)
    key = key_fn(g, r)
    assert key[-1] == 7


# ---------------------------------------------------------------------------
# 7. Externally supplied parameters
# ---------------------------------------------------------------------------


def test_param_undeclared_rejected():
    doc = {
        "name": "undeclared_param",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"param": "not_declared"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PARAM_UNDECLARED")


def test_param_schema_missing_field_rejected():
    doc = {
        "name": "bad_param_schema",
        "tie_breaker": "arrival_order",
        "parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}],  # missing 'default'
        "default": {"request_score": {"param": "x"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PARAM_SCHEMA_INVALID")


def test_param_duplicate_name_rejected():
    doc = {
        "name": "dup_param",
        "tie_breaker": "arrival_order",
        "parameters": [
            {"name": "x", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            {"name": "x", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
        ],
        "default": {"request_score": {"param": "x"}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "PARAM_DUPLICATE_NAME")


def test_param_default_resolution_and_override():
    doc = bounded_external_parameter_example()
    compiled_default = compile_heuristic(doc)
    assert compiled_default.resolved_params["kv_weight"] == pytest.approx(0.5)

    compiled_override = compile_heuristic(doc, param_overrides={"kv_weight": 0.9})
    assert compiled_override.resolved_params["kv_weight"] == pytest.approx(0.9)


def test_unknown_parameter_override_rejected():
    doc = bounded_external_parameter_example()
    with pytest.raises(CompilationError):
        compile_heuristic(doc, param_overrides={"not_a_param": 0.1})


def test_parameter_override_out_of_bounds_rejected():
    doc = bounded_external_parameter_example()
    with pytest.raises(CompilationError):
        compile_heuristic(doc, param_overrides={"kv_weight": 5.0})


# ---------------------------------------------------------------------------
# 8. Deterministic tie-breaking / determinism / replay
# ---------------------------------------------------------------------------


def test_deterministic_replay_same_program_context_state():
    doc = sparse_topk_ranking_mixture()
    pol_a = build_heuristic_policy(doc)
    pol_b = build_heuristic_policy(doc)
    reqs = [req(1, deadline=5.0, priority=2.0), req(2, deadline=50.0, priority=1.0), req(3, deadline=1.0)]
    s_a = state(copy.deepcopy(reqs))
    s_b = state(copy.deepcopy(reqs))
    action_a = pol_a.select_action(s_a)
    action_b = pol_b.select_action(s_b)
    assert action_a.admit == action_b.admit


def test_compile_is_pure_and_repeatable():
    doc = weighted_deadline_length_ranking()
    c1 = compile_heuristic(doc)
    c2 = compile_heuristic(doc)
    ctx_req = {
        "req.prompt_tokens": 64.0, "req.predicted_output_tokens": 64.0, "req.waiting_time": 1.0,
        "req.deadline_slack": 5.0, "req.deadline_urgency": 2.0, "req.priority_weight": 1.0,
        "req.estimated_prefill_cost": 64.0, "req.estimated_decode_cost": 64.0, "req.estimated_kv_cost": 128.0,
    }
    r = req(1)
    s = state([r])
    ctx = dict(ctx_req)
    ctx.update(bridge.build_runtime_context(c1.primitive_refs, c1.resolved_params, r, s))
    assert c1.score_request(ctx, {}, {}) == c2.score_request(ctx, {}, {})


# ---------------------------------------------------------------------------
# 9. Execution-cost budget
# ---------------------------------------------------------------------------


def test_primitive_budget_exceeded_rejected():
    terms = [[{"primitive": n}, 1.0] for n in ("priority", "queue_age", "prompt_length", "predicted_output_length")]
    # 4 distinct primitives is fine; force the limit down to 2 via extra_limits.
    doc = {
        "name": "budget_test",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"op": "weighted_sum", "terms": terms}},
    }
    r = verify_heuristic(doc, extra_limits={"max_active_primitives": 2})
    assert not r.valid
    assert has_error(r, "PRIMITIVE_BUDGET_EXCEEDED")


# ---------------------------------------------------------------------------
# 10. Canonical serialization / hash stability
# ---------------------------------------------------------------------------


def test_heuristic_hash_stable_across_key_order():
    doc_a = {"name": "hash_test", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 1.0}}}
    doc_b = {"default": {"request_score": {"const": 1.0}}, "tie_breaker": "arrival_order", "name": "hash_test"}
    assert heuristic_hash(doc_a) == heuristic_hash(doc_b)


def test_heuristic_hash_changes_with_content():
    doc_a = {"name": "hash_test", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 1.0}}}
    doc_b = {"name": "hash_test", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 2.0}}}
    assert heuristic_hash(doc_a) != heuristic_hash(doc_b)


# ---------------------------------------------------------------------------
# 11. Legacy DSL compatibility (regression-pinned)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [fifo_like, edf_like, slo_kv_balanced, throughput_oriented])
def test_legacy_examples_still_verify_and_compile(builder):
    doc = builder()
    r = verify_heuristic(doc)
    assert r.valid, r.errors
    compile_heuristic(doc)  # must not raise


def test_legacy_edf_like_score_unchanged():
    compiled = compile_heuristic(edf_like())
    req_vars = {"req.deadline_urgency": 3.5}
    assert compiled.score_request(req_vars, {}, {}) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 12. Representative-policy equivalence (primitive-ref vs raw-var EDF)
# ---------------------------------------------------------------------------


def test_edf_primitive_matches_edf_like_var_ranking():
    raw_doc = edf_like()
    prim_doc = edf_primitive()
    reqs = [req(1, deadline=1.0), req(2, deadline=50.0), req(3, deadline=10.0)]

    pol_raw = build_heuristic_policy(raw_doc)
    pol_prim = build_heuristic_policy(prim_doc)
    s_raw = state(copy.deepcopy(reqs))
    s_prim = state(copy.deepcopy(reqs))
    action_raw = pol_raw.select_action(s_raw)
    action_prim = pol_prim.select_action(s_prim)
    # edf_like ranks by raw deadline_urgency (ascending slo_deadline preferred);
    # edf_primitive ranks by -deadline_urgency (same ordering, negated primitive).
    assert action_raw.admit[0] == action_prim.admit[0]


# ---------------------------------------------------------------------------
# 13. Admission budget (stateful primitive)
# ---------------------------------------------------------------------------


def test_admission_budget_caps_admits_per_step():
    doc = {
        "name": "budget_capped",
        "tie_breaker": "arrival_order",
        "admission_budget": {
            "primitive": "admission_credit_budget",
            "params": {"refill_per_step": 1.0, "max_budget": 1.0, "cost_per_admit": 1.0},
        },
        "default": {"request_score": {"const": 1.0}},
    }
    pol = build_heuristic_policy(doc)
    reqs = [req(i) for i in range(1, 6)]
    s = state(reqs, gpus=[gpu(0, max_seq=10, max_kv=10000, max_batch=10)])
    action = pol.select_action(s)
    total_admitted = sum(len(v) for v in action.admit.values())
    assert total_admitted == 1  # budget caps to max(1, int(1.0)) == 1


def test_admission_budget_invalid_primitive_name_rejected():
    doc = {
        "name": "bad_budget",
        "tie_breaker": "arrival_order",
        "admission_budget": {"primitive": "priority"},
        "default": {"request_score": {"const": 1.0}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "ADMISSION_BUDGET_INVALID")

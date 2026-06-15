"""Tests for the DSL heuristic verifier."""
import pytest
from llmserveopt.heuristics.verifier import verify_heuristic
from llmserveopt.heuristics.examples import fifo_like, edf_like, slo_kv_balanced, throughput_oriented


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_error(result, code):
    return any(c == code for c, _ in result.errors)


# ---------------------------------------------------------------------------
# Valid heuristics
# ---------------------------------------------------------------------------

def test_fifo_like_passes():
    r = verify_heuristic(fifo_like())
    assert r.valid, r.errors


def test_edf_like_passes():
    r = verify_heuristic(edf_like())
    assert r.valid, r.errors


def test_slo_kv_balanced_passes():
    r = verify_heuristic(slo_kv_balanced())
    assert r.valid, r.errors


def test_throughput_oriented_passes():
    r = verify_heuristic(throughput_oriented())
    assert r.valid, r.errors


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

def test_missing_name():
    doc = {k: v for k, v in fifo_like().items() if k != "name"}
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "SCHEMA_MISSING_FIELD")


def test_missing_tie_breaker():
    doc = {k: v for k, v in edf_like().items() if k != "tie_breaker"}
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "SCHEMA_MISSING_FIELD")


def test_missing_default():
    doc = {k: v for k, v in edf_like().items() if k != "default"}
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "SCHEMA_MISSING_FIELD")


# ---------------------------------------------------------------------------
# Forbidden variables
# ---------------------------------------------------------------------------

def test_actual_output_tokens_forbidden():
    doc = {
        "name": "bad_heuristic",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"var": "req.actual_output_tokens"},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    # Should fail with FORBIDDEN_VARIABLE, FORBIDDEN_SUBSTRING, or UNKNOWN_VARIABLE
    assert any(c in ("FORBIDDEN_VARIABLE", "FORBIDDEN_SUBSTRING", "UNKNOWN_VARIABLE") for c, _ in r.errors)


def test_future_var_forbidden():
    doc = {
        "name": "future_heuristic",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"var": "future_arrivals"},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid


def test_oracle_var_forbidden():
    doc = {
        "name": "oracle_cheat",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"var": "oracle_policy"},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid


def test_unknown_var_forbidden():
    doc = {
        "name": "unknown_var",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"var": "req.nonexistent_field"},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "UNKNOWN_VARIABLE")


# ---------------------------------------------------------------------------
# Forbidden operations
# ---------------------------------------------------------------------------

def test_eval_op_forbidden():
    doc = {
        "name": "eval_cheat",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"op": "eval", "args": [{"const": 1.0}]},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "FORBIDDEN_OP")


def test_unknown_op_rejected():
    doc = {
        "name": "unknown_op",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"op": "banana", "args": []},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "UNKNOWN_OP")


# ---------------------------------------------------------------------------
# Forbidden tie-breaker
# ---------------------------------------------------------------------------

def test_bad_tie_breaker():
    doc = {
        "name": "bad_tb",
        "tie_breaker": "random_shuffle",
        "default": {
            "request_score": {"const": 1.0},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "FORBIDDEN_TIE_BREAKER")


# ---------------------------------------------------------------------------
# Constant range checks
# ---------------------------------------------------------------------------

def test_constant_too_large():
    doc = {
        "name": "huge_const",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"const": 99999.0},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "CONSTANT_OUT_OF_RANGE")


def test_constant_inf_rejected():
    doc = {
        "name": "inf_const",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {"const": float("inf")},
        },
    }
    r = verify_heuristic(doc)
    assert not r.valid


# ---------------------------------------------------------------------------
# Regime checks
# ---------------------------------------------------------------------------

def test_too_many_regimes():
    regime = {
        "condition": {"const": 1.0},
        "request_score": {"var": "req.deadline_urgency"},
    }
    doc = {
        "name": "too_many",
        "tie_breaker": "earliest_deadline",
        "regimes": [regime] * 10,
        "default": {"request_score": {"const": 0.0}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "TOO_MANY_REGIMES")


def test_regime_missing_condition():
    doc = {
        "name": "no_cond",
        "tie_breaker": "arrival_order",
        "regimes": [{"request_score": {"const": 1.0}}],
        "default": {"request_score": {"const": 0.0}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "REGIME_MISSING_CONDITION")


def test_regime_missing_request_score():
    doc = {
        "name": "no_score",
        "tie_breaker": "arrival_order",
        "regimes": [{"condition": {"const": 1.0}}],
        "default": {"request_score": {"const": 0.0}},
    }
    r = verify_heuristic(doc)
    assert not r.valid
    assert has_error(r, "REGIME_MISSING_SCORE")


# ---------------------------------------------------------------------------
# Expression depth / size
# ---------------------------------------------------------------------------

def test_expression_depth_exceeded():
    expr = {"const": 1.0}
    for _ in range(20):
        expr = {"op": "neg", "args": [expr]}
    doc = {
        "name": "deep",
        "tie_breaker": "arrival_order",
        "default": {"request_score": expr},
    }
    r = verify_heuristic(doc, extra_limits={"max_expression_depth": 5})
    assert not r.valid
    assert has_error(r, "EXPRESSION_DEPTH_EXCEEDED")


# ---------------------------------------------------------------------------
# Not-a-dict input
# ---------------------------------------------------------------------------

def test_not_a_dict():
    r = verify_heuristic("I am not a dict")
    assert not r.valid
    assert has_error(r, "SCHEMA_INVALID_TYPE")


def test_empty_dict_fails():
    r = verify_heuristic({})
    assert not r.valid

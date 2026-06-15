"""
Tests that verify the heuristic DSL cannot access forbidden information:
- actual_output_tokens
- future arrival information
- oracle policy labels
- non-online variables
"""
import pytest
from llmserveopt.heuristics.verifier import verify_heuristic
from llmserveopt.heuristics.dsl_schema import (
    FORBIDDEN_SUBSTRINGS,
    FORBIDDEN_VARS,
    ALLOWED_VARS,
)


def _doc_with_var(var_name):
    return {
        "name": "test",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"var": var_name}},
    }


# --- forbidden vars from FORBIDDEN_VARS explicitly ---

@pytest.mark.parametrize("var_name", sorted(FORBIDDEN_VARS))
def test_explicit_forbidden_vars_rejected(var_name):
    r = verify_heuristic(_doc_with_var(var_name))
    assert not r.valid, f"Expected {var_name!r} to be rejected but it passed"


# --- forbidden substrings ---

@pytest.mark.parametrize("substring", FORBIDDEN_SUBSTRINGS)
def test_forbidden_substring_rejected(substring):
    var_name = f"req.{substring}_field"
    r = verify_heuristic(_doc_with_var(var_name))
    assert not r.valid, f"Expected var containing '{substring}' to be rejected"


# --- actual_output_tokens specifically ---

def test_actual_output_tokens_in_var():
    r = verify_heuristic(_doc_with_var("req.actual_output_tokens"))
    assert not r.valid


def test_actual_output_in_var_name():
    r = verify_heuristic(_doc_with_var("actual_output_length"))
    assert not r.valid


# --- completion_time forbidden ---

def test_completion_time_rejected():
    r = verify_heuristic(_doc_with_var("req.completion_time"))
    assert not r.valid


def test_completion_time_substring():
    r = verify_heuristic(_doc_with_var("req.completion_time_predicted"))
    assert not r.valid


# --- future_ prefix rejected ---

def test_future_arrivals_rejected():
    r = verify_heuristic(_doc_with_var("future_arrivals"))
    assert not r.valid


def test_future_queue_rejected():
    r = verify_heuristic(_doc_with_var("future_queue_length"))
    assert not r.valid


# --- oracle_ prefix rejected ---

def test_oracle_policy_rejected():
    r = verify_heuristic(_doc_with_var("oracle_policy"))
    assert not r.valid


def test_oracle_label_rejected():
    r = verify_heuristic(_doc_with_var("oracle_label"))
    assert not r.valid


# --- ground_truth rejected ---

def test_ground_truth_rejected():
    r = verify_heuristic(_doc_with_var("ground_truth_output"))
    assert not r.valid


# --- hidden_ rejected ---

def test_hidden_var_rejected():
    r = verify_heuristic(_doc_with_var("hidden_output_tokens"))
    assert not r.valid


# --- all ALLOWED_VARS are accepted ---

@pytest.mark.parametrize("var_name", sorted(ALLOWED_VARS))
def test_allowed_var_accepted(var_name):
    doc = {
        "name": "test_allowed",
        "tie_breaker": "arrival_order",
        "default": {"request_score": {"var": var_name}},
    }
    r = verify_heuristic(doc)
    assert r.valid, f"Expected {var_name!r} to be allowed but got errors: {r.errors}"

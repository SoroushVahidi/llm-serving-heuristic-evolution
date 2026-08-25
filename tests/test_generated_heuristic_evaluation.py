"""Tests for candidate evaluation via the simulator."""
import math
from llmserveopt.llm_generation.evaluation import (
    CandidateResult,
    EvaluationConfig,
    evaluate_candidates,
)


# Minimal valid heuristic — FIFO-like
VALID_HEURISTIC = {
    "name": "test_fifo_like",
    "tie_breaker": "arrival_order",
    "default": {
        "request_score": {"const": 1.0},
    },
}

# Invalid heuristic (missing required fields)
INVALID_HEURISTIC = {
    "name": "broken",
    # missing 'tie_breaker' and 'default'
}

# Heuristic using a forbidden variable — should fail compilation
FORBIDDEN_HEURISTIC = {
    "name": "forbidden_var",
    "tie_breaker": "arrival_order",
    "default": {
        "request_score": {"var": "req.actual_output_tokens"},
    },
}

# Tiny config for fast unit tests
_TINY_CFG = EvaluationConfig(
    arrival_rate=5.0,
    duration=5.0,
    seed=42,
    max_active_sequences=2,
    max_batch_tokens=256,
    max_kv_tokens=2048,
    baseline_names=["fifo"],
    drain_steps=1000,
)


def test_evaluate_valid_heuristic():
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_candidates(records, _TINY_CFG)
    assert "heuristics" in results
    assert len(results["heuristics"]) == 1
    r = results["heuristics"][0]
    assert isinstance(r, CandidateResult)
    assert r.name == "test_fifo_like"
    assert r.source == "heuristic"
    assert r.error is None


def test_evaluate_returns_baseline():
    records = []
    results = evaluate_candidates(records, _TINY_CFG)
    assert len(results["baselines"]) == 1
    assert results["baselines"][0].name == "fifo"
    assert results["baselines"][0].source == "baseline"


def test_evaluate_valid_heuristic_has_finite_goodput():
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_candidates(records, _TINY_CFG)
    r = results["heuristics"][0]
    assert not math.isnan(r.weighted_goodput)
    assert not math.isnan(r.priority_weighted_slo_goodput)
    assert 0.0 <= r.priority_weighted_slo_goodput <= 1.0


def test_evaluate_weighted_goodput_equals_priority_weighted(  ):
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_candidates(records, _TINY_CFG)
    r = results["heuristics"][0]
    assert abs(r.weighted_goodput - r.priority_weighted_slo_goodput) < 1e-9


def test_evaluate_forbidden_heuristic_returns_error():
    records = [{"candidate": FORBIDDEN_HEURISTIC}]
    results = evaluate_candidates(records, _TINY_CFG)
    r = results["heuristics"][0]
    assert r.error is not None
    assert math.isnan(r.priority_weighted_slo_goodput)


def test_evaluate_invalid_heuristic_returns_error():
    records = [{"candidate": INVALID_HEURISTIC}]
    results = evaluate_candidates(records, _TINY_CFG)
    r = results["heuristics"][0]
    assert r.error is not None
    assert math.isnan(r.priority_weighted_slo_goodput)


def test_evaluate_empty_records():
    results = evaluate_candidates([], _TINY_CFG)
    assert results["heuristics"] == []
    assert len(results["baselines"]) >= 1  # at least fifo baseline


def test_evaluate_multiple_heuristics():
    records = [
        {"candidate": VALID_HEURISTIC},
        {"candidate": {
            "name": "test_edf_like",
            "tie_breaker": "earliest_deadline",
            "default": {
                "request_score": {"var": "req.deadline_urgency"},
            },
        }},
    ]
    results = evaluate_candidates(records, _TINY_CFG)
    assert len(results["heuristics"]) == 2


def test_candidate_result_num_completed_nonnegative():
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_candidates(records, _TINY_CFG)
    r = results["heuristics"][0]
    assert r.num_completed >= 0


def test_oracle_srtf_not_in_default_baselines():
    cfg = EvaluationConfig()
    assert "oracle_srtf" not in cfg.baseline_names

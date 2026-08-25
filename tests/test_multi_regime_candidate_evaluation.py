"""Tests for multi-regime candidate evaluation."""
import math
from llmserveopt.llm_generation.multi_regime_evaluation import (
    TRAIN_REGIMES,
    VALIDATION_REGIMES,
    DEFAULT_REGIMES,
    DEFAULT_BASELINES,
    MultiRegimeConfig,
    RegimeSpec,
    RegimeResult,
    AggregatedCandidateResult,
    evaluate_multi_regime,
    aggregate_regime_results,
)
from llmserveopt.workloads.synthetic import WorkloadConfig


VALID_HEURISTIC = {
    "name": "test_regime_heuristic",
    "tie_breaker": "arrival_order",
    "default": {"request_score": {"const": 1.0}},
}

FORBIDDEN_HEURISTIC = {
    "name": "forbidden_regime",
    "tie_breaker": "arrival_order",
    "default": {"request_score": {"var": "req.actual_output_tokens"}},
}

# Tiny regimes for fast tests
_TINY_TRAIN = [
    RegimeSpec(
        name="test_train_a",
        split="train",
        workload=WorkloadConfig(arrival_rate=5.0, duration=5.0, tag="test_train_a"),
        max_active_sequences=2, max_batch_tokens=128, max_kv_tokens=1024,
        seed=42,
    ),
]
_TINY_VAL = [
    RegimeSpec(
        name="test_val_a",
        split="validation",
        workload=WorkloadConfig(arrival_rate=5.0, duration=5.0, tag="test_val_a"),
        max_active_sequences=2, max_batch_tokens=128, max_kv_tokens=1024,
        seed=99,
    ),
]
_TINY_CFG = MultiRegimeConfig(
    regimes=_TINY_TRAIN + _TINY_VAL,
    baseline_names=["fifo"],
    verbose=False,
)


def test_default_regimes_have_correct_splits():
    train = [r for r in DEFAULT_REGIMES if r.split == "train"]
    val = [r for r in DEFAULT_REGIMES if r.split == "validation"]
    assert len(train) == len(TRAIN_REGIMES)
    assert len(val) == len(VALIDATION_REGIMES)
    assert all(r.split == "train" for r in TRAIN_REGIMES)
    assert all(r.split == "validation" for r in VALIDATION_REGIMES)


def test_default_baselines_no_oracle():
    assert "oracle_srtf" not in DEFAULT_BASELINES


def test_evaluate_multi_regime_returns_regime_results():
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    assert isinstance(results, list)
    assert len(results) == 2  # 1 train + 1 val


def test_evaluate_multi_regime_regime_names():
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    names = [r.regime_name for r in results]
    assert "test_train_a" in names
    assert "test_val_a" in names


def test_evaluate_multi_regime_has_heuristics_and_baselines():
    records = [{"candidate": VALID_HEURISTIC}]
    results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    for rr in results:
        assert isinstance(rr, RegimeResult)
        assert isinstance(rr.heuristics, list)
        assert isinstance(rr.baselines, list)


def test_evaluate_multi_regime_forbidden_candidate_has_error():
    records = [{"candidate": FORBIDDEN_HEURISTIC}]
    results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    for rr in results:
        r = rr.heuristics[0]
        assert r.error is not None
        assert math.isnan(r.priority_weighted_slo_goodput)


def test_evaluate_multi_regime_empty_candidates():
    results = evaluate_multi_regime([], _TINY_CFG, verbose=False)
    for rr in results:
        assert rr.heuristics == []
        assert len(rr.baselines) >= 1  # fifo should always be there


def test_aggregate_regime_results_keys():
    records = [{"candidate": VALID_HEURISTIC}]
    regime_results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    agg = aggregate_regime_results(regime_results)
    assert "test_regime_heuristic" in agg
    assert "fifo" in agg


def test_aggregate_has_train_and_val_wg():
    records = [{"candidate": VALID_HEURISTIC}]
    regime_results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    agg = aggregate_regime_results(regime_results)
    r = agg["test_regime_heuristic"]
    assert isinstance(r, AggregatedCandidateResult)
    assert not math.isnan(r.train_mean_wg)
    assert not math.isnan(r.val_mean_wg)


def test_aggregate_train_val_gap_is_computed():
    records = [{"candidate": VALID_HEURISTIC}]
    regime_results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    agg = aggregate_regime_results(regime_results)
    r = agg["test_regime_heuristic"]
    assert not math.isnan(r.train_val_gap)
    assert abs(r.train_val_gap - (r.val_mean_wg - r.train_mean_wg)) < 1e-9


def test_aggregate_per_regime_dict():
    records = [{"candidate": VALID_HEURISTIC}]
    regime_results = evaluate_multi_regime(records, _TINY_CFG, verbose=False)
    agg = aggregate_regime_results(regime_results)
    r = agg["test_regime_heuristic"]
    assert "test_train_a" in r.per_regime
    assert "test_val_a" in r.per_regime

"""Tests for workloads/augmentation.py"""
import numpy as np
import pytest

from llmserveopt.workloads.augmentation import (
    AugmentationConfig,
    DEFAULT_SLO_AUG,
    PredictionNoiseConfig,
    SLOAugConfig,
    SLOClassConfig,
    apply_prediction_noise,
    augment_slo_classes,
    augment_trace,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def actual_tokens():
    return np.array([100, 200, 300, 400, 500, 50, 1000, 2000, 10, 700])


def test_exact_mode(actual_tokens, rng):
    cfg = PredictionNoiseConfig(mode="exact")
    predicted = apply_prediction_noise(actual_tokens, cfg, rng)
    np.testing.assert_array_equal(predicted, actual_tokens)


def test_lognormal_mode_within_bounds(actual_tokens, rng):
    cfg = PredictionNoiseConfig(mode="lognormal", sigma=0.35, min_tokens=1, max_tokens=4096)
    predicted = apply_prediction_noise(actual_tokens, cfg, rng)
    assert np.all(predicted >= 1)
    assert np.all(predicted <= 4096)


def test_lognormal_clipping(rng):
    actual = np.array([1, 2, 3, 10000])
    cfg = PredictionNoiseConfig(mode="lognormal", sigma=0.35, min_tokens=5, max_tokens=100)
    predicted = apply_prediction_noise(actual, cfg, rng)
    assert np.all(predicted >= 5)
    assert np.all(predicted <= 100)


def test_biased_under_mean_less_than_actual():
    actual = np.array([500] * 1000)
    cfg = PredictionNoiseConfig(mode="biased_under", sigma=0.35, min_tokens=1, max_tokens=100000)
    rng = np.random.default_rng(0)
    predicted = apply_prediction_noise(actual, cfg, rng)
    assert predicted.mean() < actual.mean()


def test_biased_over_mean_greater_than_actual():
    actual = np.array([500] * 1000)
    cfg = PredictionNoiseConfig(mode="biased_over", sigma=0.35, min_tokens=1, max_tokens=100000)
    rng = np.random.default_rng(0)
    predicted = apply_prediction_noise(actual, cfg, rng)
    assert predicted.mean() > actual.mean()


def test_bucket_mode_midpoints(rng):
    actual = np.array([1, 10, 63, 64, 100, 255, 256, 500, 1023, 1024, 2000, 4095, 4096, 5000])
    cfg = PredictionNoiseConfig(mode="bucket", min_tokens=1, max_tokens=4096)
    predicted = apply_prediction_noise(actual, cfg, rng)
    valid_midpoints = {32, 128, 512, 2048, 4096}
    for p in predicted:
        assert int(p) in valid_midpoints, f"bucket midpoint {p} not in {valid_midpoints}"


def test_deterministic_same_seed(actual_tokens):
    cfg = PredictionNoiseConfig(mode="lognormal", sigma=0.35)
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    p1 = apply_prediction_noise(actual_tokens, cfg, rng1)
    p2 = apply_prediction_noise(actual_tokens, cfg, rng2)
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_different_results(actual_tokens):
    cfg = PredictionNoiseConfig(mode="lognormal", sigma=0.35)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    p1 = apply_prediction_noise(actual_tokens, cfg, rng1)
    p2 = apply_prediction_noise(actual_tokens, cfg, rng2)
    assert not np.all(p1 == p2)


def test_slo_class_proportions_approximate():
    n = 10000
    arrival_times = np.zeros(n)
    rng = np.random.default_rng(0)
    class_ids, priorities, slo_deadlines = augment_slo_classes(n, DEFAULT_SLO_AUG, arrival_times, rng)
    counts = {}
    for cid in class_ids:
        counts[cid] = counts.get(cid, 0) + 1
    assert abs(counts.get("interactive", 0) / n - 0.50) < 0.03
    assert abs(counts.get("standard", 0) / n - 0.35) < 0.03
    assert abs(counts.get("batch", 0) / n - 0.15) < 0.03


def test_slo_deadline_equals_arrival_plus_slack():
    n = 100
    arrival_times = np.linspace(0, 10, n)
    rng = np.random.default_rng(0)
    slo_cfg = SLOAugConfig(classes=[
        SLOClassConfig("interactive", weight=0.5, priority=3.0, slo_slack=2.0),
        SLOClassConfig("standard",    weight=0.5, priority=2.0, slo_slack=6.0),
    ])
    class_ids, priorities, slo_deadlines = augment_slo_classes(n, slo_cfg, arrival_times, rng)
    slack_map = {"interactive": 2.0, "standard": 6.0}
    for i, (cid, dl) in enumerate(zip(class_ids, slo_deadlines)):
        expected = arrival_times[i] + slack_map[cid]
        assert abs(dl - expected) < 1e-9


def test_augment_trace_returns_all_fields():
    n = 50
    actual = np.array([100] * n)
    arrivals = np.linspace(0, 5, n)
    cfg = AugmentationConfig()
    rng = np.random.default_rng(0)
    result = augment_trace(actual, arrivals, cfg, rng)
    assert "predicted_output_tokens" in result
    assert "class_ids" in result
    assert "priorities" in result
    assert "slo_deadlines" in result
    assert len(result["predicted_output_tokens"]) == n
    assert len(result["class_ids"]) == n


def test_augment_trace_deterministic():
    n = 20
    actual = np.arange(1, n + 1) * 50
    arrivals = np.linspace(0, 2, n)
    cfg = AugmentationConfig()
    r1 = augment_trace(actual, arrivals, cfg, np.random.default_rng(7))
    r2 = augment_trace(actual, arrivals, cfg, np.random.default_rng(7))
    np.testing.assert_array_equal(r1["predicted_output_tokens"], r2["predicted_output_tokens"])
    assert r1["class_ids"] == r2["class_ids"]


def test_unknown_mode_raises():
    actual = np.array([100, 200])
    cfg = PredictionNoiseConfig(mode="unknown_mode")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown prediction noise mode"):
        apply_prediction_noise(actual, cfg, rng)

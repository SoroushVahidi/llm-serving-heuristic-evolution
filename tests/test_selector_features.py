"""Tests for selector feature extraction."""
import pytest
import numpy as np

from llmserveopt.core.types import Request
from llmserveopt.selector.features import (
    extract_features, FEATURE_NAMES, FeatureMode,
    _cv, _estimate_arrival_rate, _burstiness_cv,
)


def _req(i: int, t: float, prompt: int = 64, pred_out: int = 32,
         slo_slack: float = 5.0, class_id: str = "medium", priority: float = 1.0) -> Request:
    return Request(
        request_id=i,
        arrival_time=t,
        prompt_tokens=prompt,
        predicted_output_tokens=pred_out,
        actual_output_tokens=pred_out,   # hidden field — must NOT appear in features
        slo_deadline=t + slo_slack,
        priority=priority,
        class_id=class_id,
    )


def _window(n: int = 10, start_t: float = 5.0) -> list:
    return [_req(i, start_t + i * 0.1) for i in range(n)]


def _prefix(n: int = 5, start_t: float = 0.0) -> list:
    return [_req(100 + i, start_t + i * 0.5) for i in range(n)]


# --- all 18 features present ---

def test_all_feature_names_present_causal():
    win = _window()
    feats = extract_features(win, window_start_time=5.0, mode=FeatureMode.CAUSAL)
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"
    assert len(feats) == len(FEATURE_NAMES)


def test_all_feature_names_present_offline_window_lookahead():
    win = _window()
    feats = extract_features(
        win, window_start_time=5.0, mode=FeatureMode.OFFLINE_WINDOW_LOOKAHEAD
    )
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"


def test_all_feature_names_present_descriptive():
    win = _window()
    feats = extract_features(win, window_start_time=5.0, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"


# --- feature value ranges ---

def test_fraction_tight_slo_range():
    win = [_req(i, float(i), class_id="tight" if i < 5 else "medium") for i in range(10)]
    feats = extract_features(win, window_start_time=0.0, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    assert 0.0 <= feats["fraction_tight_slo"] <= 1.0
    assert abs(feats["fraction_tight_slo"] - 0.5) < 1e-9


def test_all_tight_slo():
    win = [_req(i, float(i), class_id="tight") for i in range(10)]
    feats = extract_features(win, window_start_time=0.0, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    assert feats["fraction_tight_slo"] == pytest.approx(1.0)


def test_mean_prompt_tokens():
    win = [_req(i, float(i), prompt=100) for i in range(10)]
    feats = extract_features(win, window_start_time=0.0, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    assert feats["mean_prompt_tokens"] == pytest.approx(100.0)


def test_kv_utilization_offline_default():
    win = _window()
    feats = extract_features(win, window_start_time=5.0, mode=FeatureMode.CAUSAL)
    assert feats["kv_utilization"] == pytest.approx(0.0)


def test_free_sequence_ratio_offline_default():
    win = _window()
    feats = extract_features(win, window_start_time=5.0, mode=FeatureMode.CAUSAL)
    assert feats["free_sequence_ratio"] == pytest.approx(1.0)


def test_pred_output_cv_uniform():
    win = [_req(i, float(i), pred_out=50) for i in range(20)]
    feats = extract_features(win, window_start_time=0.0, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    # All same value → std=0 → CV=0
    assert feats["pred_output_cv"] == pytest.approx(0.0)


def test_mean_slack_positive():
    win = [_req(i, float(i), slo_slack=3.0) for i in range(10)]
    feats = extract_features(win, window_start_time=0.0, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    assert feats["mean_slack"] == pytest.approx(3.0)
    assert feats["min_slack"] == pytest.approx(3.0)


# --- causal: waiting time uses prefix ---

def test_waiting_time_from_prefix():
    prefix = _prefix(n=5, start_t=0.0)
    win = _window(n=10, start_t=5.0)
    feats = extract_features(
        win,
        window_start_time=5.0,
        mode=FeatureMode.CAUSAL,
        prefix_requests=prefix,
    )
    # Waiting time of prefix[0]: 5.0 - 0.0 = 5.0, prefix[1]: 5.0 - 0.5 = 4.5, ...
    assert feats["mean_waiting_time"] >= 0.0
    assert feats["p95_waiting_time"] >= 0.0


# --- arrival rate ---

def test_arrival_rate_positive_with_prefix():
    prefix = [_req(i, float(i) * 0.5) for i in range(20)]
    win = _window(start_t=10.0)
    feats = extract_features(
        win,
        window_start_time=10.0,
        mode=FeatureMode.CAUSAL,
        prefix_requests=prefix,
    )
    assert feats["arrival_rate_est"] > 0.0


# --- helpers ---

def test_cv_zero_mean():
    arr = np.array([0.0, 0.0, 0.0])
    assert _cv(arr) == pytest.approx(0.0)


def test_cv_uniform():
    arr = np.array([5.0, 5.0, 5.0])
    assert _cv(arr) == pytest.approx(0.0)


def test_cv_nonzero():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    cv = _cv(arr)
    assert cv > 0.0


def test_estimate_arrival_rate_empty():
    assert _estimate_arrival_rate([], reference_time=5.0) == pytest.approx(0.0)


def test_burstiness_cv_single():
    reqs = [_req(0, 0.0)]
    assert _burstiness_cv(reqs, reference_time=1.0) == pytest.approx(0.0)

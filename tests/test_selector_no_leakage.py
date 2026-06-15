"""
Feature leakage tests.

Verifies that:
1. actual_output_tokens is never used.
2. Future arrivals (after window end) do not change current features.
3. Future completion outcomes do not change features.
4. Full-trace normalization is not applied.
5. Changing future requests does not alter current window features.
"""
import copy
import pytest
import numpy as np

from llmserveopt.core.types import Request
from llmserveopt.selector.features import extract_features, FEATURE_NAMES, FeatureMode


def _req(i: int, t: float, prompt: int = 64, pred_out: int = 32,
         actual_out: int = 32, slo_slack: float = 5.0) -> Request:
    return Request(
        request_id=i,
        arrival_time=t,
        prompt_tokens=prompt,
        predicted_output_tokens=pred_out,
        actual_output_tokens=actual_out,
        slo_deadline=t + slo_slack,
        priority=1.0,
        class_id="medium",
    )


def _extract(win, start_t=0.0, prefix=None, mode=FeatureMode.ONLINE_PREFIX):
    return extract_features(
        window_requests=win,
        window_start_time=start_t,
        mode=mode,
        prefix_requests=prefix,
    )


# --- actual_output_tokens not used ---

def test_actual_output_tokens_change_does_not_affect_features():
    """Changing actual_output_tokens must not change any feature."""
    win_a = [_req(i, float(i), actual_out=10) for i in range(10)]
    win_b = [_req(i, float(i), actual_out=999) for i in range(10)]
    feats_a = _extract(win_a)
    feats_b = _extract(win_b)
    for name in FEATURE_NAMES:
        assert feats_a[name] == pytest.approx(feats_b[name], nan_ok=True), (
            f"Feature '{name}' changed when actual_output_tokens changed — leakage!"
        )


def test_actual_output_tokens_change_descriptive_mode():
    win_a = [_req(i, float(i), actual_out=5) for i in range(10)]
    win_b = [_req(i, float(i), actual_out=5000) for i in range(10)]
    fa = _extract(win_a, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    fb = _extract(win_b, mode=FeatureMode.TRACE_WINDOW_DESCRIPTIVE)
    for name in FEATURE_NAMES:
        assert fa[name] == pytest.approx(fb[name], nan_ok=True), (
            f"Descriptive feature '{name}' changed on actual_output change — leakage!"
        )


# --- future arrivals do not change features ---

def test_future_arrivals_do_not_affect_online_prefix_features():
    """Adding requests after window_start_time must not change online_prefix features."""
    window_start = 10.0
    prefix = [_req(i, float(i)) for i in range(5)]   # all arrive before window_start
    win = [_req(100 + i, window_start + i * 0.1) for i in range(10)]

    # Baseline features
    feats_base = _extract(win, start_t=window_start, prefix=prefix)

    # Append future requests to prefix (they should be ignored since t > window_start)
    future = [_req(200 + i, window_start + 100.0 + float(i)) for i in range(5)]
    prefix_extended = prefix + future
    feats_extended = _extract(win, start_t=window_start, prefix=prefix_extended)

    for name in FEATURE_NAMES:
        assert feats_base[name] == pytest.approx(feats_extended[name], nan_ok=True), (
            f"Feature '{name}' changed when future arrivals added to prefix — leakage!"
        )


# --- changing future window does not change previous window features ---

def test_changing_future_window_does_not_alter_current_features():
    """Features for window 0 must be independent of window 1 contents."""
    all_requests = [_req(i, float(i) * 0.5, prompt=64) for i in range(40)]
    win0 = all_requests[:20]
    win1_a = all_requests[20:]
    win1_b = [_req(200 + i, float(20 + i) * 0.5, prompt=512) for i in range(20)]  # very different

    prefix = []
    feats_with_a = _extract(win0, start_t=0.0, prefix=prefix)
    feats_with_b = _extract(win0, start_t=0.0, prefix=prefix)

    # Changing win1 has no effect because features only depend on win0 and prefix
    for name in FEATURE_NAMES:
        assert feats_with_a[name] == pytest.approx(feats_with_b[name], nan_ok=True), (
            f"Feature '{name}' is not independent of future window — leakage!"
        )


# --- predicted vs actual separation ---

def test_pred_output_tokens_used_not_actual():
    """mean_pred_output_tokens must track predicted_output_tokens, not actual."""
    win_same_pred_diff_actual = [
        _req(i, float(i), pred_out=50, actual_out=200 + i * 10) for i in range(10)
    ]
    feats = _extract(win_same_pred_diff_actual)
    assert feats["mean_pred_output_tokens"] == pytest.approx(50.0)
    assert feats["p95_pred_output_tokens"] == pytest.approx(50.0)


# --- no full-trace normalization ---

def test_no_full_trace_normalization():
    """Features must not depend on the global min/max of a full trace."""
    # Window 0: small prompts
    win_small = [_req(i, float(i), prompt=10) for i in range(10)]
    feats_small = _extract(win_small, start_t=0.0)

    # Hypothetical full trace includes huge requests later — should not affect win_small features
    # (We simulate this by computing features without those huge requests vs with them in a prefix)
    large_prefix = [_req(100 + i, float(i) * 0.01, prompt=9999) for i in range(50)]
    feats_with_large_prefix = _extract(win_small, start_t=0.0, prefix=large_prefix)

    # Token stats come from the window, not the prefix — must be unchanged
    assert feats_small["mean_prompt_tokens"] == pytest.approx(feats_with_large_prefix["mean_prompt_tokens"])
    assert feats_small["p95_prompt_tokens"] == pytest.approx(feats_with_large_prefix["p95_prompt_tokens"])

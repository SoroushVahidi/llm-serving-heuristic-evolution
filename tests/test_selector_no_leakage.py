"""
Feature leakage tests.

Verifies that:
1. actual_output_tokens is never used.
2. Causal features ignore later within-window arrivals.
3. Offline lookahead mode intentionally reflects within-window future arrivals.
4. Feature normalization is not fit on evaluation rows inside selector models.
"""
import copy
import pytest

from llmserveopt.core.types import Request
from llmserveopt.selector.features import (
    extract_features,
    FEATURE_NAMES,
    FeatureMode,
    parse_feature_mode,
    feature_mode_is_deployable,
)
from llmserveopt.selector.models import RandomForestSelector


def _req(
    i: int,
    t: float,
    prompt: int = 64,
    pred_out: int = 32,
    actual_out: int = 32,
    slo_slack: float = 5.0,
) -> Request:
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


def _extract(win, start_t=0.0, prefix=None, mode=FeatureMode.CAUSAL):
    return extract_features(
        window_requests=win,
        window_start_time=start_t,
        mode=mode,
        prefix_requests=prefix,
    )


# --- actual_output_tokens not used ---

def test_actual_output_tokens_change_does_not_affect_causal_features():
    win_a = [_req(i, float(i), actual_out=10) for i in range(10)]
    win_b = [_req(i, float(i), actual_out=999) for i in range(10)]
    feats_a = _extract(win_a, start_t=0.0, mode=FeatureMode.CAUSAL)
    feats_b = _extract(win_b, start_t=0.0, mode=FeatureMode.CAUSAL)
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


# --- causal vs within-window lookahead ---

def test_causal_features_unchanged_when_later_window_requests_mutated():
    """Mutating later within-window arrivals must not change causal features."""
    window_start = 10.0
    prefix = [_req(i, float(i)) for i in range(5)]
    win_base = [
        _req(100, window_start, prompt=100),
        _req(101, window_start + 1.0, prompt=200),
        _req(102, window_start + 2.0, prompt=300),
    ]
    win_mutated = [
        _req(100, window_start, prompt=100),
        _req(101, window_start + 1.0, prompt=9999),
        _req(102, window_start + 2.0, prompt=8888),
    ]

    feats_base = _extract(
        win_base, start_t=window_start, prefix=prefix, mode=FeatureMode.CAUSAL
    )
    feats_mut = _extract(
        win_mutated, start_t=window_start, prefix=prefix, mode=FeatureMode.CAUSAL
    )
    for name in FEATURE_NAMES:
        assert feats_base[name] == pytest.approx(feats_mut[name], nan_ok=True), (
            f"Causal feature '{name}' changed after mutating future within-window arrivals"
        )


def test_offline_window_lookahead_reflects_later_window_requests():
    """Offline lookahead intentionally uses full-window token/SLO statistics."""
    window_start = 10.0
    prefix = [_req(i, float(i)) for i in range(5)]
    win_base = [
        _req(100, window_start, prompt=100),
        _req(101, window_start + 1.0, prompt=200),
        _req(102, window_start + 2.0, prompt=300),
    ]
    win_mutated = [
        _req(100, window_start, prompt=100),
        _req(101, window_start + 1.0, prompt=9999),
        _req(102, window_start + 2.0, prompt=8888),
    ]

    feats_base = _extract(
        win_base,
        start_t=window_start,
        prefix=prefix,
        mode=FeatureMode.OFFLINE_WINDOW_LOOKAHEAD,
    )
    feats_mut = _extract(
        win_mutated,
        start_t=window_start,
        prefix=prefix,
        mode=FeatureMode.OFFLINE_WINDOW_LOOKAHEAD,
    )
    assert feats_base["mean_prompt_tokens"] != pytest.approx(
        feats_mut["mean_prompt_tokens"]
    )


def test_online_prefix_alias_maps_to_offline_window_lookahead():
    assert parse_feature_mode("online_prefix") == FeatureMode.OFFLINE_WINDOW_LOOKAHEAD
    assert not feature_mode_is_deployable(parse_feature_mode("online_prefix"))
    assert feature_mode_is_deployable(parse_feature_mode("causal"))


# --- prefix future arrivals do not change causal prefix-only stats ---

def test_future_prefix_arrivals_after_window_start_ignored_for_causal():
    window_start = 10.0
    prefix = [_req(i, float(i)) for i in range(5)]
    win = [_req(100 + i, window_start + i * 0.1, prompt=128) for i in range(10)]

    feats_base = _extract(win, start_t=window_start, prefix=prefix, mode=FeatureMode.CAUSAL)
    future = [_req(200 + i, window_start + 100.0 + float(i), prompt=512) for i in range(5)]
    feats_extended = _extract(
        win, start_t=window_start, prefix=prefix + future, mode=FeatureMode.CAUSAL
    )

    for name in FEATURE_NAMES:
        assert feats_base[name] == pytest.approx(feats_extended[name], nan_ok=True), (
            f"Causal feature '{name}' changed when future prefix arrivals were added"
        )


# --- predicted vs actual separation ---

def test_pred_output_tokens_used_not_actual():
    win = [
        _req(i, float(i), pred_out=50, actual_out=200 + i * 10) for i in range(10)
    ]
    feats = _extract(win, start_t=0.0, mode=FeatureMode.CAUSAL)
    assert feats["mean_pred_output_tokens"] == pytest.approx(50.0)
    assert feats["p95_pred_output_tokens"] == pytest.approx(50.0)


# --- selector model fitting uses train rows only ---

def test_selector_model_fit_uses_only_training_rows():
    """Tree selectors fit on provided train rows; test rows are not used in fit()."""
    train_rows = [
        {"feat_queue_length": 1.0, "feat_arrival_rate_est": 2.0, "best_policy": "fifo"},
        {"feat_queue_length": 2.0, "feat_arrival_rate_est": 3.0, "best_policy": "edf"},
        {"feat_queue_length": 3.0, "feat_arrival_rate_est": 4.0, "best_policy": "edf"},
    ]
    test_rows = [
        {"feat_queue_length": 99.0, "feat_arrival_rate_est": 99.0, "best_policy": "fifo"},
    ]
    for row in train_rows + test_rows:
        for name in FEATURE_NAMES:
            row.setdefault(f"feat_{name}", 0.0)

    model = RandomForestSelector(n_estimators=10, max_depth=3, random_state=0)
    model.fit(copy.deepcopy(train_rows))
    preds_before = model.predict(copy.deepcopy(test_rows))

    mutated_test = copy.deepcopy(test_rows)
    mutated_test[0]["feat_queue_length"] = 12345.0
    preds_after = model.predict(mutated_test)

    assert preds_before == preds_after
    assert model._clf.n_features_in_ == len(FEATURE_NAMES)

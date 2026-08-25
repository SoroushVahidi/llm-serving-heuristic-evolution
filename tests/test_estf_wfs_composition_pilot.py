"""Tests for minimal ESTF↔WFS composition falsification harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmserveopt.composition.estf_wfs_features import (
    FORBIDDEN_FEATURE_KEYS,
    assert_no_hidden_leakage,
    scenario_observable_features,
)
from llmserveopt.composition.estf_wfs_metrics import envelope_gain, paired_bootstrap_ci
from llmserveopt.composition.estf_wfs_policies import make_static_estf_wfs_blend
from llmserveopt.composition.estf_wfs_splits import (
    assert_no_split_leakage,
    assign_family_a_v2_splits,
)
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.composition import rank_with_named_expert
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (
    case_fairness_vs_size_v2,
)


def _obs_state(reqs):
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=2,
        max_batch_tokens=2048,
        max_kv_tokens=100000,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
        prefilling_count=0,
        decoding_count=0,
    )
    return ObservableState(
        time=0.0, waiting_queue=list(reqs), gpu_states=[gpu], completed_count=0, step=0
    )


def test_wfs_named_expert_ranks():
    reqs = [
        ObservableRequest(0, 0.0, 100, 50, 10.0, 10.0, "tenant_favored"),
        ObservableRequest(1, 0.1, 100, 50, 10.0, 1.0, "tenant_other"),
        ObservableRequest(2, 0.2, 20, 10, 10.0, 1.0, "tenant_other"),
    ]
    out = rank_with_named_expert("weighted_fair_share", _obs_state(reqs))
    assert set(out.ranked_request_ids) == {0, 1, 2}
    assert out.normalized_ranks[0] >= out.normalized_ranks[1] - 1e-12


def test_alpha_1_matches_estf_order():
    scen = case_fairness_vs_size_v2(
        target_utilization=1.2,
        tenant_weight_skew=5.0,
        favored_tenant_size="long",
        prediction_noise_sigma=0.0,
        seed=7,
        n_total_jobs=40,
        allow_synthetic_tokens=True,
    )
    waiting = [
        ObservableRequest(
            r.request_id,
            r.arrival_time,
            r.prompt_tokens,
            r.predicted_output_tokens,
            r.slo_deadline,
            r.priority,
            r.class_id,
        )
        for r in scen.requests[:8]
    ]
    state = _obs_state(waiting)
    estf = rank_with_named_expert("estimated_service_time_first", state)
    blend = make_static_estf_wfs_blend(1.0)
    ranked, _log = blend._ranked_requests_and_log(state)
    assert [r.request_id for r in ranked] == estf.ranked_request_ids


def test_alpha_0_matches_wfs_order():
    scen = case_fairness_vs_size_v2(
        target_utilization=1.2,
        tenant_weight_skew=10.0,
        favored_tenant_size="long",
        prediction_noise_sigma=0.0,
        seed=11,
        n_total_jobs=40,
        allow_synthetic_tokens=True,
    )
    waiting = [
        ObservableRequest(
            r.request_id,
            r.arrival_time,
            r.prompt_tokens,
            r.predicted_output_tokens,
            r.slo_deadline,
            r.priority,
            r.class_id,
        )
        for r in scen.requests[:8]
    ]
    state = _obs_state(waiting)
    wfs = rank_with_named_expert("weighted_fair_share", state)
    blend = make_static_estf_wfs_blend(0.0)
    ranked, _log = blend._ranked_requests_and_log(state)
    assert [r.request_id for r in ranked] == wfs.ranked_request_ids


def test_no_hidden_leakage():
    scen = case_fairness_vs_size_v2(
        target_utilization=1.1,
        tenant_weight_skew=5.0,
        favored_tenant_size="short",
        prediction_noise_sigma=0.3,
        seed=1,
        allow_synthetic_tokens=True,
    )
    feats = scenario_observable_features(scen.requests)
    assert_no_hidden_leakage(feats)
    with pytest.raises(ValueError):
        assert_no_hidden_leakage({**feats, "favored_tenant_size": "long"})
    assert not (set(feats) & FORBIDDEN_FEATURE_KEYS)


def test_split_integrity_family_a_csv():
    import csv

    path = Path(
        "experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377"
        "/scenario_features.csv"
    )
    if not path.is_file():
        pytest.skip("Family A v2 features absent")
    rows = list(csv.DictReader(path.open()))
    splits = assign_family_a_v2_splits(rows)
    assert_no_split_leakage(splits)
    assert len(splits.train) + len(splits.val) + len(splits.test) + len(splits.ood) == 72
    # OOD = long × skew10 × 3 util × 2 noise × 2 seeds = 12
    assert len(splits.ood) == 12


def test_envelope_gain_and_bootstrap():
    child = {"a": 0.9, "b": 0.5, "c": 0.8}
    p1 = {"a": 0.8, "b": 0.6, "c": 0.7}
    p2 = {"a": 0.85, "b": 0.55, "c": 0.75}
    eg = envelope_gain(child, p1, p2, ["a", "b", "c"])
    assert eg["n_beat_both"] == 2.0  # a and c
    assert eg["n_lose_both"] == 1.0  # only b
    mean, lo, hi = paired_bootstrap_ci([0.01, 0.02, -0.01], n_boot=200, seed=1)
    assert lo <= mean <= hi

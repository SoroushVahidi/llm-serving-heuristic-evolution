from __future__ import annotations

import numpy as np
import pytest

from llmserveopt.policy_separation import public_replay_load_scaling_v1 as prl
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr


@pytest.fixture(scope="module")
def windows():
    return prl.get_canonical_windows()


def test_canonical_windows_count_and_source_split(windows):
    assert len(windows) == 60
    counts = {}
    for r in windows:
        counts[r["source_dataset"]] = counts.get(r["source_dataset"], 0) + 1
    assert counts == {"burstgpt": 20, "azure_2023_conv": 20, "azure_2023_code": 20}
    assert set(counts) == set(ptr.SOURCES)


def test_matrix_completeness_and_no_duplicates():
    keys = prl.expected_cell_keys()
    assert len(keys) == prl.EXPECTED_N_CELLS == 3840
    assert len(set(keys)) == len(keys)


def test_load_factor_grid_frozen():
    assert prl.LOAD_FACTORS == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def test_policy_portfolio_frozen():
    assert prl.PEXT_POLICIES == (
        "full_prefill",
        "chunked_prefill_small",
        "estimated_service_time_first",
        "weighted_fair_share",
        "least_laxity_first",
        "kv_constrained_online",
        "official_vtc_joint_token_budget_remap",
        "vllm_style_continuous_batching",
    )
    assert len(prl.PEXT_POLICIES) == 8


def test_lambda_1_is_identity_on_arrival_times(windows):
    base = windows[0]["scenario"]
    scaled = prl.transform_arrival_only(base, 1.0)
    base_sorted = sorted(base.requests, key=lambda r: (r.arrival_time, r.request_id))
    for orig, new in zip(base_sorted, scaled.requests):
        assert orig.arrival_time == pytest.approx(new.arrival_time, abs=1e-9)
        assert orig.slo_deadline == pytest.approx(new.slo_deadline, abs=1e-9)


def test_timestamp_scaling_formula(windows):
    base = windows[0]["scenario"]
    lam = 8.0
    scaled = prl.transform_arrival_only(base, lam)
    scaled_by_id = {r.request_id: r for r in scaled.requests}
    t_start = min(float(r.arrival_time) for r in base.requests)
    for r in base.requests:
        expected = t_start + (float(r.arrival_time) - t_start) / lam
        assert scaled_by_id[r.request_id].arrival_time == pytest.approx(expected, abs=1e-9)


def test_request_count_invariance_across_lambda(windows):
    base = windows[0]["scenario"]
    n0 = len(base.requests)
    for lam in prl.LOAD_FACTORS:
        scaled = prl.transform_arrival_only(base, lam)
        assert len(scaled.requests) == n0


def test_request_order_invariance_across_lambda(windows):
    base = windows[0]["scenario"]
    base_order = [r.request_id for r in sorted(base.requests, key=lambda r: (r.arrival_time, r.request_id))]
    for lam in prl.LOAD_FACTORS:
        scaled = prl.transform_arrival_only(base, lam)
        scaled_order = [r.request_id for r in scaled.requests]
        assert scaled_order == base_order, f"order changed at lambda={lam}"


def test_length_and_class_invariance_across_lambda(windows):
    base = windows[0]["scenario"]
    base_by_id = {r.request_id: r for r in base.requests}
    for lam in prl.LOAD_FACTORS:
        scaled = prl.transform_arrival_only(base, lam)
        for r in scaled.requests:
            b = base_by_id[r.request_id]
            assert r.prompt_tokens == b.prompt_tokens
            assert r.actual_output_tokens == b.actual_output_tokens
            assert r.predicted_output_tokens == b.predicted_output_tokens
            assert r.class_id == b.class_id
            assert r.priority == b.priority


def test_no_accidental_slack_mutation(windows):
    """slo_deadline - arrival_time (per-request SLO tightness) must be
    preserved exactly across all lambda -- only the anchor (arrival_time)
    moves, never the slack."""
    base = windows[0]["scenario"]
    base_slack = {r.request_id: r.slo_deadline - r.arrival_time for r in base.requests}
    for lam in prl.LOAD_FACTORS:
        scaled = prl.transform_arrival_only(base, lam)
        for r in scaled.requests:
            assert r.slo_deadline - r.arrival_time == pytest.approx(base_slack[r.request_id], abs=1e-9)


def test_capacity_unchanged_across_lambda(windows):
    base = windows[0]["scenario"]
    g0 = base.gpu_configs[0]
    for lam in prl.LOAD_FACTORS:
        scaled = prl.transform_arrival_only(base, lam)
        g = scaled.gpu_configs[0]
        assert g.max_active_sequences == g0.max_active_sequences == 512
        assert g.max_batch_tokens == g0.max_batch_tokens == 512
        assert g.max_kv_tokens == g0.max_kv_tokens == 8_000_000


def test_higher_lambda_never_decreases_offered_load_compression(windows):
    """The transform is a monotone decreasing function of lambda applied to
    the same finite window: larger lambda must not increase the span of
    arrival times (it compresses or holds fixed, never expands)."""
    base = windows[0]["scenario"]
    prev_span = None
    for lam in prl.LOAD_FACTORS:
        scaled = prl.transform_arrival_only(base, lam)
        times = [r.arrival_time for r in scaled.requests]
        span = max(times) - min(times)
        if prev_span is not None:
            assert span <= prev_span + 1e-9
        prev_span = span


def test_evaluate_cell_lambda_1_matches_prior_public_replay(windows):
    """Sanity check L.1: lambda=1 must reproduce the frozen
    public_trace_replay_v1 result (ANWG=1.0, completion=1.0 for every P6
    policy on every window) within tolerance -- see
    docs/current/public_trace_replay_v1_analysis_20260820.md section 5."""
    w = windows[0]
    for pid in prl.P6_POLICIES:
        row = prl.evaluate_cell(w, 1.0, pid)
        assert row["status"] == "success", row.get("error")
        assert row["anwg"] == pytest.approx(1.0, abs=1e-6)
        assert row["completion_fraction"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_cell_pressure_increases_with_lambda(windows):
    w = windows[0]
    row_lo = prl.evaluate_cell(w, 1.0, "full_prefill")
    row_hi = prl.evaluate_cell(w, 32.0, "full_prefill")
    assert row_lo["status"] == "success"
    assert row_hi["status"] == "success"
    assert row_hi["active_max"] >= row_lo["active_max"]


def test_evaluate_cell_no_nan_serialization(windows):
    w = windows[0]
    row = prl.evaluate_cell(w, 4.0, "estimated_service_time_first")
    assert row["status"] == "success"
    for key in ("anwg", "completion_fraction"):
        val = row[key]
        assert val is not None
        assert not (isinstance(val, float) and np.isnan(val))


def test_cell_key_uniqueness_and_format():
    k1 = prl.cell_key("SID", 1.0, "full_prefill")
    k16 = prl.cell_key("SID", 16.0, "full_prefill")
    assert k1 != k16
    assert "lambda1" in k1
    assert "lambda16" in k16


def test_summarize_pressure_empty_samples():
    out = prl.summarize_pressure([])
    assert out["n_steps"] == 0
    assert out["active_max"] == 0.0


def test_summarize_pressure_basic_aggregation():
    samples = [
        {"waiting": 0.0, "active": 1.0, "kv_util": 0.1, "active_util": 0.01},
        {"waiting": 2.0, "active": 5.0, "kv_util": 0.2, "active_util": 0.05},
        {"waiting": 0.0, "active": 3.0, "kv_util": 0.15, "active_util": 0.03},
    ]
    out = prl.summarize_pressure(samples)
    assert out["n_steps"] == 3
    assert out["active_max"] == 5.0
    assert out["queue_length_max"] == 2.0
    assert out["frac_steps_queue_positive"] == pytest.approx(1.0 / 3.0)
    assert out["kv_utilization_max"] == pytest.approx(0.2)


def test_window_grouped_bootstrap_reuses_existing_ci_helper():
    """Section K requires a window-grouped bootstrap for post-completion
    analysis; this experiment reuses the already-frozen
    public_trace_replay_v1_analysis.paired_bootstrap_ci helper rather than
    redefining bootstrap semantics. Sanity-check it groups by window (one
    value per window, not per-request) and returns a sane CI."""
    from llmserveopt.analysis import public_trace_replay_v1_analysis as ana

    rng = np.random.default_rng(20260825)
    per_window_values = rng.normal(loc=0.02, scale=0.01, size=60)
    lo, hi = ana.paired_bootstrap_ci(list(per_window_values), n_boot=500, seed=20260825)
    assert lo <= hi
    assert lo <= float(np.mean(per_window_values)) <= hi


def test_build_pext_policy_constructs_all_eight(windows):
    w = windows[0]["scenario"]
    for pid in prl.PEXT_POLICIES:
        policy, sm_override = prl.build_pext_policy(pid, w)
        assert policy is not None
        assert isinstance(sm_override, dict)

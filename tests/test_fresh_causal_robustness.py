"""Tests for the POST-HOC fresh-causal robustness analysis (scripts/fresh_causal_robustness_v1.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import fresh_causal_robustness_v1 as rob


def _toy() -> pd.DataFrame:
    # 3 windows in one workload + 1 in another; two regimes; one FP-noise "beneficial" state; one tie at 0.5 ms
    rows = [
        ("A", 1, "kv", "k", 0.0), ("A", 1, "kv", "k", 0.0005000000000001), ("A", 1, "kv", "k", 0.01),
        ("A", 2, "kv", "k", 5.551115123125783e-17), ("A", 2, "act", "a", 0.0),
        ("A", 3, "act", "a", 0.002), ("B", 1, "act", "a", 0.0001), ("B", 1, "act", "a", 0.0),
    ]
    df = pd.DataFrame(rows, columns=["source_dataset", "window_index", "axis", "condition_id", "oracle_headroom"])
    df["beneficial_opportunity"] = (df["oracle_headroom"] > 0).astype(int)
    df["max_a_lat"] = df["oracle_headroom"]
    df["state_id"] = [f"s{i}" for i in range(len(df))]
    return rob.add_units(df)


def test_units_are_workload_qualified():
    df = _toy()
    assert df["window"].nunique() == 4  # (A,1) and (B,1) are distinct clusters
    assert df["regime"].nunique() == 3


def test_exceeds_guards_fp_noise_and_ties():
    h = np.array([0.0, 5.551115123125783e-17, 0.0005000000000001, 0.0005, 0.0006])
    assert list(rob.exceeds(h, 0.0)) == [False, False, True, True, True]
    assert list(rob.exceeds(h, 0.5)) == [False, False, False, False, True]  # raw FP '>' would flag 0.0005000000000001


def test_threshold_table_counts_and_raw_fp_diagnostic():
    t = rob.threshold_table(_toy())
    g = t[t["scope"] == "global"].set_index("threshold_ms")
    assert g.loc[0.0, "n_states"] == 8
    assert g.loc[0.0, "n_exceed"] == 4  # 0.0005000000000001, 0.01, 0.002, 0.0001; FP noise 5.55e-17 excluded
    assert g.loc[0.0, "n_exceed_raw_fp_strict"] == 5  # raw '>' counts the noise state too
    assert g.loc[0.5, "n_exceed_raw_fp_strict"] == g.loc[0.5, "n_exceed"] + 1


def test_bootstrap_stream_matches_canonical_sequential_draws():
    K, B = 36, 500
    rng = np.random.default_rng(rob.CANON_SEED)
    seq = np.stack([rng.integers(0, K, K) for _ in range(B)])
    bulk = np.random.default_rng(rob.CANON_SEED).integers(0, K, size=(B, K))
    assert (seq == bulk).all()


def test_boot_ratio_chunking_is_invariant_and_ratio_of_sums():
    rng = np.random.default_rng(0)
    S, N = rng.random((7, 1)), rng.integers(1, 9, 7).astype(float)
    a = rob.boot_ratio(S, N, 1000, 5, chunk=1000)
    b = rob.boot_ratio(S, N, 1000, 5, chunk=137)
    assert np.allclose(a, b)
    idx = np.random.default_rng(5).integers(0, 7, size=(1, 7))
    assert np.isclose(a[0, 0], S[idx[0], 0].sum() / N[idx[0]].sum())


def test_jackknife_recomputes_ratio_not_mean_of_ratios():
    S, N = np.array([1.0, 2.0, 9.0]), np.array([1.0, 1.0, 8.0])
    th = rob.jackknife_ratio(S, N)
    assert np.allclose(th, [(11) / 9, (10) / 9, 3 / 2])
    js = rob.jackknife_summary(S.sum() / N.sum(), th)
    assert js["loo_min"] == pytest.approx(10 / 9) and js["loo_max"] == pytest.approx(3 / 2)


def test_weighting_estimands_definitions():
    df = _toy()
    w = rob.weighting_estimands(df).set_index("estimand")
    assert w.loc["state-weighted (canonical primary)", "mean_H_ms"] == pytest.approx(df["oracle_headroom"].mean() * 1e3)
    per_win = df.groupby("window")["oracle_headroom"].mean()
    assert w.loc["equal-window", "mean_H_ms"] == pytest.approx(per_win.mean() * 1e3)
    assert w.loc["equal-window", "n_units"] == 4
    per_reg = df.groupby("regime")["oracle_headroom"].mean()
    assert w.loc["equal-regime", "mean_H_ms"] == pytest.approx(per_reg.mean() * 1e3)


def test_leave_one_out_recomputes_pooled_ratio():
    df = _toy()
    loo = rob.leave_one_out(df, "window").set_index("omitted")
    rest = df[df["window"] != "A:w1"]
    assert loo.loc["A:w1", "mean_H_ms"] == pytest.approx(rest["oracle_headroom"].mean() * 1e3)
    assert loo.loc["A:w1", "n_states"] == 5
    assert loo["share_of_total_headroom_omitted"].sum() == pytest.approx(1.0)


def test_composition_shares_sum_to_one_and_decompose_mean():
    df = _toy()
    c = rob.composition(df)
    assert c["regime"]["share_of_states"].sum() == pytest.approx(1.0)
    assert c["regime"]["share_of_total_headroom"].sum() == pytest.approx(1.0)
    assert c["regime"]["contribution_to_pooled_mean_ms"].sum() == pytest.approx(df["oracle_headroom"].mean() * 1e3)


def test_dist_stats_zero_accounting():
    d = rob.dist_stats(np.array([0.0, 0.0, 5.5e-17, 1e-3, 3e-3]))
    assert d["n_exact_zero"] == 2 and d["n_noise_level_positive(0<H<=eps)"] == 1 and d["n_positive_above_eps"] == 2
    assert d["max_ms"] == pytest.approx(3.0)


# ---------------------------------------------------------------- real canonical artifact (skipped if absent)
CANON = rob.DESIGN / "FRESH_LATENCY_STATE_LEVEL_V1.csv"


@pytest.mark.skipif(not CANON.exists(), reason="canonical fresh-causal artifact not present")
def test_canonical_primary_values_reproduce():
    states, actions = rob.load_states(), rob.load_actions()
    rep = rob.reproduction_check(states, actions)
    assert rep["overall"] in {"EXACT", "WITHIN_TOLERANCE"}
    assert rep["n_failed"] == 0
    assert len(states) == 720 and int((states["oracle_headroom"] > 0).sum()) == 590
    assert states.groupby(["source_dataset", "window_index"]).ngroups == 36
    assert rep["canonical_bootstrap_loop"]["lo"] == pytest.approx(0.00019088421760325944, abs=1e-15)
    assert rep["original_pre_correction_key_check"]["matches_documented_to_1e-6_ms"]


@pytest.mark.skipif(not CANON.exists(), reason="canonical fresh-causal artifact not present")
def test_canonical_tolerance_audit_and_reference_recovery():
    states, actions = rob.load_states(), rob.load_actions()
    tol = rob.tolerance_audit(states, actions)
    assert tol["n_actions_off_grid_by_more_than_1e-6_units"] == 0
    assert tol["beneficial_states_with_eps_guard"] == 589 and tol["canonical_beneficial_states_strict_gt0"] == 590
    # recovered SBS reference must be constant within a state (the state-level 'mean_ref_latency' column is not the reference)
    spread = actions.groupby("state_id")["ref_mean_latency"].agg(lambda v: v.max() - v.min()).max()
    assert spread < 1e-12

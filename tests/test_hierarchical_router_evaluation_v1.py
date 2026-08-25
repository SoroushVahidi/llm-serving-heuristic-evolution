"""Focused tests for Hierarchical Regime Router v1 offline baselines and
metric formulas (design doc SS R item 14, SS I baseline identity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    FALLBACK_POLICY,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_NONE,
    STAGE2_CANDIDATES,
)
from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import (
    baseline_a_anwg,
    baseline_c_anwg,
    baseline_e_anwg,
    baseline_g_anwg,
    catastrophic_misroute_rate,
    delta_anwg,
    fit_baseline_b,
    group_resampled_bootstrap_ci,
    load_scenario_level_dataset,
    mean_regret,
    multi_regime_benefit_count,
    oracle_gap_closure,
    regime_fixed_best_from_train,
)
from llmserveopt.selector.multifamily_contextual_selector_v1 import POLICY_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
UUM_WIDE = ROOT / "experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv"
MF_PSD_SCENARIOS = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
REQUIRES_DATA = pytest.mark.skipif(
    not (UUM_WIDE.exists() and MF_PSD_SCENARIOS.exists()), reason="frozen scenario-level dataset not present"
)


def _tiny_df():
    return pd.DataFrame({
        "estimated_service_time_first": [0.5, 0.9, 0.2],
        "weighted_fair_share": [0.7, 0.6, 0.3],
        "full_prefill": [0.4, 0.4, 0.4],
        "chunked_prefill_small": [0.5, 0.5, 0.5],
        "kv_constrained_online": [0.1, 0.1, 0.9],
        "least_laxity_first": [0.2, 0.2, 0.8],
        "regime_ground_truth": [REGIME_A, REGIME_A, REGIME_C],
        "group_key": ["g1", "g1", "g2"],
    })


# ---------------------------------------------------------------------------
# Baseline identity (design doc SS I)
# ---------------------------------------------------------------------------

def test_baseline_a_is_weighted_fair_share_column():
    df = _tiny_df()
    out = baseline_a_anwg(df)
    assert (out == df["weighted_fair_share"]).all()
    assert FALLBACK_POLICY == "weighted_fair_share"


def test_baseline_c_oracle_picks_max_of_native_pair_per_row():
    df = _tiny_df()
    out = baseline_c_anwg(df)
    # Row 0/1 regime A -> max(estf, wfs); row 2 regime C -> max(kv_constrained, llf)
    assert out.iloc[0] == pytest.approx(max(0.5, 0.7))
    assert out.iloc[1] == pytest.approx(max(0.9, 0.6))
    assert out.iloc[2] == pytest.approx(max(0.9, 0.8))


def test_baseline_g_oracle_is_max_over_all_six_policies():
    df = _tiny_df()
    out = baseline_g_anwg(df)
    for i in range(len(df)):
        assert out.iloc[i] == pytest.approx(df.loc[i, POLICY_COLUMNS].max())


def test_baseline_c_falls_back_to_fixed_policy_for_none_regime():
    df = _tiny_df()
    df.loc[0, "regime_ground_truth"] = REGIME_NONE
    out = baseline_c_anwg(df)
    assert out.iloc[0] == pytest.approx(df.loc[0, "weighted_fair_share"])


def test_regime_fixed_best_from_train_picks_higher_mean_per_regime():
    df = _tiny_df()
    best = regime_fixed_best_from_train(df)
    # Regime A rows: estf mean=(0.5+0.9)/2=0.7, wfs mean=(0.7+0.6)/2=0.65 -> estf wins
    assert best[REGIME_A] == "estimated_service_time_first"
    # Regime C rows: kv=0.9, llf=0.8 -> kv wins
    assert best[REGIME_C] == "kv_constrained_online"


def test_baseline_e_dispatches_to_regime_fixed_best_or_fallback():
    df = _tiny_df()
    best = regime_fixed_best_from_train(df)
    predicted_regime = pd.Series([REGIME_A, REGIME_NONE, REGIME_C])
    out = baseline_e_anwg(df, predicted_regime, best)
    assert out.iloc[0] == pytest.approx(df.loc[0, best[REGIME_A]])
    assert out.iloc[1] == pytest.approx(df.loc[1, FALLBACK_POLICY])
    assert out.iloc[2] == pytest.approx(df.loc[2, best[REGIME_C]])


# ---------------------------------------------------------------------------
# SS Q metric formulas -- hand-computed reference values
# ---------------------------------------------------------------------------

def test_mean_regret_hand_computed():
    achieved = pd.Series([0.5, 0.6, 0.7])
    oracle = pd.Series([0.8, 0.6, 0.9])
    assert mean_regret(achieved, oracle) == pytest.approx((0.3 + 0.0 + 0.2) / 3)


def test_delta_anwg_hand_computed():
    hierarchy = pd.Series([0.6, 0.7])
    comparator = pd.Series([0.5, 0.5])
    assert delta_anwg(hierarchy, comparator) == pytest.approx((0.1 + 0.2) / 2)


def test_oracle_gap_closure_hand_computed():
    # hierarchy=0.7, fixed=0.5, oracle=0.9 -> (0.7-0.5)/(0.9-0.5) = 0.5
    assert oracle_gap_closure(0.7, 0.5, 0.9) == pytest.approx(0.5)


def test_oracle_gap_closure_returns_none_on_zero_denominator():
    assert oracle_gap_closure(0.5, 0.5, 0.5) is None


def test_multi_regime_benefit_count_hand_computed():
    df = _tiny_df()
    hierarchy = pd.Series([0.9, 0.9, 0.95])  # beats fixed(wfs) everywhere it's evaluated
    fixed = df["weighted_fair_share"]
    n, per_regime = multi_regime_benefit_count(df, hierarchy, fixed)
    assert per_regime[REGIME_A] > 0
    assert per_regime[REGIME_C] > 0
    assert n == 2  # regimes A and C both present and both benefit; B absent from this tiny df


def test_catastrophic_misroute_rate_hand_computed():
    predicted = pd.Series([REGIME_A, REGIME_B, REGIME_C, REGIME_NONE])
    true = pd.Series([REGIME_A, REGIME_C, REGIME_C, REGIME_A])
    # Row0: A/A correct. Row1: B/C both active, wrong -> counts. Row2: C/C correct.
    # Row3: predicted NONE -> excluded (not both active).
    assert catastrophic_misroute_rate(predicted, true) == pytest.approx(1 / 3)


def test_catastrophic_misroute_rate_zero_when_no_active_pairs():
    predicted = pd.Series([REGIME_NONE, REGIME_NONE])
    true = pd.Series([REGIME_A, REGIME_B])
    assert catastrophic_misroute_rate(predicted, true) == 0.0


def test_group_resampled_bootstrap_ci_shape_and_direction():
    df = _tiny_df()
    hierarchy = pd.Series([0.9, 0.9, 0.95])
    fixed = df["weighted_fair_share"]
    lo, hi = group_resampled_bootstrap_ci(df, hierarchy, fixed, n_boot=200, seed=1)
    assert lo <= hi
    assert lo > 0  # hierarchy strictly beats fixed on every row here


# ---------------------------------------------------------------------------
# Real-data smoke (TRAIN/VAL only; no TEST-split conclusion drawn)
# ---------------------------------------------------------------------------

@REQUIRES_DATA
def test_load_scenario_level_dataset_has_expected_shape_and_columns():
    df = load_scenario_level_dataset()
    assert len(df) == 176
    assert set(df["regime_ground_truth"].unique()) <= {REGIME_A, REGIME_B, REGIME_C}
    assert set(df["split"].unique()) <= {"train", "val", "test"}
    for p in POLICY_COLUMNS:
        assert p in df.columns


@REQUIRES_DATA
def test_baseline_b_fits_on_train_and_predicts_only_known_policies():
    df = load_scenario_level_dataset()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    if len(train) == 0 or len(val) == 0:
        pytest.skip("empty train/val split")
    pipe, num_cols, cat_cols = fit_baseline_b(train)
    from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import baseline_b_anwg
    out = baseline_b_anwg(val, pipe, num_cols, cat_cols)
    assert len(out) == len(val)
    assert out.notna().all()

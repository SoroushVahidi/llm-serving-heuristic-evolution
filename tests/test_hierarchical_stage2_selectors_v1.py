"""Focused tests for Hierarchical Regime Router v1 Stage-2 native-pair
selectors (design doc SS R items 7-9)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (
    REGIME_A,
    REGIME_B,
    REGIME_C,
    STAGE2_CANDIDATES,
    STAGE2_EXCLUDED_CROSS_REGIME,
)
from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import load_scenario_level_dataset
from llmserveopt.selector.hierarchical_stage2_selectors_v1 import (
    Stage2Selector,
    compute_native_pair_winner,
    fit_all_stage2_selectors,
)

ROOT = Path(__file__).resolve().parents[1]
UUM_WIDE = ROOT / "experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv"
MF_PSD_SCENARIOS = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"

REQUIRES_DATA = pytest.mark.skipif(
    not (UUM_WIDE.exists() and MF_PSD_SCENARIOS.exists()), reason="frozen scenario-level dataset not present"
)


# ---------------------------------------------------------------------------
# Item 7/8: exact native-pair candidate sets; foreign-policy exclusion
# ---------------------------------------------------------------------------

def test_stage2_candidate_sets_are_exactly_the_frozen_native_pairs():
    assert STAGE2_CANDIDATES[REGIME_A] == ("estimated_service_time_first", "weighted_fair_share")
    assert STAGE2_CANDIDATES[REGIME_B] == ("full_prefill", "chunked_prefill_small")
    assert STAGE2_CANDIDATES[REGIME_C] == ("kv_constrained_online", "least_laxity_first")


def test_stage2_candidate_sets_have_exactly_two_policies_each():
    for regime, candidates in STAGE2_CANDIDATES.items():
        assert len(candidates) == 2, f"regime {regime} has {len(candidates)} candidates, expected exactly 2"


def test_kv_constrained_online_never_a_candidate_for_regime_a():
    assert "kv_constrained_online" not in STAGE2_CANDIDATES[REGIME_A]
    assert "kv_constrained_online" in STAGE2_EXCLUDED_CROSS_REGIME[REGIME_A]


def test_estf_and_wfs_never_candidates_for_regime_c():
    assert "estimated_service_time_first" not in STAGE2_CANDIDATES[REGIME_C]
    assert "weighted_fair_share" not in STAGE2_CANDIDATES[REGIME_C]
    assert set(STAGE2_EXCLUDED_CROSS_REGIME[REGIME_C]) == {"estimated_service_time_first", "weighted_fair_share"}


def test_stage2_selector_rejects_unknown_regime():
    with pytest.raises(ValueError):
        Stage2Selector("NOT_A_REGIME")


def test_stage2_selector_predict_before_fit_raises():
    sel = Stage2Selector(REGIME_A)
    with pytest.raises(RuntimeError):
        sel.predict(pd.DataFrame({"x": [1]}))


def test_stage2_selector_fit_on_zero_rows_raises():
    sel = Stage2Selector(REGIME_A)
    with pytest.raises(ValueError):
        sel.fit(pd.DataFrame())


# ---------------------------------------------------------------------------
# Item 9: deterministic model training + native-pair-only prediction on real data
# ---------------------------------------------------------------------------

@REQUIRES_DATA
def test_native_pair_winner_only_uses_the_two_candidate_columns():
    df = load_scenario_level_dataset()
    winner = compute_native_pair_winner(df, REGIME_C)
    assert set(winner.unique()) <= set(STAGE2_CANDIDATES[REGIME_C])


@REQUIRES_DATA
def test_stage2_selector_fit_and_predict_never_emits_foreign_policy():
    df = load_scenario_level_dataset()
    train = df[(df["split"] == "train") & (df["regime_ground_truth"] == REGIME_A)]
    val = df[(df["split"] == "val") & (df["regime_ground_truth"] == REGIME_A)]
    if len(train) == 0 or len(val) == 0:
        pytest.skip("no TRAIN/VAL rows for Regime A on this split")
    sel = Stage2Selector(REGIME_A).fit(train)
    preds = sel.predict(val)
    assert set(preds.tolist()) <= set(STAGE2_CANDIDATES[REGIME_A])


@REQUIRES_DATA
def test_stage2_selector_is_deterministic():
    df = load_scenario_level_dataset()
    train = df[(df["split"] == "train") & (df["regime_ground_truth"] == REGIME_C)]
    if len(train) == 0:
        pytest.skip("no TRAIN rows for Regime C on this split")
    s1 = Stage2Selector(REGIME_C).fit(train)
    s2 = Stage2Selector(REGIME_C).fit(train)
    p1 = s1.predict(train)
    p2 = s2.predict(train)
    assert (p1 == p2).all()


@REQUIRES_DATA
def test_fit_all_stage2_selectors_produces_one_per_populated_regime():
    df = load_scenario_level_dataset()
    train_by_regime = {
        regime: df[(df["split"] == "train") & (df["regime_ground_truth"] == regime)]
        for regime in (REGIME_A, REGIME_B, REGIME_C)
    }
    selectors = fit_all_stage2_selectors(train_by_regime)
    for regime, sub in train_by_regime.items():
        if len(sub) > 0:
            assert regime in selectors
            preds = selectors[regime].predict(sub)
            assert set(preds.tolist()) <= set(STAGE2_CANDIDATES[regime])

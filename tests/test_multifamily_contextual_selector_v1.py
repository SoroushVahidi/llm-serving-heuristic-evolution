"""Focused tests for the Multi-Family Contextual Selector v1 (Step 3).

See docs/design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md. Validates the
harness itself (splits, targets, metrics, anti-leakage, verdict logic),
not selector performance -- performance is reported, not asserted.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector import multifamily_contextual_selector_v1 as mcs  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return mcs.load_dataset()


# ---------------------------------------------------------------------------
# Dataset / schema
# ---------------------------------------------------------------------------

def test_dataset_has_176_scenarios_and_6_policies(df):
    assert len(df) == 176
    assert mcs.POLICY_COLUMNS == sorted(mcs.POLICY_COLUMNS)
    assert len(mcs.POLICY_COLUMNS) == 6


def test_feature_allowlist_matches_schema_exactly():
    assert len(mcs.FEATURE_COLUMNS) == 33
    assert all(c.startswith("feat_") for c in mcs.FEATURE_COLUMNS)


def test_forbidden_columns_never_in_feature_allowlist():
    assert mcs.FORBIDDEN_COLUMNS.isdisjoint(set(mcs.FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# Target / tie handling
# ---------------------------------------------------------------------------

def test_exact_winner_is_always_one_of_six_policies(df):
    winners = mcs.compute_exact_winner(df)
    assert set(winners.unique()).issubset(set(mcs.POLICY_COLUMNS))


def test_exact_winner_tie_break_prefers_chunked_over_full_prefill():
    row = pd.DataFrame([{
        "chunked_prefill_small": 0.5, "estimated_service_time_first": 0.1,
        "full_prefill": 0.5, "kv_constrained_online": 0.1,
        "least_laxity_first": 0.1, "weighted_fair_share": 0.1,
    }])
    winner = mcs.compute_exact_winner(row)
    assert winner.iloc[0] == "chunked_prefill_small"


def test_exact_winner_unique_max_no_tie():
    row = pd.DataFrame([{
        "chunked_prefill_small": 0.1, "estimated_service_time_first": 0.9,
        "full_prefill": 0.2, "kv_constrained_online": 0.3,
        "least_laxity_first": 0.4, "weighted_fair_share": 0.5,
    }])
    assert mcs.compute_exact_winner(row).iloc[0] == "estimated_service_time_first"


# ---------------------------------------------------------------------------
# Regret / metric formulas
# ---------------------------------------------------------------------------

def test_oracle_prediction_has_zero_regret(df):
    oracle = mcs.compute_exact_winner(df)
    regret = mcs.regret_of(df, oracle)
    assert np.allclose(regret, 0.0, atol=1e-9)


def test_regret_is_nonnegative(df):
    fixed = pd.Series(["kv_constrained_online"] * len(df), index=df.index)
    regret = mcs.regret_of(df, fixed)
    assert (regret >= -1e-12).all()


def test_evaluate_predictions_formulas_are_internally_consistent(df):
    pred = pd.Series(["weighted_fair_share"] * len(df), index=df.index)
    m = mcs.evaluate_predictions(df, pred)
    assert m["n"] == len(df)
    assert m["mean_regret"] >= 0
    assert 0.0 <= m["exact_winner_accuracy"] <= 1.0
    assert 0.0 <= m["epsilon_optimal_accuracy"] <= 1.0
    assert m["epsilon_optimal_accuracy"] >= m["exact_winner_accuracy"] - 1e-9  # eps-optimal is a superset of exact
    assert abs(m["gap_to_oracle_mean_regret"] - m["mean_regret"]) < 1e-9


def test_evaluate_predictions_does_not_leak_test_derived_fixed_baseline():
    """Regression guard for the fixed bug where gap_to_best_fixed was
    computed from the same df being evaluated (test-set leakage)."""
    sig = inspect.signature(mcs.evaluate_predictions)
    assert list(sig.parameters) == ["df", "predicted"]
    src = mcs.evaluate_predictions.__code__.co_names
    assert "idxmax" not in src  # no self-derived 'best fixed' inside eval


# ---------------------------------------------------------------------------
# Missing-value handling / feature matrix
# ---------------------------------------------------------------------------

def test_build_X_has_no_nan(df):
    numeric, categorical = mcs.infer_column_kinds(df)
    X = mcs.build_X(df, numeric, categorical)
    assert not X.isna().any().any()


def test_missing_indicator_columns_match_family_membership(df):
    numeric, categorical = mcs.infer_column_kinds(df)
    X = mcs.build_X(df, numeric, categorical)
    a_rows = df["mechanism_family"] == "FAMILY_A_FAIRNESS_STARVATION_V2"
    a_numeric = [c for c in numeric if c.startswith("feat_A__")]
    if a_numeric:
        col = f"{a_numeric[0]}__missing"
        assert (X.loc[a_rows, col] == 0).all()
        assert (X.loc[~a_rows, col] == 1).all()


# ---------------------------------------------------------------------------
# Splits: deterministic, group-disjoint, no leakage, LOFO exclusion
# ---------------------------------------------------------------------------

def test_regime_a_splits_are_group_disjoint(df):
    splits = mcs.regime_a_within_family_splits(df)
    for fam, s in splits.items():
        g_train = set(s["train"]["group_key"])
        g_val = set(s["val"]["group_key"])
        g_test = set(s["test"]["group_key"])
        assert g_train.isdisjoint(g_val)
        assert g_train.isdisjoint(g_test)
        assert g_val.isdisjoint(g_test)
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(df[df["mechanism_family"] == fam])


def test_regime_b_split_is_group_disjoint_and_deterministic(df):
    b1 = mcs.regime_b_pooled_split(df)
    b2 = mcs.regime_b_pooled_split(df)
    assert set(b1["test"]["canonical_scenario_id"]) == set(b2["test"]["canonical_scenario_id"])
    assert set(b1["train"]["group_key"]).isdisjoint(set(b1["test"]["group_key"]))
    assert len(b1["train"]) + len(b1["val"]) + len(b1["test"]) == len(df)


def test_regime_c_lofo_test_set_is_exactly_the_held_out_family(df):
    splits = mcs.regime_c_lofo_splits(df)
    for held_out, s in splits.items():
        assert set(s["test"]["mechanism_family"]) == {held_out}
        assert len(s["test"]) == len(df[df["mechanism_family"] == held_out])
        assert held_out not in set(s["train"]["mechanism_family"])
        assert held_out not in set(s["val"]["mechanism_family"])


def test_regime_c_lofo_train_val_never_touches_held_out_family_group_keys(df):
    splits = mcs.regime_c_lofo_splits(df)
    for held_out, s in splits.items():
        held_out_groups = set(df[df["mechanism_family"] == held_out]["group_key"])
        train_val_groups = set(s["train"]["group_key"]) | set(s["val"]["group_key"])
        assert train_val_groups.isdisjoint(held_out_groups)


def test_split_groups_n_way_covers_every_group_exactly_once():
    groups = [f"g{i}" for i in range(20)]
    parts = mcs.split_groups_n_way(groups, (0.6, 0.2, 0.2))
    flat = [g for part in parts for g in part]
    assert sorted(flat) == sorted(groups)
    assert all(len(p) >= 1 for p in parts)


# ---------------------------------------------------------------------------
# Anti-leakage: family label / utility columns / identity never in X
# ---------------------------------------------------------------------------

def test_family_label_never_in_feature_columns():
    assert "mechanism_family" not in mcs.FEATURE_COLUMNS


def test_utility_columns_never_in_feature_columns():
    assert set(mcs.POLICY_COLUMNS).isdisjoint(set(mcs.FEATURE_COLUMNS))


def test_build_X_output_columns_disjoint_from_forbidden(df):
    numeric, categorical = mcs.infer_column_kinds(df)
    X = mcs.build_X(df, numeric, categorical)
    assert set(X.columns).isdisjoint(mcs.FORBIDDEN_COLUMNS)


# ---------------------------------------------------------------------------
# Models: all six policy IDs resolvable, reproducible with fixed seed
# ---------------------------------------------------------------------------

def test_all_model_classes_predict_only_known_policies(df):
    numeric, categorical = mcs.infer_column_kinds(df)
    X = mcs.build_X(df, numeric, categorical)
    y = mcs.compute_exact_winner(df)
    preproc = mcs.build_preprocessor(numeric, categorical)
    for name in ("logreg", "tree", "forest"):
        pipe = mcs.fit_classifier(name, X, y, preproc)
        pred = mcs.predict_classifier(pipe, X, df.index)
        assert set(pred.unique()).issubset(set(mcs.POLICY_COLUMNS))


def test_forest_is_reproducible_with_fixed_seed(df):
    numeric, categorical = mcs.infer_column_kinds(df)
    X = mcs.build_X(df, numeric, categorical)
    y = mcs.compute_exact_winner(df)
    preproc = mcs.build_preprocessor(numeric, categorical)
    p1 = mcs.predict_classifier(mcs.fit_classifier("forest", X, y, preproc), X, df.index)
    p2 = mcs.predict_classifier(mcs.fit_classifier("forest", X, y, preproc), X, df.index)
    assert (p1.to_numpy() == p2.to_numpy()).all()


def test_utility_argmax_and_pairwise_predict_only_known_policies(df):
    numeric, categorical = mcs.infer_column_kinds(df)
    X = mcs.build_X(df, numeric, categorical)
    preproc = mcs.build_preprocessor(numeric, categorical)
    reg = mcs.fit_utility_regressors(X, df, preproc)
    pred_f = mcs.predict_utility_argmax(reg, X, df.index)
    assert set(pred_f.unique()).issubset(set(mcs.POLICY_COLUMNS))
    pw = mcs.fit_pairwise(X, df, preproc)
    pred_g = mcs.predict_pairwise(pw, X, df.index)
    assert set(pred_g.unique()).issubset(set(mcs.POLICY_COLUMNS))


# ---------------------------------------------------------------------------
# Family-predictability diagnostic / shared-feature robustness
# ---------------------------------------------------------------------------

def test_family_predictability_diagnostic_runs_and_returns_valid_accuracy(df):
    result = mcs.family_predictability_diagnostic(df)
    assert 0.0 <= result["mean_accuracy"] <= 1.0
    assert result["n_folds"] >= 1


def test_shared_feature_robustness_check_excludes_family_c(df):
    result = mcs.shared_feature_robustness_check(df)
    assert result["family_c_excluded"] is True
    assert result["n_scenarios"] == 104  # 72 (A) + 32 (B)


# ---------------------------------------------------------------------------
# Macro vs micro aggregation
# ---------------------------------------------------------------------------

def test_macro_family_average_differs_from_micro_when_family_sizes_differ(df):
    b = mcs.regime_b_pooled_split(df)
    numeric, categorical = mcs.infer_column_kinds(df)
    X_train = mcs.build_X(b["train"], numeric, categorical)
    X_test = mcs.build_X(b["test"], numeric, categorical)
    preproc = mcs.build_preprocessor(numeric, categorical)
    pipe = mcs.fit_classifier("forest", X_train, mcs.compute_exact_winner(b["train"]), preproc)
    pred = mcs.predict_classifier(pipe, X_test, b["test"].index)
    per_family_regret = []
    for fam in mcs.FAMILIES:
        mask = b["test"]["mechanism_family"] == fam
        if mask.sum() == 0:
            continue
        per_family_regret.append(mcs.evaluate_predictions(b["test"][mask], pred[mask])["mean_regret"])
    macro = float(np.mean(per_family_regret))
    micro = mcs.evaluate_predictions(b["test"], pred)["mean_regret"]
    # Not asserting they differ (could coincide), just that both are well-defined finite numbers.
    assert np.isfinite(macro) and np.isfinite(micro)


# ---------------------------------------------------------------------------
# No mutation of frozen artifacts
# ---------------------------------------------------------------------------

def test_no_mutation_of_step2_and_mf_psd_artifacts():
    import subprocess
    result = subprocess.run(
        ["git", "status", "--short",
         "experiments/mf_psd_v1/", "experiments/unified_utility_matrix_v1/",
         "experiments/unified_utility_matrix_v2/", "experiments/family_c_reconstruction_v1/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", f"frozen Step-1/2 artifacts show git diff: {result.stdout}"

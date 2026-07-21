import pandas as pd
import pytest

from llmserveopt.selector.advanced import (
    PairwisePolicyRanker,
    PolicyRewardRegressorSelector,
    UncertaintyFallbackSelector,
    azure_conv_like_gate,
    policy_margin_weights,
    validate_feature_columns,
)


POLICIES = ["orca_style", "scorpio_style_slo_guard", "admission_control"]
FEATURES = ["feat_mean_prompt_tokens", "feat_fraction_tight_slo", "feat_arrival_rate_est"]


def _rows(n=80):
    rows = []
    for i in range(n):
        azure_like = i % 2 == 0
        prompt = 1500.0 if azure_like else 128.0
        tight = 0.5 if azure_like else 0.2
        row = {
            "feat_mean_prompt_tokens": prompt,
            "feat_fraction_tight_slo": tight,
            "feat_arrival_rate_est": float(5 + i % 7),
            "label_best_all_non_oracle_policy": "orca_style" if azure_like else "scorpio_style_slo_guard",
            "anwg_orca_style": 0.9 if azure_like else 0.65,
            "anwg_scorpio_style_slo_guard": 0.7 if azure_like else 0.92,
            "anwg_admission_control": 0.6,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_validate_feature_columns_rejects_leaky_columns():
    with pytest.raises(ValueError):
        validate_feature_columns(["feat_mean_prompt_tokens", "reward_orca_style"])


def test_validate_feature_columns_allows_recent_slo_violation_rate_feature():
    assert validate_feature_columns(["feat_recent_slo_violation_rate"]) == ["feat_recent_slo_violation_rate"]


def test_policy_margin_weights_downweight_near_ties():
    rows = [
        {"anwg_orca_style": 0.9, "anwg_scorpio_style_slo_guard": 0.1},
        {"anwg_orca_style": 0.501, "anwg_scorpio_style_slo_guard": 0.5},
    ]
    weights = policy_margin_weights(rows, ["orca_style", "scorpio_style_slo_guard"])
    assert weights[0] > weights[1]
    assert weights[1] > 0.0


def test_reward_regressor_predicts_from_features_only():
    pytest.importorskip("sklearn")
    df = _rows()
    selector = PolicyRewardRegressorSelector(
        name="reg_test",
        allowed_policies=POLICIES,
        feature_cols=FEATURES,
        estimator="random_forest",
        n_estimators=20,
        max_depth=4,
        random_state=0,
    ).fit(df)
    feature_only = df[FEATURES].copy()
    assert selector.predict(df) == selector.predict(feature_only)


def test_pairwise_ranker_recovers_orca_scorpio_pattern():
    pytest.importorskip("sklearn")
    df = _rows()
    selector = PairwisePolicyRanker(
        name="pairwise_test",
        allowed_policies=POLICIES,
        feature_cols=FEATURES,
        pairs=[("orca_style", "scorpio_style_slo_guard")],
        estimator="random_forest",
        n_estimators=20,
        max_depth=4,
        random_state=0,
        min_pair_margin=0.01,
    ).fit(df)
    preds = selector.predict(df)
    assert preds[0] == "orca_style"
    assert preds[1] == "scorpio_style_slo_guard"


def test_uncertainty_fallback_uses_predicted_margin_not_rewards():
    pytest.importorskip("sklearn")
    df = _rows()
    base = PolicyRewardRegressorSelector(
        name="reg_test",
        allowed_policies=POLICIES,
        feature_cols=FEATURES,
        estimator="random_forest",
        n_estimators=20,
        max_depth=4,
        random_state=0,
    ).fit(df)
    wrapped = UncertaintyFallbackSelector(
        name="fallback_test",
        base_selector=base,
        fallback_policy="scorpio_style_slo_guard",
        margin_threshold=10.0,
    )
    assert set(wrapped.predict(df[FEATURES])) == {"scorpio_style_slo_guard"}


def test_azure_conv_like_gate_is_feature_based():
    assert azure_conv_like_gate({"feat_mean_prompt_tokens": 1200, "feat_fraction_tight_slo": 0.5})
    assert not azure_conv_like_gate({"feat_mean_prompt_tokens": 1200, "feat_fraction_tight_slo": 0.9})

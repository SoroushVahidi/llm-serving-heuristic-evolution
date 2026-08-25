"""Focused tests for CC5 model-agnostic uncertainty and regime fallback."""
from __future__ import annotations

import json

import numpy as np
import pytest

from llmserveopt.experiments import cc5_contextual_predictor as cc5
from llmserveopt.experiments import cc5_uncertainty_regime_refinement as refine

DATASET_DIR = "results/cc4_oracle_composition_dataset/20260803T170735Z"
CC4B_DIR = "results/cc4b_oracle_composition_expansion/20260803T182426Z"

pytestmark = pytest.mark.skipif(
    not (cc5.ROOT / DATASET_DIR / "manifest.json").exists(),
    reason="reference CC4 dataset not present locally",
)


@pytest.fixture(scope="module")
def ds() -> cc5.CC4Dataset:
    return cc5.load_cc4_dataset(cc5.ROOT / DATASET_DIR)


@pytest.fixture(scope="module")
def encoder(ds) -> cc5.FeatureEncoder:
    dev_ids = ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"]
    return cc5.FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])


@pytest.fixture(scope="module")
def calibrators(ds, encoder):
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    factories = cc5.build_regret_regressor_factories(seed=0)
    fb = cc5.fit_best_fixed_policy(ds, dev_ids)
    selected, all_cals, grid = cc5.fit_model_agnostic_uncertainty(
        model_name="gradient_boosting",
        model_factory=factories["gradient_boosting"],
        encoder=encoder,
        ds=ds,
        dev_window_ids=dev_ids,
        seed=0,
        n_bootstrap=5,
        fallback_for_threshold=fb,
    )
    return selected, all_cals, grid, factories, fb, dev_ids


def test_uncertainty_calibration_no_eval_leakage(ds, calibrators):
    selected, all_cals, _, _, _, _ = calibrators
    eval_ids = set(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])
    for cal in all_cals.values():
        assert set(cal.train_window_ids).isdisjoint(eval_ids)
        assert set(cal.calibration_window_ids).isdisjoint(eval_ids)
        assert set(cal.train_window_ids).isdisjoint(set(cal.calibration_window_ids))
        assert set(cal.calibration_window_ids) <= set(
            ds.causal_features[ds.causal_features["split"] == "VALIDATION"]["window_id"]
        )


def test_both_uncertainty_methods_produce_finite_scores(ds, encoder, calibrators):
    selected, all_cals, _, _, _, dev_ids = calibrators
    causal_row = ds.causal_features.set_index("window_id").loc[dev_ids[0]]
    _, X = cc5.build_candidate_matrix(ds, encoder, causal_row)
    for cal in all_cals.values():
        scores = cal.score(X)
        assert len(scores) == len(X)
        assert np.isfinite(scores).all()
        assert (scores >= 0).all()


def test_uncertainty_intervals_deterministic(ds, encoder):
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    factories = cc5.build_regret_regressor_factories(seed=0)
    fb = cc5.fit_best_fixed_policy(ds, dev_ids)
    a1, _, _ = cc5.fit_model_agnostic_uncertainty(
        model_name="gradient_boosting", model_factory=factories["gradient_boosting"],
        encoder=encoder, ds=ds, dev_window_ids=dev_ids, seed=0, n_bootstrap=5, fallback_for_threshold=fb,
    )
    a2, _, _ = cc5.fit_model_agnostic_uncertainty(
        model_name="gradient_boosting", model_factory=factories["gradient_boosting"],
        encoder=encoder, ds=ds, dev_window_ids=dev_ids, seed=0, n_bootstrap=5, fallback_for_threshold=fb,
    )
    assert a1.method == a2.method
    assert a1.conformal_quantile == pytest.approx(a2.conformal_quantile)
    assert a1.uncertainty_threshold == pytest.approx(a2.uncertainty_threshold)
    causal_row = ds.causal_features.set_index("window_id").loc[dev_ids[0]]
    _, X = cc5.build_candidate_matrix(ds, encoder, causal_row)
    assert np.allclose(a1.score(X), a2.score(X))


def test_threshold_selection_uses_validation_only(calibrators):
    selected, _, grid, _, _, _ = calibrators
    assert "selection_threshold" in selected.calibration_manifest
    assert not grid.empty
    assert set(grid["method"]) <= {"normalized_split_conformal", "bootstrap_ensemble"}


def test_stale_calibration_schema_rejected(calibrators):
    selected, _, _, _, _, _ = calibrators
    bad = cc5.UncertaintyCalibrator(**{**selected.__dict__, "schema_version": selected.schema_version - 1})
    with pytest.raises(cc5.CC5Error, match="stale uncertainty calibration"):
        cc5.assert_uncertainty_calibrator_compatible(bad)


def test_fallback_behavior_respects_gate_modes(ds, encoder, calibrators):
    selected, _, _, factories, fb, dev_ids = calibrators
    model = factories["gradient_boosting"]()
    X, y, _ = cc5.build_regret_training_table(ds, encoder, dev_ids)
    model.fit(X, y)
    gate = cc5.UncertaintyOODGate.fit(
        encoder, ds.causal_features, dev_ids,
        ood_z_threshold=2.0, uncertainty_threshold=selected.uncertainty_threshold,
    )
    artifact = cc5.PredictorArtifact(
        model_name="gradient_boosting", model=model, encoder=encoder, gate=gate, fallback=fb,
        supports_ensemble_uncertainty=False, dsl_schema_version=2, compiler_version="cc3.1",
        dataset_config_hash="", dataset_dir=str(ds.dataset_dir), git_sha="test",
        feature_schema=encoder.feature_names, target_definition="t", split_definition={},
        hyperparameters={}, uncertainty_method=selected.method,
        ood_method="max_abs_zscore_vs_dev_causal_feature_distribution",
        objective_definition=cc5.PRIMARY, training_timestamp="t", dependency_versions={},
        uncertainty_calibrator=selected, gate_mode="ood_or_uncertainty",
    )
    causal_row = ds.causal_features.iloc[0]
    d_ood = cc5.select_composition_with_fallback(artifact, ds, causal_row, gate_mode="ood_only")
    d_unc = cc5.select_composition_with_fallback(artifact, ds, causal_row, gate_mode="uncertainty_only")
    d_both = cc5.select_composition_with_fallback(artifact, ds, causal_row, gate_mode="ood_or_uncertainty")
    for d in (d_ood, d_unc, d_both):
        assert d["selected_candidate_id"] in set(ds.candidate_compositions["candidate_id"])
        assert "uncertainty" in d
        assert d["gate_mode"] in {"ood_only", "uncertainty_only", "ood_or_uncertainty"}


def test_regime_rule_validation_only_and_inference_determinism(ds, encoder, calibrators):
    selected, _, _, factories, fb, dev_ids = calibrators
    model = factories["gradient_boosting"]()
    X, y, _ = cc5.build_regret_training_table(ds, encoder, dev_ids)
    model.fit(X, y)
    gate = cc5.UncertaintyOODGate.fit(
        encoder, ds.causal_features, dev_ids, ood_z_threshold=2.0,
        uncertainty_threshold=selected.uncertainty_threshold,
    )
    best_global = cc5.fit_best_global_composition(ds, dev_ids)
    artifact = cc5.PredictorArtifact(
        model_name="gradient_boosting", model=model, encoder=encoder, gate=gate, fallback=best_global,
        supports_ensemble_uncertainty=False, dsl_schema_version=2, compiler_version="cc3.1",
        dataset_config_hash="", dataset_dir=str(ds.dataset_dir), git_sha="test",
        feature_schema=encoder.feature_names, target_definition="t", split_definition={},
        hyperparameters={}, uncertainty_method=selected.method,
        ood_method="z", objective_definition=cc5.PRIMARY, training_timestamp="t", dependency_versions={},
        uncertainty_calibrator=selected,
    )
    val_ids = sorted(ds.causal_features[ds.causal_features["split"] == "VALIDATION"]["window_id"])
    rules = refine.fit_regime_fallback_rules(artifact=artifact, ds=ds, val_ids=val_ids, best_global=best_global)
    assert set(rules.values()) <= {"fallback", "trust_predictor"}
    artifact.regime_fallback_rules = rules
    causal_row = ds.causal_features.iloc[0]
    d1 = cc5.select_composition_with_fallback(artifact, ds, causal_row, gate_mode="regime_aware")
    d2 = cc5.select_composition_with_fallback(artifact, ds, causal_row, gate_mode="regime_aware")
    for key in ("selected_candidate_id", "model_recommended_candidate_id", "predicted_regret", "uncertainty", "ood_score", "abstained", "fallback_reason", "gate_mode"):
        assert d1[key] == d2[key]


def test_training_emits_calibration_artifacts_and_valid_method(tmp_path, ds):
    result = cc5.run_training(dataset_dir=ds.dataset_dir, output_root=str(tmp_path), timestamp="unc", seed=0)
    assert (result.output_dir / "calibration_manifest.json").exists()
    assert (result.output_dir / "uncertainty_method_comparison.csv").exists()
    assert result.manifest["uncertainty_method"] in cc5.SUPPORTED_UNCERTAINTY_METHODS
    assert result.manifest["uncertainty_schema_version"] == cc5.UNCERTAINTY_SCHEMA_VERSION
    cal = json.loads((result.output_dir / "calibration_manifest.json").read_text())
    assert "empirical_coverage" in cal
    assert "calibration_error" in cal


def test_gradient_boosting_has_usable_uncertainty(calibrators):
    selected, _, _, _, _, _ = calibrators
    assert selected.method in cc5.SUPPORTED_UNCERTAINTY_METHODS
    assert selected.method != "unsupported_for_selected_model_type"
    assert np.isfinite(selected.uncertainty_threshold)


@pytest.mark.skipif(
    not (cc5.ROOT / CC4B_DIR / "manifest.json").exists(),
    reason="CC4b dataset not present locally",
)
def test_cc4b_artifacts_still_validate():
    ds = cc5.load_cc4_dataset(cc5.ROOT / CC4B_DIR)
    findings = cc5.validate_cc4_dataset(ds)
    assert any("evaluation windows" in f for f in findings)

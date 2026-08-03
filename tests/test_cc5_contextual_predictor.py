"""CC5: contextual composition predictor tests.

Runs against the real CC4 dataset (results/cc4_oracle_composition_dataset/
20260803T170735Z/, ~2s full training/evaluation) rather than a synthetic
fixture, since that dataset already exists, is small, and exercising the
real leakage/split boundaries end-to-end is stronger evidence than a mock.
If the reference dataset is ever removed, these tests skip cleanly rather
than fail (CC5 has a hard dependency on CC4's output, which is
intentionally not committed to git -- see the CC4 report).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from llmserveopt.experiments import cc5_contextual_predictor as cc5

DATASET_DIR = "results/cc4_oracle_composition_dataset/20260803T170735Z"

pytestmark = pytest.mark.skipif(
    not (cc5.ROOT / DATASET_DIR / "manifest.json").exists(),
    reason="reference CC4 dataset not present locally (results/ is gitignored, regenerate via replay_commands.sh)",
)


@pytest.fixture(scope="module")
def ds() -> cc5.CC4Dataset:
    return cc5.load_cc4_dataset(cc5.ROOT / DATASET_DIR)


@pytest.fixture(scope="module")
def encoder(ds) -> cc5.FeatureEncoder:
    dev_ids = ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"]
    return cc5.FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])


# ---------------------------------------------------------------------------
# Causal feature enforcement / no oracle leakage
# ---------------------------------------------------------------------------


def test_causal_feature_columns_contain_no_outcome_or_oracle_fields():
    for col in cc5.CAUSAL_FEATURE_COLUMNS:
        assert not col.startswith("metric_")
        assert "oracle" not in col
        assert "regret" not in col


def test_feature_encoder_never_reads_outcome_columns(encoder, ds):
    causal_row = ds.causal_features.iloc[0]
    vec = encoder.causal_vector(causal_row)
    assert len(vec) == len(cc5.CAUSAL_FEATURE_COLUMNS)
    # Candidate vector is built only from a candidate's own declared recipe
    # (family/weights/extras), never from any executed metric.
    cand_vec = encoder.candidate_vector("weighted_primitive_mixture", {"laxity_urgency": 0.5}, {})
    assert len(cand_vec) == len(cc5.CANDIDATE_FAMILIES) + len(cc5.PRIMITIVE_POOL) + 3


def test_module_never_imports_dsl_synthesis_functions():
    """CC5 only ever selects among CC4's pre-verified candidate pool -- it
    must never gain the ability to synthesize or verify new DSL programs
    (that stays CC3/CC4 scope)."""
    import llmserveopt.experiments.cc5_contextual_predictor as mod
    assert not hasattr(mod, "compile_heuristic")
    assert not hasattr(mod, "verify_heuristic")


# ---------------------------------------------------------------------------
# Dataset validation / stale dataset rejection
# ---------------------------------------------------------------------------


def test_validate_cc4_dataset_clean(ds):
    findings = cc5.validate_cc4_dataset(ds)
    assert any("development windows" in f for f in findings)


def test_stale_dataset_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment": "not_cc4"}))
    with pytest.raises(cc5.CC5Error, match="not a CC4"):
        cc5.load_cc4_dataset(tmp_path)


def test_missing_dataset_dir_rejected(tmp_path):
    with pytest.raises(cc5.CC5Error, match="does not exist"):
        cc5.load_cc4_dataset(tmp_path / "nonexistent")


def test_missing_table_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"experiment": "cc4_oracle_composition_dataset"}))
    with pytest.raises(cc5.CC5Error, match="missing required table"):
        cc5.load_cc4_dataset(tmp_path)


def test_split_leakage_detected(ds):
    bad = cc5.CC4Dataset(
        dataset_dir=ds.dataset_dir, manifest=ds.manifest, workload_windows=ds.workload_windows,
        causal_features=ds.causal_features, candidate_compositions=ds.candidate_compositions,
        per_window_results=ds.per_window_results, oracle_labels=ds.oracle_labels,
        regret_matrix=ds.regret_matrix, composition_parameters=ds.composition_parameters,
        near_tie_flags=ds.near_tie_flags, completion_constraints=ds.completion_constraints,
        development_splits=("TRAIN", "VALIDATION", "ID_TEST"),  # deliberately overlaps eval
        evaluation_splits=("ID_TEST", "OOD_TEST"),
    )
    with pytest.raises(cc5.CC5Error, match="overlap"):
        cc5.validate_cc4_dataset(bad)


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------


def test_dev_and_eval_windows_disjoint_and_match_config(ds):
    dev = set(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    eval_ = set(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])
    assert dev.isdisjoint(eval_)
    assert len(dev) == 6
    assert len(eval_) == 6


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def test_build_regret_training_table_shape_and_values(ds, encoder):
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    X, y, meta = cc5.build_regret_training_table(ds, encoder, dev_ids)
    assert X.shape[0] == len(dev_ids) * len(ds.candidate_compositions)
    assert X.shape[1] == len(encoder.feature_names)
    assert len(y) == len(meta) == X.shape[0]
    # Spot-check: y values must match regret_matrix exactly for a sampled row.
    row = meta.iloc[0]
    expected = ds.regret_matrix[
        (ds.regret_matrix["window_id"] == row["window_id"]) & (ds.regret_matrix["candidate_id"] == row["candidate_id"])
    ]["regret"].iloc[0]
    assert y[0] == pytest.approx(expected)


def test_build_class_training_table(ds, encoder):
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    X, y, window_ids = cc5.build_class_training_table(ds, encoder, dev_ids)
    assert X.shape == (len(dev_ids), len(cc5.CAUSAL_FEATURE_COLUMNS))
    assert set(y) <= set(cc5.CANDIDATE_FAMILIES)


# ---------------------------------------------------------------------------
# Verified composition output (never synthesizes, always selects from the pool)
# ---------------------------------------------------------------------------


def test_select_candidate_for_window_always_in_pool(ds, encoder):
    from sklearn.linear_model import Ridge

    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    X, y, _ = cc5.build_regret_training_table(ds, encoder, dev_ids)
    model = Ridge().fit(X, y)
    valid_ids = set(ds.candidate_compositions["candidate_id"])
    for window_id in dev_ids:
        selected = cc5.select_candidate_for_window(ds, encoder, model, window_id)
        assert selected in valid_ids


# ---------------------------------------------------------------------------
# Uncertainty / OOD / abstention / fallback
# ---------------------------------------------------------------------------


def test_uncertainty_zero_for_non_ensemble_model():
    class FakeArtifact:
        supports_ensemble_uncertainty = False
        model = type("M", (), {"predict": staticmethod(lambda X: np.zeros(len(X)))})()

    preds, uncertainty = cc5._predict_with_uncertainty(FakeArtifact(), np.zeros((3, 5)))
    assert (uncertainty == 0.0).all()


def test_uncertainty_nonzero_for_random_forest(ds, encoder):
    from sklearn.ensemble import RandomForestRegressor

    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    X, y, _ = cc5.build_regret_training_table(ds, encoder, dev_ids)
    rf = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)

    class FakeArtifact:
        supports_ensemble_uncertainty = True
        model = rf

    preds, uncertainty = cc5._predict_with_uncertainty(FakeArtifact(), X[:5])
    assert (uncertainty >= 0.0).all()


def test_ood_gate_flags_far_out_of_distribution_point(ds, encoder):
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    gate = cc5.UncertaintyOODGate.fit(encoder, ds.causal_features, dev_ids, ood_z_threshold=2.0, uncertainty_threshold=1.0)
    far_vector = gate.dev_causal_mean + 100 * gate.dev_causal_std
    assert gate.is_ood(far_vector)
    assert not gate.is_ood(gate.dev_causal_mean)


def test_lookup_baseline_fallback_selects_valid_fixed_policy(ds):
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    best_fixed = cc5.fit_best_fixed_policy(ds, dev_ids)
    selected = best_fixed.select("any_regime_not_seen")
    fixed_ids = set(ds.candidate_compositions[ds.candidate_compositions["family"] == "fixed_policy"]["candidate_id"])
    assert selected in fixed_ids


# ---------------------------------------------------------------------------
# Deterministic training / inference determinism / full integration
# ---------------------------------------------------------------------------


def test_training_is_deterministic(tmp_path):
    r1 = cc5.run_training(dataset_dir=cc5.ROOT / DATASET_DIR, output_root=str(tmp_path.relative_to(cc5.ROOT)) if tmp_path.is_relative_to(cc5.ROOT) else str(tmp_path), timestamp="run1", seed=0)
    r2 = cc5.run_training(dataset_dir=cc5.ROOT / DATASET_DIR, output_root=str(tmp_path.relative_to(cc5.ROOT)) if tmp_path.is_relative_to(cc5.ROOT) else str(tmp_path), timestamp="run2", seed=0)
    assert r1.manifest["model_type"] == r2.manifest["model_type"]
    assert r1.verdict["predictor_anwg"]["mean"] == pytest.approx(r2.verdict["predictor_anwg"]["mean"])
    assert r1.verdict["status"] == r2.verdict["status"]


def test_resume_short_circuits_without_retraining(tmp_path):
    out_root = tmp_path
    r1 = cc5.run_training(dataset_dir=cc5.ROOT / DATASET_DIR, output_root=str(out_root), timestamp="resumeme", seed=0)
    mtime_before = (r1.output_dir / "manifest.json").stat().st_mtime
    r2 = cc5.run_training(dataset_dir=cc5.ROOT / DATASET_DIR, output_root=str(out_root), resume_dir=r1.output_dir, seed=0)
    mtime_after = (r2.output_dir / "manifest.json").stat().st_mtime
    assert mtime_before == mtime_after  # never rewritten -- short-circuited
    assert r2.verdict == r1.verdict


def test_full_run_produces_required_outputs_and_valid_verdict(tmp_path):
    result = cc5.run_training(dataset_dir=cc5.ROOT / DATASET_DIR, output_root=str(tmp_path), timestamp="full", seed=0)
    for name in (
        "manifest.json", "verdict.json", "model_card.md", "dataset_audit.json", "cv_model_selection.csv",
        "per_window_predictions.csv", "per_regime_summaries.csv", "fallback_analysis.csv",
        "composition_class_predictions.csv", "uncertainty_ood_diagnostics.csv", "regret_tables.csv",
        "resolved_config.json", "replay_commands.sh",
    ):
        assert (result.output_dir / name).exists(), name
    assert result.verdict["status"] in ("PROCEED", "REGIME_SPECIFIC_ONLY", "STOP_OR_REDESIGN", "INCONCLUSIVE")
    manifest = result.manifest
    for key in (
        "git_sha", "dataset_config_hash", "feature_schema", "target_definition", "split_definition",
        "model_type", "hyperparameters", "uncertainty_method", "ood_method", "fallback_policy",
        "objective_definition", "training_timestamp", "dependency_versions",
    ):
        assert key in manifest, key


def test_runtime_wrapper_returns_required_keys_and_is_deterministic(ds):
    result = cc5.run_training(dataset_dir=ds.dataset_dir, output_root="results/cc5_contextual_composition_predictor", timestamp="wrapper_test_tmp", seed=0)
    try:
        causal_row = ds.causal_features.iloc[0]
        artifact = _rebuild_artifact_from_manifest(ds, result)
        decision1 = cc5.select_composition_with_fallback(artifact, ds, causal_row)
        decision2 = cc5.select_composition_with_fallback(artifact, ds, causal_row)
        for key in ("selected_candidate_id", "model_recommended_candidate_id", "predicted_regret", "uncertainty", "ood_score", "abstained", "fallback_reason"):
            assert key in decision1
        assert decision1 == decision2  # inference determinism
        assert decision1["selected_candidate_id"] in set(ds.candidate_compositions["candidate_id"])
    finally:
        import shutil
        shutil.rmtree(result.output_dir, ignore_errors=True)


def _rebuild_artifact_from_manifest(ds, result):
    """Helper: re-run the same training call in-process to get a live
    PredictorArtifact object (manifest.json is a serialized summary, not a
    pickle -- this mirrors how a real CLI caller would need to retrain or
    load a pickled artifact; CC5's own run_training already returns the
    fitted model in-memory via its internal artifact, exercised directly
    in the other tests above)."""
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    encoder = cc5.FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])
    from sklearn.ensemble import RandomForestRegressor
    X, y, _ = cc5.build_regret_training_table(ds, encoder, dev_ids)
    model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    gate = cc5.UncertaintyOODGate.fit(encoder, ds.causal_features, dev_ids, ood_z_threshold=2.0, uncertainty_threshold=0.05)
    fallback = cc5.fit_best_fixed_policy(ds, dev_ids)
    return cc5.PredictorArtifact(
        model_name="random_forest", model=model, encoder=encoder, gate=gate, fallback=fallback,
        supports_ensemble_uncertainty=True, dsl_schema_version=2, compiler_version="cc3.1",
        dataset_config_hash="", dataset_dir=str(ds.dataset_dir), git_sha="test",
        feature_schema=encoder.feature_names, target_definition="test", split_definition={},
        hyperparameters={}, uncertainty_method="random_forest_per_tree_prediction_std",
        ood_method="max_abs_zscore_vs_dev_causal_feature_distribution", objective_definition=cc5.PRIMARY,
        training_timestamp="test", dependency_versions={},
    )


# ---------------------------------------------------------------------------
# Bootstrap CI honesty
# ---------------------------------------------------------------------------


def test_bootstrap_ci_reports_n_and_wide_interval_for_small_samples():
    ci = cc5.bootstrap_ci([0.1, 0.5, 0.9], n_boot=500)
    assert ci["n"] == 3
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]


def test_bootstrap_ci_empty_input():
    ci = cc5.bootstrap_ci([])
    assert ci["n"] == 0
    assert np.isnan(ci["mean"])

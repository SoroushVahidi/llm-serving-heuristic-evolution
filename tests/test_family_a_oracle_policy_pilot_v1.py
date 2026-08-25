"""Tests for the Family-A oracle-labeled ESTF/WFS pilot dataset v1.

These are dataset-construction and integrity tests only. They do not train a
controller and do not run a new scientific simulation.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/build_family_a_oracle_policy_pilot_v1.py"
DATASET_DIR = REPO_ROOT / "datasets/family_a_oracle_policy_pilot_v1"


def _module():
    spec = importlib.util.spec_from_file_location("family_a_oracle_policy_pilot_v1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _rows() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "pilot_rows.csv")


def _schema() -> dict:
    return json.loads((DATASET_DIR / "schema.json").read_text())


def test_build_rows_is_deterministic_and_matches_persisted_dataset():
    mod = _module()
    df1, feature_cols1, _, _ = mod.build_rows()
    df2, feature_cols2, _, _ = mod.build_rows()
    persisted = _rows()
    assert feature_cols1 == feature_cols2
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), persisted.reset_index(drop=True))


def test_script_reproduces_exact_rows_and_manifest_in_tmpdir(tmp_path):
    out = tmp_path / "pilot"
    subprocess.check_call([sys.executable, str(SCRIPT), "--output-dir", str(out)], cwd=REPO_ROOT)
    produced = pd.read_csv(out / "pilot_rows.csv")
    pd.testing.assert_frame_equal(produced, _rows())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["pilot_rows_sha256"] == json.loads((DATASET_DIR / "manifest.json").read_text())["pilot_rows_sha256"]


def test_no_test_rows_and_unique_sample_ids():
    df = _rows()
    assert set(df["split"].unique()) <= {"train", "val"}
    assert "test" not in set(df["split"].str.lower())
    assert df["sample_id"].is_unique
    assert not df.duplicated(subset=["scenario_id", "step"]).any()


def test_label_delta_j_consistency():
    df = _rows()
    assert np.allclose(df["J_ESTF"] - df["J_WFS"], df["delta_J"])
    expected = np.where(df["delta_J"] > 0, "ESTF", np.where(df["delta_J"] < 0, "WFS", "TIE_OR_UNCERTAIN"))
    assert (df["oracle_label"].to_numpy() == expected).all()
    assert (df.loc[df["oracle_label"] == "TIE_OR_UNCERTAIN", "delta_J"] == 0.0).all()


def test_no_forbidden_model_features_and_feature_classification_clean():
    schema = _schema()
    feature_cols = schema["feature_columns"]
    forbidden = [
        "scenario_id",
        "canonical_scenario_id",
        "split",
        "seed",
        "analysis_fav",
        "J_",
        "delta_J",
        "oracle_label",
        "br_",
        "raw_completion",
    ]
    for col in feature_cols:
        assert col.startswith("feat_")
        assert not any(tok in col for tok in forbidden)

    fc = pd.read_csv(DATASET_DIR / "feature_classification.csv")
    mapped = dict(zip(fc["column"], fc["classification"]))
    for col in feature_cols:
        assert mapped[col] == "ONLINE_CAUSAL_MODEL_FEATURE"
    for col in ("scenario_id", "split", "analysis_fav", "analysis_seed"):
        assert mapped[col] in {"METADATA_ONLY", "EXPERIMENT_METADATA"}
    for col in ("J_ESTF", "J_WFS", "delta_J", "oracle_label"):
        assert mapped[col] == "LABEL_OR_FUTURE_OUTCOME"


def test_feature_dimensional_unit_validity_for_pilot_schema():
    schema = _schema()
    feature_cols = schema["feature_columns"]
    assert not any("deadline_slack_if_admitted_now" in c for c in feature_cols)
    assert "feat_laxity_own_diff_estf_minus_wfs" in feature_cols
    assert "feat_predicted_service_proxy_diff_estf_minus_wfs" in feature_cols
    # Ratio features are only between same-unit quantities.
    for col in schema["pairwise_difference_feature_columns"]:
        if col.endswith("_ratio_estf_over_wfs"):
            base = col.removeprefix("feat_").removesuffix("_ratio_estf_over_wfs")
            assert base in {"priority", "predicted_service_proxy", "queue_age", "laxity_own"}


def test_grouped_cv_has_no_row_or_group_leakage():
    df = _rows()
    binary = df[df["oracle_label"].isin(["ESTF", "WFS"])].reset_index(drop=True)
    groups = binary["group_key"]
    for train_idx, test_idx in GroupKFold(n_splits=5).split(binary, binary["oracle_label"], groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)
        assert set(binary.iloc[train_idx]["sample_id"]).isdisjoint(set(binary.iloc[test_idx]["sample_id"]))


def test_stored_branch_symmetry_and_snapshot_noninterference_invariants():
    """The repaired artifacts should retain the branch-order/non-interference
    invariants established by the counterfactual tests: each admitted
    contested request's own outcome is independent of which policy continues
    after that first admission in this no-preemption Family-A setting."""
    req = pd.read_csv(REPO_ROOT / "experiments/family_a_contested_request_value_diagnosis/contested_requests.csv")
    estf_only = req[req["contested_side"] == "estf_only"]
    wfs_only = req[req["contested_side"] == "wfs_only"]
    for suffix in ["completed", "completion_time", "slo_violated", "weighted_contribution"]:
        assert (
            estf_only[f"br_estf_estf_{suffix}"].fillna(-1).to_numpy()
            == estf_only[f"br_estf_wfs_{suffix}"].fillna(-1).to_numpy()
        ).all()
        assert (
            wfs_only[f"br_wfs_wfs_{suffix}"].fillna(-1).to_numpy()
            == wfs_only[f"br_wfs_estf_{suffix}"].fillna(-1).to_numpy()
        ).all()


def test_dataset_artifacts_have_expected_schema_files():
    expected = {
        "README.md",
        "pilot_rows.csv",
        "schema.json",
        "feature_classification.csv",
        "provenance.json",
        "quality_summary.json",
        "model_sanity_summary.json",
        "delta_j_regression_summary.json",
        "representation_check_summary.json",
        "manifest.json",
    }
    assert expected <= {p.name for p in DATASET_DIR.iterdir()}

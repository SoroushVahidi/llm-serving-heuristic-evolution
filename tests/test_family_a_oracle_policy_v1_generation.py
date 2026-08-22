"""Prelaunch integrity tests for scaled Family-A oracle dataset generation.

These tests validate schema, label arithmetic, sharding/resume mechanics, and
merge safety. They do not launch the long dataset job and do not train a model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/generate_family_a_oracle_policy_v1.py"
DATASET_DIR = REPO_ROOT / "datasets/family_a_oracle_policy_v1"
PILOT_SCHEMA = REPO_ROOT / "datasets/family_a_oracle_policy_pilot_v1/schema.json"


def _module():
    spec = importlib.util.spec_from_file_location("family_a_oracle_policy_v1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dry_rows() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "_dry_run/dry_run_rows.csv")


def test_feature_schema_is_pilot_backward_compatible():
    mod = _module()
    pilot_cols = json.loads(PILOT_SCHEMA.read_text())["feature_columns"]
    assert mod.feature_columns() == pilot_cols
    assert len(mod.feature_columns()) == 63


def test_manifest_is_trainval_only_and_has_configuration_groups():
    df = pd.read_csv(DATASET_DIR / "scenario_manifest.csv")
    assert len(df) == 704
    assert set(df["split"].unique()) <= {"train", "val"}
    assert "test" not in set(df["split"].str.lower())
    assert df["scenario_id"].is_unique
    assert df["configuration_group_id"].nunique() == 32
    grouped = df.groupby("configuration_group_id")["seed"].nunique()
    assert grouped.min() == grouped.max() == 22


def test_whole_branch_utility_and_label_arithmetic():
    mod = _module()
    completed = [
        SimpleNamespace(request=SimpleNamespace(priority=2.0, request_id=1), slo_violated=False),
        SimpleNamespace(request=SimpleNamespace(priority=5.0, request_id=2), slo_violated=True),
    ]
    assert mod.success_weight(completed) == 2.0
    estf = SimpleNamespace(success_weight_numerator=7.0)
    wfs = SimpleNamespace(success_weight_numerator=3.0)
    lab = mod.whole_branch_label(estf, wfs)
    assert lab["delta_J_whole"] == 4.0
    assert lab["oracle_label"] == "ESTF"


def test_dry_run_rows_have_consistent_primary_and_contested_labels():
    df = _dry_rows()
    assert len(df) >= 1
    expected_primary = df["delta_J_whole"].map(
        lambda x: "ESTF" if x > 0 else ("WFS" if x < 0 else "TIE_OR_UNCERTAIN")
    )
    expected_contested = df["delta_J_contested"].map(
        lambda x: "ESTF" if x > 0 else ("WFS" if x < 0 else "TIE_OR_UNCERTAIN")
    )
    assert (df["oracle_label"] == expected_primary).all()
    assert (df["oracle_label_contested"] == expected_contested).all()


def test_no_forbidden_model_features_or_invalid_unit_feature():
    schema = json.loads((DATASET_DIR / "schema.json").read_text())
    forbidden_tokens = ["scenario_id", "seed", "split", "configuration_group_id", "actual_output", "br_", "J_", "delta_J", "oracle_label"]
    for col in schema["feature_columns"]:
        assert col.startswith("feat_")
        assert not any(tok in col for tok in forbidden_tokens)
        assert "deadline_slack_if_admitted_now" not in col


def test_shard_assignment_is_disjoint_cover():
    mod = _module()
    manifest = pd.read_csv(DATASET_DIR / "scenario_manifest.csv")
    mod.verify_shard_disjointness(manifest, 4)
    assignments = mod.shard_assignments(manifest, 4)
    assert {k: len(v) for k, v in assignments.items()} == {0: 176, 1: 176, 2: 176, 3: 176}


def _copy_row(row: pd.Series, *, sample_suffix: str, fp_suffix: str) -> dict:
    out = row.to_dict()
    out["sample_id"] = f"{out['sample_id']}::{sample_suffix}"
    out["state_fingerprint"] = f"{out['state_fingerprint']}::{fp_suffix}"
    return out


def test_resume_marker_requires_matching_checksum(tmp_path):
    mod = _module()
    row = _copy_row(_dry_rows().iloc[0], sample_suffix="a", fp_suffix="a")
    rows_path = tmp_path / "shard.rows.csv"
    done_path = tmp_path / "shard.done.json"
    mod.write_rows_csv(rows_path, [row])
    mod.json_dump(done_path, {"rows_sha256": mod.sha256_file(rows_path)})
    assert mod.done_marker_valid(rows_path, done_path)
    mod.json_dump(done_path, {"rows_sha256": "bad"})
    assert not mod.done_marker_valid(rows_path, done_path)


def test_deterministic_merge_refuses_duplicate_sample_ids(tmp_path):
    mod = _module()
    out = tmp_path / "dataset"
    manifest = mod.build_scenario_manifest(output_dir=out, target_scenarios=4)
    mod.write_dataset_metadata(out, manifest, 2, ["unit-test"])
    shard_dir = out / "shards"
    row = _dry_rows().iloc[0].to_dict()
    for shard_id in (0, 1):
        rows_path = shard_dir / f"shard_{shard_id:03d}.rows.csv"
        done_path = shard_dir / f"shard_{shard_id:03d}.done.json"
        mod.write_rows_csv(rows_path, [row])
        mod.json_dump(done_path, {"rows_sha256": mod.sha256_file(rows_path)})
    args = argparse.Namespace(output_dir=str(out), target_scenarios=4, workers=2)
    with pytest.raises(RuntimeError, match="duplicate sample_id"):
        mod.merge_shards(args)


def test_deterministic_merge_succeeds_for_disjoint_rows(tmp_path):
    mod = _module()
    out = tmp_path / "dataset"
    manifest = mod.build_scenario_manifest(output_dir=out, target_scenarios=4)
    mod.write_dataset_metadata(out, manifest, 2, ["unit-test"])
    shard_dir = out / "shards"
    base = _dry_rows().iloc[0]
    for shard_id in (0, 1):
        row = _copy_row(base, sample_suffix=str(shard_id), fp_suffix=str(shard_id))
        rows_path = shard_dir / f"shard_{shard_id:03d}.rows.csv"
        done_path = shard_dir / f"shard_{shard_id:03d}.done.json"
        mod.write_rows_csv(rows_path, [row])
        mod.json_dump(done_path, {"rows_sha256": mod.sha256_file(rows_path)})
    args = argparse.Namespace(output_dir=str(out), target_scenarios=4, workers=2)
    mod.merge_shards(args)
    merged = pd.read_csv(out / "oracle_rows.csv")
    assert len(merged) == 2
    assert merged["sample_id"].is_unique
    assert merged["state_fingerprint"].is_unique

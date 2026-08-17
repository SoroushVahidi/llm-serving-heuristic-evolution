"""Focused validation tests for the Multi-Family Policy Separation Dataset
(MF-PSD) v1 builder.

See docs/audits/multi_family_policy_separation_dataset_v1_20260817.md and
src/llmserveopt/policy_separation/mf_psd.py. This is a DATA UNIFICATION
artifact -- these tests validate the builder's conservation, uniqueness,
determinism, and anti-leakage properties, not any downstream selector.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from llmserveopt.policy_separation.mf_psd import (
    CANONICAL_ANCHOR_POLICIES,
    FAMILY_A,
    FAMILY_B,
    FAMILY_C,
    FORBIDDEN_AUDIT_ONLY_FIELDS,
    LEARNABLE_FEATURE_ALLOWLIST,
    LONG_FORM_COLUMNS,
    MECHANISM_FAMILIES,
    PRIMARY_METRIC,
    SCENARIO_TABLE_COLUMNS,
    SCENARIO_TABLE_IDENTITY_COLUMNS,
    MFPSDValidationError,
    build_long_form_rows,
    build_mf_psd,
    build_scenario_table_rows,
    default_source_specs,
    group_key_for_scenario_id,
    validate_long_form,
    validate_scenario_table,
)


@pytest.fixture(scope="module")
def sources():
    specs = default_source_specs()
    for s in specs:
        assert s.per_policy_results_path.is_file(), f"missing frozen source: {s.per_policy_results_path}"
        if s.scenario_features_path is not None:
            assert s.scenario_features_path.is_file(), f"missing frozen source: {s.scenario_features_path}"
    return specs


@pytest.fixture(scope="module")
def long_rows(sources):
    return build_long_form_rows(sources)


@pytest.fixture(scope="module")
def scenario_rows(long_rows):
    return build_scenario_table_rows(long_rows)


# ---------------------------------------------------------------------------
# Source-row / source-scenario conservation
# ---------------------------------------------------------------------------


def test_exact_source_row_conservation(sources, long_rows):
    expected = 0
    for s in sources:
        with open(s.per_policy_results_path, newline="") as f:
            expected += sum(1 for _ in csv.DictReader(f))
    assert len(long_rows) == expected == 496


def test_exact_source_scenario_conservation(sources, scenario_rows):
    expected_total = 0
    for s in sources:
        with open(s.per_policy_results_path, newline="") as f:
            ids = {r["scenario_id"] for r in csv.DictReader(f)}
        expected_total += len(ids)
    assert len(scenario_rows) == expected_total == 176


def test_per_family_row_and_scenario_counts(long_rows, scenario_rows):
    expected_rows = {FAMILY_A: 288, FAMILY_B: 64, FAMILY_C: 144}
    expected_scenarios = {FAMILY_A: 72, FAMILY_B: 32, FAMILY_C: 72}
    actual_rows = {}
    actual_scenarios = {}
    for r in long_rows:
        actual_rows[r["mechanism_family"]] = actual_rows.get(r["mechanism_family"], 0) + 1
    for r in scenario_rows:
        actual_scenarios[r["mechanism_family"]] = actual_scenarios.get(r["mechanism_family"], 0) + 1
    assert actual_rows == expected_rows
    assert actual_scenarios == expected_scenarios


# ---------------------------------------------------------------------------
# Uniqueness / duplication
# ---------------------------------------------------------------------------


def test_no_duplicate_canonical_row_ids(long_rows):
    ids = [r["mf_psd_row_id"] for r in long_rows]
    assert len(ids) == len(set(ids))


def test_no_duplicate_scenario_policy_cells(long_rows):
    cells = [(r["canonical_scenario_id"], r["canonical_policy_id"]) for r in long_rows]
    assert len(cells) == len(set(cells))


def test_no_duplicate_canonical_scenario_ids(scenario_rows):
    ids = [r["canonical_scenario_id"] for r in scenario_rows]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Metric validity
# ---------------------------------------------------------------------------


def test_finite_anwg_for_all_success_rows(long_rows):
    for r in long_rows:
        if r["status"] == "success":
            assert math.isfinite(r["primary_utility_anwg"]), r["mf_psd_row_id"]


def test_all_rows_are_success_in_v1_sources(long_rows):
    # Known, verified property of the three frozen v2 source runs (0
    # failures in every final_summary.json). If a future source has
    # failures this test documents the expectation changing, not silently
    # passing.
    statuses = {r["status"] for r in long_rows}
    assert statuses == {"success"}


def test_canonical_primary_metric_identity():
    assert PRIMARY_METRIC == "arrival_normalized_weighted_goodput"


# ---------------------------------------------------------------------------
# Traceability / provenance
# ---------------------------------------------------------------------------


def test_source_family_traceability(long_rows, sources):
    known_run_ids = {s.source_run_id for s in sources}
    known_families = set(MECHANISM_FAMILIES)
    for r in long_rows:
        assert r["source_run_id"] in known_run_ids
        assert r["mechanism_family"] in known_families
        assert r["source_result_path"]  # non-empty
        assert isinstance(r["source_row_index"], int)


def test_row_id_matches_scenario_and_policy(long_rows):
    for r in long_rows:
        assert r["mf_psd_row_id"] == f"{r['canonical_scenario_id']}::{r['source_policy_name']}"
        assert r["canonical_scenario_id"] == f"{r['mechanism_family']}::{r['source_scenario_id']}"


# ---------------------------------------------------------------------------
# Determinism / stable ordering
# ---------------------------------------------------------------------------


def test_deterministic_rebuild(sources):
    rows_a = build_long_form_rows(sources)
    rows_b = build_long_form_rows(sources)
    assert rows_a == rows_b

    scen_a = build_scenario_table_rows(rows_a)
    scen_b = build_scenario_table_rows(rows_b)
    assert scen_a == scen_b


def test_stable_ordering_is_sorted(long_rows, scenario_rows):
    keys = [(r["mechanism_family"], r["canonical_scenario_id"], r["canonical_policy_id"]) for r in long_rows]
    assert keys == sorted(keys)

    scen_keys = [(r["mechanism_family"], r["canonical_scenario_id"]) for r in scenario_rows]
    assert scen_keys == sorted(scen_keys)


def test_build_mf_psd_full_pipeline_deterministic(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build_mf_psd(out1)
    build_mf_psd(out2)
    assert (out1 / "mf_psd_long_v1.csv").read_text() == (out2 / "mf_psd_long_v1.csv").read_text()
    assert (out1 / "mf_psd_scenarios_v1.csv").read_text() == (out2 / "mf_psd_scenarios_v1.csv").read_text()
    assert (out1 / "mf_psd_schema_v1.json").read_text() == (out2 / "mf_psd_schema_v1.json").read_text()


# ---------------------------------------------------------------------------
# Anti-leakage: learnable allowlist vs forbidden/audit-only fields
# ---------------------------------------------------------------------------


def test_learnable_allowlist_disjoint_from_forbidden_fields():
    assert set(LEARNABLE_FEATURE_ALLOWLIST).isdisjoint(set(FORBIDDEN_AUDIT_ONLY_FIELDS))


def test_forbidden_fields_cover_all_long_form_identity_columns():
    non_metric_long_form_cols = [
        c for c in LONG_FORM_COLUMNS if c not in ("mf_psd_row_id",)  # row id itself listed separately
    ]
    # Every long-form column must be either a forbidden/audit field or
    # (there are none in the long-form table) a learnable feature -- the
    # long-form table carries zero learnable scenario-context columns by
    # design (those live only in the scenario table).
    for c in LONG_FORM_COLUMNS:
        assert c in FORBIDDEN_AUDIT_ONLY_FIELDS, f"long-form column {c!r} is neither forbidden nor documented"
        assert c not in LEARNABLE_FEATURE_ALLOWLIST


def test_mechanism_family_excluded_from_learnable_allowlist():
    assert "mechanism_family" not in LEARNABLE_FEATURE_ALLOWLIST
    assert "mechanism_family" in FORBIDDEN_AUDIT_ONLY_FIELDS


def test_scenario_and_seed_and_split_excluded_from_learnable_allowlist():
    for forbidden in (
        "canonical_scenario_id",
        "source_scenario_id",
        "seed",
        "group_key",
        "source_split_raw",
        "status",
        "primary_utility_anwg",
        "canonical_policy_id",
        "is_canonical_anchor",
    ):
        assert forbidden not in LEARNABLE_FEATURE_ALLOWLIST
        assert forbidden in FORBIDDEN_AUDIT_ONLY_FIELDS


def test_learnable_allowlist_only_contains_family_prefixed_columns():
    for col in LEARNABLE_FEATURE_ALLOWLIST:
        assert col.startswith(("feat_A__", "feat_B__", "feat_C__")), col


def test_scenario_table_columns_are_identity_plus_allowlist():
    assert set(SCENARIO_TABLE_COLUMNS) == set(SCENARIO_TABLE_IDENTITY_COLUMNS) | set(
        LEARNABLE_FEATURE_ALLOWLIST
    )


# ---------------------------------------------------------------------------
# Scenario-feature invariance across policy rows
# ---------------------------------------------------------------------------


def test_scenario_features_invariant_across_policy_rows(long_rows):
    by_scenario = {}
    for r in long_rows:
        by_scenario.setdefault(r["canonical_scenario_id"], []).append(r)
    for canonical_scenario_id, rows in by_scenario.items():
        feats = {r["source_scenario_features_json"] for r in rows}
        assert len(feats) == 1, f"{canonical_scenario_id} has varying scenario features across policies"


def test_scenario_table_feature_missingness_is_family_scoped(scenario_rows):
    for r in scenario_rows:
        family = r["mechanism_family"]
        for col in LEARNABLE_FEATURE_ALLOWLIST:
            owning_family = {"feat_A__": FAMILY_A, "feat_B__": FAMILY_B, "feat_C__": FAMILY_C}[col[:8]]
            if owning_family != family:
                assert r[col] == "", f"{col} should be missing for {family} scenario {r['canonical_scenario_id']}"
            else:
                # Not all owning-family values are guaranteed non-empty in
                # principle (a source could legitimately record an empty
                # string), but for these three frozen v2 sources every
                # allowlisted column is populated for its own family.
                assert r[col] != "", f"{col} unexpectedly empty for its own family {family}"


# ---------------------------------------------------------------------------
# Provenance / checksums populated; source artifacts never mutated
# ---------------------------------------------------------------------------


def test_provenance_manifest_populated(tmp_path):
    manifest = build_mf_psd(tmp_path)
    provenance_path = tmp_path / "mf_psd_provenance_v1.json"
    assert provenance_path.is_file()
    provenance = json.loads(provenance_path.read_text())
    assert provenance["builder_version"]
    assert provenance["build_git_head_sha"] and provenance["build_git_head_sha"] != "unknown"
    assert len(provenance["sources"]) == 3
    for s in provenance["sources"]:
        assert s["per_policy_results_sha256"]
        assert len(s["per_policy_results_sha256"]) == 64
        assert s["launch_git_sha"]
        assert s["audit_doc"]
    for fname, checksum in provenance["output_files"].items():
        assert len(checksum) == 64
    assert manifest["n_long_form_rows"] == 496
    assert manifest["n_scenarios"] == 176


def test_source_artifacts_not_mutated_by_build(sources, tmp_path):
    before = {}
    for s in sources:
        before[s.per_policy_results_path] = s.per_policy_results_path.read_bytes()
        if s.scenario_features_path:
            before[s.scenario_features_path] = s.scenario_features_path.read_bytes()

    build_mf_psd(tmp_path)  # build into an isolated tmp dir

    for path, content in before.items():
        assert path.read_bytes() == content, f"source artifact mutated: {path}"


# ---------------------------------------------------------------------------
# Six-policy / canonical-anchor coverage (sparse, honestly documented)
# ---------------------------------------------------------------------------


def test_six_canonical_anchors_each_appear_in_exactly_one_family(long_rows):
    assert len(CANONICAL_ANCHOR_POLICIES) == 6
    family_by_policy = {}
    for r in long_rows:
        if r["is_canonical_anchor"]:
            family_by_policy.setdefault(r["canonical_policy_id"], set()).add(r["mechanism_family"])
    assert set(family_by_policy.keys()) == set(CANONICAL_ANCHOR_POLICIES)
    for policy, families in family_by_policy.items():
        assert len(families) == 1, f"{policy} unexpectedly evaluated in >1 family: {families}"


def test_six_policy_matrix_is_sparse_not_dense(long_rows):
    """No scenario has more than 4 of the 6 canonical anchors evaluated on
    it (in fact at most 2, since each family only ran its own 2 anchors)
    -- the full 6-policy x every-scenario matrix does NOT exist in the
    frozen source evidence. This test documents that sparsity rather than
    silently assuming density; Step 2 of the roadmap must run the missing
    cross-family evaluations, not assume they already exist."""
    anchors_per_scenario = {}
    for r in long_rows:
        if r["is_canonical_anchor"]:
            anchors_per_scenario.setdefault(r["canonical_scenario_id"], set()).add(r["canonical_policy_id"])
    max_anchors = max(len(v) for v in anchors_per_scenario.values())
    assert max_anchors == 2
    assert len(anchors_per_scenario) == 176  # every scenario has exactly its own family's 2 anchors evaluated


def test_extra_non_anchor_policies_are_family_a_only(long_rows):
    extra = {r["canonical_policy_id"] for r in long_rows if not r["is_canonical_anchor"]}
    assert extra == {"fifo", "aging_priority"}
    families = {r["mechanism_family"] for r in long_rows if not r["is_canonical_anchor"]}
    assert families == {FAMILY_A}


# ---------------------------------------------------------------------------
# Group-key / seed-grouping sanity (for later leakage-safe splitting)
# ---------------------------------------------------------------------------


def test_group_key_strips_seed_suffix():
    assert (
        group_key_for_scenario_id(FAMILY_A, "fs2.util1.1000.skew1.0000.favshort.noise0.00.s20260816")
        == f"{FAMILY_A}::fs2.util1.1000.skew1.0000.favshort.noise0.00"
    )


def test_group_keys_have_multiple_scenarios_each(scenario_rows):
    # Every family's scenarios are generated at >=2 seeds per underlying
    # config, so group-aware splitting has real within-group multiplicity
    # to exploit (a leakage risk if splitting were done at the scenario
    # level instead of the group level).
    from collections import defaultdict

    per_family_groups = defaultdict(set)
    for r in scenario_rows:
        per_family_groups[r["mechanism_family"]].add(r["group_key"])
    assert len(per_family_groups[FAMILY_A]) == 36  # 72 scenarios / 2 seeds
    assert len(per_family_groups[FAMILY_B]) == 8  # 32 scenarios / 4 seeds
    assert len(per_family_groups[FAMILY_C]) == 12  # 72 scenarios / 6 seeds


# ---------------------------------------------------------------------------
# Explicit failure handling (builder-level, using a validation error path)
# ---------------------------------------------------------------------------


def test_validate_long_form_rejects_duplicate_row_ids(long_rows, sources):
    corrupted = list(long_rows) + [dict(long_rows[0])]
    with pytest.raises(MFPSDValidationError):
        validate_long_form(corrupted, sources)


def test_validate_scenario_table_rejects_duplicate_scenario_ids(long_rows, scenario_rows, sources):
    corrupted = list(scenario_rows) + [dict(scenario_rows[0])]
    with pytest.raises(MFPSDValidationError):
        validate_scenario_table(corrupted, long_rows, sources)

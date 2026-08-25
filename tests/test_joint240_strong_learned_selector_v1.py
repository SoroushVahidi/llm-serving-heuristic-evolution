"""Tests for joint-240 strong learned selector v1."""
from __future__ import annotations

import numpy as np

from llmserveopt.analysis.joint240_strong_learned_selector_v1 import (
    DESIGN_DOC,
    ET_GRID,
    FEATURE_ALLOWLIST,
    HGB_GRID,
    P6,
    build_feature_matrix,
    classify_recovery,
    expand_utility_rows,
    load_parent_folds,
    load_parent_oof,
    method_summary,
    select_policies_from_preds,
    sha256_file,
)


def test_design_doc_exists_and_hashes():
    assert DESIGN_DOC.exists()
    h = sha256_file(DESIGN_DOC)
    assert len(h) == 64


def test_parent_artifacts_intact():
    folds = load_parent_folds()
    oof = load_parent_oof()
    assert len(folds) == 240
    assert len(oof) == 240
    assert abs(float(oof["a_scen_anwg"].mean()) - 0.3059465519866274) < 1e-12
    assert abs(float(oof["a_live_anwg"].mean()) - 0.2839667616302265) < 1e-12
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
    assert folds.groupby("scenario_id")["fold"].nunique().max() == 1


def test_feature_matrix_allowlist_only():
    data = build_feature_matrix()
    assert len(data) == 240
    for c in FEATURE_ALLOWLIST:
        assert c in data.columns
    for p in P6:
        assert p in data.columns
    assert "actual_output_tokens" not in FEATURE_ALLOWLIST


def test_expand_utility_rows_shape():
    data = build_feature_matrix().head(3)
    X, y, sids, pids = expand_utility_rows(data, data["scenario_id"].tolist())
    assert X.shape == (3 * 6, len(FEATURE_ALLOWLIST) + 6)
    assert len(y) == 18
    assert set(pids) == set(P6)


def test_select_policies_tiebreak_deterministic():
    pred = {"s0": {p: 0.1 for p in P6}}
    chosen = select_policies_from_preds(pred)
    assert chosen["s0"] == P6[0]


def test_preregistered_grids_modest():
    assert len(HGB_GRID) == 8
    assert len(ET_GRID) == 4


def test_classify_recovery_labels():
    base = {
        "realized_gain": -0.01,
        "bootstrap_gain_vs_sbs": {"ci95_low": -0.02, "ci95_high": 0.0},
        "gap_closure": -0.5,
    }
    assert classify_recovery(base) == "NO_RECOVERY"
    base["realized_gain"] = 0.01
    base["bootstrap_gain_vs_sbs"]["ci95_low"] = -0.001
    base["gap_closure"] = 0.2
    assert classify_recovery(base) == "PARTIAL_RECOVERY"
    base["bootstrap_gain_vs_sbs"]["ci95_low"] = 0.001
    base["gap_closure"] = 0.55
    assert classify_recovery(base) == "STRONG_RECOVERY"


def test_method_summary_catastrophic_def():
    sbs = np.asarray([0.5, 0.5])
    vbs = np.asarray([0.6, 0.6])
    a = np.asarray([0.49, 0.40])  # second is catastrophic (< 0.5 - 0.01)
    s = method_summary(a, sbs, vbs, n_boot=200)
    assert s["catastrophic_lt_sbs_minus_eps"] == 1

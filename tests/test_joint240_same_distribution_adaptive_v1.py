"""Tests for joint-240 same-distribution adaptive exploitability v1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import (
    FEATURE_ALLOWLIST,
    P6,
    PolicyDwellFSM,
    freeze_oof_folds,
    generator_feature_table,
    load_utility_matrix,
    rebuild_all_scenarios,
    summarize_oof,
)


def test_utility_matrix_p6_and_sbs():
    df = load_utility_matrix()
    assert len(df) == 240
    for p in P6:
        assert p in df.columns
    means = {p: float(df[p].mean()) for p in P6}
    assert max(means, key=means.get) == "kv_constrained_online"
    assert abs(means["kv_constrained_online"] - 0.3140716694729327) < 1e-12
    assert abs(float(df["vbs_anwg"].mean()) - 0.33310550374603504) < 1e-12


def test_generator_features_allowlist_only():
    scenarios = rebuild_all_scenarios()[:3]
    feats = generator_feature_table(scenarios)
    assert set(feats.columns) == {"scenario_id"} | set(FEATURE_ALLOWLIST)
    assert "actual_output_tokens" not in feats.columns
    assert len(feats) == 3


def test_oof_folds_partition():
    ids = [f"s{i}" for i in range(20)]
    strata = [i % 4 for i in range(20)]
    folds = freeze_oof_folds(ids, strata, n_folds=5, seed=20260825)
    assert len(folds) == 20
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
    assert folds["scenario_id"].nunique() == 20


def test_policy_dwell_fsm():
    fsm = PolicyDwellFSM(["a", "b"], dwell_steps=3)
    seq = []
    for raw in ["a", "b", "b", "b", "b"]:
        seq.append(fsm.step(raw))
    assert seq == ["a", "a", "a", "a", "b"]
    assert fsm.transitions == 1


def test_summarize_uses_fixed_sbs_not_rowmax():
    rows = pd.DataFrame(
        {
            "full_prefill": [0.1, 0.1],
            "chunked_prefill_small": [0.2, 0.2],
            "estimated_service_time_first": [0.3, 0.3],
            "weighted_fair_share": [0.4, 0.4],
            "least_laxity_first": [0.5, 0.5],
            "kv_constrained_online": [0.35, 0.35],
            "vbs_anwg": [0.5, 0.5],
            "a_scen_anwg": [0.45, 0.40],
        }
    )
    s = summarize_oof(rows, "a_scen_anwg")
    assert s["SBS_policy"] == "least_laxity_first"
    assert abs(s["R_SBS"] - 0.5) < 1e-12
    assert abs(s["headroom"] - 0.0) < 1e-12

"""Tests for horizon_stability_v1 reconstruction fix and fail-closed behavior.

Tests cover:
- canonical constructor resolution (case_fairness_vs_size_v2 from templates)
- regression for the missing constructor bug (fac.case_fairness_vs_size_v2)
- reconstruction from D0 metadata (configuration_group_id parsing)
- fail-closed behavior for zero valid evaluations
- no scientific verdict from empty results
- deterministic sample reuse
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ======================================================================
# 1. Canonical constructor resolution
# ======================================================================

def test_case_fairness_vs_size_v2_is_in_templates_module():
    """The canonical constructor must be importable from templates_fairness_starvation_v2."""
    from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (
        case_fairness_vs_size_v2,
    )
    assert callable(case_fairness_vs_size_v2)


def test_case_fairness_vs_size_v2_not_in_family_a_observability_module():
    """Regression: the old bug called fac.case_fairness_vs_size_v2 which does not exist."""
    from llmserveopt.analysis import family_a_observability_continuation_v1 as fac
    assert not hasattr(fac, "case_fairness_vs_size_v2")


def test_horizon_stability_v1_imports_canonical_constructor():
    """horizon_stability_v1 must import from templates, not from fac."""
    from scripts.horizon_stability_v1 import case_fairness_vs_size_v2
    from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (
        case_fairness_vs_size_v2 as canonical,
    )
    assert case_fairness_vs_size_v2 is canonical


def test_d0_generator_uses_same_canonical_constructor():
    """D0 generator and horizon validator must use the same constructor source."""
    import scripts.generate_family_a_oracle_policy_v1 as gen
    from scripts.horizon_stability_v1 import case_fairness_vs_size_v2
    assert gen.case_fairness_vs_size_v2 is case_fairness_vs_size_v2


# ======================================================================
# 2. Reconstruction from D0 metadata
# ======================================================================

def test_reconstruct_scenario_from_d0_row_parses_config_group_id():
    """configuration_group_id parsing must extract n_total_jobs and max_active_sequences."""
    from scripts.horizon_stability_v1 import reconstruct_scenario_from_d0_row

    row = pd.Series({
        "configuration_group_id": "util1.1000.skew10.0000.favlong.noise0.00.n120.maxseq1",
        "analysis_utilization": 1.1,
        "analysis_skew": 10.0,
        "analysis_fav": "long",
        "analysis_noise": 0.0,
        "analysis_seed": 20260822,
    })
    scenario = reconstruct_scenario_from_d0_row(row)
    assert scenario is not None
    assert hasattr(scenario, "requests")
    assert hasattr(scenario, "gpu_configs")
    assert hasattr(scenario, "service_model_kwargs")


def test_reconstruct_scenario_default_n_total_jobs():
    """When n-total-jobs regex doesn't match, default to 120."""
    import re
    from scripts.horizon_stability_v1 import reconstruct_scenario_from_d0_row

    row = pd.Series({
        "configuration_group_id": "util1.1000.skew10.0000.favlong.noise0.00",
        "analysis_utilization": 1.1,
        "analysis_skew": 10.0,
        "analysis_fav": "long",
        "analysis_noise": 0.0,
        "analysis_seed": 20260822,
    })
    # This should still work, using default n_total_jobs=120, max_active_sequences=1
    scenario = reconstruct_scenario_from_d0_row(row)
    assert scenario is not None


# ======================================================================
# 3. Fail-closed behavior
# ======================================================================

def test_decide_d0_horizon_returns_reconstruction_failure_on_zero_valid():
    """When total_evaluable is 0, no scientific verdict may be emitted."""
    from scripts.horizon_stability_v1 import decide_d0_horizon
    result = decide_d0_horizon(results=[], certified_count=0, total_evaluable=0)
    assert result == "HORIZON_VALIDATION_RECONSTRUCTION_FAILURE"


def test_decide_d0_horizon_returns_reconstruction_failure_on_below_min_samples():
    """When fewer than MIN_VALID_SAMPLES, no scientific verdict may be emitted."""
    from scripts.horizon_stability_v1 import decide_d0_horizon, HorizonResult
    fake_results = [HorizonResult(
        row_id="fake", horizon="H1500",
        j_estf=1.0, j_wfs=0.0, delta_j=1.0, oracle_label="ESTF",
        steps_run=1500, ran_to_natural_completion=False,
        estf_completed=1, wfs_completed=0,
        estf_slo_violations=0, wfs_slo_violations=0,
        residual_cert={}, reconstruction_match=True, error=None,
    )]
    result = decide_d0_horizon(results=fake_results, certified_count=1, total_evaluable=1)
    assert result == "HORIZON_VALIDATION_RECONSTRUCTION_FAILURE"


def test_no_d0_horizon_verdict_from_empty_results():
    """None of the four D0_HORIZON_* verdicts may appear with zero valid evaluations."""
    from scripts.horizon_stability_v1 import decide_d0_horizon
    forbidden_verdicts = {
        "D0_HORIZON_VALIDATED",
        "D0_HORIZON_VALID_WITH_CERTIFIED_FILTER",
        "D0_HORIZON_REQUIRES_RELABELING",
        "D0_HORIZON_INVALID",
    }
    result = decide_d0_horizon(results=[], certified_count=0, total_evaluable=0)
    assert result not in forbidden_verdicts


def test_margin_bin_label_accepts_scalar_values():
    """Regression: summary generation must not call pd.cut on a scalar."""
    from scripts.horizon_stability_v1 import margin_bin_label

    assert margin_bin_label(0.0) == "exact_zero"
    assert margin_bin_label(1e-9) == "0_to_1e-6"
    assert margin_bin_label(0.05) == "1e-6_to_0.1"
    assert margin_bin_label(0.5) == "0.1_to_1.0"
    assert margin_bin_label(5.0) == "1.0_to_10.0"
    assert margin_bin_label(11.0) == "gt_10.0"


# ======================================================================
# 4. Deterministic sample reuse
# ======================================================================

def test_stratified_sample_d0_is_deterministic():
    """The same D0 input must produce the same 128-row sample every time."""
    from scripts.horizon_stability_v1 import stratified_sample_d0

    # Create a small synthetic D0-like DataFrame
    rows = []
    for i in range(50):
        rows.append({
            "scenario_id": f"SCEN_{i:03d}",
            "step": 100 + i * 10,
            "configuration_group_id": f"util1.1000.skew5.0000.favlong.noise0.00.n120.maxseq1",
            "oracle_label": "ESTF" if i % 3 == 0 else ("WFS" if i % 3 == 1 else "TIE_OR_UNCERTAIN"),
            "delta_J_whole": float(i % 3 - 1),
            "J_ESTF_whole": float(i),
            "J_WFS_whole": float(i - (i % 3 - 1)),
        })
    df = pd.DataFrame(rows)

    sample1 = stratified_sample_d0(df, sample_size=20)
    sample2 = stratified_sample_d0(df, sample_size=20)

    assert len(sample1) == len(sample2)
    assert list(sample1["scenario_id"]) == list(sample2["scenario_id"])
    assert list(sample1["step"]) == list(sample2["step"])


# ======================================================================
# 5. No TEST leakage
# ======================================================================

def test_d0_dataset_has_no_test_split():
    """D0 oracle_rows.csv must not contain any TEST split rows."""
    d0_path = REPO_ROOT / "datasets/family_a_oracle_policy_v1/oracle_rows.csv"
    if not d0_path.exists():
        pytest.skip("D0 dataset not present")
    df = pd.read_csv(d0_path)
    assert "TEST" not in df["split"].values, "TEST split leakage in D0 dataset"


# ======================================================================
# 6. Old failed run preserved
# ======================================================================

def test_old_failed_run_directory_exists():
    """The original failed run directory must be preserved (not overwritten)."""
    old_dir = REPO_ROOT / "experiments/family_a_horizon_stability_v1"
    assert old_dir.exists(), "Old failed run directory must be preserved"
    assert (old_dir / "summary.json").exists(), "Old summary.json must be preserved"
    assert (old_dir / "horizon_results.csv").exists(), "Old horizon_results.csv must be preserved"


def test_old_failed_run_has_zero_valid_evaluations():
    """The old failed run must show 0 valid evaluations (confirming it was a reconstruction failure)."""
    old_summary = REPO_ROOT / "experiments/family_a_horizon_stability_v1/summary.json"
    if not old_summary.exists():
        pytest.skip("Old summary not present")
    with open(old_summary) as f:
        summary = json.load(f)
    assert summary["n_h1500_valid"] == 0
    assert summary["n_h3000_valid"] == 0
    assert summary["n_hnatural_valid"] == 0

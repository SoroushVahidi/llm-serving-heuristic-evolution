"""Focused tests for the Family-B-Balanced Replication v1 preregistration
completion and readiness tooling (docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md).

Covers: TEST-set metadata-only selection, held-out group identity, no-
TEST-fitting integrity, frozen Stage-1/Stage-2 contracts, no-majority-vote
guard, dwell/fallback constancy, provenance completeness, deterministic
replay, and frozen-artifact immutability.

Does NOT run the scientific evaluation of the replication set -- these
tests exercise only smoke/synthetic/train-split paths and the pure
selection logic, consistent with the task that authored this module (no
held-out Family-B TEST outcome is read anywhere in this file).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation.hierarchical_regime_router_v1 import (  # noqa: E402
    DWELL_MINIMUM_STEPS,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    STAGE1_INPUT_COLUMNS,
    STAGE2_CANDIDATES,
    build_splits,
)
from llmserveopt.policy_separation.family_b_balanced_replication_v1 import (  # noqa: E402
    FAMILY_A,
    FAMILY_B,
    FAMILY_C,
    FAMILY_C_PRIMARY_HELD_OUT_SEED,
    select_balanced_replication_set,
    select_family_a_replication,
    select_family_b_replication,
    select_family_c_replication,
    verify_no_train_leakage,
)

MF_PSD_SCENARIOS = REPO_ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
FROZEN_SELECTION = REPO_ROOT / "experiments/family_b_balanced_replication_v1/frozen_scenario_selection_v1.json"
RUNNER_SCRIPT = REPO_ROOT / "scripts/run_family_b_balanced_replication_v1.py"
DESIGN_DOC = REPO_ROOT / "docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md"
GATES_JSON = REPO_ROOT / "configs/hierarchical_regime_router_v1_gates.json"
PARENT_DESIGN_DOC = REPO_ROOT / "docs/design/HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md"


@pytest.fixture(scope="module")
def scen_with_split() -> pd.DataFrame:
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    split_map = build_splits(scen)
    scen["split"] = scen["canonical_scenario_id"].map(split_map)
    return scen


# ---------------------------------------------------------------------------
# 1. Selection metadata-only, deterministic, correctly balanced
# ---------------------------------------------------------------------------

def test_selection_is_balanced_12_12_12(scen_with_split):
    rep = select_balanced_replication_set(scen_with_split)
    counts = rep["mechanism_family"].value_counts().to_dict()
    assert counts == {FAMILY_A: 12, FAMILY_B: 12, FAMILY_C: 12}


def test_selection_is_deterministic_across_repeated_calls(scen_with_split):
    a = select_balanced_replication_set(scen_with_split)
    b = select_balanced_replication_set(scen_with_split)
    assert list(a["canonical_scenario_id"]) == list(b["canonical_scenario_id"])


def test_selection_has_no_duplicate_scenario_ids(scen_with_split):
    rep = select_balanced_replication_set(scen_with_split)
    assert rep["canonical_scenario_id"].is_unique


# ---------------------------------------------------------------------------
# 2. TEST-set / held-out disjointness ("no TEST fitting")
# ---------------------------------------------------------------------------

def test_replication_set_has_zero_scenario_row_overlap_with_train(scen_with_split):
    rep = select_balanced_replication_set(scen_with_split)
    verify_no_train_leakage(scen_with_split, rep)  # must not raise


def test_family_b_replication_pool_is_entirely_val_never_train_or_test(scen_with_split):
    sel_b = select_family_b_replication(scen_with_split)
    assert set(sel_b["split"]) == {"val"}


def test_family_c_replication_uses_only_primary_held_out_seed(scen_with_split):
    sel_c = select_family_c_replication(scen_with_split)
    assert set(sel_c["seed"].unique()) == {FAMILY_C_PRIMARY_HELD_OUT_SEED}
    assert set(sel_c["split"]) == {"test"}


def test_family_a_replication_prefers_val_over_test(scen_with_split):
    sel_a = select_family_a_replication(scen_with_split)
    val_count = int((sel_a["split"] == "val").sum())
    test_count = int((sel_a["split"] == "test").sum())
    assert val_count == 10
    assert test_count == 2


# ---------------------------------------------------------------------------
# 3. Held-out Family-B group identity matches the frozen manifest
# ---------------------------------------------------------------------------

def test_frozen_manifest_exists_and_matches_live_selection(scen_with_split):
    if not FROZEN_SELECTION.exists():
        pytest.skip("frozen_scenario_selection_v1.json not materialized in this checkout")
    frozen = json.loads(FROZEN_SELECTION.read_text())
    rep = select_balanced_replication_set(scen_with_split)
    for fam in (FAMILY_A, FAMILY_B, FAMILY_C):
        actual = sorted(rep[rep["mechanism_family"] == fam]["canonical_scenario_id"])
        assert actual == frozen["scenario_ids_by_family"][fam]


def test_frozen_manifest_family_c_seed_matches_gates_config_held_out_seed():
    if not FROZEN_SELECTION.exists():
        pytest.skip("frozen_scenario_selection_v1.json not materialized in this checkout")
    gates = json.loads(GATES_JSON.read_text())
    held_out_seeds = {int(s) for s in gates["splits"]["family_c_held_out"]["seeds"]}
    assert FAMILY_C_PRIMARY_HELD_OUT_SEED in held_out_seeds


# ---------------------------------------------------------------------------
# 4. Frozen Stage-1 feature set / Stage-2 native pair / foreign-policy exclusion
# ---------------------------------------------------------------------------

def test_stage1_input_columns_unchanged():
    assert STAGE1_INPUT_COLUMNS == (
        "contention_score_v2", "priority_skew", "kv_pressure", "queue_length",
    )


def test_stage2_family_b_native_pair_unchanged():
    assert STAGE2_CANDIDATES[REGIME_B] == ("full_prefill", "chunked_prefill_small")


def test_dwell_minimum_unchanged():
    assert DWELL_MINIMUM_STEPS == 20


# ---------------------------------------------------------------------------
# 5. No majority-vote path in the runner (AST guard)
# ---------------------------------------------------------------------------

def test_runner_never_imports_majority_vote_evaluation_surface():
    tree = ast.parse(RUNNER_SCRIPT.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    forbidden = {"baseline_d_anwg", "apply_dwell_and_fallback", "regime_label_from_activity"}
    # baseline_d_anwg / apply_dwell_and_fallback / regime_label_from_activity are the
    # majority-vote-approximation surface used by the OLD evaluation scripts;
    # this runner must use the live harness exclusively.
    hit = imported_names & forbidden
    assert not hit, f"runner must not import majority-vote surface, found: {hit}"


def test_runner_imports_the_live_harness_run_function():
    tree = ast.parse(RUNNER_SCRIPT.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    assert "run_live_scenario" in imported_names


# ---------------------------------------------------------------------------
# 6. Authorization guard on the scientific launch path
# ---------------------------------------------------------------------------

def test_replication_source_without_authorization_flag_is_refused():
    result = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--source", "replication"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 3
    assert "REFUSED" in result.stderr
    assert "--i-am-authorized" in result.stderr


def test_runner_source_argument_has_no_default_value():
    """Regression guard: --source must stay `required=True` with no
    default, so a bare invocation cannot silently fall through to the
    scientific replication path."""
    tree = ast.parse(RUNNER_SCRIPT.read_text())
    add_argument_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    source_calls = [
        c for c in add_argument_calls
        if any(isinstance(a, ast.Constant) and a.value == "--source" for a in c.args)
    ]
    assert len(source_calls) == 1
    kwarg_names = {kw.arg for kw in source_calls[0].keywords}
    assert "default" not in kwarg_names
    required_kw = next(kw for kw in source_calls[0].keywords if kw.arg == "required")
    assert isinstance(required_kw.value, ast.Constant) and required_kw.value.value is True


# ---------------------------------------------------------------------------
# 7. Smoke execution: synthetic microcase, no NaN/Inf, dwell/fallback correct,
#    provenance complete, deterministic replay
# ---------------------------------------------------------------------------

def test_smoke_synthetic_run_end_to_end(tmp_path):
    result = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--source", "smoke_synthetic"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr

    out_path = REPO_ROOT / "experiments/family_b_balanced_replication_v1/run_smoke_synthetic_results.json"
    assert out_path.exists()
    report = json.loads(out_path.read_text())

    assert report["run_tag"] == "smoke_synthetic"
    assert report["n_scenarios"] >= 1
    for row in report["per_scenario_results"]:
        assert row["anwg_live"] == row["anwg_live"]  # not NaN
        assert row["anwg_live"] not in (float("inf"), float("-inf"))
        assert row["dwell_violations"] == 0

    prov = report["provenance"]
    for key in (
        "git_head_sha", "git_tree_dirty", "design_doc_sha256", "gates_json_sha256",
        "mf_psd_scenarios_sha256", "telemetry_csv_sha256", "reused_model_hashes",
    ):
        assert key in prov, f"missing provenance field: {key}"
    assert "stage1_model_hash" in prov["reused_model_hashes"]
    assert set(prov["reused_model_hashes"]["stage2_model_hashes"]) == {REGIME_A, REGIME_B, REGIME_C}

    traj_path = REPO_ROOT / "experiments/family_b_balanced_replication_v1/run_smoke_synthetic_trajectories.csv"
    assert traj_path.exists()
    traj = pd.read_csv(traj_path)
    assert (traj["effective_regime"] == REGIME_B).any(), "Family-B regime must activate in the smoke run"
    assert traj["selected_policy"].notna().all()


def test_smoke_run_is_deterministic_bit_identical_anwg(tmp_path):
    r1 = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--source", "smoke_synthetic"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert r1.returncode == 0, r1.stderr
    out_path = REPO_ROOT / "experiments/family_b_balanced_replication_v1/run_smoke_synthetic_results.json"
    first = json.loads(out_path.read_text())["per_scenario_results"]

    r2 = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), "--source", "smoke_synthetic"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert r2.returncode == 0, r2.stderr
    second = json.loads(out_path.read_text())["per_scenario_results"]

    first_anwg = [r["anwg_live"] for r in sorted(first, key=lambda r: r["canonical_scenario_id"])]
    second_anwg = [r["anwg_live"] for r in sorted(second, key=lambda r: r["canonical_scenario_id"])]
    assert first_anwg == second_anwg


# ---------------------------------------------------------------------------
# 8. Frozen-artifact immutability guard
# ---------------------------------------------------------------------------

def test_parent_frozen_docs_untouched_by_this_replication_work():
    """This work must not modify any already-frozen router/gates/audit
    artifact -- checked via git diff against HEAD (staged+unstaged)."""
    frozen_paths = [
        "configs/hierarchical_regime_router_v1_gates.json",
        "docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md",
        "docs/design/HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md",
        "src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py",
        "src/llmserveopt/policy_separation/hierarchical_router_gates_v1.py",
        "src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py",
    ]
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--stat", "HEAD", "--"] + frozen_paths,
        capture_output=True, text=True, timeout=30,
    )
    assert result.stdout.strip() == "", f"frozen artifact modified:\n{result.stdout}"

"""Focused pre-launch tests for the Step-2 unified utility matrix builder.

See docs/design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md. Data-generation-only
harness; these tests validate the harness itself, not selector performance.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation import unified_utility_matrix as uum  # noqa: E402

MF_PSD_DIR = ROOT / "experiments" / "mf_psd_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Canonical policy-set identity
# ---------------------------------------------------------------------------

def test_six_canonical_anchors_match_reassessment_roadmap():
    assert set(uum.CANONICAL_ANCHOR_POLICIES.keys()) == {
        "estf", "wfs", "full_prefill", "chunked_prefill_small",
        "least_laxity", "kv_constrained",
    }
    assert set(uum.CANONICAL_ANCHOR_IDS) == {
        "estimated_service_time_first", "weighted_fair_share",
        "full_prefill", "chunked_prefill_small",
        "least_laxity_first", "kv_constrained_online",
    }
    assert len(uum.CANONICAL_ANCHOR_IDS) == 6


def test_native_family_assignment_matches_mf_psd_v1_audit():
    assert uum.NATIVE_FAMILY_OF_POLICY["estimated_service_time_first"] == "FAMILY_A_FAIRNESS_STARVATION_V2"
    assert uum.NATIVE_FAMILY_OF_POLICY["weighted_fair_share"] == "FAMILY_A_FAIRNESS_STARVATION_V2"
    assert uum.NATIVE_FAMILY_OF_POLICY["full_prefill"] == "FAMILY_B_PREFILL_DECODE_V2"
    assert uum.NATIVE_FAMILY_OF_POLICY["chunked_prefill_small"] == "FAMILY_B_PREFILL_DECODE_V2"
    assert uum.NATIVE_FAMILY_OF_POLICY["least_laxity_first"] == "FAMILY_C_KV_PRESSURE_V2"
    assert uum.NATIVE_FAMILY_OF_POLICY["kv_constrained_online"] == "FAMILY_C_KV_PRESSURE_V2"


# ---------------------------------------------------------------------------
# Expected task count / no duplicates / native exclusion / blocked family
# ---------------------------------------------------------------------------

def test_expected_new_cell_tasks_excludes_native_pairs():
    tasks = uum.expected_new_cell_tasks()
    for t in tasks:
        assert uum.NATIVE_FAMILY_OF_POLICY[t["canonical_policy_id"]] != t["mechanism_family"]


def test_expected_new_cell_tasks_count_matches_roadmap_704():
    tasks = uum.expected_new_cell_tasks()
    # 6 anchors x 2 non-native families each = 12 (family, policy) pairs
    assert len(tasks) == 12
    scenario_counts = {"FAMILY_A_FAIRNESS_STARVATION_V2": 72, "FAMILY_B_PREFILL_DECODE_V2": 32, "FAMILY_C_KV_PRESSURE_V2": 72}
    total_cells = sum(scenario_counts[t["mechanism_family"]] for t in tasks)
    assert total_cells == 704


def test_blocked_family_is_exactly_kv_family():
    assert uum.BLOCKED_TARGET_FAMILIES == {"FAMILY_C_KV_PRESSURE_V2"}


def test_no_duplicate_task_pairs():
    tasks = uum.expected_new_cell_tasks()
    seen = {(t["mechanism_family"], t["canonical_policy_id"]) for t in tasks}
    assert len(seen) == len(tasks)


def test_degenerate_policies_are_exactly_prefill_variants():
    assert uum.DEGENERATE_MECHANISM_POLICIES == {"full_prefill", "chunked_prefill_small"}


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------

def test_all_six_canonical_policies_construct():
    for policy_id in uum.CANONICAL_ANCHOR_IDS:
        policy, sm_override = uum._build_policy(policy_id)  # noqa: SLF001
        assert policy is not None
        assert isinstance(sm_override, dict)


def test_unknown_policy_id_raises():
    with pytest.raises(KeyError):
        uum._build_policy("not_a_real_policy")  # noqa: SLF001


# ---------------------------------------------------------------------------
# Deterministic scenario reconstruction
# ---------------------------------------------------------------------------

def test_family_a_scenario_ids_match_mf_psd_exactly():
    scenarios = uum.regenerate_family_a_scenarios()
    ids = {s.scenario_id for s in scenarios}
    assert len(scenarios) == 72
    with open(MF_PSD_DIR / "mf_psd_scenarios_v1.csv") as f:
        expected = {
            row["source_scenario_id"] for row in csv.DictReader(f)
            if row["mechanism_family"] == "FAMILY_A_FAIRNESS_STARVATION_V2"
        }
    assert ids == expected


def test_family_b_scenario_ids_match_mf_psd_exactly():
    scenarios = uum.regenerate_family_b_scenarios()
    ids = {s.scenario_id for s in scenarios}
    assert len(scenarios) == 32
    with open(MF_PSD_DIR / "mf_psd_scenarios_v1.csv") as f:
        expected = {
            row["source_scenario_id"] for row in csv.DictReader(f)
            if row["mechanism_family"] == "FAMILY_B_PREFILL_DECODE_V2"
        }
    assert ids == expected


def test_family_a_regeneration_is_deterministic_across_two_calls():
    s1 = uum.regenerate_family_a_scenarios()
    s2 = uum.regenerate_family_a_scenarios()
    ids1 = [s.scenario_id for s in s1]
    ids2 = [s.scenario_id for s in s2]
    assert ids1 == ids2  # same order too


# ---------------------------------------------------------------------------
# Cross-family compatibility / anti-leakage / canonical metric identity
# ---------------------------------------------------------------------------

def test_cross_family_cell_estf_on_family_b_executes_and_is_valid():
    scenarios = uum.regenerate_family_b_scenarios()
    row = uum.run_cell(scenarios[0], "estimated_service_time_first", "FAMILY_B_PREFILL_DECODE_V2")
    assert row["status"] == "success"
    assert row["primary_utility_anwg"] == row["primary_utility_anwg"]  # not NaN
    assert 0.0 <= row["primary_utility_anwg"] <= 1.0 + 1e-9
    assert row["degenerate_mechanism"] is False
    assert row["cell_source"] == "STEP2_CROSS_FAMILY_EVALUATION"


def test_cross_family_cell_kv_constrained_on_family_a_executes_and_is_valid():
    scenarios = uum.regenerate_family_a_scenarios()
    row = uum.run_cell(scenarios[0], "kv_constrained_online", "FAMILY_A_FAIRNESS_STARVATION_V2")
    assert row["status"] == "success"
    assert row["degenerate_mechanism"] is False


def test_cross_family_prefill_variants_degenerate_identically_on_family_a():
    scenarios = uum.regenerate_family_a_scenarios()
    row_full = uum.run_cell(scenarios[0], "full_prefill", "FAMILY_A_FAIRNESS_STARVATION_V2")
    row_small = uum.run_cell(scenarios[0], "chunked_prefill_small", "FAMILY_A_FAIRNESS_STARVATION_V2")
    assert row_full["status"] == "success" and row_small["status"] == "success"
    assert row_full["degenerate_mechanism"] is True
    assert row_small["degenerate_mechanism"] is True
    assert row_full["primary_utility_anwg"] == row_small["primary_utility_anwg"]


def test_run_cell_never_leaks_scenario_identity_to_policy(monkeypatch):
    """Anti-leakage: the policy object never receives scenario_id,
    canonical_scenario_id, mechanism_family, or seed as part of its
    observable state -- select_action only ever sees ObservableState."""
    scenarios = uum.regenerate_family_a_scenarios()
    scenario = scenarios[0]
    from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy
    seen_states = []
    orig = LeastLaxityFirstPolicy.select_action

    def spy(self, state):
        seen_states.append(state)
        return orig(self, state)

    monkeypatch.setattr(LeastLaxityFirstPolicy, "select_action", spy)
    uum.run_cell(scenario, "least_laxity_first", "FAMILY_A_FAIRNESS_STARVATION_V2")
    assert seen_states
    for state in seen_states:
        assert not hasattr(state, "scenario_id")
        assert not hasattr(state, "mechanism_family")
        assert not hasattr(state, "canonical_scenario_id")
        for req in state.waiting_queue:
            assert not hasattr(req, "scenario_id")


def test_unsupported_row_has_no_fabricated_metric():
    row = uum.unsupported_row("FAMILY_C_KV_PRESSURE_V2::x", "x", "FAMILY_C_KV_PRESSURE_V2", "estimated_service_time_first")
    assert row["status"] == "unsupported_scenario_reconstruction"
    assert row["primary_utility_anwg"] != row["primary_utility_anwg"]  # NaN
    assert "reconstruction" in row["error"].lower()


# ---------------------------------------------------------------------------
# Provenance / no mutation of frozen artifacts
# ---------------------------------------------------------------------------

def test_frozen_mf_psd_v1_artifacts_not_mutated_by_import_or_run_cell():
    files = [
        MF_PSD_DIR / "mf_psd_long_v1.csv",
        MF_PSD_DIR / "mf_psd_scenarios_v1.csv",
        MF_PSD_DIR / "mf_psd_schema_v1.json",
        MF_PSD_DIR / "mf_psd_provenance_v1.json",
    ]
    before = {f: _sha256(f) for f in files}
    scenarios = uum.regenerate_family_a_scenarios()
    uum.run_cell(scenarios[0], "kv_constrained_online", "FAMILY_A_FAIRNESS_STARVATION_V2")
    after = {f: _sha256(f) for f in files}
    assert before == after


def test_frozen_source_run_dirs_not_mutated():
    result = subprocess.run(
        ["git", "status", "--short",
         "experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/",
         "experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/",
         "experiments/kv_pressure_pilot_v2_20260817T165053Z/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", f"frozen source dirs show git diff: {result.stdout}"


# ---------------------------------------------------------------------------
# Result schema sanity
# ---------------------------------------------------------------------------

def test_result_fieldnames_are_stable_and_cover_required_columns():
    required = {
        "uum_row_id", "canonical_scenario_id", "mechanism_family",
        "canonical_policy_id", "cell_source", "degenerate_mechanism",
        "status", "primary_utility_anwg",
    }
    assert required.issubset(set(uum.RESULT_FIELDNAMES))

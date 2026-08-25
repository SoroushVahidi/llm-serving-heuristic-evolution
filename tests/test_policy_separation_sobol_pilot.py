"""Focused tests for the Policy Separation Sobol Pilot v1 design
(scripts/run_policy_separation_sobol_pilot_v1.py,
src/llmserveopt/policy_separation/sobol_pilot.py). This is a
design/validation-only pipeline -- no scientific sweep has been run with
it -- so these tests cover determinism, no-duplicate-IDs, categorical
expansion, anti-leakage field classification, config/roster correctness,
and output-schema plumbing via a tiny local dry-run subprocess."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.sobol_pilot import (
    FAMILY_B_RANGES, FAMILY_C_RANGES, GENERATOR_ONLY_FIELDS, POLICY_VISIBLE_FIELDS,
    classify_field, generate_family_b_sobol_scenarios, generate_family_c_sobol_scenarios,
    generate_fcfs_categorical_add_on, sobol_unit_points, validate_scenario,
)

CONFIG_PATH = ROOT / "configs" / "policy_separation_sobol_pilot_v1.yaml"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


# ---------------------------------------------------------------------------
# Sobol determinism / no duplicates / range transforms
# ---------------------------------------------------------------------------

def test_sobol_points_deterministic_given_seed():
    a = sobol_unit_points(d=2, m=5, scramble_seed=20260810)
    b = sobol_unit_points(d=2, m=5, scramble_seed=20260810)
    assert (a == b).all()
    assert a.shape == (32, 2)


def test_sobol_points_no_duplicates_and_in_unit_interval():
    pts = sobol_unit_points(d=2, m=7, scramble_seed=20260810)
    assert len(pts) == 128
    uniq = {tuple(p) for p in pts}
    assert len(uniq) == 128
    assert (pts >= 0.0).all() and (pts <= 1.0).all()


def test_sobol_different_scramble_seeds_give_different_sequences():
    a = sobol_unit_points(d=2, m=5, scramble_seed=20260810)
    b = sobol_unit_points(d=2, m=5, scramble_seed=20260812)
    assert not (a == b).all()


def test_family_b_scenarios_respect_declared_ranges():
    scenarios = generate_family_b_sobol_scenarios(
        m=5, scramble_seed=20260810, heterogeneity_levels=["moderate", "strong"], seeds=[1],
    )
    lo_u, hi_u = FAMILY_B_RANGES["target_utilization"]
    lo_i, hi_i = FAMILY_B_RANGES["inversion_fraction"]
    for s in scenarios:
        assert lo_u <= s.params["target_utilization"] <= hi_u
        assert lo_i <= s.params["inversion_fraction"] <= hi_i


def test_family_c_scenarios_respect_declared_ranges():
    scenarios = generate_family_c_sobol_scenarios(m=5, scramble_seed=20260812, seeds=[1])
    lo_o, hi_o = FAMILY_C_RANGES["overload_factor"]
    lo_f, hi_f = FAMILY_C_RANGES["fraction_impossible"]
    for s in scenarios:
        assert lo_o <= s.params["overload_factor"] <= hi_o
        assert lo_f <= s.params["fraction_impossible"] <= hi_f
        assert s.stress_control_relationship == "stress"


# ---------------------------------------------------------------------------
# Categorical expansion / stable IDs / no duplicates across the full design
# ---------------------------------------------------------------------------

def test_family_b_categorical_expansion_count():
    scenarios = generate_family_b_sobol_scenarios(
        m=4, scramble_seed=20260810, heterogeneity_levels=["moderate", "strong"], seeds=[1, 2, 3],
    )
    assert len(scenarios) == 16 * 2 * 3


def test_scenario_ids_stable_across_repeated_generation():
    a = generate_family_b_sobol_scenarios(m=3, scramble_seed=20260810, heterogeneity_levels=["moderate"], seeds=[1])
    b = generate_family_b_sobol_scenarios(m=3, scramble_seed=20260810, heterogeneity_levels=["moderate"], seeds=[1])
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]


def test_no_duplicate_scenario_ids_across_full_pilot_design(cfg):
    from llmserveopt.policy_separation.sobol_pilot import (
        generate_family_b_sobol_scenarios as gb, generate_family_c_sobol_scenarios as gc,
        generate_fcfs_categorical_add_on as ga,
    )
    b_cfg, c_cfg, a_cfg, sobol_cfg = (
        cfg["family_b_prediction_sensitive"], cfg["family_c_deadline_admission"],
        cfg["fcfs_categorical_add_on"], cfg["sobol"],
    )
    s_b = gb(m=4, scramble_seed=sobol_cfg["family_b_scramble_seed"],
             heterogeneity_levels=b_cfg["heterogeneity"], seeds=b_cfg["seeds"][:2])
    s_c = gc(m=4, scramble_seed=sobol_cfg["family_c_scramble_seed"], seeds=c_cfg["seeds"][:2])
    s_a = ga(a_cfg["a1_ratios"], a_cfg["a1_short_counts"], a_cfg["a1_seeds"][:2],
             a_cfg["a2_ratio"], a_cfg["a2_short_count"], a_cfg["a2_offsets"],
             a_cfg["a2_max_active_sequences"], a_cfg["a2_seeds"][:2])
    ids = [s.scenario_id for s in s_b + s_c + s_a]
    assert len(ids) == len(set(ids))


def test_fcfs_offset_treated_categorically_not_continuously(cfg):
    a_cfg = cfg["fcfs_categorical_add_on"]
    scenarios = generate_fcfs_categorical_add_on(
        a_cfg["a1_ratios"], a_cfg["a1_short_counts"], a_cfg["a1_seeds"],
        a_cfg["a2_ratio"], a_cfg["a2_short_count"], a_cfg["a2_offsets"],
        a_cfg["a2_max_active_sequences"], a_cfg["a2_seeds"],
    )
    offsets_used = {s.params["offset"] for s in scenarios}
    # exactly the discrete set {0.0 (template A1)} union the configured A2 offsets -- never
    # a continuously-sampled/Sobol value.
    assert offsets_used == {0.0} | set(a_cfg["a2_offsets"])


# ---------------------------------------------------------------------------
# Validity guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,seed", [(5, 20260810), (5, 20260812)])
def test_no_validity_problems_across_sampled_ranges(m, seed):
    b = generate_family_b_sobol_scenarios(m=m, scramble_seed=seed, heterogeneity_levels=["moderate", "strong"], seeds=[1])
    c = generate_family_c_sobol_scenarios(m=m, scramble_seed=seed, seeds=[1])
    for s in b + c:
        assert validate_scenario(s) == []


# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------

def test_generator_only_and_policy_visible_fields_overlap_is_exactly_the_two_documented_collisions():
    # "max_active_sequences" and "role" are real GPUConfig fields that
    # coincidentally share a name with an unrelated generator-only
    # scenario.params label -- see sobol_pilot.py's POLICY_VISIBLE_FIELDS
    # docstring. Any OTHER overlap would indicate a genuine new leak risk.
    overlap = GENERATOR_ONLY_FIELDS & POLICY_VISIBLE_FIELDS
    assert overlap == {"max_active_sequences", "role"}


def test_classify_field():
    assert classify_field("sobol_index") == "generator_only"
    assert classify_field("target_utilization") == "generator_only"
    assert classify_field("rank_agreement_kendall_tau") == "generator_only"
    assert classify_field("request_id") == "policy_visible"
    assert classify_field("slo_deadline") == "policy_visible"
    assert classify_field("totally_unrelated_name") == "unknown"


def test_no_generator_field_leaks_into_policy_visible_state():
    scenarios = generate_family_b_sobol_scenarios(
        m=3, scramble_seed=20260810, heterogeneity_levels=["moderate"], seeds=[1],
    ) + generate_family_c_sobol_scenarios(m=3, scramble_seed=20260812, seeds=[1])
    from llmserveopt.policy_separation.sobol_pilot import _GPU_CONFIG_NAME_COLLISIONS
    for s in scenarios:
        for field in GENERATOR_ONLY_FIELDS - _GPU_CONFIG_NAME_COLLISIONS:
            for r in s.requests:
                assert not hasattr(r, field), f"{field!r} leaked onto Request in {s.scenario_id}"
            for g in s.gpu_configs:
                assert not hasattr(g, field), f"{field!r} leaked onto GPUConfig in {s.scenario_id}"
        # Request objects never have the two documented name-collision
        # fields either (only GPUConfig legitimately does) -- check those two explicitly.
        for r in s.requests:
            assert not hasattr(r, "max_active_sequences")
            assert not hasattr(r, "role")


def test_sobol_provenance_lives_only_in_params_not_top_level_scenario_fields():
    s = generate_family_b_sobol_scenarios(
        m=2, scramble_seed=20260810, heterogeneity_levels=["moderate"], seeds=[1],
    )[0]
    assert "sobol_index" in s.params
    assert not hasattr(s, "sobol_index")


# ---------------------------------------------------------------------------
# Config / roster correctness
# ---------------------------------------------------------------------------

def test_weighted_shortest_processing_excluded_from_pilot(cfg):
    for key in ("family_b_prediction_sensitive", "fcfs_categorical_add_on"):
        assert "weighted_shortest_processing" not in cfg[key]["policies"], (
            f"{key} should exclude weighted_shortest_processing -- proven 100% ANWG-identical "
            "to estimated_service_time_first in job 1171116 (uniform priority=1.0 collapses "
            "WSP's score formula to ESTF's); see docs/design/POLICY_SEPARATION_SOBOL_PILOT_V1.md section 3."
        )


def test_admission_control_retained_in_family_c(cfg):
    assert "admission_control" in cfg["family_c_deadline_admission"]["policies"]


def test_config_rosters_present_and_nonempty(cfg):
    for key in ("family_b_prediction_sensitive", "family_c_deadline_admission", "fcfs_categorical_add_on"):
        assert len(cfg[key]["policies"]) > 0


# ---------------------------------------------------------------------------
# Expected scenario/evaluation count at full (non-dry-run) config scale
# ---------------------------------------------------------------------------

def test_full_scale_scenario_and_eval_counts_match_design_doc(cfg):
    from llmserveopt.policy_separation.sobol_pilot import (
        generate_family_b_sobol_scenarios as gb, generate_family_c_sobol_scenarios as gc,
        generate_fcfs_categorical_add_on as ga,
    )
    sobol_cfg, b_cfg, c_cfg, a_cfg = cfg["sobol"], cfg["family_b_prediction_sensitive"], \
        cfg["family_c_deadline_admission"], cfg["fcfs_categorical_add_on"]

    s_b = gb(m=sobol_cfg["family_b_m"], scramble_seed=sobol_cfg["family_b_scramble_seed"],
              heterogeneity_levels=b_cfg["heterogeneity"], seeds=b_cfg["seeds"])
    s_c = gc(m=sobol_cfg["family_c_m"], scramble_seed=sobol_cfg["family_c_scramble_seed"], seeds=c_cfg["seeds"])
    s_a = ga(a_cfg["a1_ratios"], a_cfg["a1_short_counts"], a_cfg["a1_seeds"],
             a_cfg["a2_ratio"], a_cfg["a2_short_count"], a_cfg["a2_offsets"],
             a_cfg["a2_max_active_sequences"], a_cfg["a2_seeds"])

    assert len(s_b) == 1024
    assert len(s_c) == 512
    assert len(s_a) == 80

    n_evals = (len(s_b) * len(b_cfg["policies"]) + len(s_c) * len(c_cfg["policies"])
               + len(s_a) * len(a_cfg["policies"]))
    assert n_evals == 6976

    all_ids = [s.scenario_id for s in s_b + s_c + s_a]
    assert len(all_ids) == len(set(all_ids))
    assert len(all_ids) == 1616


# ---------------------------------------------------------------------------
# Output-schema smoke via a tiny local dry-run subprocess (no Slurm)
# ---------------------------------------------------------------------------

def test_dry_run_produces_full_output_schema(tmp_path):
    run_dir = tmp_path / "sobol_dryrun_pytest"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "run_policy_separation_sobol_pilot_v1.py"),
            "--config", str(CONFIG_PATH), "--run-dir", str(run_dir), "--workers", "2", "--dry-run",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    expected_files = [
        "run_manifest.json", "sobol_design.json", "config_snapshot.yaml", "scenarios.jsonl",
        "scenario_features.csv", "per_policy_results.csv", "pairwise_separation.csv",
        "policy_winner_summary.csv", "oracle_headroom.csv", "family_summary.csv",
        "coverage_summary.csv", "final_summary.json", "run.log",
    ]
    for name in expected_files:
        assert (run_dir / name).exists(), f"missing output {name}"
    assert not (run_dir / "failures.jsonl").exists()

    final = json.loads((run_dir / "final_summary.json").read_text())
    assert final["n_failed"] == 0
    assert final["n_validity_warnings"] == 0
    assert final["dry_run"] is True
    assert final["scientific_result"] is False

    with open(run_dir / "per_policy_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = {(r["scenario_id"], r["policy_name"]) for r in rows}
    assert len(keys) == len(rows), "duplicate (scenario_id, policy_name) keys in dry-run output"

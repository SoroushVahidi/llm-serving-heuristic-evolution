"""Focused tests for the Apt-Serve Phase G infrastructure:
src/llmserveopt/workloads/apt_serve_stress.py's regime generator,
src/llmserveopt/workloads/apt_serve_phase_g_regimes.py's curated catalog,
and scripts/run_apt_serve_phase_g.py's orchestration/resumability helpers.

Does not run the full overnight matrix (that's the point of the separate
tmux-launched run) -- exercises generator determinism, catalog integrity,
resumable-unit bookkeeping, and one small end-to-end unit computation.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from llmserveopt.workloads.apt_serve_stress import (
    ARRIVAL_PATTERNS,
    CACHE_USE_STRUCTURES,
    KV_PRESSURE_TIERS,
    LENGTH_PATTERNS,
    SLO_PATTERNS,
    generate_apt_serve_regime_workload,
)
from llmserveopt.workloads.apt_serve_phase_g_regimes import REGIME_CATALOG

_SPEC = importlib.util.spec_from_file_location(
    "run_apt_serve_phase_g",
    Path(__file__).parent.parent / "scripts" / "run_apt_serve_phase_g.py",
)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["run_apt_serve_phase_g"] = runner
_SPEC.loader.exec_module(runner)


# ======================================================================
# Regime generator
# ======================================================================

def test_regime_workload_is_deterministic():
    w1 = generate_apt_serve_regime_workload(seed=42, n_requests=20, arrival_pattern="steady",
                                             kv_pressure="medium", slo_pattern="bimodal",
                                             length_pattern="bimodal", cache_use_structure="none")
    w2 = generate_apt_serve_regime_workload(seed=42, n_requests=20, arrival_pattern="steady",
                                             kv_pressure="medium", slo_pattern="bimodal",
                                             length_pattern="bimodal", cache_use_structure="none")
    assert len(w1) == 20
    assert w1 == w2


def test_regime_workload_arrival_times_nondecreasing():
    for arrival_pattern in ARRIVAL_PATTERNS:
        w = generate_apt_serve_regime_workload(seed=7, n_requests=30, arrival_pattern=arrival_pattern,
                                                kv_pressure="high", slo_pattern="relaxed_homogeneous",
                                                length_pattern="homogeneous", cache_use_structure="none")
        times = [r.arrival_time for r in w]
        assert times == sorted(times), f"arrival times not sorted for pattern={arrival_pattern}"


def test_cache_use_structure_produces_two_cohorts():
    for hint in CACHE_USE_STRUCTURES:
        if hint == "none":
            continue
        w = generate_apt_serve_regime_workload(seed=3, n_requests=20, arrival_pattern="steady",
                                                kv_pressure="medium", slo_pattern="bimodal",
                                                length_pattern="bimodal", cache_use_structure=hint)
        class_ids = {r.class_id for r in w}
        assert any("long" in c for c in class_ids)
        assert any("short" in c for c in class_ids)


def test_unknown_axis_values_raise():
    with pytest.raises(ValueError):
        generate_apt_serve_regime_workload(seed=1, n_requests=5, kv_pressure="not_a_tier")


# ======================================================================
# Regime catalog
# ======================================================================

def test_regime_catalog_size_in_target_range():
    assert 30 <= len(REGIME_CATALOG) <= 60


def test_regime_catalog_ids_unique():
    ids = [r["regime_id"] for r in REGIME_CATALOG]
    assert len(ids) == len(set(ids))


def test_regime_catalog_every_axis_value_represented():
    for r in REGIME_CATALOG:
        assert r["kv_pressure"] in KV_PRESSURE_TIERS
        assert r["slo_pattern"] in SLO_PATTERNS
        assert r["length_pattern"] in LENGTH_PATTERNS
        assert r["arrival_pattern"] in ARRIVAL_PATTERNS
        assert r["cache_use_structure"] in CACHE_USE_STRUCTURES

    seen_kv = {r["kv_pressure"] for r in REGIME_CATALOG}
    seen_slo = {r["slo_pattern"] for r in REGIME_CATALOG}
    seen_length = {r["length_pattern"] for r in REGIME_CATALOG}
    seen_arrival = {r["arrival_pattern"] for r in REGIME_CATALOG}
    seen_cache = {r["cache_use_structure"] for r in REGIME_CATALOG}
    assert seen_kv == set(KV_PRESSURE_TIERS)
    assert seen_slo == set(SLO_PATTERNS)
    assert seen_length == set(LENGTH_PATTERNS)
    assert seen_arrival == set(ARRIVAL_PATTERNS)
    assert seen_cache == set(CACHE_USE_STRUCTURES)


def test_regime_catalog_is_json_serializable():
    json.dumps(REGIME_CATALOG)  # must not raise


# ======================================================================
# Runner: strong-baseline set, resumability, atomic writes
# ======================================================================

def test_strong_baseline_names_exist_in_registry():
    names = runner.strong_baseline_policy_names()
    assert len(names) == 12
    assert "oracle_srtf" not in names
    assert len(names) == len(set(names))


def test_transition_cost_labels_include_idealized_and_primary():
    assert "0x_idealized" in runner.TRANSITION_COST_LABELS
    assert runner.PRIMARY_TRANSITION_COST in runner.TRANSITION_COST_LABELS
    assert runner.TRANSITION_COST_MULTIPLIERS["0x_idealized"] == 0.0
    assert runner.TRANSITION_COST_MULTIPLIERS[runner.PRIMARY_TRANSITION_COST] == 1.0


def test_unit_key_stable_and_distinguishes_stage():
    k1 = runner.unit_key("screening", "pressure_high_baseline", 42)
    k2 = runner.unit_key("confirmation", "pressure_high_baseline", 42)
    assert k1 != k2
    assert runner.unit_key("screening", "pressure_high_baseline", 42) == k1


def test_resumability_skips_completed_units(tmp_path):
    results_path = tmp_path / "results.jsonl"
    rec = {"stage": "screening", "regime_id": "pressure_high_baseline", "seed": 1001, "cells": []}
    runner.append_jsonl(results_path, rec)
    done = runner.load_completed_units(results_path)
    assert runner.unit_key("screening", "pressure_high_baseline", 1001) in done
    assert runner.unit_key("screening", "pressure_high_baseline", 1002) not in done


def test_atomic_write_json_never_leaves_partial_file(tmp_path):
    path = tmp_path / "progress.json"
    runner.atomic_write_json(path, {"a": 1})
    runner.atomic_write_json(path, {"a": 2, "b": [1, 2, 3]})
    loaded = json.loads(path.read_text())
    assert loaded == {"a": 2, "b": [1, 2, 3]}
    assert not path.with_suffix(".json.tmp").exists()


# ======================================================================
# End-to-end: one small unit, exercising baselines + all transition costs
# ======================================================================

def test_run_one_unit_smoke():
    small_regime = {
        "regime_id": "smoke_test_regime", "kv_pressure": "medium", "slo_pattern": "bimodal",
        "length_pattern": "bimodal", "arrival_pattern": "steady",
        "cache_use_structure": "kv_to_hidden_opportunity", "n_requests": 10,
    }
    result = runner.run_one_unit(small_regime, seed=99001, stage="unit_test")
    assert result["critical_failure"] is None
    labels = {(c["policy_label"], c["transition_cost"]) for c in result["cells"]}
    for name in runner.strong_baseline_policy_names():
        assert (name, "na") in labels
    for tc in runner.TRANSITION_COST_LABELS:
        assert ("apt_serve_faithful", tc) in labels
    for c in result["cells"]:
        assert c["num_completed"] + c["num_dropped"] <= small_regime["n_requests"]
        if c["policy_kind"] == "apt_serve":
            assert "apt_stats" in c

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
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.apt_serve_faithful import (
    AptServeAdapterConfig,
    AptServeSchedulerInput,
    AptServeSubprocessClient,
    CacheTier,
)

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

# ======================================================================
# SS15 incident regression: overnight sweep self-terminated on
# pressure_sustained_overload_baseline / seed=1005 / transition_cost=4x
# with AptServeCapacityViolation ("Insufficient destination hidden
# capacity" for request 11). Root cause was two independent bugs -- see
# docs/audits/apt_serve_phase_g_ss15_incident_20260807.md:
#   1. apt_serve_faithful.py applied cache-tier transitions in raw client
#      dict order instead of an order that respects capacity dependencies
#      (fixed: apply HIDDEN->KV releases before KV->HIDDEN acquisitions).
#   2. fake_scheduler_worker.py's "evict one relaxed KV resident to admit
#      an urgent waiting request" fallback never checked that the evicted
#      resident's freed KV blocks actually covered the new request's need
#      (fixed: added a fits_after_eviction check).
# ======================================================================

def _sustained_overload_regime():
    return next(r for r in REGIME_CATALOG if r["regime_id"] == "pressure_sustained_overload_baseline")


def test_ss15_known_failing_cell_now_completes():
    """Exact reproduction of the cell that crashed the overnight sweep:
    must now run to completion with zero dropped requests, no exception."""
    regime = _sustained_overload_regime()
    requests = generate_apt_serve_regime_workload(
        seed=1005, n_requests=regime["n_requests"], arrival_pattern=regime["arrival_pattern"],
        kv_pressure=regime["kv_pressure"], slo_pattern=regime["slo_pattern"],
        length_pattern=regime["length_pattern"], cache_use_structure=regime["cache_use_structure"],
    )
    service_model = runner.ServiceModel(step_size=runner.SERVICE_MODEL_STEP_SIZE)
    policy = runner.build_apt_policy("4x")
    m = run_policy(policy=policy, requests=requests, gpu_configs=runner.GPU_CONFIGS,
                    service_model=service_model, workload_tag=regime["regime_id"], seed=1005)
    policy.terminate()
    assert m.num_completed == regime["n_requests"]
    assert m.num_dropped == 0


def test_ss15_neighboring_cells_no_critical_failure():
    """Neighboring seeds and transition costs around the known failing
    cell must all complete without a critical Apt-Serve invariant
    failure -- guards against the fix being a narrow special case."""
    regime = _sustained_overload_regime()
    for seed in (1004, 1005, 1006):
        requests = generate_apt_serve_regime_workload(
            seed=seed, n_requests=regime["n_requests"], arrival_pattern=regime["arrival_pattern"],
            kv_pressure=regime["kv_pressure"], slo_pattern=regime["slo_pattern"],
            length_pattern=regime["length_pattern"], cache_use_structure=regime["cache_use_structure"],
        )
        for tc_label in runner.TRANSITION_COST_LABELS:
            service_model = runner.ServiceModel(step_size=runner.SERVICE_MODEL_STEP_SIZE)
            policy = runner.build_apt_policy(tc_label)
            try:
                m = run_policy(policy=policy, requests=requests, gpu_configs=runner.GPU_CONFIGS,
                                service_model=service_model, workload_tag=regime["regime_id"], seed=seed)
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"seed={seed} tc={tc_label} raised {type(e).__name__}: {e}")
            finally:
                policy.terminate()
            assert m.num_completed + m.num_dropped == regime["n_requests"]


def test_ss15_run_one_unit_reports_no_critical_failure():
    """run_one_unit (the exact function the overnight sweep called) must
    report critical_failure=None and a full complement of cells for the
    known-bad unit."""
    regime = _sustained_overload_regime()
    result = runner.run_one_unit(regime, seed=1005, stage="regression_test")
    assert result["critical_failure"] is None
    assert result["failures"] == []
    labels = {(c["policy_label"], c["transition_cost"]) for c in result["cells"]}
    for tc in runner.TRANSITION_COST_LABELS:
        assert ("apt_serve_faithful", tc) in labels


def test_fake_worker_preemption_requires_eviction_to_cover_admission():
    """The fake scheduler's urgent-preemption fallback must not evict a
    relaxed KV resident and admit a new request when the eviction alone
    doesn't free enough KV blocks to cover it -- doing so silently
    produces a decision that overcommits KV capacity by the client's own
    ledger, which the adapter then can only catch downstream as a raw
    KVBlockManagerError ('Out of memory')."""
    config = AptServeAdapterConfig(checkout_path="", execution_mode="test")

    # max_kv_tokens=112 -> raw 7 blocks, watermark 1 -> 6 usable blocks.
    # hidden_cache_capacity_blocks=2 -> raw 2 blocks, watermark 1 -> 1 usable block.
    gpus = [{"max_kv_tokens": 112, "hidden_cache_capacity_blocks": 2}]

    running_requests = [
        # A: filler, non-relaxed, stays on KV (3 blocks).
        {"request_id": 1, "prompt_tokens": 48, "arrival_time": 0.0, "predicted_output_tokens": 10,
         "slo_deadline": 1.0, "priority": 1.0, "current_cache_tier": "kv"},
        # B: non-relaxed HIDDEN resident that restores to KV in pass 1,
        # freeing the hidden headroom C's eviction will need in pass 2
        # (2 blocks -> 1 hidden block).
        {"request_id": 2, "prompt_tokens": 32, "arrival_time": 0.0, "predicted_output_tokens": 10,
         "slo_deadline": 1.0, "priority": 1.0, "current_cache_tier": "hidden"},
        # C: relaxed KV resident, small (1 block) -- the only eviction
        # candidate for the urgent admission below.
        {"request_id": 3, "prompt_tokens": 16, "arrival_time": 0.0, "predicted_output_tokens": 10,
         "slo_deadline": 100.0, "priority": 1.0, "current_cache_tier": "kv"},
    ]
    # D: urgent waiting request needing 3 KV blocks -- more than evicting
    # C (1 block) can free (current_kv would go 6 - 1 + 3 = 8 > 6).
    waiting_requests = [
        {"request_id": 4, "prompt_tokens": 48, "arrival_time": 0.0, "predicted_output_tokens": 10,
         "slo_deadline": 1.0, "priority": 1.0},
    ]

    state_input = AptServeSchedulerInput(
        schema_version=1, request_id=1, simulator_step=1, timestamp=0.0,
        gpus=gpus, waiting_requests=waiting_requests, running_requests=running_requests,
        cache_snapshot={},
    )

    with AptServeSubprocessClient(config) as client:
        decision = client.schedule_step(state_input)

    assert decision.cache_assignments.get(4) == CacheTier.NONE
    assert 4 in decision.deprioritized_requests
    assert 4 not in decision.selected_request_ids
    # C must be left exactly where it was -- no partial/speculative eviction.
    assert decision.cache_assignments.get(3) == CacheTier.KV


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

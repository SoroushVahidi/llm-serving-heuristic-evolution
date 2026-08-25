"""Focused tests for Family C Reconstruction v1.

See docs/design/FAMILY_C_RECONSTRUCTION_V1.md. Data-generation-only
harness; validates the harness itself (generation, serialization, replay
determinism, anti-leakage, provenance), not selector performance.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation import family_c_reconstruction_v1 as fc  # noqa: E402
from llmserveopt.policy_separation import unified_utility_matrix as uum  # noqa: E402

MF_PSD_DIR = ROOT / "experiments" / "mf_psd_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Scenario generation: exactly 72, matching MF-PSD's Family-C set
# ---------------------------------------------------------------------------

def test_exactly_72_scenarios_generated_matching_mf_psd():
    scenarios = fc.regenerate_family_c_scenarios()
    assert len(scenarios) == 72
    ids = {s.scenario_id for s in scenarios}
    with open(MF_PSD_DIR / "mf_psd_scenarios_v1.csv") as f:
        expected = {
            row["source_scenario_id"] for row in csv.DictReader(f)
            if row["mechanism_family"] == "FAMILY_C_KV_PRESSURE_V2"
        }
    assert ids == expected


def test_generation_is_deterministic_across_two_calls():
    s1 = fc.regenerate_family_c_scenarios()
    s2 = fc.regenerate_family_c_scenarios()
    for a, b in zip(s1, s2):
        assert a.scenario_id == b.scenario_id
        assert tuple(asdict(r) for r in a.requests) == tuple(asdict(r) for r in b.requests)


# ---------------------------------------------------------------------------
# Serialization / deterministic replay
# ---------------------------------------------------------------------------

def test_serialize_and_reload_round_trips_exactly(tmp_path):
    scenarios = fc.regenerate_family_c_scenarios()
    out = tmp_path / "scenarios.jsonl"
    fc.serialize_scenarios(scenarios, out)
    reloaded = fc.load_serialized_scenarios(out)
    assert len(reloaded) == len(scenarios)
    by_id = {s.scenario_id: s for s in scenarios}
    for r in reloaded:
        orig = by_id[r.scenario_id]
        assert tuple(asdict(x) for x in r.requests) == tuple(asdict(x) for x in orig.requests)
        assert tuple(asdict(x) for x in r.gpu_configs) == tuple(asdict(x) for x in orig.gpu_configs)
        assert r.service_model_kwargs == orig.service_model_kwargs
        assert r.seed == orig.seed
        assert r.params == orig.params


def test_serialization_is_byte_stable_across_two_writes(tmp_path):
    scenarios = fc.regenerate_family_c_scenarios()
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    fc.serialize_scenarios(scenarios, out1)
    fc.serialize_scenarios(fc.regenerate_family_c_scenarios(), out2)
    assert _sha256(out1) == _sha256(out2)


def test_load_serialized_scenarios_never_touches_burstgpt(tmp_path, monkeypatch):
    """Replay must not resample BurstGPT: the loader must not call
    _load_burstgpt_arrays / resolve_burstgpt_path at all."""
    scenarios = fc.regenerate_family_c_scenarios()
    out = tmp_path / "scenarios.jsonl"
    fc.serialize_scenarios(scenarios, out)

    from llmserveopt.policy_separation import templates_prefill_decode as tpd

    def _boom(*args, **kwargs):
        raise AssertionError("load_serialized_scenarios must not call BurstGPT loading code")

    monkeypatch.setattr(tpd, "_load_burstgpt_arrays", _boom)
    monkeypatch.setattr(tpd, "resolve_burstgpt_path", _boom)
    # Must succeed without ever invoking the patched (booby-trapped) functions.
    reloaded = fc.load_serialized_scenarios(out)
    assert len(reloaded) == 72


def test_frozen_scenario_replay_has_no_generator_import_dependency():
    # Inspect bytecode names (not docstring text) referenced by the
    # function body: the loader must not name any generator/BurstGPT
    # symbol anywhere in its actual code.
    forbidden = {"_load_burstgpt_arrays", "resolve_burstgpt_path", "templates_kv_pressure_v2", "templates_prefill_decode"}
    names = set(fc.load_serialized_scenarios.__code__.co_names)
    assert names.isdisjoint(forbidden)


# ---------------------------------------------------------------------------
# Six policies, same frozen request-level input
# ---------------------------------------------------------------------------

def test_six_canonical_policies_match_step2_design():
    assert set(uum.CANONICAL_ANCHOR_IDS) == {
        "estimated_service_time_first", "weighted_fair_share",
        "full_prefill", "chunked_prefill_small",
        "least_laxity_first", "kv_constrained_online",
    }


def test_all_six_policies_receive_identical_request_tuple_for_one_scenario(tmp_path):
    scenarios = fc.regenerate_family_c_scenarios()
    out = tmp_path / "scenarios.jsonl"
    fc.serialize_scenarios(scenarios, out)
    reloaded = fc.load_serialized_scenarios(out)
    scenario = reloaded[0]
    # All 6 evaluations for this scenario must read from the exact same
    # in-memory requests tuple -- run_cell_reconstruction never mutates or
    # replaces scenario.requests.
    before = scenario.requests
    for policy_id in uum.CANONICAL_ANCHOR_IDS:
        fc.run_cell_reconstruction(scenario, policy_id)
        assert scenario.requests is before


def test_run_cell_reconstruction_native_and_cross_family_all_succeed(tmp_path):
    scenarios = fc.regenerate_family_c_scenarios()
    out = tmp_path / "scenarios.jsonl"
    fc.serialize_scenarios(scenarios, out)
    scenario = fc.load_serialized_scenarios(out)[0]
    rows = {p: fc.run_cell_reconstruction(scenario, p) for p in uum.CANONICAL_ANCHOR_IDS}
    for p, row in rows.items():
        assert row["status"] == "success", row.get("error")
        assert row["primary_utility_anwg"] == row["primary_utility_anwg"]  # not NaN
        assert row["source_family"] == "FAMILY_C_KV_PRESSURE_V2"
        assert row["reconstruction_version"] == "CURRENT_RECONSTRUCTED_FAMILY_C_V1"
    assert rows["kv_constrained_online"]["native_to_family_c"] is True
    assert rows["least_laxity_first"]["native_to_family_c"] is True
    assert rows["estimated_service_time_first"]["native_to_family_c"] is False
    assert rows["full_prefill"]["degenerate_mechanism"] is True
    assert rows["chunked_prefill_small"]["degenerate_mechanism"] is True
    assert rows["kv_constrained_online"]["degenerate_mechanism"] is False


# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------

def test_run_cell_reconstruction_never_leaks_scenario_identity(tmp_path, monkeypatch):
    scenarios = fc.regenerate_family_c_scenarios()
    out = tmp_path / "scenarios.jsonl"
    fc.serialize_scenarios(scenarios, out)
    scenario = fc.load_serialized_scenarios(out)[0]

    from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy
    seen_states = []
    orig = KVConstrainedOnlinePolicy.select_action

    def spy(self, state):
        seen_states.append(state)
        return orig(self, state)

    monkeypatch.setattr(KVConstrainedOnlinePolicy, "select_action", spy)
    fc.run_cell_reconstruction(scenario, "kv_constrained_online")
    assert seen_states
    for state in seen_states:
        assert not hasattr(state, "scenario_id")
        for req in state.waiting_queue:
            assert not hasattr(req, "scenario_id")


# ---------------------------------------------------------------------------
# No mutation of frozen artifacts
# ---------------------------------------------------------------------------

def test_no_mutation_of_mf_psd_v1_or_historical_kv_v2(tmp_path):
    files = [
        MF_PSD_DIR / "mf_psd_long_v1.csv",
        MF_PSD_DIR / "mf_psd_scenarios_v1.csv",
        ROOT / "experiments/kv_pressure_pilot_v2_20260817T165053Z/per_policy_results.csv",
    ]
    before = {f: _sha256(f) for f in files}
    scenarios = fc.regenerate_family_c_scenarios()
    out = tmp_path / "scenarios.jsonl"
    fc.serialize_scenarios(scenarios, out)
    scenario = fc.load_serialized_scenarios(out)[0]
    fc.run_cell_reconstruction(scenario, "kv_constrained_online")
    after = {f: _sha256(f) for f in files}
    assert before == after


def test_no_mutation_of_unified_utility_matrix_v1():
    uum_dir = ROOT / "experiments" / "unified_utility_matrix_v1"
    result = subprocess.run(
        ["git", "status", "--short", str(uum_dir.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", f"unified_utility_matrix_v1 shows git diff: {result.stdout}"


def test_frozen_family_c_source_run_dirs_not_mutated():
    result = subprocess.run(
        ["git", "status", "--short",
         "experiments/kv_pressure_pilot_v1_20260817T162650Z/",
         "experiments/kv_pressure_pilot_v2_20260817T165053Z/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

def test_result_fieldnames_cover_required_columns():
    required = {
        "reconstruction_scenario_id", "source_family", "reconstruction_version",
        "canonical_policy_id", "native_to_family_c", "status", "primary_utility_anwg",
    }
    assert required.issubset(set(fc.RESULT_FIELDNAMES))

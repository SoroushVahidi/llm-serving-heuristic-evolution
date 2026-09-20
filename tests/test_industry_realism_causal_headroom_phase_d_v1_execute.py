from __future__ import annotations

import csv

from scripts import industry_realism_causal_headroom_phase_d_v1_execute as exec_d


def test_execute_freeze_hashes_match_expected():
    observed = exec_d.verify_frozen_hashes()
    assert observed["preregistration"] == exec_d.EXPECTED_HASHES["preregistration"][1]
    assert observed["eligible_universe"] == exec_d.EXPECTED_HASHES["eligible_universe"][1]


def test_execute_universe_counts_and_selected_regimes():
    states, branches = exec_d.load_universe()
    summary = exec_d.universe_integrity_summary(states, branches)
    assert summary["eligible_disagreement_states"] == 11328
    assert summary["unique_non_sbs_canonical_branches"] == 12169
    assert summary["total_terminal_continuations"] == 23497
    assert summary["contributing_faithful_windows"] == 44
    assert summary["workload_distribution"] == {
        "azure_2023_code": 697,
        "azure_2023_conv": 10169,
        "burstgpt": 462,
    }
    assert summary["axis_distribution"] == {
        "active_sequence_capacity": 466,
        "kv_capacity": 10862,
    }
    assert summary["selected_regimes"]["azure_2023_code"] == ["active_4", "active_8", "kv_16000"]
    assert summary["selected_regimes"]["azure_2023_conv"] == ["active_4", "kv_16000", "kv_8000"]
    assert summary["selected_regimes"]["burstgpt"] == ["active_16", "active_4", "active_8", "kv_16000", "kv_8000"]


def test_action_roundtrip_canonical_hash():
    _states, branches = exec_d.load_universe()
    for row in branches[:100]:
        action = exec_d.parse_action(row["candidate_canonical_action"])
        assert exec_d.canonical_action_id(exec_d.phase_a.canonical_action(action)) == row["candidate_canonical_action_id"]


def test_state_based_sharding_keeps_branches_with_state():
    states, branches = exec_d.load_universe()
    payloads = exec_d.build_shard_payloads(states, branches, 96)
    seen_states = sum(len(p["states"]) for p in payloads)
    seen_branches = sum(len(v) for p in payloads for v in p["branches_by_state"].values())
    assert seen_states == 11328
    assert seen_branches == 12169
    for p in payloads:
        state_ids = {s["state_id"] for s in p["states"]}
        assert set(p["branches_by_state"]).issubset(state_ids)


def test_completeness_gate_rejects_missing_without_analysis():
    states, branches = exec_d.load_universe()
    empty = exec_d.completeness_gate([], [])
    assert not empty["passed"]
    assert empty["expected_total_continuations"] == 23497
    assert empty["completed_total_continuations"] == 0


def test_no_selector_training_tokens_in_execution_source():
    src = open("scripts/industry_realism_causal_headroom_phase_d_v1_execute.py").read()
    forbidden = [".fit(", "predict_proba(", "HistGradientBoosting", "ExtraTrees", "RandomForest"]
    for token in forbidden:
        assert token not in src

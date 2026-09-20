from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts import industry_realism_causal_headroom_phase_d_v1 as phase_d


def test_phase_b_result_artifacts_reconcile_before_phase_d():
    summary = phase_d.phase_b_summary()
    assert summary["window_conditions"] == 1080
    assert summary["valid_disagreement_states"] == 11328
    assert summary["causal_labeling_executed"] is False
    assert summary["new_selector_training_executed"] is False
    assert summary["real_trace_structure_preserved"] is True


def test_regime_selection_is_mechanical_and_coverage_based():
    selected = phase_d.selected_regime_rows()
    keys = {(r["source_dataset"], r["axis"], r["condition_id"], r["stage_label"]) for r in selected}
    assert ("azure_2023_code", "kv_capacity", "kv_16000", "onset+sustained+high_valid_pressure") in keys
    assert ("azure_2023_code", "active_sequence_capacity", "active_8", "onset") in keys
    assert ("azure_2023_code", "active_sequence_capacity", "active_4", "sustained+high_valid_pressure") in keys
    assert ("azure_2023_conv", "kv_capacity", "kv_16000", "onset") in keys
    assert ("azure_2023_conv", "kv_capacity", "kv_8000", "sustained+high_valid_pressure") in keys
    assert ("azure_2023_conv", "active_sequence_capacity", "active_4", "onset+sustained+high_valid_pressure") in keys
    assert ("burstgpt", "kv_capacity", "kv_16000", "onset+sustained") in keys
    assert ("burstgpt", "kv_capacity", "kv_8000", "high_valid_pressure") in keys
    assert ("burstgpt", "active_sequence_capacity", "active_16", "onset") in keys
    assert ("burstgpt", "active_sequence_capacity", "active_8", "sustained") in keys
    assert ("burstgpt", "active_sequence_capacity", "active_4", "high_valid_pressure") in keys
    assert all(r["axis"] != "arrival_pressure" for r in selected)


def test_support_universe_regimes_sum_to_phase_b_valid_disagreements():
    support = phase_d.support_regime_rows()
    assert sum(r["phase_b_disagreement_states"] for r in support) == 11328
    assert all(not r["pressure_regime_class"].startswith("INVALID") for r in support)
    assert all(r["axis"] != "arrival_pressure" for r in support)


def test_sampling_or_exhaustive_decision_rule():
    summary = {
        "selected_phase_d_total_terminal_continuations": 100,
        "selected_phase_d_states": 40,
        "selected_phase_d_unique_non_sbs_branches": 60,
        "selected_phase_d_sbs_reference_branches": 40,
    }
    assert phase_d.selected_branch_plan(summary)["chosen_path"] == "EXHAUSTIVE_SELECTED_REGIME_LABELING"
    summary["selected_phase_d_total_terminal_continuations"] = phase_d.EXHAUSTIVE_MAX_TERMINAL_CONTINUATIONS + 1
    assert phase_d.selected_branch_plan(summary)["chosen_path"] == "DETERMINISTIC_COVERAGE_SAMPLING"


def test_metric_protocol_denominators_and_bootstrap_are_frozen():
    protocol = phase_d.metric_protocol()
    assert protocol["prevalence_population"].startswith("all SBS decision states")
    assert protocol["causal_population"].startswith("canonical disagreement states")
    assert protocol["statistical_protocol"]["cluster_unit"] == "faithful window"
    assert protocol["statistical_protocol"]["bootstrap_replicates"] == 2000
    assert protocol["statistical_protocol"]["bootstrap_seed"] == 20260920
    assert protocol["statistical_protocol"]["minimum_clusters_for_ci"] == 5
    assert protocol["zero_policy"]["epsilon"] is None


def test_phase_d_design_does_not_execute_outcome_or_selector_paths():
    src = inspect.getsource(phase_d)
    forbidden_calls = [
        "run_one_step_then_sbs_terminal(",
        "predict_proba(",
        ".fit(",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
    ]
    for token in forbidden_calls:
        assert token not in src


def test_preregistration_does_not_embed_self_hash_if_generated():
    path = Path("experiments/industry_realism_causal_headroom_phase_d_v1/PREREGISTRATION_V1.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text())
    assert "preregistration" not in obj["artifact_hashes_excluding_self"]
    assert obj["self_hash_record"].startswith("PREREGISTRATION_V1.json SHA-256 is recorded externally")

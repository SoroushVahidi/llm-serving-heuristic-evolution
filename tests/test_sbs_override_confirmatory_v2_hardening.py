from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_a = _load_script("sbs_override_fresh_id_stage_a_predict_v2", "sbs_override_fresh_id_stage_a_predict_v2.py")
stage_b = _load_script(
    "sbs_override_fresh_id_confirmatory_evaluate_v2",
    "sbs_override_fresh_id_confirmatory_evaluate_v2.py",
)


def test_restored_feature_runtime_order_matches_frozen_selector():
    selector = json.loads(
        (ROOT / "experiments/sbs_override_conservative_selector_dev_v2/run_v2/FINAL_CONFIRMATORY_SELECTOR_V2.json").read_text()
    )
    runtime = stage_a.runtime_feature_order()
    assert len([c for c in runtime if c.startswith("state__")]) == 95
    assert len([c for c in runtime if c.startswith("action__")]) == 70
    assert runtime == selector["feature_columns"]


def test_stage_a_structural_universe_counts_are_clean_and_outcome_blind():
    _states, _maps, _scenarios, summary = stage_a.load_structural_universe(
        stage_a.DEFAULT_STATE_MANIFEST,
        stage_a.DEFAULT_POLICY_MAP,
        stage_a.DEFAULT_SCENARIO_MANIFEST,
    )
    assert summary["clean_unique_disagreement_states"] == 2862
    assert summary["raw_duplicate_state_id_rows"] == 56
    assert summary["supported_scenarios"] == 78
    assert summary["selected_scenarios"] == 80
    assert summary["unique_non_sbs_canonical_branches"] == 6996
    assert summary["total_terminal_continuations"] == 9858


def test_stage_a_rejects_outcome_columns():
    with pytest.raises(ValueError, match="outcome_columns_forbidden"):
        stage_a.reject_outcome_columns(["state_id", "a_sbs_anwg"])


def test_stage_a_selector_hash_rejection(tmp_path):
    selector = tmp_path / "selector.json"
    bundle = tmp_path / "selector.joblib"
    selector.write_text("{}\n")
    bundle.write_bytes(b"not a joblib")
    with pytest.raises(ValueError, match="selector_json_hash_mismatch"):
        stage_a.load_selector(selector, bundle)


def test_stage_a_feature_order_rejection():
    selector = {"feature_columns": list(reversed(stage_a.runtime_feature_order()))}
    with pytest.raises(ValueError, match="feature_order_mismatch"):
        stage_a.assert_feature_order(selector)


def test_stage_a_prediction_validation_membership():
    states = pd.DataFrame(
        {
            "state_id": [f"s{i}" for i in range(2862)],
            "scenario_id": [f"sc{i % 78}" for i in range(2862)],
        }
    )
    maps = pd.DataFrame(
        {
            "state_id": [f"s{i}" for i in range(2862)],
            "candidate_canonical_action_full": [f"a{i}" for i in range(2862)],
        }
    )
    decisions = [
        {
            "state_id": f"s{i}",
            "scenario_id": f"sc{i % 78}",
            "override": 1 if i == 0 else 0,
            "abstain": 0 if i == 0 else 1,
            "selected_canonical_action_id": "x" if i == 0 else "SBS_ABSTAIN",
            "selected_canonical_action_full": "a0" if i == 0 else "",
        }
        for i in range(2862)
    ]
    stage_a.validate_predictions(decisions, states, maps)
    decisions[1]["override"] = 1
    decisions[1]["abstain"] = 0
    decisions[1]["selected_canonical_action_full"] = "not_in_support"
    with pytest.raises(ValueError, match="selected_action_not_in_support"):
        stage_a.validate_predictions(decisions, states, maps)


def test_stage_b_abstention_gain_zero_and_join_correctness(monkeypatch):
    monkeypatch.setattr(stage_b, "PRIMARY_DENOMINATOR", 3)
    predictions = pd.DataFrame(
        {
            "state_id": ["s0", "s1", "s2"],
            "scenario_id": ["a", "a", "b"],
            "override": [1, 0, 1],
            "abstain": [0, 1, 0],
            "selected_canonical_action_id": ["act0", "SBS_ABSTAIN", "act2"],
        }
    )
    labels = pd.DataFrame(
        {
            "state_id": ["s0", "s2"],
            "canonical_action_id": ["act0", "act2"],
            "a_sbs_anwg": [0.3, -0.1],
        }
    )
    joined = stage_b.join_predictions_to_labels(predictions, labels)
    assert joined.sort_values("state_id")["realized_gain"].tolist() == [0.3, 0.0, -0.1]
    assert stage_b.primary_mean_gain(joined) == pytest.approx((0.3 - 0.1) / 3.0)


def test_stage_b_bootstrap_clusters_by_scenario_and_is_seeded(monkeypatch):
    monkeypatch.setattr(stage_b, "CI_LOW_Q", 0.025)
    monkeypatch.setattr(stage_b, "CI_HIGH_Q", 0.975)
    joined = pd.DataFrame(
        {
            "scenario_id": ["a", "a", "b"],
            "realized_gain": [1.0, 3.0, -2.0],
        }
    )
    a = stage_b.scenario_cluster_bootstrap(joined, seed=7, replicates=100)
    b = stage_b.scenario_cluster_bootstrap(joined, seed=7, replicates=100)
    assert a == b
    assert a["replicates"] == 100


def test_stage_b_primary_verdict_strict_lower_bound():
    assert stage_b.primary_verdict(0.1, 0.0001) == "CONFIRMED_POSITIVE_GENERALIZATION"
    assert stage_b.primary_verdict(0.1, 0.0) == "POSITIVE_GENERALIZATION_NOT_CONFIRMED"
    assert stage_b.primary_verdict(-0.1, 0.1) == "POSITIVE_GENERALIZATION_NOT_CONFIRMED"


def test_stage_b_secondary_metrics_cannot_override_primary(monkeypatch):
    monkeypatch.setattr(stage_b, "PRIMARY_DENOMINATOR", 2)
    joined = pd.DataFrame(
        {
            "scenario_id": ["a", "b"],
            "override": [1, 1],
            "realized_gain": [10.0, -11.0],
        }
    )
    diagnostics = stage_b.secondary_diagnostics(joined)
    assert diagnostics["beneficial_overrides"] == 1
    assert stage_b.primary_verdict(stage_b.primary_mean_gain(joined), -1.0) == "POSITIVE_GENERALIZATION_NOT_CONFIRMED"


def test_stage_b_hash_mismatch_rejection(tmp_path):
    p = tmp_path / "pred.csv"
    p.write_text("state_id\ns0\n")
    with pytest.raises(ValueError, match="stage_a_prediction_hash_mismatch"):
        stage_b.load_stage_a_predictions(p, "bad")


def test_stage_b_refuses_existing_result_directory(tmp_path):
    existing = tmp_path / "result"
    existing.mkdir()
    args = SimpleNamespace(result_dir=str(existing))
    with pytest.raises(FileExistsError):
        stage_b.evaluate_once(args)


from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "sbs_override_conservative_selector_dev_v2.py"
spec = importlib.util.spec_from_file_location("sbs_override_conservative_selector_dev_v2", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _pred_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_id": ["s0", "s1", "s2", "s3", "s4"],
            "scenario_id": ["a", "a", "b", "b", "c"],
            "fold": [0, 0, 1, 1, 2],
            "canonical_action_id": ["a0", "a1", "a2", "a3", "a4"],
            module.v1.PRIMARY_TARGET: [0.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_advantage": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


def test_scenario_grouping_integrity_rejects_overlap():
    train = pd.DataFrame({"scenario_id": ["sc0", "sc1"], "state_id": ["s0", "s1"]})
    val = pd.DataFrame({"scenario_id": ["sc1"], "state_id": ["s2"]})
    with pytest.raises(ValueError, match="scenario_group_leakage"):
        module.assert_group_split_integrity(train, val)


def test_grouped_crossfit_predictions_are_out_of_fold(monkeypatch):
    df = pd.DataFrame(
        {
            "state_id": ["s0", "s1", "s2", "s3"],
            "scenario_id": ["sc0", "sc0", "sc1", "sc1"],
            "fold": [0, 0, 1, 1],
            "canonical_action_id": ["a0", "a1", "a2", "a3"],
            module.v1.PRIMARY_TARGET: [0.0, 0.1, -0.1, 0.2],
            "state__x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    seen_validation_folds = []

    def fake_fit_predict(config, train, pred_df, cols, strategy):
        assert set(train["scenario_id"]).isdisjoint(set(pred_df["scenario_id"]))
        seen_validation_folds.extend(sorted(pred_df["fold"].unique()))
        return object(), np.full(len(pred_df), 0.25)

    monkeypatch.setattr(module.v1, "fit_predict", fake_fit_predict)
    candidate = module.ModelCandidate("EXTRA_TREES", "STATE_EQUAL", {"family": "EXTRA_TREES"})
    out = module.grouped_crossfit_predictions_for_model(df, ["state__x"], candidate)
    assert len(out) == len(df)
    assert sorted(seen_validation_folds) == [0, 1]
    assert (out["predicted_advantage"] == 0.25).all()


def test_deterministic_candidate_selection_uses_stable_key():
    base = {
        "min_fold_mean_gain": 1.0,
        "mean_gain": 2.0,
        "negative_scenarios": 0,
        "harmful_overrides": 0,
        "override_rate": 0.1,
    }
    selected = module.select_final_candidate(
        [
            {**base, "stable_key": "b"},
            {**base, "stable_key": "a"},
        ]
    )
    assert selected["stable_key"] == "a"


def test_lexicographic_objective_order():
    records = [
        {
            "stable_key": "worse_min_better_mean",
            "min_fold_mean_gain": 0.0,
            "mean_gain": 99.0,
            "negative_scenarios": 0,
            "harmful_overrides": 0,
            "override_rate": 0.0,
        },
        {
            "stable_key": "better_min",
            "min_fold_mean_gain": 0.1,
            "mean_gain": 0.2,
            "negative_scenarios": 99,
            "harmful_overrides": 99,
            "override_rate": 1.0,
        },
    ]
    assert module.select_final_candidate(records)["stable_key"] == "better_min"
    records[0]["min_fold_mean_gain"] = 0.1
    records[0]["mean_gain"] = 0.3
    assert module.select_final_candidate(records)["stable_key"] == "worse_min_better_mean"
    records[1]["mean_gain"] = 0.3
    records[0]["negative_scenarios"] = 1
    records[1]["negative_scenarios"] = 0
    assert module.select_final_candidate(records)["stable_key"] == "better_min"


def test_q90_q95_residual_margin_correctness():
    margins = module.residual_margins_from_oof(_pred_df(), "predicted_advantage")
    assert margins["q90"] == pytest.approx(3.6)
    assert margins["q95"] == pytest.approx(3.8)


def test_threshold_direction_is_strictly_greater_than_tau():
    df = pd.DataFrame(
        {
            "state_id": ["s0", "s1"],
            "scenario_id": ["sc0", "sc1"],
            "fold": [0, 1],
            "canonical_action_id": ["a0", "a1"],
            module.v1.PRIMARY_TARGET: [1.0, 1.0],
            "predicted_advantage": [0.5, 0.5001],
        }
    )
    decisions = module.v1.state_decisions(df, "predicted_advantage", tau=0.5, margin=0.0)
    assert decisions.sort_values("state_id")["override"].tolist() == [0, 1]


def test_one_action_per_state_semantics():
    df = pd.DataFrame(
        {
            "state_id": ["s0", "s0"],
            "scenario_id": ["sc0", "sc0"],
            "fold": [0, 0],
            "canonical_action_id": ["low", "high"],
            module.v1.PRIMARY_TARGET: [-1.0, 2.0],
            "predicted_advantage": [0.1, 0.2],
        }
    )
    decisions = module.v1.state_decisions(df, "predicted_advantage", tau=0.0, margin=0.0)
    assert len(decisions) == 1
    assert decisions["selected_canonical_action_id"].iloc[0] == "high"
    assert decisions["realized_gain"].iloc[0] == 2.0


def test_sbs_abstention_semantics():
    df = pd.DataFrame(
        {
            "state_id": ["s0"],
            "scenario_id": ["sc0"],
            "fold": [0],
            "canonical_action_id": ["a0"],
            module.v1.PRIMARY_TARGET: [5.0],
            "predicted_advantage": [-0.1],
        }
    )
    decisions = module.v1.state_decisions(df, "predicted_advantage", tau=0.0, margin=0.0)
    assert decisions["selected_canonical_action_id"].iloc[0] == "SBS_ABSTAIN"
    assert decisions["realized_gain"].iloc[0] == 0.0


def test_outer_oof_exclusion_from_final_selection():
    with pytest.raises(ValueError, match="outer_oof_result_forbidden"):
        module.reject_forbidden_inputs(["/tmp/development_outer_oof_state_decisions.csv"])


def test_fresh_input_rejection():
    bad = Path("/tmp") / module.v1.FRESH_FORBIDDEN_SUBSTRING / "state_action_rows.csv"
    with pytest.raises(ValueError):
        module.reject_forbidden_inputs([bad])


def test_exact_feature_order_guard():
    module.assert_feature_order(["state__a", "action__b"], ["state__a", "action__b"])
    with pytest.raises(ValueError, match="feature_order_mismatch"):
        module.assert_feature_order(["action__b", "state__a"], ["state__a", "action__b"])


class DummyModel:
    def predict(self, x):
        return np.asarray(x[:, 0], dtype=float)


def test_serialization_reload_equivalence(tmp_path):
    selector = {
        "feature_columns": ["state__x", "action__y"],
    }
    selector_path = tmp_path / "selector.json"
    selector_path.write_text('{"feature_columns": ["state__x", "action__y"]}\n')
    joblib_path = tmp_path / "selector.joblib"
    joblib.dump(
        {
            "selector": selector,
            "kind": "single",
            "models": {"extra_trees": DummyModel()},
        },
        joblib_path,
    )
    df = pd.DataFrame({"state__x": [1.0, 2.0], "action__y": [0.0, 0.0]})
    result = module.reload_validation(selector_path, joblib_path, df)
    assert result == {"passed": True, "rows": 2}

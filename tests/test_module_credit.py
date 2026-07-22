from __future__ import annotations

import json

import numpy as np
import pytest

from llmserveopt.selector.module_credit import (
    ModuleCreditModel,
    ModuleGateConfig,
    build_intervention_dataset,
    build_pairwise_interaction_rows,
    evaluate_credit_predictions,
    evaluate_offline_synthesis_decisions,
    evaluate_topk_ranking,
    gate_candidates,
    load_intervention_artifacts,
    module_structural_features,
    synthetic_intervention_fixture,
    validate_no_target_leakage,
    validate_split_integrity,
)
from llmserveopt.selector.module_credit.dataset import ModuleInterventionDataError


def _rows():
    return build_intervention_dataset(synthetic_intervention_fixture())


def test_intervention_ingestion_jsonl(tmp_path):
    raw = synthetic_intervention_fixture()[:3]
    path = tmp_path / "module_interventions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in raw))
    loaded = load_intervention_artifacts(path)
    assert len(loaded) == 3
    rows = build_intervention_dataset(loaded)
    assert rows[0]["C_base"] == pytest.approx(rows[0]["intervention_reward"] - rows[0]["base_reward"])


def test_credit_target_construction_parent_and_envelope():
    row = _rows()[0]
    assert row["C_parent"] == pytest.approx(row["intervention_reward"] - max(row["base_reward"], row["donor_reward"]))
    assert row["C_env"] == pytest.approx(row["intervention_reward"] - row["library_best_reward"])


def test_split_integrity_rejects_state_crossing_splits():
    rows = _rows()
    leaked = [dict(r) for r in rows]
    leaked[0]["split"] = "TRAIN"
    leaked[1]["state_id"] = leaked[0]["state_id"]
    leaked[1]["split_group_key"] = leaked[0]["split_group_key"]
    leaked[1]["split"] = "ID_TEST"
    with pytest.raises(ValueError):
        validate_split_integrity(leaked)


def test_no_target_leakage_detection():
    rows = _rows()
    rows[0]["state_features"]["C_base"] = 1.0
    with pytest.raises(ModuleInterventionDataError):
        validate_no_target_leakage(rows)


def test_module_encoding_contains_ast_and_edf_flag():
    rows = _rows()
    model = ModuleCreditModel(name="m", encoding="suitability_augmented", n_estimators=5, random_state=1).fit(rows)
    assert any("donor_is_edf_regime_specific" in f for f in model.encoder.feature_names)
    encoded = model.encoder.transform(rows[:2])
    assert encoded.shape[0] == 2
    assert len(model.encoder.feature_names) > 0


def test_module_structural_features_for_none_module():
    feats = module_structural_features(None)
    assert feats["module_present"] == 0.0
    assert feats["module_ast_node_count"] == 0.0


@pytest.mark.parametrize("encoding", ["identity", "structural", "contextual", "suitability_augmented"])
def test_model_fit_predict_uncertainty(encoding):
    rows = _rows()
    model = ModuleCreditModel(name=encoding, encoding=encoding, n_estimators=20, random_state=2).fit(rows)
    mu = model.predict_mean(rows)
    u = model.predict_uncertainty(rows)
    assert mu.shape == (len(rows),)
    assert u.shape == (len(rows),)
    assert (u >= 0.0).all()
    assert np.allclose(model.predict_score(rows, lambda_m=0.5), mu - 0.5 * u)


def test_prediction_and_topk_evaluation():
    rows = _rows()
    train = [r for r in rows if r["split"] in ("TRAIN", "VALIDATION")]
    test = [r for r in rows if r["split"] == "ID_TEST"]
    model = ModuleCreditModel(name="ctx", encoding="contextual", n_estimators=30, random_state=3).fit(train)
    pred = evaluate_credit_predictions(model, test)
    topk = evaluate_topk_ranking(model, test, ks=(1, 3))
    assert pred["n_rows"] == len(test)
    assert "top_1" in topk and "positive_transfer_precision" in topk["top_1"]


def test_edf_remains_explicit_donor_candidate():
    rows = _rows()
    edf_rows = [r for r in rows if r["donor_policy"] == "edf"]
    assert edf_rows
    model = ModuleCreditModel(name="edf", encoding="suitability_augmented", n_estimators=20, random_state=4).fit(rows)
    scores = model.predict_score(edf_rows)
    assert scores.shape == (len(edf_rows),)


def test_synthesis_gate_respects_compatibility_and_uncertainty():
    rows = _rows()
    model = ModuleCreditModel(name="gate", encoding="contextual", n_estimators=20, random_state=5).fit(rows)
    gated = gate_candidates(model, rows[:8], ModuleGateConfig(max_uncertainty=10.0, min_conservative_C_base=-1.0))
    assert len(gated) == 8
    assert all("S_C" in r and "passes" in r for r in gated)
    incompatible = [dict(rows[0])]
    incompatible[0]["compatibility_metadata"] = {"compatible": 0.0}
    assert not gate_candidates(model, incompatible, ModuleGateConfig(max_uncertainty=10.0, min_conservative_C_base=-1.0))[0]["passes"]


def test_offline_synthesis_decision_evaluation():
    rows = _rows()
    model = ModuleCreditModel(name="offline", encoding="suitability_augmented", n_estimators=30, random_state=6).fit(rows)
    ev = evaluate_offline_synthesis_decisions(rows, model, seed=0)
    assert {"random_compatible", "highest_whole_policy_suitability_donor", "structural_nearest_proxy", "module_credit_model"} <= set(ev)
    assert ev["module_credit_model"]["n_selected"] > 0


def test_pairwise_interaction_target():
    pair_rows = [{
        "state_id": "s",
        "base_policy": "fifo",
        "donor_policy_a": "edf",
        "donor_policy_b": "scorpio_style_slo_guard",
        "module_type_a": "priority_rule",
        "module_type_b": "admission_rule",
        "single_reward_a": 0.6,
        "single_reward_b": 0.7,
        "both_reward": 0.9,
        "base_reward": 0.5,
    }]
    out = build_pairwise_interaction_rows(pair_rows)
    assert out[0]["interaction"] == pytest.approx(0.1)
    assert out[0]["interaction_class"] == "synergistic"

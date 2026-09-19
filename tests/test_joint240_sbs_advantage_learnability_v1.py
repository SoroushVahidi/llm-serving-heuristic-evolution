from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llmserveopt.analysis import joint240_sbs_advantage_learnability_v1 as learn


def _rows() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for scen in range(3):
            scenario_id = f"s{fold}_{scen}"
            for state in range(2):
                state_id = f"{scenario_id}::state{state}"
                for action in range(2):
                    rows.append(
                        {
                            "state_id": state_id,
                            "scenario_id": scenario_id,
                            "fold": fold,
                            "canonical_action_id": f"a{action}",
                            "is_sbs_action": 0,
                            "a_sbs_anwg": 0.01 * (action == 1) - 0.002 * fold,
                            "state__x": float(fold + scen),
                            "state__y": float(state),
                            "action__diff": float(action),
                        }
                    )
    return pd.DataFrame(rows)


def test_leakage_guard_rejects_ids_and_outcomes():
    assert learn.leakage_columns(["state__x", "action__diff"]) == []
    bad = learn.leakage_columns(["state_id", "state__future_arrivals", "action__q_sbs_hint"])
    assert set(bad) == {"state_id", "state__future_arrivals", "action__q_sbs_hint"}


def test_outer_split_scenario_and_state_exclusive():
    audit = learn.outer_split_audit(_rows())
    assert audit["scenario_assigned_to_multiple_folds"] == 0
    assert all(x["scenario_overlap"] == 0 for x in audit["outer_folds"])
    assert all(x["state_overlap"] == 0 for x in audit["outer_folds"])


def test_state_equal_weights_sum_to_one_per_state():
    rows = _rows()
    weights = learn.state_equal_weights(rows)
    rows = rows.assign(w=weights)
    sums = rows.groupby("state_id")["w"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_selector_abstention_and_argmax():
    rows = pd.DataFrame(
        [
            {"state_id": "s1", "scenario_id": "c1", "fold": 0, "canonical_action_id": "a", "a_sbs_anwg": 0.2, "pred": 0.01},
            {"state_id": "s1", "scenario_id": "c1", "fold": 0, "canonical_action_id": "b", "a_sbs_anwg": -0.5, "pred": 0.02},
            {"state_id": "s2", "scenario_id": "c1", "fold": 0, "canonical_action_id": "c", "a_sbs_anwg": 0.4, "pred": -0.01},
        ]
    )
    selected = learn.selector_decisions(rows, "pred", tau=0.0)
    s1 = selected[selected["state_id"] == "s1"].iloc[0]
    s2 = selected[selected["state_id"] == "s2"].iloc[0]
    assert s1["selected_canonical_action_id"] == "b"
    assert s1["realized_gain"] == -0.5
    assert s2["selected_canonical_action_id"] == "SBS_ABSTAIN"
    assert s2["realized_gain"] == 0.0


def test_threshold_tuning_uses_crossfit_predictions_table_only():
    rows = pd.DataFrame(
        [
            {"state_id": "s1", "scenario_id": "c1", "fold": 0, "canonical_action_id": "a", "a_sbs_anwg": 0.1, "pred": 0.02},
            {"state_id": "s2", "scenario_id": "c2", "fold": 1, "canonical_action_id": "b", "a_sbs_anwg": -0.1, "pred": 0.001},
        ]
    )
    tau, summary = learn.tune_threshold(rows.rename(columns={"pred": "predicted_advantage"}), "predicted_advantage")
    assert tau in learn.THRESHOLDS
    assert summary["selected"]["mean_realized_gain"] >= 0.0


def test_always_sbs_and_oracle_gap_math():
    rows = pd.DataFrame(
        [
            {"state_id": "s1", "scenario_id": "c1", "fold": 0, "canonical_action_id": "a", "a_sbs_anwg": 0.2, "pred": -1.0},
            {"state_id": "s1", "scenario_id": "c1", "fold": 0, "canonical_action_id": "b", "a_sbs_anwg": -0.1, "pred": -2.0},
            {"state_id": "s2", "scenario_id": "c2", "fold": 1, "canonical_action_id": "c", "a_sbs_anwg": -0.3, "pred": -1.0},
        ]
    )
    decisions = learn.selector_decisions(rows, "pred", tau=0.0)
    metrics = learn.selector_metrics(decisions)
    assert metrics["total_realized_gain"] == 0.0
    assert metrics["oracle_total_gain"] == 0.2
    assert metrics["gap_closure"] == 0.0


def test_scenario_bootstrap_resamples_scenarios_not_rows():
    decisions = pd.DataFrame(
        [
            {"state_id": "s1", "scenario_id": "c1", "fold": 0, "realized_gain": 0.1, "oracle_gain": 0.2, "regret_to_oracle": 0.1, "override": 1, "beneficial_override": 1, "harmful_override": 0, "zero_effect_override": 0},
            {"state_id": "s2", "scenario_id": "c1", "fold": 0, "realized_gain": 0.0, "oracle_gain": 0.1, "regret_to_oracle": 0.1, "override": 0, "beneficial_override": 0, "harmful_override": 0, "zero_effect_override": 0},
            {"state_id": "s3", "scenario_id": "c2", "fold": 1, "realized_gain": -0.1, "oracle_gain": 0.0, "regret_to_oracle": 0.1, "override": 1, "beneficial_override": 0, "harmful_override": 1, "zero_effect_override": 0},
        ]
    )
    dist, summary = learn.scenario_bootstrap(decisions, replicates=25, seed=1)
    assert len(dist) == 25
    assert "mean_realized_gain" in summary


def test_input_checksum_validation(tmp_path: Path, monkeypatch):
    rows = tmp_path / "rows.csv"
    maps = tmp_path / "maps.csv"
    rows.write_text("a\n1\n")
    maps.write_text("b\n2\n")
    monkeypatch.setattr(learn, "EXPECTED_ROWS_SHA256", learn.sha256_file(rows))
    monkeypatch.setattr(learn, "EXPECTED_MAP_SHA256", learn.sha256_file(maps))
    checks = learn.validate_input_checksums(rows, maps)
    assert checks["rows_sha256_matches"]
    assert checks["map_sha256_matches"]


def test_loader_schema_and_feature_sets_on_synthetic_table(tmp_path: Path, monkeypatch):
    rows = _rows()
    state_pad = pd.DataFrame({f"state__pad{i}": float(i) for i in range(93)}, index=rows.index)
    action_pad = pd.DataFrame({f"action__pad{i}": float(i) for i in range(69)}, index=rows.index)
    rows = pd.concat([rows, state_pad, action_pad], axis=1)
    sbs = rows.drop_duplicates("state_id").copy()
    sbs["canonical_action_id"] = "sbs"
    sbs["is_sbs_action"] = 1
    sbs["a_sbs_anwg"] = 0.0
    all_rows = pd.concat([rows, sbs], ignore_index=True)
    maps = pd.DataFrame(
        [
            {"state_id": sid, "policy_id": f"p{i}", "canonical_action_id": "a0" if i else "sbs"}
            for sid in all_rows["state_id"].unique()
            for i in range(6)
        ]
    )
    rows_path = tmp_path / "rows.csv"
    map_path = tmp_path / "map.csv"
    all_rows.to_csv(rows_path, index=False)
    maps.to_csv(map_path, index=False)
    monkeypatch.setattr(learn, "EXPECTED_ROWS_SHA256", learn.sha256_file(rows_path))
    monkeypatch.setattr(learn, "EXPECTED_MAP_SHA256", learn.sha256_file(map_path))
    monkeypatch.setattr(learn, "EXPECTED_NON_SBS_ROWS", len(rows))
    monkeypatch.setattr(learn, "EXPECTED_STATES", rows["state_id"].nunique())
    monkeypatch.setattr(learn, "EXPECTED_SCENARIOS", rows["scenario_id"].nunique())
    loaded, _, integrity = learn.load_learning_table(rows_path, map_path)
    assert integrity["non_sbs_rows"] == len(rows)
    assert len(learn.learnable_columns(loaded, "STATE_CORE_V1")) == 95
    assert len(learn.learnable_columns(loaded, "STATE_ACTION_V1")) == 165

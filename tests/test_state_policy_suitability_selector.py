from __future__ import annotations

import pytest

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES
from llmserveopt.selector.suitability.dataset import genome_table, group_by_state
from llmserveopt.selector.suitability.encoders import structural_features
from llmserveopt.selector.suitability.models import JointRewardModel
from llmserveopt.selector.suitability.selector import (
    build_delta_rows,
    evaluate_selection,
    held_out_policy_split,
    joint_select,
    load_policy_families,
    oracle_best,
    top2_margin,
)


def _rows_all_27(n_states=6, seed=0):
    import random

    rng = random.Random(seed)
    genomes = genome_table(POLICY_LIBRARY_V2_NAMES)
    rows = []
    for widx in range(n_states):
        state_features = {"feat_load": rng.uniform(0, 1)}
        for i, policy in enumerate(POLICY_LIBRARY_V2_NAMES):
            reward = (rng.random() * 0.05) + (i % 5) * 0.15
            rows.append({
                "state_id": f"s{widx}", "state_features": state_features,
                "policy_name": policy, "reward_anwg": min(1.0, reward),
                "policy_representation": structural_features(genomes[policy]),
            })
    return rows


def test_joint_select_scores_and_picks_among_all_27_policies():
    pytest.importorskip("sklearn")
    rows = _rows_all_27()
    model = JointRewardModel(name="m", encoding="hybrid", all_policies=POLICY_LIBRARY_V2_NAMES, n_estimators=20, max_depth=4, random_state=0).fit(rows)
    rbs = group_by_state(rows)
    selections = joint_select(model, rbs, lam=0.5)
    assert len(selections) == len(rbs)
    assert all(p in POLICY_LIBRARY_V2_NAMES for p in selections.values())


def test_evaluate_selection_margin_thresholds_present_and_monotonic_state_count():
    rows = _rows_all_27(n_states=8)
    rbs = group_by_state(rows)
    oracle_selections = {sid: oracle_best({r["policy_name"]: r["reward_anwg"] for r in rs})[0] for sid, rs in rbs.items()}
    result = evaluate_selection(rbs, oracle_selections, best_fixed_policy="fifo")
    for key in ("overall", "margin_gt_0.0", "margin_gt_0.001", "margin_gt_0.005", "margin_gt_0.01"):
        assert key in result
    # Perfect (oracle) selection must have zero regret and full match accuracy.
    assert result["overall"]["mean_regret_to_oracle"] == 0.0
    assert result["overall"]["policy_match_accuracy"] == 1.0
    # Stricter margin thresholds can only keep a subset of states.
    counts = [result[f"margin_gt_{t}"]["n_states"] for t in (0.0, 0.001, 0.005, 0.01)]
    assert counts == sorted(counts, reverse=True)


def test_top2_margin_and_oracle_best():
    rewards = {"a": 0.9, "b": 0.5, "c": 0.9}
    assert top2_margin(rewards) == 0.0  # tie for best
    policy, value = oracle_best(rewards)
    assert value == 0.9
    assert policy in ("a", "c")


def test_build_delta_rows_target_construction():
    rows = _rows_all_27(n_states=5)
    rbs = group_by_state(rows)
    delta_rows = build_delta_rows(rbs, policy_a="fifo", policy_b="edf")
    assert len(delta_rows) == 5
    for row in delta_rows:
        rewards = {r["policy_name"]: r["reward_anwg"] for r in rbs[row["state_id"]]}
        assert row["delta"] == pytest.approx(rewards["fifo"] - rewards["edf"])
        assert "state_features" in row and "margin" in row


def test_held_out_policy_split_integrity_no_leakage():
    rows = _rows_all_27(n_states=4)
    train, test = held_out_policy_split(rows, ["fifo"])
    assert all(r["policy_name"] != "fifo" for r in train)
    assert all(r["policy_name"] == "fifo" for r in test)
    # Every state still represented in train via its other 26 policies.
    train_states = {r["state_id"] for r in train}
    test_states = {r["state_id"] for r in test}
    assert train_states == test_states
    assert len(train) + len(test) == len(rows)


def test_held_out_policy_split_multiple_policies():
    rows = _rows_all_27(n_states=3)
    family = load_policy_families("kv_memory_pressure")
    assert len(family) >= 2
    train, test = held_out_policy_split(rows, family)
    assert all(r["policy_name"] not in family for r in train)
    assert all(r["policy_name"] in family for r in test)


def test_load_policy_families_is_from_documented_matrix_not_invented():
    family = load_policy_families("slo_deadline_handling")
    assert "edf" in family
    assert set(family).issubset(set(POLICY_LIBRARY_V2_NAMES))
    assert len(family) < len(POLICY_LIBRARY_V2_NAMES)

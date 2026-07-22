from __future__ import annotations

import pytest

from llmserveopt.policies.registry import ORACLE_POLICY_NAMES, POLICY_LIBRARY_V2_NAMES
from llmserveopt.selector.suitability.dataset import (
    build_long_format_rows,
    genome_table,
    group_by_state,
    rows_with_reward,
)


def _tiny_wide_fixture():
    window_rows = [
        {"window_idx": 0, "split": "TRAIN", "feat_mean_prompt_tokens": 100.0, "feat_arrival_rate_est": 5.0},
        {"window_idx": 1, "split": "TEST", "feat_mean_prompt_tokens": 200.0, "feat_arrival_rate_est": 8.0},
    ]
    policies = ["fifo", "edf", "weighted_shortest_processing"]
    policy_rows = []
    for widx in (0, 1):
        for i, policy in enumerate(policies):
            policy_rows.append({
                "window_idx": widx, "policy_name": policy,
                "metric_arrival_normalized_weighted_goodput": 0.5 + 0.1 * i + 0.01 * widx,
                "metric_completion_fraction": 0.9,
                "metric_weighted_goodput": 0.8,
            })
    return window_rows, policy_rows, policies


def test_build_long_format_rows_schema_and_coverage():
    window_rows, policy_rows, policies = _tiny_wide_fixture()
    rows = build_long_format_rows(
        window_rows, policy_rows, deployable_policies=policies,
        source="unit_test", trace_family="fixture", seed=7,
    )
    assert len(rows) == len(window_rows) * len(policies)
    expected_keys = {
        "state_id", "state_features", "policy_name", "policy_hash", "policy_representation",
        "reward_anwg", "completion_fraction", "completed_request_quality",
        "source", "trace_family", "temporal_block", "split", "seed",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys
        assert row["policy_name"] in policies


def test_build_long_format_rows_covers_all_27_deployable_policies():
    window_rows = [{"window_idx": 0, "split": "TRAIN", "feat_x": 1.0}]
    policy_rows = [
        {"window_idx": 0, "policy_name": name, "metric_arrival_normalized_weighted_goodput": 0.5,
         "metric_completion_fraction": 1.0, "metric_weighted_goodput": 1.0}
        for name in POLICY_LIBRARY_V2_NAMES
    ]
    rows = build_long_format_rows(
        window_rows, policy_rows, deployable_policies=POLICY_LIBRARY_V2_NAMES,
        source="unit_test", trace_family="fixture", seed=1,
    )
    assert len(rows) == 27
    assert {r["policy_name"] for r in rows} == set(POLICY_LIBRARY_V2_NAMES)


def test_oracle_policies_are_rejected_from_deployable_set():
    window_rows = [{"window_idx": 0, "split": "TRAIN", "feat_x": 1.0}]
    policy_rows = [{"window_idx": 0, "policy_name": "oracle_srtf", "metric_arrival_normalized_weighted_goodput": 1.0}]
    with pytest.raises(ValueError, match="oracle"):
        build_long_format_rows(
            window_rows, policy_rows,
            deployable_policies=list(POLICY_LIBRARY_V2_NAMES) + list(ORACLE_POLICY_NAMES),
            source="unit_test", trace_family="fixture", seed=1,
        )


def test_split_inherited_exactly_from_source_no_recomputation():
    window_rows, policy_rows, policies = _tiny_wide_fixture()
    rows = build_long_format_rows(
        window_rows, policy_rows, deployable_policies=policies,
        source="unit_test", trace_family="fixture", seed=1,
    )
    splits_by_window = {0: "TRAIN", 1: "TEST"}
    for row in rows:
        widx = int(row["state_id"].rsplit("__w", 1)[-1])
        assert row["split"] == splits_by_window[widx]


def test_no_cross_split_duplication_per_state():
    window_rows, policy_rows, policies = _tiny_wide_fixture()
    rows = build_long_format_rows(
        window_rows, policy_rows, deployable_policies=policies,
        source="unit_test", trace_family="fixture", seed=1,
    )
    splits_seen = {}
    for row in rows:
        splits_seen.setdefault(row["state_id"], set()).add(row["split"])
    assert all(len(v) == 1 for v in splits_seen.values())


def test_deterministic_output_ordering():
    window_rows, policy_rows, policies = _tiny_wide_fixture()
    rows_a = build_long_format_rows(window_rows, policy_rows, deployable_policies=policies, source="s", trace_family="t", seed=1)
    rows_b = build_long_format_rows(list(reversed(window_rows)), list(reversed(policy_rows)), deployable_policies=policies, source="s", trace_family="t", seed=1)
    assert rows_a == rows_b
    ids = [(r["state_id"], r["policy_name"]) for r in rows_a]
    assert ids == sorted(ids)


def test_policy_hash_is_stable_and_distinct():
    genomes = genome_table(POLICY_LIBRARY_V2_NAMES)
    hashes = {name: g.stable_hash() for name, g in genomes.items()}
    # Recomputing must give byte-identical hashes.
    genomes2 = genome_table(POLICY_LIBRARY_V2_NAMES)
    for name in POLICY_LIBRARY_V2_NAMES:
        assert genomes2[name].stable_hash() == hashes[name]
    # Every policy gets a distinct hash (name/metadata differ even for
    # structurally-identical UNSUPPORTED placeholders).
    assert len(set(hashes.values())) == len(POLICY_LIBRARY_V2_NAMES)


def test_causal_feature_validator_rejects_leaky_state_features():
    window_rows = [{"window_idx": 0, "split": "TRAIN", "feat_mean_prompt_tokens": 1.0, "reward_fifo": 0.9}]
    policy_rows = [{"window_idx": 0, "policy_name": "fifo", "metric_arrival_normalized_weighted_goodput": 0.5}]
    # "reward_fifo" is not a feat_* column so it's silently excluded by the
    # feat_ prefix filter -- verify the state_features dict never contains it.
    rows = build_long_format_rows(window_rows, policy_rows, deployable_policies=["fifo"], source="s", trace_family="t", seed=1)
    assert "reward_fifo" not in rows[0]["state_features"]
    assert set(rows[0]["state_features"].keys()) == {"feat_mean_prompt_tokens"}


def test_rows_with_reward_filters_missing():
    rows = [
        {"reward_anwg": 0.5, "state_id": "a"},
        {"reward_anwg": None, "state_id": "b"},
    ]
    assert len(rows_with_reward(rows)) == 1


def test_group_by_state_groups_correctly():
    window_rows, policy_rows, policies = _tiny_wide_fixture()
    rows = build_long_format_rows(window_rows, policy_rows, deployable_policies=policies, source="s", trace_family="t", seed=1)
    grouped = group_by_state(rows)
    assert len(grouped) == 2
    assert all(len(v) == len(policies) for v in grouped.values())

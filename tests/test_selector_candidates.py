"""Tests that selector candidates equal deployable policy names minus oracle names."""
import pytest

from llmserveopt.policies.registry import BASELINE_NAMES, ORACLE_POLICY_NAMES
from llmserveopt.selector.candidates import SELECTOR_CANDIDATES, SELECTOR_CANDIDATE_COUNT


def test_candidates_exclude_oracle():
    for oname in ORACLE_POLICY_NAMES:
        assert oname not in SELECTOR_CANDIDATES, (
            f"Oracle policy '{oname}' must not appear in SELECTOR_CANDIDATES"
        )


def test_candidates_equal_baseline_minus_oracle():
    expected = [n for n in BASELINE_NAMES if n not in set(ORACLE_POLICY_NAMES)]
    assert SELECTOR_CANDIDATES == expected


def test_candidate_count():
    assert SELECTOR_CANDIDATE_COUNT == len(SELECTOR_CANDIDATES)
    assert SELECTOR_CANDIDATE_COUNT > 0


def test_expected_policies_present():
    expected = [
        "fifo", "edf", "shortest_output_first", "shortest_prompt_first",
        "greedy_token_fill", "least_loaded", "multi_bin_batching", "random_feasible",
        "orca_style", "vllm_style_token_budget", "sarathi_style", "splitfuse_style",
        "slo_slack_score", "weighted_shortest_processing", "first_fit", "best_fit",
        # Phase 2A.3B: hardened deadline/laxity and service-time baselines
        "least_laxity_first", "estimated_service_time_first",
    ]
    for name in expected:
        assert name in SELECTOR_CANDIDATES, f"Expected '{name}' in SELECTOR_CANDIDATES"
    assert SELECTOR_CANDIDATE_COUNT >= 18  # 18 in Phase 2A.4; 19+ after Phase 2B.5

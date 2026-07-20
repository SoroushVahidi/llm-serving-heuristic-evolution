"""
Invariant tests for the Selector v2 candidate-policy source of truth.

These guard the three-way distinction established in
src/llmserveopt/selector/dataset_v2/candidates.py's module docstring:
  - BASELINE_NAMES (20): the full historical/internal policy portfolio.
  - MONOLITHIC_DIAGNOSTIC_POLICY_POOL / monolithic_candidate_policies() (14):
    a broader diagnostic/exploration pool used by pre-Option-B pilots.
  - SELECTOR_V2_OPTION_B_POLICIES (8): the current, approved, canonical
    Selector v2 trainable action space.

Test invariants (counts, subset relationships, exclusions), not prose --
these should stay green even as individual policy names evolve, as long as
the underlying design invariants hold.
"""
from llmserveopt.policies.external_baselines_registry import EXTERNAL_BASELINE_NAMES
from llmserveopt.policies.registry import BASELINE_NAMES
from llmserveopt.selector.dataset_v2.calibrated_targeted_pilot import (
    CANDIDATE_POLICIES as PILOT_CANDIDATE_POLICIES,
)
from llmserveopt.selector.dataset_v2.candidates import (
    MONOLITHIC_DIAGNOSTIC_POLICY_POOL,
    SELECTOR_V2_OPTION_B_POLICIES,
    monolithic_candidate_policies,
)

EXPECTED_OPTION_B_POLICIES = frozenset(
    {
        "fifo",
        "edf",
        "scorpio_style_slo_guard",
        "admission_control",
        "weighted_shortest_processing",
        "estimated_service_time_first",
        "best_fit",
        "multi_bin_batching",
    }
)


def test_historical_registry_has_20_policies():
    assert len(BASELINE_NAMES) == 20


def test_external_baseline_registry_has_6_baselines():
    assert len(EXTERNAL_BASELINE_NAMES) == 6


def test_option_b_is_exactly_the_approved_8_policies():
    assert set(SELECTOR_V2_OPTION_B_POLICIES) == EXPECTED_OPTION_B_POLICIES
    assert len(SELECTOR_V2_OPTION_B_POLICIES) == 8
    assert len(set(SELECTOR_V2_OPTION_B_POLICIES)) == 8, "duplicate entries in Option B set"


def test_option_b_contains_no_faithful_external_baseline():
    assert set(SELECTOR_V2_OPTION_B_POLICIES).isdisjoint(EXTERNAL_BASELINE_NAMES)


def test_option_b_is_a_strict_subset_of_historical_registry():
    assert set(SELECTOR_V2_OPTION_B_POLICIES).issubset(set(BASELINE_NAMES))


def test_option_b_is_a_strict_subset_of_the_diagnostic_pool():
    # Every current Option B policy also appears in the broader diagnostic
    # pool -- Option B was carved out of that pool, not defined independently.
    assert set(SELECTOR_V2_OPTION_B_POLICIES).issubset(set(MONOLITHIC_DIAGNOSTIC_POLICY_POOL))
    assert set(SELECTOR_V2_OPTION_B_POLICIES) != set(MONOLITHIC_DIAGNOSTIC_POLICY_POOL), (
        "Option B should be strictly narrower than the diagnostic pool "
        "(orca_style, slo_slack_score, shortest_output_first are excluded)"
    )


def test_diagnostic_pool_unchanged_at_11_historical_plus_3_external():
    assert len(MONOLITHIC_DIAGNOSTIC_POLICY_POOL) == 11
    assert len(monolithic_candidate_policies()) == 14


def test_calibrated_pilot_imports_the_canonical_option_b_constant():
    # Identity check (not just equality) -- the pilot must import the
    # constant, not maintain its own duplicate literal.
    assert PILOT_CANDIDATE_POLICIES is SELECTOR_V2_OPTION_B_POLICIES

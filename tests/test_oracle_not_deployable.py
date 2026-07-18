"""
Tests: oracle_srtf is non-deployable and cannot access online policy paths.

Key invariants:
1. oracle_srtf is not in any online-policy list (BASELINE_NAMES, SELECTOR_CANDIDATE_NAMES).
2. make_policy("oracle_srtf") raises KeyError.
3. ObservableRequest (the only view given to online policies) does not expose
   actual_output_tokens — it only has predicted_output_tokens.
4. Running oracle without an actual_output_map raises TypeError (no default).
5. all_baseline_policies() does not include oracle_srtf.
"""
import pytest

from llmserveopt.core.types import ObservableRequest, Request
from llmserveopt.policies.oracle import OracleShortestJobFirstPolicy
from llmserveopt.policies.registry import (
    BASELINE_NAMES,
    ORACLE_POLICY_NAMES,
    SELECTOR_CANDIDATE_NAMES,
    all_baseline_policies,
    make_oracle_policy,
    make_policy,
)


class TestOracleNotInOnlineRegistry:
    def test_oracle_not_in_baseline_names(self):
        assert "oracle_srtf" not in BASELINE_NAMES

    def test_oracle_not_in_selector_candidate_names(self):
        assert "oracle_srtf" not in SELECTOR_CANDIDATE_NAMES

    def test_all_baseline_policies_excludes_oracle(self):
        policies = all_baseline_policies()
        names = [p.name for p in policies]
        assert "oracle_srtf" not in names

    def test_make_policy_rejects_oracle_name(self):
        with pytest.raises(KeyError):
            make_policy("oracle_srtf")

    def test_make_oracle_rejects_online_policy_name(self):
        req = Request(
            request_id=0, arrival_time=0.0, prompt_tokens=10,
            predicted_output_tokens=5, actual_output_tokens=5,
            slo_deadline=10.0, priority=1.0, class_id="medium",
        )
        with pytest.raises(ValueError, match="not an oracle policy"):
            make_oracle_policy("fifo", [req])

    def test_make_oracle_rejects_serving_style_name(self):
        req = Request(
            request_id=0, arrival_time=0.0, prompt_tokens=10,
            predicted_output_tokens=5, actual_output_tokens=5,
            slo_deadline=10.0, priority=1.0, class_id="medium",
        )
        with pytest.raises(ValueError, match="not an oracle policy"):
            make_oracle_policy("sarathi_style", [req])


class TestOracleRequiresActualOutputMap:
    def test_oracle_construction_requires_actual_output_map(self):
        # No default for actual_output_map — must be provided explicitly
        with pytest.raises(TypeError):
            OracleShortestJobFirstPolicy()  # type: ignore[call-arg]

    def test_oracle_construction_with_empty_map_still_warns(self):
        with pytest.warns(UserWarning, match="NOT a deployable policy"):
            OracleShortestJobFirstPolicy(actual_output_map={})


class TestObservableRequestHidesActualOutput:
    def test_observable_request_has_no_actual_output_tokens(self):
        r = Request(
            request_id=0, arrival_time=0.0, prompt_tokens=10,
            predicted_output_tokens=5, actual_output_tokens=99,
            slo_deadline=10.0, priority=1.0, class_id="medium",
        )
        obs = ObservableRequest.from_request(r)
        # ObservableRequest must NOT expose actual_output_tokens
        assert not hasattr(obs, "actual_output_tokens"), (
            "ObservableRequest must not expose actual_output_tokens to online policies"
        )
        # It only has predicted_output_tokens
        assert obs.predicted_output_tokens == 5

    def test_observable_request_cannot_infer_actual_from_predicted(self):
        # Even if prediction == actual, the field must not be present on observable view
        r = Request(
            request_id=1, arrival_time=0.0, prompt_tokens=10,
            predicted_output_tokens=7, actual_output_tokens=7,
            slo_deadline=10.0, priority=1.0, class_id="medium",
        )
        obs = ObservableRequest.from_request(r)
        assert not hasattr(obs, "actual_output_tokens")


class TestOraclePolicyNames:
    def test_oracle_policy_names_is_nonempty(self):
        assert len(ORACLE_POLICY_NAMES) >= 1

    def test_baseline_and_oracle_names_disjoint(self):
        assert set(BASELINE_NAMES).isdisjoint(set(ORACLE_POLICY_NAMES))

    def test_selector_and_oracle_names_disjoint(self):
        assert set(SELECTOR_CANDIDATE_NAMES).isdisjoint(set(ORACLE_POLICY_NAMES))

"""
Tests: oracle_srtf wiring — runs correctly when explicitly enabled with actual outputs.
"""
import math
import warnings
import pytest

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.oracle import OracleShortestJobFirstPolicy, build_oracle
from llmserveopt.policies.registry import (
    ORACLE_POLICY_NAMES,
    BASELINE_NAMES,
    SELECTOR_CANDIDATE_NAMES,
    make_oracle_policy,
    make_policy,
)
from llmserveopt.evaluation.run_policy import run_policy


def _make_requests():
    return [
        Request(
            request_id=i,
            arrival_time=float(i) * 0.1,
            prompt_tokens=10,
            predicted_output_tokens=5,
            actual_output_tokens=5 + i % 3,
            slo_deadline=100.0,
            priority=1.0,
            class_id="medium",
        )
        for i in range(10)
    ]


def _gpu():
    return [GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=64, max_kv_tokens=1024)]


class TestOracleRegistry:
    def test_oracle_srtf_in_oracle_names(self):
        assert "oracle_srtf" in ORACLE_POLICY_NAMES

    def test_oracle_srtf_not_in_baseline_names(self):
        assert "oracle_srtf" not in BASELINE_NAMES

    def test_oracle_srtf_not_in_selector_candidates(self):
        assert "oracle_srtf" not in SELECTOR_CANDIDATE_NAMES

    def test_selector_candidate_names_equals_baseline_names(self):
        assert set(SELECTOR_CANDIDATE_NAMES) == set(BASELINE_NAMES)


class TestMakeOraclePolicy:
    def test_make_oracle_returns_oracle_instance(self):
        requests = _make_requests()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            oracle = make_oracle_policy("oracle_srtf", requests)
        assert isinstance(oracle, OracleShortestJobFirstPolicy)

    def test_make_oracle_wrong_name_raises(self):
        with pytest.raises(ValueError, match="not an oracle policy"):
            make_oracle_policy("fifo", _make_requests())

    def test_make_policy_cannot_make_oracle(self):
        with pytest.raises(KeyError):
            make_policy("oracle_srtf")


class TestOracleSRTFRuns:
    def test_oracle_srtf_emits_user_warning(self):
        requests = _make_requests()
        actual_map = {r.request_id: r.actual_output_tokens for r in requests}
        with pytest.warns(UserWarning, match="NOT a deployable policy"):
            OracleShortestJobFirstPolicy(actual_output_map=actual_map)

    def test_oracle_srtf_completes_tiny_trace(self):
        requests = _make_requests()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            oracle = build_oracle(requests)
        m = run_policy(
            policy=oracle,
            requests=requests,
            gpu_configs=_gpu(),
            drain_steps=5000,
        )
        assert m.num_completed > 0
        assert not math.isnan(m.mean_latency)

    def test_oracle_srtf_beats_or_matches_fifo_on_mean_latency(self):
        requests = _make_requests()
        fifo = make_policy("fifo")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            oracle = build_oracle(requests)

        m_fifo = run_policy(policy=fifo, requests=requests, gpu_configs=_gpu(), drain_steps=5000)
        m_oracle = run_policy(policy=oracle, requests=requests, gpu_configs=_gpu(), drain_steps=5000)

        # Oracle must complete all requests (same count as FIFO on this tiny trace)
        assert m_oracle.num_completed == m_fifo.num_completed
        # Oracle mean latency ≤ FIFO mean latency (oracle is an upper bound)
        assert m_oracle.mean_latency <= m_fifo.mean_latency + 1e-6


class TestBuildOracle:
    def test_build_oracle_from_requests(self):
        requests = _make_requests()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            oracle = build_oracle(requests)
        assert oracle._actual == {r.request_id: r.actual_output_tokens for r in requests}

"""Tests for the shared real-LLM calibration infrastructure
(src/llmserveopt/real_llm/calibration_common.py), independent of any single
provider script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402


def test_expand_call_plan_shape():
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short", "medium"],
        max_tokens_list=[64, 128], concurrency_list=[1, 2], requests_per_cell=3, seed=1,
    )
    assert len(plan) == 2 * 2 * 2 * 3 == 24
    assert all(isinstance(p, cc.PlannedRequest) for p in plan)


def test_request_result_fields_are_stable():
    """The schema every provider must produce. Changing this is a breaking
    change to requests.jsonl consumers (aggregation, downstream analysis)."""
    expected = {
        "request_id", "experiment_id", "model", "prompt_bucket",
        "intended_prompt_tokens", "actual_prompt_tokens", "max_tokens",
        "concurrency_level", "request_index", "start_time_iso", "end_time_iso",
        "elapsed_seconds", "ttft_seconds", "total_latency_seconds",
        "output_text_length_chars", "output_tokens", "billed_units",
        "finish_reason", "status", "error_type", "error_message",
        "retry_count", "was_resumed",
    }
    assert cc.REQUEST_RESULT_FIELDS == expected


def test_estimate_cost_usd():
    cost = cc.estimate_cost_usd(1_000_000, 1_000_000, price_per_m_input_usd=1.0, price_per_m_output_usd=2.0)
    assert cost == pytest.approx(3.0)


def test_validate_call_plan_requires_price_kwargs():
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    class Args:
        max_total_requests = 1000
        max_total_input_tokens = 10_000_000
        max_total_output_tokens = 10_000_000
        max_estimated_cost_usd = 1000.0
    violations = cc.validate_call_plan(
        plan, Args(), price_per_m_input_usd=0.01, price_per_m_output_usd=0.01,
    )
    assert violations == []


def test_budget_tracker_generic_reserve_release():
    class Args:
        max_total_requests = 1000
        max_total_input_tokens = 10_000_000
        max_total_output_tokens = 100
        max_estimated_cost_usd = 1000.0
    tracker = cc.BudgetTracker(Args(), price_per_m_input_usd=0.1, price_per_m_output_usd=0.1)
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=10, seed=1,
    )
    reserved = [tracker.try_reserve(p) for p in plan]
    assert sum(reserved) == 1  # cap of 100 output tokens allows only one 64-token reservation
    tracker.record_actual(plan[0], input_tokens=10, output_tokens=8)
    # Reservation released; a second request can now be reserved.
    assert tracker.try_reserve(plan[1]) is True


def test_mock_call_generic_and_stream_aware():
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    out_stream = cc.mock_call(plan[0], stream=True)
    out_nostream = cc.mock_call(plan[0], stream=False)
    assert out_stream["ttft_seconds"] is not None
    assert out_nostream["ttft_seconds"] is None


def test_add_common_arguments_defaults():
    import argparse
    parser = argparse.ArgumentParser()
    cc.add_common_arguments(parser, default_model="test-model")
    args = parser.parse_args(["--dry-run", "--output-dir", "/tmp/x"])
    assert args.model == "test-model"
    assert args.prompt_buckets == list(cc.KNOWN_PROMPT_BUCKETS)
    assert args.max_tokens_list == [64, 128, 256]
    assert args.concurrency_list == [1, 2, 4, 8]
    assert args.requests_per_cell == 5
    assert args.max_total_requests == 180
    assert args.mock is False


def test_add_common_arguments_mock_provider_alias():
    import argparse
    parser = argparse.ArgumentParser()
    cc.add_common_arguments(parser, default_model="test-model")
    args = parser.parse_args(["--dry-run", "--output-dir", "/tmp/x", "--mock-provider"])
    assert args.mock is True


def test_add_common_arguments_custom_defaults():
    import argparse
    parser = argparse.ArgumentParser()
    cc.add_common_arguments(
        parser, default_model="m",
        default_max_total_requests=60,
        default_max_total_input_tokens=1000,
        default_max_total_output_tokens=500,
        default_max_estimated_cost_usd=1.0,
    )
    args = parser.parse_args(["--dry-run", "--output-dir", "/tmp/x"])
    assert args.max_total_requests == 60
    assert args.max_total_input_tokens == 1000
    assert args.max_total_output_tokens == 500
    assert args.max_estimated_cost_usd == 1.0


def test_run_calibration_main_refuses_live_when_not_implemented(tmp_path, monkeypatch):
    import argparse
    monkeypatch.setenv("SOME_FAKE_PROVIDER_API_KEY", "fake")
    parser = argparse.ArgumentParser()
    cc.add_common_arguments(parser, default_model="m")
    args = parser.parse_args([
        "--allow-live-api",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    result = cc.run_calibration_main(
        args, root=ROOT, provider_display_name="FakeProvider",
        api_key_env_var="SOME_FAKE_PROVIDER_API_KEY", sdk_package_name=None,
        price_per_m_input_usd=0.1, price_per_m_output_usd=0.1,
        live_implemented=False,
    )
    assert result == 6


def test_run_calibration_main_dry_run_generic(tmp_path):
    import argparse
    parser = argparse.ArgumentParser()
    cc.add_common_arguments(parser, default_model="m")
    args = parser.parse_args([
        "--dry-run",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    result = cc.run_calibration_main(
        args, root=ROOT, provider_display_name="FakeProvider",
        api_key_env_var="SOME_FAKE_PROVIDER_API_KEY", sdk_package_name=None,
        price_per_m_input_usd=0.1, price_per_m_output_usd=0.1,
        live_implemented=False,
    )
    assert result == 0
    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_requests"] == 2 * 2 * 2 * 2  # 16
    assert not (tmp_path / "requests.jsonl").exists()

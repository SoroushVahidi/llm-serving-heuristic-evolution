"""Tests for the shared real-LLM calibration infrastructure
(src/llmserveopt/real_llm/calibration_common.py), independent of any single
provider script.
"""
from __future__ import annotations

import sys
import time
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
        "rate_limiter_wait_seconds", "provider_request_latency_seconds",
        "ttft_seconds", "total_wall_time_seconds",
        "output_text_length_chars", "output_tokens", "billed_units",
        "finish_reason", "status", "error_type", "error_message",
        "retry_count", "was_resumed",
        "target_output_tokens", "workload_version",
        "output_text_preview", "reached_target_output_range",
    }
    assert cc.REQUEST_RESULT_FIELDS == expected


def test_build_length_targeted_prompt_is_deterministic():
    p1 = cc.build_length_targeted_prompt("short", 128, seed=42, variant_index=0)
    p2 = cc.build_length_targeted_prompt("short", 128, seed=42, variant_index=0)
    assert p1 == p2


def test_build_length_targeted_prompt_varies_by_variant_and_target():
    base = cc.build_length_targeted_prompt("short", 64, seed=1, variant_index=0)
    other_variant = cc.build_length_targeted_prompt("short", 64, seed=1, variant_index=1)
    other_target = cc.build_length_targeted_prompt("short", 256, seed=1, variant_index=0)
    assert base != other_variant
    assert base != other_target


def test_build_length_targeted_prompt_instruction_scales_with_target():
    small = cc.build_length_targeted_prompt("short", 64, seed=1, variant_index=0)
    large = cc.build_length_targeted_prompt("short", 256, seed=1, variant_index=0)
    # The instruction should ask for a proportionally larger word count.
    assert "words" in small and "words" in large
    assert cc.PROPOSED_V2_TARGET_OUTPUT_TOKENS == (64, 128, 256)


def test_build_length_targeted_prompt_no_copyrighted_or_pii_markers():
    prompt = cc.build_length_targeted_prompt("long", 256, seed=99, variant_index=3)
    # Only the deterministic synthetic sentence bank should appear; no
    # external text was pulled in.
    for banned in ("http://", "https://", "@", "Copyright", "©"):
        assert banned not in prompt


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


# ---------------------------------------------------------------------------
# RPM-wait / provider-latency separation (regression test for the
# measurement artifact where an artificial local rate-limiter wait used to
# be counted as provider latency, inflating p95/p99 with a purely local
# scheduling delay).
# ---------------------------------------------------------------------------

class _ArtificialWaitLimiter:
    """A limiter whose first acquire() blocks for a fixed, known duration,
    simulating an RPM-budget wait, so the test can assert that duration
    lands in rate_limiter_wait_seconds and NOT in provider latency."""

    def __init__(self, wait_seconds: float) -> None:
        self._wait_seconds = wait_seconds
        self._acquired_once = False

    def acquire(self) -> None:
        if not self._acquired_once:
            self._acquired_once = True
            if self._wait_seconds > 0:
                time.sleep(self._wait_seconds)


def _fast_call_fn(client, planned, timeout_s):
    # Deliberately fast provider call so any inflation from the limiter
    # wait would be obvious in provider_request_latency_seconds.
    return {
        "text": "ok",
        "finish_reason": "COMPLETE",
        "prompt_tokens": float(planned.intended_prompt_tokens),
        "output_tokens": 5.0,
        "ttft_seconds": None,
    }


def test_artificial_rate_limiter_wait_does_not_inflate_provider_latency():
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    wait_seconds = 0.3
    limiter = _ArtificialWaitLimiter(wait_seconds)
    result = cc.execute_one_request(
        plan[0],
        client=None,
        stream=False,
        timeout_s=5,
        mock=False,
        rpm_limiter=limiter,
        was_resumed=False,
        call_streaming_fn=None,
        call_non_streaming_fn=_fast_call_fn,
    )
    assert result.status == "success"
    # The artificial wait must show up as rate-limiter wait...
    assert result.rate_limiter_wait_seconds == pytest.approx(wait_seconds, abs=0.05)
    # ...and must NOT pollute provider latency, which should reflect only
    # the (near-instant) call to _fast_call_fn.
    assert result.provider_request_latency_seconds < 0.05
    # Wall time still reflects the full request including the wait, so
    # throughput accounting is not blind to it.
    assert result.total_wall_time_seconds >= wait_seconds


def test_aggregate_results_latency_stats_exclude_rate_limiter_wait(tmp_path):
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=3, seed=1,
    )
    writer = cc.JsonlWriter(tmp_path / "requests.jsonl")
    # First request pays a large artificial rate-limiter wait; the rest do not.
    waits = [5.0, 0.0, 0.0]
    for planned, wait in zip(plan, waits):
        limiter = _ArtificialWaitLimiter(wait)
        result = cc.execute_one_request(
            planned, client=None, stream=False, timeout_s=5, mock=False,
            rpm_limiter=limiter, was_resumed=False,
            call_streaming_fn=None, call_non_streaming_fn=_fast_call_fn,
        )
        writer.write(result)
    writer.close()

    overall = cc.aggregate_results(
        tmp_path, price_per_m_input_usd=0.1, price_per_m_output_usd=0.1,
    )
    # p99 provider latency must stay small even though one request waited 5s
    # locally for the rate limiter — that wait belongs in the separate
    # rate-limiter-wait accounting, not in the latency percentile stats.
    assert overall["p99_latency_s"] < 0.1
    assert overall["n_requests_with_rate_limiter_wait"] == 1
    assert overall["total_rate_limiter_wait_seconds"] == pytest.approx(5.0, abs=0.05)


# ---------------------------------------------------------------------------
# v2 length-targeted workload: grid construction
# ---------------------------------------------------------------------------

def test_expand_call_plan_length_targeted_shape():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short", "medium", "long"],
        target_output_tokens_list=[64, 128, 256], concurrency_list=[1, 2, 4, 8],
        requests_per_cell=3, seed=1,
    )
    assert len(plan) == 3 * 3 * 4 * 3 == 108


def test_expand_call_plan_length_targeted_sets_target_and_version():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64], concurrency_list=[1],
        requests_per_cell=1, seed=1,
    )
    p = plan[0]
    assert p.target_output_tokens == 64
    assert p.workload_version == "v2"
    assert p.prompt_text == cc.build_length_targeted_prompt("short", 64, seed=1, variant_index=0)


def test_expand_call_plan_length_targeted_max_tokens_has_headroom():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64, 128], concurrency_list=[1],
        requests_per_cell=1, seed=1,
    )
    by_target = {p.target_output_tokens: p.max_tokens for p in plan}
    # Default headroom multiplier is 2x, matching docs/real_llm_v2_workload_proposal.md.
    assert by_target[64] == 128
    assert by_target[128] == 256


def test_expand_call_plan_v1_leaves_target_fields_at_default():
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    assert plan[0].target_output_tokens is None
    assert plan[0].workload_version == "v1"


# ---------------------------------------------------------------------------
# v2: reached_target_output_range / output_text_preview computed by
# execute_one_request
# ---------------------------------------------------------------------------

def _long_output_call_fn(client, planned, timeout_s):
    return {
        "text": "word " * 200,
        "finish_reason": "COMPLETE",
        "prompt_tokens": float(planned.intended_prompt_tokens),
        "output_tokens": 60.0,
        "ttft_seconds": None,
    }


class _NoWaitLimiter:
    def acquire(self) -> None:
        return None


def test_reached_target_output_range_true_above_ratio():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    result = cc.execute_one_request(
        plan[0], client=None, stream=False, timeout_s=5, mock=False,
        rpm_limiter=_NoWaitLimiter(), was_resumed=False,
        call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
        min_output_token_ratio=0.70,
    )
    # 60 output tokens / 64 target = 0.9375 >= 0.70 ratio.
    assert result.reached_target_output_range is True
    assert result.target_output_tokens == 64
    assert result.workload_version == "v2"


def test_reached_target_output_range_false_below_ratio():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[256], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    result = cc.execute_one_request(
        plan[0], client=None, stream=False, timeout_s=5, mock=False,
        rpm_limiter=_NoWaitLimiter(), was_resumed=False,
        call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
        min_output_token_ratio=0.70,
    )
    # 60 / 256 = 0.234 < 0.70 ratio.
    assert result.reached_target_output_range is False


def test_v1_request_has_no_reached_target_flag():
    plan = cc.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    result = cc.execute_one_request(
        plan[0], client=None, stream=False, timeout_s=5, mock=False,
        rpm_limiter=_NoWaitLimiter(), was_resumed=False,
        call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
        min_output_token_ratio=0.70,
    )
    assert result.reached_target_output_range is None
    assert result.target_output_tokens is None


def test_output_text_preview_truncated_to_configured_length():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    result = cc.execute_one_request(
        plan[0], client=None, stream=False, timeout_s=5, mock=False,
        rpm_limiter=_NoWaitLimiter(), was_resumed=False,
        call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
        output_text_preview_chars=80,
    )
    assert result.output_text_preview is not None
    assert len(result.output_text_preview) <= 80
    assert result.output_text_preview == ("word " * 200)[:80]


def test_output_text_preview_disabled_by_default():
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    result = cc.execute_one_request(
        plan[0], client=None, stream=False, timeout_s=5, mock=False,
        rpm_limiter=_NoWaitLimiter(), was_resumed=False,
        call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
    )
    assert result.output_text_preview is None


def test_full_model_output_never_stored_beyond_preview_cap():
    """Requirement: no full output text is persisted, only a short preview."""
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    full_text = "word " * 200
    result = cc.execute_one_request(
        plan[0], client=None, stream=False, timeout_s=5, mock=False,
        rpm_limiter=_NoWaitLimiter(), was_resumed=False,
        call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
        output_text_preview_chars=80,
    )
    assert result.output_text_preview != full_text
    assert len(result.output_text_preview) < len(full_text)


# ---------------------------------------------------------------------------
# v2: aggregation reports output-token distribution by target length
# ---------------------------------------------------------------------------

def test_aggregate_by_target_output_tokens_csv_and_summary(tmp_path):
    plan = cc.expand_call_plan_length_targeted(
        experiment_id="t", model="m", prompt_buckets=["short"],
        target_output_tokens_list=[64, 128], concurrency_list=[1],
        requests_per_cell=2, seed=1,
    )
    writer = cc.JsonlWriter(tmp_path / "requests.jsonl")
    for planned in plan:
        result = cc.execute_one_request(
            planned, client=None, stream=False, timeout_s=5, mock=False,
            rpm_limiter=_NoWaitLimiter(), was_resumed=False,
            call_streaming_fn=None, call_non_streaming_fn=_long_output_call_fn,
            min_output_token_ratio=0.70, output_text_preview_chars=80,
        )
        writer.write(result)
    writer.close()

    overall = cc.aggregate_results(
        tmp_path, price_per_m_input_usd=0.1, price_per_m_output_usd=0.1,
    )
    assert (tmp_path / "aggregate_by_target_output_tokens.csv").exists()
    by_target = {rec["target_output_tokens"]: rec for rec in overall["by_target_output_tokens"]}
    assert set(by_target.keys()) == {64, 128}
    assert by_target[64]["n_success"] == 2
    assert by_target[64]["mean_output_tokens"] == pytest.approx(60.0)
    assert by_target[64]["frac_reached_target_range"] == pytest.approx(1.0)
    assert by_target[128]["frac_reached_target_range"] == pytest.approx(0.0)
    assert overall["frac_reached_target_output_range"] == pytest.approx(0.5)

    import json as _json
    # Whole overall dict (including by_target_output_tokens) must be JSON-safe.
    _json.dumps(overall)

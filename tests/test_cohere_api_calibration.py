"""Tests for the Cohere API calibration dry-run/live infrastructure.

No test in this file makes a real network call. Live-mode code paths are
exercised only via --mock, which never imports the `cohere` SDK's network
client.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
RUNNER_PATH = ROOT / "scripts" / "run_cohere_api_calibration.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_cohere_api_calibration", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Module loads without requiring network / real API key
# ---------------------------------------------------------------------------

def test_module_loads_without_cohere_import_side_effects():
    mod = _load_runner()
    assert hasattr(mod, "expand_call_plan")
    assert hasattr(mod, "validate_call_plan")
    assert hasattr(mod, "run_requests")
    assert hasattr(mod, "aggregate_results")


# ---------------------------------------------------------------------------
# Prompt generation determinism
# ---------------------------------------------------------------------------

def test_build_prompt_deterministic():
    mod = _load_runner()
    a = mod.build_prompt("short", seed=7, variant_index=0)
    b = mod.build_prompt("short", seed=7, variant_index=0)
    assert a == b


def test_build_prompt_varies_by_variant_index():
    mod = _load_runner()
    a = mod.build_prompt("short", seed=7, variant_index=0)
    b = mod.build_prompt("short", seed=7, variant_index=1)
    assert a != b, "Distinct variant indices must produce distinct prompts (defeats server-side caching)"


def test_build_prompt_length_scales_with_bucket():
    mod = _load_runner()
    short = mod.build_prompt("short", seed=1, variant_index=0)
    medium = mod.build_prompt("medium", seed=1, variant_index=0)
    long_ = mod.build_prompt("long", seed=1, variant_index=0)
    assert len(short) < len(medium) < len(long_)


# ---------------------------------------------------------------------------
# Call plan expansion
# ---------------------------------------------------------------------------

def test_expand_call_plan_size():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test",
        model="command-r7b-12-2024",
        prompt_buckets=["short", "medium", "long"],
        max_tokens_list=[64, 128, 256],
        concurrency_list=[1, 2, 4, 8],
        requests_per_cell=5,
        seed=1,
    )
    assert len(plan) == 3 * 3 * 4 * 5 == 180


def test_expand_call_plan_required_fields():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=2, seed=1,
    )
    for p in plan:
        assert p.request_id
        assert p.experiment_id == "test"
        assert p.prompt_bucket == "short"
        assert p.max_tokens == 64
        assert p.concurrency_level == 1
        assert p.intended_prompt_tokens > 0
        assert p.prompt_text


def test_expand_call_plan_request_ids_unique():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short", "medium"],
        max_tokens_list=[64, 128], concurrency_list=[1, 2], requests_per_cell=3, seed=1,
    )
    ids = [p.request_id for p in plan]
    assert len(ids) == len(set(ids))


def test_expand_call_plan_request_ids_stable_across_calls():
    """Request IDs must be reproducible across runs to support --resume."""
    mod = _load_runner()
    plan1 = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=2, seed=1,
    )
    plan2 = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=2, seed=1,
    )
    assert [p.request_id for p in plan1] == [p.request_id for p in plan2]


# ---------------------------------------------------------------------------
# Hard cap validation
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    defaults = dict(
        max_total_requests=1000,
        max_total_input_tokens=10_000_000,
        max_total_output_tokens=10_000_000,
        max_estimated_cost_usd=1000.0,
    )
    defaults.update(overrides)

    class Args:
        pass
    a = Args()
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


def test_validate_call_plan_passes_generous_caps():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=2, seed=1,
    )
    violations = mod.validate_call_plan(plan, _make_args())
    assert violations == []


def test_validate_call_plan_rejects_too_many_requests():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=5, seed=1,
    )
    violations = mod.validate_call_plan(plan, _make_args(max_total_requests=1))
    assert any("max-total-requests" in v for v in violations)


def test_validate_call_plan_rejects_input_token_cap():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["long"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    violations = mod.validate_call_plan(plan, _make_args(max_total_input_tokens=1))
    assert any("input tokens" in v for v in violations)


def test_validate_call_plan_rejects_output_token_cap():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["short"],
        max_tokens_list=[256], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    violations = mod.validate_call_plan(plan, _make_args(max_total_output_tokens=1))
    assert any("output tokens" in v for v in violations)


def test_validate_call_plan_rejects_cost_cap():
    mod = _load_runner()
    plan = mod.expand_call_plan(
        experiment_id="test", model="m", prompt_buckets=["long"],
        max_tokens_list=[256], concurrency_list=[8], requests_per_cell=5, seed=1,
    )
    violations = mod.validate_call_plan(plan, _make_args(max_estimated_cost_usd=0.0))
    assert any("cost" in v for v in violations)


# ---------------------------------------------------------------------------
# CLI: mode selection
# ---------------------------------------------------------------------------

def test_requires_dry_run_or_allow_live_api(tmp_path):
    mod = _load_runner()
    result = mod.main(["--output-dir", str(tmp_path)])
    assert result == 2


def test_unknown_prompt_bucket_rejected(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--dry-run", "--prompt-buckets", "extra_long",
        "--output-dir", str(tmp_path),
    ])
    assert result == 2


def test_dry_run_makes_no_cohere_import(tmp_path, monkeypatch):
    """Dry-run must never import the cohere SDK's live client path."""
    for key in list(sys.modules.keys()):
        if key == "cohere" or key.startswith("cohere."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    mod = _load_runner()
    result = mod.main([
        "--dry-run",
        "--prompt-buckets", "short,medium,long",
        "--max-tokens-list", "64,128,256",
        "--concurrency-list", "1,2,4,8",
        "--requests-per-cell", "5",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert "cohere" not in sys.modules


def test_dry_run_plans_180_requests(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--dry-run",
        "--model", "command-r7b-12-2024",
        "--prompt-buckets", "short,medium,long",
        "--max-tokens-list", "64,128,256",
        "--concurrency-list", "1,2,4,8",
        "--requests-per-cell", "5",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_requests"] == 180
    assert not (tmp_path / "requests.jsonl").exists()


def test_dry_run_writes_reproducibility_and_config(tmp_path):
    mod = _load_runner()
    mod.main(["--dry-run", "--output-dir", str(tmp_path)])
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "reproducibility.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.md").exists() or (tmp_path / "summary.json").exists()


# ---------------------------------------------------------------------------
# Live mode requires credentials + explicit flag
# ---------------------------------------------------------------------------

def test_live_mode_refuses_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    mod = _load_runner()
    result = mod.main([
        "--allow-live-api",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 5


def test_mock_mode_does_not_require_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    mod = _load_runner()
    result = mod.main([
        "--allow-live-api", "--mock",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0


# ---------------------------------------------------------------------------
# Anti-overwrite / resume
# ---------------------------------------------------------------------------

def test_refuses_to_overwrite_nonempty_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    args = [
        "--allow-live-api", "--mock",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(args) == 0
    result = mod.main(args)
    assert result == 3


def test_resume_skips_completed_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    base_args = [
        "--allow-live-api", "--mock", "--stream",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "3",
        "--seed", "5",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(base_args) == 0
    first_lines = (tmp_path / "requests.jsonl").read_text().strip().splitlines()
    assert len(first_lines) == 2 * 2 * 2 * 3  # 24

    # Re-run with an expanded grid + --resume: original 24 must not be duplicated.
    expanded_args = [
        "--allow-live-api", "--mock", "--stream", "--resume",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64,128,256",
        "--concurrency-list", "1,2,4", "--requests-per-cell", "3",
        "--seed", "5",
        "--max-total-requests", "300",
        "--max-total-input-tokens", "500000",
        "--max-total-output-tokens", "100000",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(expanded_args) == 0
    all_lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    ids = [r["request_id"] for r in all_lines]
    assert len(ids) == len(set(ids)), "resume must not duplicate request_ids"
    assert len(all_lines) == 2 * 3 * 3 * 3  # 54


# ---------------------------------------------------------------------------
# Request JSONL schema
# ---------------------------------------------------------------------------

REQUIRED_REQUEST_FIELDS = {
    "request_id", "experiment_id", "model", "prompt_bucket",
    "intended_prompt_tokens", "actual_prompt_tokens", "max_tokens",
    "concurrency_level", "request_index", "start_time_iso", "end_time_iso",
    "rate_limiter_wait_seconds", "provider_request_latency_seconds",
    "ttft_seconds", "total_wall_time_seconds",
    "output_text_length_chars", "output_tokens", "billed_units",
    "finish_reason", "status", "error_type", "error_message",
    "retry_count", "was_resumed",
}


def test_request_jsonl_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert lines
    for row in lines:
        assert REQUIRED_REQUEST_FIELDS <= set(row.keys())
        assert row["status"] == "success"


def test_mock_streaming_produces_ttft(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert all(r["ttft_seconds"] is not None for r in lines)


def test_mock_non_streaming_has_no_ttft(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--no-stream",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert all(r["ttft_seconds"] is None for r in lines)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def test_summary_aggregation_outputs_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for fname in (
        "summary.json", "summary.md", "aggregate_by_cell.csv",
        "aggregate_by_concurrency.csv", "aggregate_by_prompt_bucket.csv",
        "errors.jsonl",
    ):
        assert (tmp_path / fname).exists(), f"missing {fname}"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status_counts"]["success"] == 2 * 2 * 2 * 2  # 16
    assert summary["mean_latency_s"] is not None
    assert summary["mean_ttft_s"] is not None

    import pandas as pd
    by_cell = pd.read_csv(tmp_path / "aggregate_by_cell.csv")
    assert len(by_cell) == 2 * 2 * 2  # 8 cells


# ---------------------------------------------------------------------------
# Fail-fast
# ---------------------------------------------------------------------------

def test_fail_fast_tracker_triggers_on_error_rate():
    mod = _load_runner()
    tracker = mod.FailFastTracker(enabled=True)
    for _ in range(9):
        tracker.record("success")
    assert not tracker.abort_event.is_set()
    for _ in range(2):
        tracker.record("error")
    assert tracker.abort_event.is_set()
    assert "error rate" in tracker.abort_reason


def test_fail_fast_tracker_triggers_on_consecutive_rate_limits():
    mod = _load_runner()
    tracker = mod.FailFastTracker(enabled=True)
    tracker.record("rate_limited")
    tracker.record("rate_limited")
    assert not tracker.abort_event.is_set()
    tracker.record("rate_limited")
    assert tracker.abort_event.is_set()
    assert "consecutive" in tracker.abort_reason


def test_fail_fast_tracker_disabled_never_triggers():
    mod = _load_runner()
    tracker = mod.FailFastTracker(enabled=False)
    for _ in range(20):
        tracker.record("error")
    assert not tracker.abort_event.is_set()


def test_fail_fast_end_to_end_skips_remaining_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()

    def _always_fail(planned, stream):
        raise RuntimeError("simulated failure for fail-fast test")

    monkeypatch.setattr(mod, "_mock_call", _always_fail)

    result = mod.main([
        "--allow-live-api", "--mock", "--fail-fast",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium,long", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "5",
        "--rpm-limit", "10000",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    statuses = {r["status"] for r in lines}
    assert "skipped" in statuses, "fail-fast should have skipped remaining requests"
    assert "error" in statuses


# ---------------------------------------------------------------------------
# Hard caps enforced at runtime (BudgetTracker)
# ---------------------------------------------------------------------------

def test_budget_tracker_reserves_up_to_request_cap():
    mod = _load_runner()
    args = _make_args(max_total_requests=3)
    tracker = mod.BudgetTracker(args)
    plan = mod.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=10, seed=1,
    )
    reserved = [tracker.try_reserve(p) for p in plan]
    assert sum(reserved) == 3
    assert reserved.count(False) == 7


def test_budget_tracker_reserves_up_to_output_token_cap():
    mod = _load_runner()
    args = _make_args(max_total_output_tokens=100)
    tracker = mod.BudgetTracker(args)
    plan = mod.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=10, seed=1,
    )
    reserved = [tracker.try_reserve(p) for p in plan]
    # Each reservation assumes worst-case max_tokens=64; cap of 100 allows only 1.
    assert sum(reserved) == 1


def test_budget_tracker_accounts_actual_usage_on_resume():
    """Simulates resuming: history contributes to the running total."""
    mod = _load_runner()
    args = _make_args(max_total_output_tokens=100)
    tracker = mod.BudgetTracker(args)
    tracker.actual_output_tokens = 90  # pretend prior successful requests used this much
    plan = mod.expand_call_plan(
        experiment_id="t", model="m", prompt_buckets=["short"],
        max_tokens_list=[64], concurrency_list=[1], requests_per_cell=1, seed=1,
    )
    assert tracker.try_reserve(plan[0]) is False, "90 + 64 > 100 must be rejected"


# ---------------------------------------------------------------------------
# API key never leaked
# ---------------------------------------------------------------------------

def test_api_key_never_written_to_output_files(tmp_path, monkeypatch):
    secret = "sk-COHERE-SECRET-TEST-VALUE-12345"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"


def test_manifest_env_var_presence_is_boolean_only(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "sk-should-not-appear-anywhere")
    mod = _load_runner()
    mod.main(["--dry-run", "--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["env_var_presence"] == {"COHERE_API_KEY_present": True}
    assert "sk-should-not-appear-anywhere" not in json.dumps(manifest)


# ---------------------------------------------------------------------------
# v2 length-targeted workload CLI
# ---------------------------------------------------------------------------

V2_GRID_ARGS = [
    "--stream",
    "--model", "command-r7b-12-2024",
    "--workload-version", "v2",
    "--prompt-buckets", "short,medium,long",
    "--target-output-tokens-list", "64,128,256",
    "--concurrency-list", "1,2,4,8",
    "--requests-per-cell", "3",
    "--timeout-seconds", "120",
    "--rpm-limit", "20",
    "--max-total-requests", "108",
    "--max-total-input-tokens", "250000",
    "--max-total-output-tokens", "50000",
    "--max-estimated-cost-usd", "5",
    "--seed", "20260703",
    "--fail-fast",
]


def test_v2_dry_run_plans_108_requests(tmp_path):
    mod = _load_runner()
    result = mod.main(
        ["--dry-run", *V2_GRID_ARGS, "--output-dir", str(tmp_path)]
    )
    assert result == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_requests"] == 3 * 3 * 4 * 3 == 108
    assert not (tmp_path / "requests.jsonl").exists()


def test_v2_dry_run_records_target_output_tokens_in_plan(tmp_path):
    mod = _load_runner()
    mod.main(["--dry-run", *V2_GRID_ARGS, "--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    targets = {r["target_output_tokens"] for r in manifest["requests_preview"]}
    assert targets <= {64, 128, 256}
    assert all(r["workload_version"] == "v2" for r in manifest["requests_preview"])


def test_v2_dry_run_no_cohere_import(tmp_path, monkeypatch):
    for key in list(sys.modules.keys()):
        if key == "cohere" or key.startswith("cohere."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    mod = _load_runner()
    result = mod.main(["--dry-run", *V2_GRID_ARGS, "--output-dir", str(tmp_path)])
    assert result == 0
    assert "cohere" not in sys.modules


def test_v2_requires_target_output_tokens_list(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--dry-run", "--workload-version", "v2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 2


def test_v2_dry_run_worst_case_cost_below_cap(tmp_path, capsys):
    mod = _load_runner()
    result = mod.main(["--dry-run", *V2_GRID_ARGS, "--output-dir", str(tmp_path)])
    assert result == 0
    out = capsys.readouterr().out
    assert "worst_case_cost_usd" in out


def test_v2_mock_run_schema_has_v2_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    result = mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert len(lines) == 1 * 2 * 1 * 2  # 4
    for row in lines:
        assert row["workload_version"] == "v2"
        assert row["target_output_tokens"] in (64, 128)
        assert row["status"] == "success"


def test_v2_provider_latency_not_polluted_by_rate_limiter_wait(tmp_path, monkeypatch):
    """Regression: the v2 path must reuse the same rate_limiter_wait_seconds
    / provider_request_latency_seconds split as v1 — not recombine them."""
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    for row in lines:
        assert row["rate_limiter_wait_seconds"] == 0.0
        assert row["provider_request_latency_seconds"] is not None
        assert row["provider_request_latency_seconds"] < 1.0


def test_v2_summary_reports_output_token_distribution_by_target(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert (tmp_path / "aggregate_by_target_output_tokens.csv").exists()
    import pandas as pd
    by_target = pd.read_csv(tmp_path / "aggregate_by_target_output_tokens.csv")
    assert set(by_target["target_output_tokens"]) == {64, 128}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["by_target_output_tokens"]) == 2


def test_v2_output_text_preview_capped_at_80_chars_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    for row in lines:
        assert row["output_text_preview"] is not None
        assert len(row["output_text_preview"]) <= 80


def test_v2_resume_skips_completed_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    base_args = [
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--seed", "5",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(base_args) == 0
    first_lines = (tmp_path / "requests.jsonl").read_text().strip().splitlines()
    assert len(first_lines) == 1 * 2 * 1 * 2  # 4

    expanded_args = [
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2", "--resume",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128,256",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--seed", "5",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(expanded_args) == 0
    all_lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    ids = [r["request_id"] for r in all_lines]
    assert len(ids) == len(set(ids)), "resume must not duplicate request_ids"
    assert len(all_lines) == 1 * 3 * 1 * 2  # 6


def test_v2_refuses_to_overwrite_nonempty_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake_key_for_test")
    mod = _load_runner()
    args = [
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(args) == 0
    assert mod.main(args) == 3


def test_v2_api_key_never_written_to_output_files(tmp_path, monkeypatch):
    # NOTE: git_diff.patch is excluded here because it is an intentional full
    # working-tree diff snapshot (collect_reproducibility_metadata); if this
    # test's own source line assigning `secret` is itself uncommitted, that
    # diff legitimately contains it. That is a source-control hygiene
    # concern, not a harness runtime-secret leak, which is what this test
    # guards against — see the equivalent v1 test above for the same pattern.
    secret = "sk-COHERE-V2-SECRET-TEST-VALUE-12345"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    mod = _load_runner()
    mod.main([
        "--allow-live-api", "--mock", "--stream", "--workload-version", "v2",
        "--rpm-limit", "100000",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file() and f.name != "git_diff.patch":
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"

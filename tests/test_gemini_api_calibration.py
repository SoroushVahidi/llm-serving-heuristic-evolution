"""Tests for the Gemini API calibration dry-run infrastructure."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
RUNNER_PATH = ROOT / "scripts" / "run_gemini_api_calibration.py"
CONFIG_PATH = ROOT / "configs" / "api_calibration" / "gemini_minimal_v1.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_gemini_api_calibration", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cfg():
    mod = _load_runner()
    return mod.load_config(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Smoke: module loads without requiring Google SDK
# ---------------------------------------------------------------------------

def test_module_loads_without_google_sdk():
    """Importing the runner must not raise even if google SDK is absent."""
    mod = _load_runner()
    assert hasattr(mod, "expand_call_plan")
    assert hasattr(mod, "validate_call_plan")
    assert hasattr(mod, "write_manifest")


def test_config_loads():
    cfg = _load_cfg()
    assert "hard_caps" in cfg
    assert "prompt_buckets" in cfg
    assert "output_buckets" in cfg


# ---------------------------------------------------------------------------
# Call plan expansion
# ---------------------------------------------------------------------------

def test_dry_run_expands_within_max_calls():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    max_calls = int(cfg["hard_caps"]["max_calls"])
    assert len(calls) <= max_calls, (
        f"Planned {len(calls)} calls but max_calls={max_calls}"
    )


def test_call_plan_has_required_fields():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    assert calls, "No calls in plan"
    for c in calls:
        assert c.call_id.startswith("call_")
        assert c.provider in ("gemini_api", "vertex")
        assert c.model
        assert c.prompt_bucket in cfg["prompt_buckets"]
        assert c.output_bucket in cfg["output_buckets"]
        assert c.planned_prompt_tokens >= 0
        assert c.max_output_tokens > 0
        assert c.concurrency_group >= 1
        assert c.repeat_index >= 0


def test_call_plan_covers_all_buckets():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    prompt_buckets_seen = {c.prompt_bucket for c in calls}
    output_buckets_seen = {c.output_bucket for c in calls}
    assert prompt_buckets_seen == set(cfg["prompt_buckets"].keys())
    assert output_buckets_seen == set(cfg["output_buckets"].keys())


def test_validate_call_plan_passes_for_valid_config():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    violations = mod.validate_call_plan(calls, cfg)
    assert violations == [], f"Unexpected violations: {violations}"


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def test_validate_rejects_too_many_calls():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    tight_cfg = {**cfg, "hard_caps": {**cfg["hard_caps"], "max_calls": 1}}
    violations = mod.validate_call_plan(calls, tight_cfg)
    assert any("max_calls" in v for v in violations)


def test_validate_rejects_oversized_prompt():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    tight_cfg = {
        **cfg,
        "hard_caps": {**cfg["hard_caps"], "max_prompt_tokens_per_call": 1},
    }
    violations = mod.validate_call_plan(calls, tight_cfg)
    # Every call with >1 planned_prompt_tokens should be flagged
    calls_over = [c for c in calls if c.planned_prompt_tokens > 1]
    if calls_over:
        assert any("max_prompt_tokens_per_call" in v for v in violations)


def test_max_calls_override_respected(tmp_path):
    mod = _load_runner()
    cfg = _load_cfg()
    cap = int(cfg["hard_caps"]["max_calls"])
    # Override must not exceed config cap
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--max-calls", str(cap),
        "--output-dir", str(tmp_path),
    ])
    assert result == 0


def test_max_calls_override_refuses_if_above_cap():
    mod = _load_runner()
    cfg = _load_cfg()
    cap = int(cfg["hard_caps"]["max_calls"])
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--allow-live-api",
        "--mock",
        "--max-calls", str(cap + 1),
        "--output-dir", "/tmp/should_not_be_created_abc123",
    ])
    assert result == 3


# ---------------------------------------------------------------------------
# Live mode refuses without explicit flag
# ---------------------------------------------------------------------------

def test_live_mode_requires_allow_live_api_flag(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--output-dir", str(tmp_path),
    ])
    assert result == 2, "Expected exit code 2 when neither --dry-run nor --allow-live-api"


def test_live_mode_refuses_without_credentials(tmp_path, monkeypatch):
    """Without --mock, live mode should refuse if no credentials env var."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--allow-live-api",
        "--max-calls", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 4, "Expected exit code 4 when credentials are missing"


# ---------------------------------------------------------------------------
# Manifest content
# ---------------------------------------------------------------------------

def test_manifest_has_required_fields(tmp_path):
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["mode"] == "dry_run"
    assert "planned_calls" in manifest
    assert "hard_caps" in manifest
    assert "estimate" in manifest
    assert "calls" in manifest
    assert isinstance(manifest["calls"], list)
    assert len(manifest["calls"]) > 0


def test_manifest_calls_have_prompt_output_concurrency_fields(tmp_path):
    mod = _load_runner()
    mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--output-dir", str(tmp_path),
    ])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for call in manifest["calls"]:
        assert "planned_prompt_tokens" in call
        assert "max_output_tokens" in call
        assert "concurrency_group" in call
        assert "prompt_bucket" in call
        assert "output_bucket" in call


def test_manifest_contains_no_secrets(tmp_path):
    """Manifest must not embed API keys, credentials, or env values."""
    mod = _load_runner()
    os.environ["GOOGLE_API_KEY"] = "test_secret_key_PLACEHOLDER"
    try:
        mod.main([
            "--config", str(CONFIG_PATH),
            "--dry-run",
            "--output-dir", str(tmp_path),
        ])
    finally:
        del os.environ["GOOGLE_API_KEY"]
    manifest_text = (tmp_path / "manifest.json").read_text()
    assert "test_secret_key_PLACEHOLDER" not in manifest_text
    assert "PLACEHOLDER" not in manifest_text


# ---------------------------------------------------------------------------
# Output paths are under results/logs (not project root)
# ---------------------------------------------------------------------------

def test_default_output_dir_is_under_results(tmp_path):
    mod = _load_runner()
    # Use a temp subdir that mimics the results/ directory structure.
    out = tmp_path / "results" / "api_calibration" / "gemini_minimal_v1"
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--output-dir", str(out),
    ])
    assert result == 0
    assert (out / "manifest.json").exists()
    # The path must NOT be directly inside the repo's source tree.
    assert "scripts" not in str(out)
    assert "src" not in str(out)


def test_dry_run_creates_summary_markdown(tmp_path):
    mod = _load_runner()
    mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--output-dir", str(tmp_path),
    ])
    summary = tmp_path / "dry_run_summary.md"
    assert summary.exists()
    text = summary.read_text()
    assert "No API calls were made" in text
    assert "To Run Live Pilot" in text


# ---------------------------------------------------------------------------
# Missing SDK does not break dry-run
# ---------------------------------------------------------------------------

def test_dry_run_succeeds_without_google_sdk(tmp_path, monkeypatch):
    """Simulate google SDK absence by hiding it from sys.modules."""
    # Remove any cached google imports but leave pyyaml intact.
    for key in list(sys.modules.keys()):
        if "google" in key:
            monkeypatch.delitem(sys.modules, key, raising=False)
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--dry-run",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0


# ---------------------------------------------------------------------------
# Mock live run (does not call any real API)
# ---------------------------------------------------------------------------

def test_mock_live_run_records_results(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "mock_key_not_real")
    mod = _load_runner()
    result = mod.main([
        "--config", str(CONFIG_PATH),
        "--allow-live-api",
        "--mock",
        "--max-calls", "3",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    log = tmp_path / "call_log.jsonl"
    assert log.exists()
    lines = [json.loads(l) for l in log.read_text().strip().splitlines()]
    assert len(lines) == 3
    for line in lines:
        assert line["status"] == "mock_ok"
        assert "latency_total_ms" in line
        assert "start_utc" in line
        assert "end_utc" in line
        # No real API key in log
        assert "mock_key_not_real" not in json.dumps(line)


def test_mock_live_run_respects_max_calls_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "mock_key_not_real")
    mod = _load_runner()
    cfg = _load_cfg()
    cap = int(cfg["hard_caps"]["max_calls"])
    mod.main([
        "--config", str(CONFIG_PATH),
        "--allow-live-api",
        "--mock",
        "--max-calls", str(cap),
        "--output-dir", str(tmp_path),
    ])
    log = tmp_path / "call_log.jsonl"
    lines = log.read_text().strip().splitlines()
    assert len(lines) <= cap


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

def test_estimate_cost_is_nonnegative():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    cost = mod.estimate_cost_usd(calls, cfg.get("provider", "gemini_api"))
    assert cost >= 0.0


def test_estimate_cost_within_budget_cap():
    mod = _load_runner()
    cfg = _load_cfg()
    calls = mod.expand_call_plan(cfg)
    cost = mod.estimate_cost_usd(calls, cfg.get("provider", "gemini_api"))
    budget = float(cfg["hard_caps"]["estimated_budget_usd"])
    assert cost <= budget, (
        f"Estimated cost ${cost:.5f} exceeds budget cap ${budget}"
    )

"""Tests for the Gemini/Vertex, Azure OpenAI, and Fireworks calibration
scripts, and for cross-provider schema/output compatibility with the Cohere
calibration script.

No test in this file makes a real network call. Gemini's live mode IS
implemented (unlike Azure/Fireworks, still skeletons), so tests that
exercise its live-vs-mock branching are careful to either use --mock (which
never calls build_client_fn/the network, regardless of live_implemented) or
to unset all credential env vars so the script's own missing-credential
check (exit 5) returns before any client is built — never setting a *fake*
credential value for Gemini's non-mock path, since a fake-but-present value
would pass the presence check and reach real network code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# Env var used for each script's live-mode credential *presence* gate (see
# API_KEY_ENV_VAR in each script). For Gemini this is GOOGLE_CLOUD_PROJECT
# (this project's environment is Vertex/ADC-configured, not a raw API key).
PROVIDER_SCRIPTS = [
    pytest.param(
        "scripts/run_gemini_real_llm_calibration.py",
        "run_gemini_real_llm_calibration",
        "GOOGLE_CLOUD_PROJECT",
        id="gemini",
    ),
    pytest.param(
        "scripts/run_azure_openai_api_calibration.py",
        "run_azure_openai_api_calibration",
        "AZURE_OPENAI_API_KEY",
        id="azure_openai",
    ),
    pytest.param(
        "scripts/run_fireworks_api_calibration.py",
        "run_fireworks_api_calibration",
        "FIREWORKS_API_KEY",
        id="fireworks",
    ),
]

# Providers whose live mode is NOT implemented yet (still skeletons).
NOT_LIVE_IMPLEMENTED_SCRIPTS = [p for p in PROVIDER_SCRIPTS if p.id != "gemini"]


def _load(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Live mode refuses (not yet implemented) for Azure/Fireworks (still skeletons)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,mod_name,env_var", NOT_LIVE_IMPLEMENTED_SCRIPTS)
def test_live_mode_not_implemented_refuses(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    monkeypatch.setenv(env_var, "fake_key_for_test")
    mod = _load(rel_path, mod_name)
    result = mod.main([
        "--allow-live-api",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 6, f"{mod_name} should refuse live mode (not yet implemented)"
    assert not (tmp_path / "requests.jsonl").exists()


@pytest.mark.parametrize("rel_path,mod_name,env_var", PROVIDER_SCRIPTS)
def test_mock_mode_always_works(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    """--mock never touches the network or build_client_fn, regardless of
    whether live mode is implemented for this provider."""
    monkeypatch.delenv(env_var, raising=False)
    mod = _load(rel_path, mod_name)
    result = mod.main([
        "--allow-live-api", "--mock",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert len(lines) == 2
    assert all(r["status"] == "success" for r in lines)


def test_gemini_live_mode_refuses_without_credentials(tmp_path, monkeypatch):
    """Gemini's live mode IS implemented, so unlike Azure/Fireworks it must
    be gated on missing credentials (exit 5), not "not implemented" (exit
    6). Both real credential env vars are unset (never set to a fake
    truthy value) so the script's own presence check short-circuits before
    any client is built or any network call is attempted."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    mod = _load("scripts/run_gemini_real_llm_calibration.py", "run_gemini_real_llm_calibration")
    result = mod.main([
        "--allow-live-api",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 5
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Dry-run planned grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,mod_name,env_var", PROVIDER_SCRIPTS)
def test_dry_run_plans_expected_grid(tmp_path, rel_path, mod_name, env_var):
    mod = _load(rel_path, mod_name)
    result = mod.main([
        "--dry-run",
        "--prompt-buckets", "short,medium,long", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["planned_requests"] == 3 * 2 * 2 * 2  # 24
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Anti-overwrite + resume
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,mod_name,env_var", PROVIDER_SCRIPTS)
def test_refuses_overwrite_and_resume_skips(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    monkeypatch.setenv(env_var, "fake_key_for_test")
    mod = _load(rel_path, mod_name)
    base = [
        "--allow-live-api", "--mock",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(base) == 0
    assert mod.main(base) == 3  # refuses without --resume

    expanded = [
        "--allow-live-api", "--mock", "--resume",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ]
    assert mod.main(expanded) == 0
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    ids = [r["request_id"] for r in lines]
    assert len(ids) == len(set(ids))
    assert len(lines) == 4  # 2 (short, original) + 2 (medium, new)


# ---------------------------------------------------------------------------
# Pre-flight hard-cap validation (runtime BudgetTracker enforcement itself is
# covered at the shared-module level in tests/test_cohere_api_calibration.py
# and tests/test_real_llm_calibration_common.py, since every provider script
# delegates to the same calibration_common.BudgetTracker).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,mod_name,env_var", PROVIDER_SCRIPTS)
def test_hard_cap_violation_refuses_before_any_request(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    monkeypatch.setenv(env_var, "fake_key_for_test")
    mod = _load(rel_path, mod_name)
    result = mod.main([
        "--allow-live-api", "--mock",
        "--prompt-buckets", "short", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "10",
        "--max-total-requests", "3",
        "--output-dir", str(tmp_path),
    ])
    assert result == 4
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# No secrets leaked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path,mod_name,env_var", PROVIDER_SCRIPTS)
def test_api_key_never_written_to_output(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    secret = f"secret-{mod_name}-12345"
    monkeypatch.setenv(env_var, secret)
    mod = _load(rel_path, mod_name)
    mod.main([
        "--allow-live-api", "--mock",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"


# ---------------------------------------------------------------------------
# Cross-provider schema compatibility (including Cohere)
# ---------------------------------------------------------------------------

ALL_PROVIDER_SCRIPTS_INCLUDING_COHERE = PROVIDER_SCRIPTS + [
    pytest.param(
        "scripts/run_cohere_api_calibration.py",
        "run_cohere_api_calibration",
        "COHERE_API_KEY",
        id="cohere",
    ),
]


def _run_mock_pilot(tmp_path, rel_path, mod_name, env_var, monkeypatch):
    monkeypatch.setenv(env_var, "fake_key_for_test")
    mod = _load(rel_path, mod_name)
    args = [
        "--allow-live-api", "--mock",
        "--prompt-buckets", "short,medium", "--max-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ]
    if mod_name == "run_cohere_api_calibration":
        args.append("--stream")
    assert mod.main(args) == 0
    return mod


@pytest.mark.parametrize("rel_path,mod_name,env_var", ALL_PROVIDER_SCRIPTS_INCLUDING_COHERE)
def test_output_files_match_shared_schema(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    _run_mock_pilot(tmp_path, rel_path, mod_name, env_var, monkeypatch)
    for fname in (
        "requests.jsonl", "summary.json", "summary.md",
        "aggregate_by_cell.csv", "aggregate_by_concurrency.csv",
        "aggregate_by_prompt_bucket.csv", "manifest.json",
        "run_config.json", "reproducibility.md", "errors.jsonl",
    ):
        assert (tmp_path / fname).exists(), f"{mod_name} missing {fname}"


@pytest.mark.parametrize("rel_path,mod_name,env_var", ALL_PROVIDER_SCRIPTS_INCLUDING_COHERE)
def test_requests_jsonl_field_set_matches_shared_dataclass(tmp_path, monkeypatch, rel_path, mod_name, env_var):
    import llmserveopt.real_llm.calibration_common as cc
    _run_mock_pilot(tmp_path, rel_path, mod_name, env_var, monkeypatch)
    lines = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert lines
    for row in lines:
        assert set(row.keys()) == cc.REQUEST_RESULT_FIELDS


def test_aggregate_by_cell_columns_identical_across_providers(tmp_path, monkeypatch):
    column_sets = {}
    for param in ALL_PROVIDER_SCRIPTS_INCLUDING_COHERE:
        rel_path, mod_name, env_var = param.values
        sub_dir = tmp_path / mod_name
        sub_dir.mkdir()
        _run_mock_pilot(sub_dir, rel_path, mod_name, env_var, monkeypatch)
        import pandas as pd
        df = pd.read_csv(sub_dir / "aggregate_by_cell.csv")
        column_sets[mod_name] = set(df.columns)

    all_column_sets = list(column_sets.values())
    first = all_column_sets[0]
    for name, cols in column_sets.items():
        assert cols == first, f"{name} aggregate_by_cell.csv columns differ: {cols} vs {first}"

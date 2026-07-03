"""Structural regression tests for the real-LLM comparison report and
existing pilot artifacts: required sections are present, and no secrets
ever leaked into committed docs/experiment output. No network access.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

COMPARISON_REPORT = ROOT / "docs" / "real_llm_cohere_gemini_comparison.md"

REQUIRED_SECTIONS = [
    "## Experiment directories",
    "## Request counts and reliability",
    "## Cost caveat",
    "## TTFT comparison",
    "## Latency comparison",
    "## Concurrency comparison",
    "## Prompt-bucket comparison",
    "## max_tokens caveat",
    "## Safe claims",
    "## Unsafe claims",
]

SAFE_CLAIM_MARKERS = [
    "180/180",
    "lower TTFT",
]

UNSAFE_CLAIM_MARKERS = [
    "Do not claim",
]

SECRET_PATTERNS = ("API_KEY=", "Bearer ", "sk-", "AIza", "COHERE_API_KEY_VALUE")


def test_comparison_report_exists_and_has_required_sections():
    assert COMPARISON_REPORT.exists(), "comparison report doc is missing"
    text = COMPARISON_REPORT.read_text()
    for section in REQUIRED_SECTIONS:
        assert section in text, f"missing required section: {section}"


def test_comparison_report_states_safe_and_unsafe_claims():
    text = COMPARISON_REPORT.read_text()
    for marker in SAFE_CLAIM_MARKERS:
        assert marker in text
    for marker in UNSAFE_CLAIM_MARKERS:
        assert marker in text
    # Explicitly must not claim our scheduler beats the providers.
    assert "beats Cohere or Gemini" in text or "scheduler \"beats\"" in text


def test_comparison_report_has_no_secrets():
    text = COMPARISON_REPORT.read_text()
    for pattern in SECRET_PATTERNS:
        assert pattern not in text


def test_pilot_experiment_dirs_have_no_secrets():
    real_llm_dir = ROOT / "experiments" / "real_llm"
    if not real_llm_dir.exists():
        return
    for path in real_llm_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".json", ".md", ".jsonl", ".csv", ".txt", ".log", ".patch"):
            continue
        text = path.read_text(errors="ignore")
        for pattern in SECRET_PATTERNS:
            assert pattern not in text, f"possible secret pattern {pattern!r} in {path}"


def test_pilot_manifests_only_record_key_presence_not_value():
    for name in ("cohere_pilot_20260703T040421Z", "gemini_pilot_20260703T044905Z"):
        manifest_path = ROOT / "experiments" / "real_llm" / name / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        presence = manifest.get("env_var_presence", {})
        assert presence, "expected env_var_presence block in manifest"
        for key, val in presence.items():
            assert isinstance(val, bool), f"{key} should record presence as a bool, not a value"

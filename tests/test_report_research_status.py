"""Tests for scripts/report_research_status.py."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = "scripts/report_research_status.py"


def _run(args=(), timeout=30):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )


# -------------------------------------------------------------------------
# Script runs without crashing
# -------------------------------------------------------------------------

def test_script_runs_cleanly():
    result = _run()
    assert result.returncode == 0, result.stderr


def test_help_exits_cleanly():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()


def test_json_output_is_valid():
    result = _run(["--json"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "deployable_baselines" in data
    assert "oracle_policies" in data
    assert "selector_candidates" in data


# -------------------------------------------------------------------------
# Correct counts and policy names
# -------------------------------------------------------------------------

def test_json_deployable_count():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    count = data["deployable_baselines"]["count"]
    assert count >= 19, f"Expected at least 19 deployable baselines, got {count}"


def test_json_oracle_excluded():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    oracle_names = data["oracle_policies"]["names"]
    candidate_names = data["selector_candidates"]["names"]
    for name in oracle_names:
        assert name not in candidate_names, f"Oracle {name} leaked into selector candidates"


def test_json_admission_control_registered():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    assert data["admission_control_registered"] is True


def test_json_rule_based_not_fifo_placeholder():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    assert data["rule_based_is_fifo_placeholder"] is False


def test_json_no_invariant_violations():
    result = _run(["--json", "--check"])
    assert result.returncode == 0, f"Invariant violation: {result.stderr}"


def test_text_output_contains_admission_control():
    result = _run()
    assert "admission_control" in result.stdout


def test_text_output_oracle_not_deployable():
    result = _run()
    assert "ORACLE" in result.stdout or "oracle" in result.stdout


# -------------------------------------------------------------------------
# gather_status() callable from Python directly
# -------------------------------------------------------------------------

def test_gather_status_importable():
    from scripts.report_research_status import gather_status
    status = gather_status()
    assert status["deployable_baselines"]["count"] >= 19
    assert "oracle_srtf" in status["oracle_policies"]["names"]
    assert status["admission_control_registered"] is True
    assert status["rule_based_is_fifo_placeholder"] is False
    assert status["invariants"]["oracle_not_in_selector_candidates"] is True

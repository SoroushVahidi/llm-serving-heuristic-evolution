"""
Regression tests: side-effecting scripts must be safe under --help.

Before this fix, inspect_gpu_environment.py and update_phase17c_docs.py
unconditionally executed their full side effects (including overwriting
tracked docs) on any invocation, because they lacked argument parsing and
therefore ignored --help entirely.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

SCRIPTS = [
    "scripts/inspect_gpu_environment.py",
    "scripts/update_phase17c_docs.py",
    "scripts/generate_phase17c_summary.py",
    "scripts/smoke_test.py",
]

DRY_RUN_SCRIPTS = [
    "scripts/inspect_gpu_environment.py",
    "scripts/update_phase17c_docs.py",
]


def _git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout


def _run_script(script: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_exits_cleanly_with_usage(script):
    result = _run_script(script, "--help", timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_does_not_modify_working_tree(script):
    before = _git_status_short()
    result = _run_script(script, "--help", timeout=30)
    after = _git_status_short()
    assert result.returncode == 0, result.stderr
    assert before == after, f"--help on {script} changed working tree state:\n{after}"


@pytest.mark.parametrize("script", DRY_RUN_SCRIPTS)
def test_dry_run_does_not_modify_working_tree(script):
    before = _git_status_short()
    result = _run_script(script, "--dry-run", timeout=60)
    after = _git_status_short()
    assert result.returncode == 0, result.stderr
    assert before == after, f"--dry-run on {script} changed working tree state:\n{after}"


@pytest.mark.parametrize("script", DRY_RUN_SCRIPTS)
def test_dry_run_reports_intent_without_writing(script):
    result = _run_script(script, "--dry-run", timeout=60)
    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout.lower()


PAID_API_SHELL_SCRIPTS = [
    "scripts/_run_cohere_v2_live_pilot.sh",
    "scripts/_run_gemini_v2_live_pilot.sh",
]


def _existing_real_llm_dirs() -> set:
    real_llm_dir = ROOT / "experiments" / "real_llm"
    if not real_llm_dir.is_dir():
        return set()
    return {p.name for p in real_llm_dir.iterdir()}


@pytest.mark.parametrize("script", PAID_API_SHELL_SCRIPTS)
def test_paid_api_script_refuses_without_opt_in(script):
    """Without LLMSERVEOPT_ALLOW_PAID_API_CALLS=1, these scripts must exit
    before making any network call or creating an experiment directory --
    they fire real, billed API requests otherwise."""
    env = {k: v for k, v in os.environ.items() if k != "LLMSERVEOPT_ALLOW_PAID_API_CALLS"}
    before = _existing_real_llm_dirs()
    result = subprocess.run(
        ["bash", script],
        cwd=ROOT, capture_output=True, text=True, timeout=30, env=env,
    )
    after = _existing_real_llm_dirs()
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "LLMSERVEOPT_ALLOW_PAID_API_CALLS" in result.stderr
    assert before == after, "script created an experiment dir without the opt-in var set"

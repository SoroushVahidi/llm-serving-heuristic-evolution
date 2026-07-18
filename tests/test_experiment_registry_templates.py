"""Tests for Phase 2B.6 experiment registry templates and report script updates."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "docs" / "templates"
DOCS_DIR = REPO_ROOT / "docs"


# ---------------------------------------------------------------------------
# Template files exist and are valid CSVs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", [
    "experiment_registry_template.csv",
    "failure_case_registry_template.csv",
    "api_usage_ledger_template.csv",
])
def test_template_file_exists(fname):
    path = TEMPLATES_DIR / fname
    assert path.exists(), f"Template file missing: {path}"


@pytest.mark.parametrize("fname", [
    "experiment_registry_template.csv",
    "failure_case_registry_template.csv",
    "api_usage_ledger_template.csv",
])
def test_template_file_has_header(fname):
    """CSV templates must have at least one non-comment header line."""
    path = TEMPLATES_DIR / fname
    if not path.exists():
        pytest.skip(f"Template missing: {fname}")
    lines = path.read_text().splitlines()
    # First non-comment line should look like a CSV header
    header_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    assert len(header_lines) >= 1, f"{fname}: no non-comment header lines found"
    # First header should be comma-separated
    first = header_lines[0]
    assert "," in first, f"{fname}: header line does not look like CSV: {first!r}"


def test_experiment_registry_template_has_required_fields():
    path = TEMPLATES_DIR / "experiment_registry_template.csv"
    if not path.exists():
        pytest.skip("Template missing")
    lines = path.read_text().splitlines()
    header = next(l for l in lines if l.strip() and not l.startswith("#"))
    fields = [f.strip() for f in header.split(",")]
    required = ["experiment_id", "phase", "config_file", "policies", "selector_wg",
                "best_fixed_policy", "best_fixed_wg", "delta_vs_best_fixed"]
    for req in required:
        assert req in fields, f"Required field '{req}' missing from experiment registry template"


def test_failure_case_registry_template_has_required_fields():
    path = TEMPLATES_DIR / "failure_case_registry_template.csv"
    if not path.exists():
        pytest.skip("Template missing")
    lines = path.read_text().splitlines()
    header = next(l for l in lines if l.strip() and not l.startswith("#"))
    fields = [f.strip() for f in header.split(",")]
    required = ["failure_id", "experiment_id", "workload_tag", "delta", "failure_category"]
    for req in required:
        assert req in fields, f"Required field '{req}' missing from failure registry template"


def test_api_usage_ledger_template_has_required_fields():
    path = TEMPLATES_DIR / "api_usage_ledger_template.csv"
    if not path.exists():
        pytest.skip("Template missing")
    lines = path.read_text().splitlines()
    header = next(l for l in lines if l.strip() and not l.startswith("#"))
    fields = [f.strip() for f in header.split(",")]
    required = ["call_id", "date", "provider", "cost_usd", "phase", "purpose"]
    for req in required:
        assert req in fields, f"Required field '{req}' missing from API ledger template"


# ---------------------------------------------------------------------------
# docs/experiment_tracking.md exists
# ---------------------------------------------------------------------------

def test_experiment_tracking_doc_exists():
    path = DOCS_DIR / "experiment_tracking.md"
    assert path.exists(), "docs/experiment_tracking.md is missing"


def test_experiment_tracking_doc_mentions_templates():
    path = DOCS_DIR / "experiment_tracking.md"
    if not path.exists():
        pytest.skip("doc missing")
    text = path.read_text()
    assert "experiment_registry_template.csv" in text
    assert "failure_case_registry_template.csv" in text
    assert "api_usage_ledger_template.csv" in text


# ---------------------------------------------------------------------------
# report_research_status.py template checks
# ---------------------------------------------------------------------------

def test_report_script_checks_templates():
    """report_research_status.py --json output must include templates section."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "report_research_status.py"), "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert "templates" in data, "report_research_status.py JSON must include 'templates' key"
    assert "all_present" in data["templates"]
    assert "present" in data["templates"]


def test_report_script_templates_all_present():
    """Templates must be present; report script must confirm all_present=True."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "report_research_status.py"), "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    data = json.loads(result.stdout)
    tmpl = data["templates"]
    assert tmpl["all_present"] is True, (
        f"Not all templates present: {tmpl['present']}"
    )


def test_report_script_check_mode_passes():
    """--check mode must pass when all invariants hold."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "report_research_status.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"--check failed unexpectedly:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# docs/external_baseline_correctness_audit.md exists
# ---------------------------------------------------------------------------

def test_external_baseline_correctness_audit_exists():
    path = DOCS_DIR / "external_baseline_correctness_audit.md"
    assert path.exists(), "docs/external_baseline_correctness_audit.md is missing"


def test_external_baseline_audit_covers_required_policies():
    """The correctness audit must cover all 11 specified policies."""
    path = DOCS_DIR / "external_baseline_correctness_audit.md"
    if not path.exists():
        pytest.skip("audit doc missing")
    text = path.read_text()
    required = [
        "orca_style",
        "vllm_style_token_budget",
        "sarathi_style",
        "splitfuse_style",
        "multi_bin_batching",
        "estimated_service_time_first",
        "least_laxity_first",
        "admission_control",
        "greedy_token_fill",
        "slo_slack_score",
        "oracle_srtf",
    ]
    for policy in required:
        assert policy in text, (
            f"Correctness audit does not cover policy '{policy}'"
        )


# ---------------------------------------------------------------------------
# docs/audits/admission_control_threshold_calibration_summary.md
# ---------------------------------------------------------------------------

def test_admission_calibration_summary_exists():
    path = DOCS_DIR / "audits" / "admission_control_threshold_calibration_summary.md"
    assert path.exists(), "Admission control calibration summary doc missing"


def test_admission_calibration_summary_mentions_unit_mismatch():
    path = DOCS_DIR / "audits" / "admission_control_threshold_calibration_summary.md"
    if not path.exists():
        pytest.skip("doc missing")
    text = path.read_text()
    assert "unit" in text.lower() or "mismatch" in text.lower(), (
        "Calibration summary must mention unit mismatch"
    )

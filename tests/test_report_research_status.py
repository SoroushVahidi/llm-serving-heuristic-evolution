"""Tests for scripts/report_research_status.py."""
import json
import subprocess
import sys
from pathlib import Path


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
    assert count >= 20, f"Expected at least 20 deployable baselines, got {count}"


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
    assert status["deployable_baselines"]["count"] >= 20
    assert "oracle_srtf" in status["oracle_policies"]["names"]
    assert status["admission_control_registered"] is True
    assert status["rule_based_is_fifo_placeholder"] is False
    assert status["invariants"]["oracle_not_in_selector_candidates"] is True


# -------------------------------------------------------------------------
# Phase 2B.7 additions: unit fix, multi-bin tests, failure cases, API ledger
# -------------------------------------------------------------------------

def test_json_admission_control_unit_fixed():
    result = _run(["--json"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "admission_control_unit_fixed" in data
    assert data["admission_control_unit_fixed"] is True


def test_json_multi_bin_tests_exist():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    assert "multi_bin_tests_exist" in data
    assert data["multi_bin_tests_exist"] is True


def test_json_failure_cases_section():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    assert "failure_cases" in data
    fc = data["failure_cases"]
    assert "count" in fc
    assert fc["count"] >= 0  # may be 0 if registry not yet populated in CI


def test_json_api_ledger_section():
    result = _run(["--json"])
    data = json.loads(result.stdout)
    assert "api_ledger" in data
    al = data["api_ledger"]
    assert "entries" in al
    assert "is_empty" in al


# -------------------------------------------------------------------------
# Phase 2B.8: rule selector KV-pressure repair detection
# -------------------------------------------------------------------------

def test_json_rule_based_kv_repair_applied():
    """Phase 2B.8: rule_based selector should have KV-pressure repair applied."""
    result = _run(["--json"])
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "rule_based_kv_repair_applied" in data, (
        "report_research_status.py missing rule_based_kv_repair_applied field"
    )
    assert data["rule_based_kv_repair_applied"] is True, (
        "rule_based KV-pressure repair not applied: kv_pressure_decode_heavy features "
        "should route to weighted_shortest_processing, not least_laxity_first"
    )


def test_json_tight_slo_not_routed_to_llf():
    """Phase 2B.8: tight SLO (no KV pressure) must NOT route to least_laxity_first."""
    result = _run(["--json"])
    data = json.loads(result.stdout)
    tight_slo_policy = data.get("rule_based_tight_slo_policy", "unknown")
    assert tight_slo_policy != "least_laxity_first", (
        f"Tight-SLO policy still routes to least_laxity_first — Phase 2B.8 repair not applied. "
        f"Expected slo_slack_score, got {tight_slo_policy}"
    )
    assert tight_slo_policy == "slo_slack_score", (
        f"Tight-SLO policy should be slo_slack_score after Phase 2B.8 repair, got {tight_slo_policy}"
    )


def test_text_output_kv_repair_mentioned():
    """Phase 2B.8: text report should mention the KV-pressure repair."""
    result = _run()
    assert "Phase 2B.8" in result.stdout or "KV" in result.stdout or "kv" in result.stdout.lower(), (
        "Text report doesn't mention Phase 2B.8 KV repair"
    )

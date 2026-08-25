"""
Phase 2B.10 SCORPIO-Style SLO Guard Tests.

Tests config, runner, docs, and registry invariants for the new baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def test_phase2b10_config_exists():
    cfg = CONFIGS_DIR / "phase2b10_scorpio_slo_guard.yaml"
    assert cfg.exists()


def test_phase2b10_config_has_phase2b9_reference():
    import yaml
    cfg = yaml.safe_load((CONFIGS_DIR / "phase2b10_scorpio_slo_guard.yaml").read_text())
    assert "phase2b9_reference" in cfg
    assert "rule_based_wg" in cfg["phase2b9_reference"]


def test_phase2b10_runner_exists():
    script = SCRIPTS_DIR / "run_phase2b10_scorpio_slo_guard.py"
    assert script.exists()


def test_phase2b10_runner_no_paid_api_calls():
    content = (SCRIPTS_DIR / "run_phase2b10_scorpio_slo_guard.py").read_text()
    for term in ["openai", "cohere", "gemini", "cloudrift", "cerebras", "mistral"]:
        assert term.lower() not in content.lower()


def test_phase2b10_audit_doc_exists():
    doc = DOCS_DIR / "audits" / "phase2b10_scorpio_slo_guard_summary.md"
    assert doc.exists()


def test_phase2b10_audit_doc_safe_wording():
    content = (DOCS_DIR / "audits" / "phase2b10_scorpio_slo_guard_summary.md").read_text()
    assert "SCORPIO-style" in content or "SCORPIO-inspired" in content
    assert "official SCORPIO reproduction" not in content.lower()
    assert "scorpio_style_slo_guard" in content


def test_research_status_mentions_phase2b10():
    content = (DOCS_DIR / "research_status.md").read_text()
    assert "2B.10" in content or "phase2b10" in content.lower()


def test_baselines_doc_lists_scorpio_style():
    content = (DOCS_DIR / "baselines.md").read_text()
    assert "scorpio_style_slo_guard" in content
    assert "SCORPIO" in content


def test_external_baseline_decision_marks_scorpio_implemented():
    content = (DOCS_DIR / "external_baseline_decision.md").read_text()
    assert "scorpio_style_slo_guard" in content or "B.2" in content


def test_scorpio_in_selector_candidates():
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES
    assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATES
    assert "scorpio_style_slo_guard" not in ORACLE_POLICY_NAMES

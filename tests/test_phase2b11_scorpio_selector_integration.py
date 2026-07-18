"""
Phase 2B.11 SCORPIO Selector Integration Tests.

Tests config, runner, docs, and selector integration for Phase 2B.11.
Verifies that scorpio_style_slo_guard is properly integrated into the
rule-based selector and that registry/docs are consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# -------------------------------------------------------------------------
# Config / runner existence
# -------------------------------------------------------------------------

def test_phase2b11_config_exists():
    cfg = CONFIGS_DIR / "phase2b11_scorpio_selector_integration.yaml"
    assert cfg.exists(), "Phase 2B.11 config file not found"


def test_phase2b11_config_has_phase2b10_reference():
    import yaml
    cfg = yaml.safe_load(
        (CONFIGS_DIR / "phase2b11_scorpio_selector_integration.yaml").read_text()
    )
    assert "phase2b10_reference" in cfg
    ref = cfg["phase2b10_reference"]
    assert "rule_based_wg" in ref
    assert "scorpio_fixed_wg" in ref or "best_fixed_wg" in ref


def test_phase2b11_config_same_workloads_as_2b10():
    """Phase 2B.11 must use same workloads as Phase 2B.10 for apples-to-apples comparison."""
    import yaml
    cfg10 = yaml.safe_load(
        (CONFIGS_DIR / "phase2b10_scorpio_slo_guard.yaml").read_text()
    )
    cfg11 = yaml.safe_load(
        (CONFIGS_DIR / "phase2b11_scorpio_selector_integration.yaml").read_text()
    )
    tags10 = {w["tag"] for w in cfg10.get("workloads", [])}
    tags11 = {w["tag"] for w in cfg11.get("workloads", [])}
    assert tags10 == tags11, (
        f"Phase 2B.11 workloads differ from Phase 2B.10: "
        f"added={tags11-tags10}, removed={tags10-tags11}"
    )


def test_phase2b11_runner_exists():
    script = SCRIPTS_DIR / "run_phase2b11_scorpio_selector_integration.py"
    assert script.exists(), "Phase 2B.11 runner script not found"


def test_phase2b11_runner_no_paid_api_calls():
    """Runner must not call paid APIs."""
    content = (SCRIPTS_DIR / "run_phase2b11_scorpio_selector_integration.py").read_text()
    for term in ["openai", "cohere", "gemini", "cloudrift", "cerebras", "mistral"]:
        assert term.lower() not in content.lower(), (
            f"Phase 2B.11 runner references paid API: {term}"
        )


def test_phase2b11_runner_no_hf_token():
    """Runner must not reference Hugging Face token."""
    content = (SCRIPTS_DIR / "run_phase2b11_scorpio_selector_integration.py").read_text()
    assert "HF_TOKEN" not in content
    assert "huggingface_hub" not in content


# -------------------------------------------------------------------------
# Selector integration: scorpio_style_slo_guard
# -------------------------------------------------------------------------

def test_scorpio_in_rule_selector_policy_choices():
    """scorpio_style_slo_guard must be in RuleBasedSelector._POLICY_CHOICES."""
    from llmserveopt.selector.models import RuleBasedSelector
    assert "scorpio_style_slo_guard" in RuleBasedSelector._POLICY_CHOICES


def test_rule_selector_can_dispatch_to_scorpio():
    """Rule selector can return scorpio_style_slo_guard for overload + tight SLO + violations."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    # Overloaded tight-SLO + high violations → Rule 0 → SCORPIO
    f = {
        "fraction_tight_slo": 0.5,
        "min_slack": 0.4,
        "recent_slo_violation_rate": 0.35,
        "kv_utilization": 0.0,
        "pred_output_cv": 0.87,
        "mean_pred_output_tokens": 96.0,
        "mean_prompt_tokens": 128.0,
        "p95_prompt_tokens": 256.0,
        "burstiness_cv": 0.3,
    }
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_rule_selector_dispatches_scorpio_for_extreme_noise():
    """Rule selector routes very high noise (pred_output_cv > 2.0) to SCORPIO (fail_004 fix)."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    f = {
        "fraction_tight_slo": 0.4,
        "min_slack": 0.5,
        "recent_slo_violation_rate": 0.0,
        "kv_utilization": 0.0,
        "pred_output_cv": 2.8,
        "mean_pred_output_tokens": 96.0,
        "mean_prompt_tokens": 128.0,
        "p95_prompt_tokens": 256.0,
        "burstiness_cv": 0.3,
    }
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_rule_selector_dispatches_scorpio_for_high_violations():
    """Rule selector routes standalone high violations to SCORPIO (Rule 3, Phase 2B.11)."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    f = {
        "fraction_tight_slo": 0.0,
        "min_slack": 50.0,
        "recent_slo_violation_rate": 0.5,
        "kv_utilization": 0.0,
        "pred_output_cv": 0.8,
        "mean_pred_output_tokens": 96.0,
        "mean_prompt_tokens": 128.0,
        "p95_prompt_tokens": 256.0,
        "burstiness_cv": 0.3,
    }
    assert sel.predict_one(f) == "scorpio_style_slo_guard"


def test_moderate_noise_still_uses_admission_control():
    """pred_output_cv in (1.0, 2.0] still routes to admission_control (Rule 2b unchanged)."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    f = {
        "fraction_tight_slo": 0.0,
        "min_slack": 50.0,
        "recent_slo_violation_rate": 0.0,
        "kv_utilization": 0.0,
        "pred_output_cv": 1.5,
        "mean_pred_output_tokens": 96.0,
        "mean_prompt_tokens": 128.0,
        "p95_prompt_tokens": 256.0,
        "burstiness_cv": 0.3,
    }
    assert sel.predict_one(f) == "admission_control"


def test_tight_slo_without_violations_still_uses_slo_slack():
    """Tight SLO without active violations still routes to slo_slack_score (Rule 4)."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    f = {
        "fraction_tight_slo": 0.5,
        "min_slack": 0.4,
        "recent_slo_violation_rate": 0.0,   # no violations
        "kv_utilization": 0.0,
        "pred_output_cv": 0.87,
        "mean_pred_output_tokens": 96.0,
        "mean_prompt_tokens": 128.0,
        "p95_prompt_tokens": 256.0,
        "burstiness_cv": 0.3,
    }
    assert sel.predict_one(f) == "slo_slack_score"


def test_kv_pressure_without_overload_still_triggers_wsp():
    """KV pressure alone (no tight SLO violations) still routes to WSP (Rule 1)."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    f = {
        "fraction_tight_slo": 0.0,
        "min_slack": 50.0,
        "recent_slo_violation_rate": 0.0,
        "kv_utilization": 0.85,
        "pred_output_cv": 0.8,
        "mean_pred_output_tokens": 250.0,
        "mean_prompt_tokens": 64.0,
        "p95_prompt_tokens": 128.0,
        "burstiness_cv": 0.3,
    }
    assert sel.predict_one(f) == "weighted_shortest_processing"


# -------------------------------------------------------------------------
# Registry and oracle exclusion invariants
# -------------------------------------------------------------------------

def test_selector_candidates_count_is_20():
    """Selector candidates should number exactly 20 after Phase 2B.11."""
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    assert len(SELECTOR_CANDIDATES) == 20, (
        f"Expected 20 selector candidates, got {len(SELECTOR_CANDIDATES)}: "
        f"{sorted(SELECTOR_CANDIDATES)}"
    )


def test_oracle_srtf_excluded_from_candidates():
    """oracle_srtf must never be in SELECTOR_CANDIDATES."""
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    assert "oracle_srtf" not in SELECTOR_CANDIDATES


def test_rule_selector_never_returns_oracle_srtf():
    """RuleBasedSelector must not return oracle_srtf under any feature combination."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    feature_cases = [
        {"fraction_tight_slo": 0.6, "min_slack": 50.0, "recent_slo_violation_rate": 0.0,
         "kv_utilization": 0.0, "pred_output_cv": 0.8, "mean_pred_output_tokens": 96.0,
         "mean_prompt_tokens": 128.0, "p95_prompt_tokens": 256.0, "burstiness_cv": 0.3},
        {"fraction_tight_slo": 0.5, "min_slack": 0.4, "recent_slo_violation_rate": 0.35,
         "kv_utilization": 0.0, "pred_output_cv": 0.87, "mean_pred_output_tokens": 96.0,
         "mean_prompt_tokens": 128.0, "p95_prompt_tokens": 256.0, "burstiness_cv": 0.3},
        {"fraction_tight_slo": 0.0, "min_slack": 50.0, "recent_slo_violation_rate": 0.5,
         "kv_utilization": 0.0, "pred_output_cv": 0.8, "mean_pred_output_tokens": 96.0,
         "mean_prompt_tokens": 128.0, "p95_prompt_tokens": 256.0, "burstiness_cv": 0.3},
        {"fraction_tight_slo": 0.0, "min_slack": 50.0, "recent_slo_violation_rate": 0.0,
         "kv_utilization": 0.0, "pred_output_cv": 2.5, "mean_pred_output_tokens": 96.0,
         "mean_prompt_tokens": 128.0, "p95_prompt_tokens": 256.0, "burstiness_cv": 0.3},
    ]
    for f in feature_cases:
        assert sel.predict_one(f) != "oracle_srtf"


def test_all_rule_selector_choices_are_candidates():
    """Every policy in _POLICY_CHOICES must be in SELECTOR_CANDIDATES (incl. SCORPIO)."""
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.selector.models import RuleBasedSelector
    for p in RuleBasedSelector._POLICY_CHOICES:
        assert p in SELECTOR_CANDIDATES, (
            f"RuleBasedSelector._POLICY_CHOICES contains non-candidate: {p}"
        )


# -------------------------------------------------------------------------
# Docs existence
# -------------------------------------------------------------------------

def test_phase2b11_audit_doc_exists():
    doc = DOCS_DIR / "audits" / "phase2b11_scorpio_selector_integration_summary.md"
    assert doc.exists(), "Phase 2B.11 audit summary doc not found"


def test_phase2b11_failure_cases_doc_exists():
    doc = DOCS_DIR / "audits" / "phase2b11_failure_cases_summary.md"
    assert doc.exists(), "Phase 2B.11 failure cases doc not found"


def test_research_status_mentions_phase2b11():
    content = (DOCS_DIR / "research_status.md").read_text()
    assert "2B.11" in content or "phase2b11" in content.lower()

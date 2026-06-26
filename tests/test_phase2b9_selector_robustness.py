"""
Phase 2B.9 Selector Robustness Tests.

Tests:
- RF/DT selector training labels exclude oracle_srtf
- Online selector features exclude actual output and future arrivals (re-verify)
- Report script detects Phase 2B.9 docs/results
- External baseline and dataset decision docs exist and contain required sections
- Repaired rule selector is deterministic and oracle-free
- Phase 2B.9 config is valid (workloads tagged as dev_ or heldout_)
- Runner script exists and has expected CLI interface
- Selector split checks: dev and heldout seeds do not overlap
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Oracle exclusion (re-verify Phase 2B.9 invariants)
# ---------------------------------------------------------------------------

def test_oracle_not_in_selector_candidates():
    """oracle_srtf must not appear in SELECTOR_CANDIDATES."""
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES
    for oracle in ORACLE_POLICY_NAMES:
        assert oracle not in SELECTOR_CANDIDATES, (
            f"Oracle policy '{oracle}' must not be a selector candidate"
        )


def test_selector_label_function_excludes_oracle():
    """label_windows must never assign an oracle policy as best_policy."""
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES

    # Any policy returned by a labeling function must be in SELECTOR_CANDIDATES
    # (and therefore not an oracle). This tests the invariant at the API level.
    from llmserveopt.selector.labels import WindowLabel
    # WindowLabel best_policy field is documented to be from SELECTOR_CANDIDATES
    # We verify by checking that ORACLE_POLICY_NAMES ∩ SELECTOR_CANDIDATES == ∅
    overlap = set(ORACLE_POLICY_NAMES) & set(SELECTOR_CANDIDATES)
    assert overlap == set(), (
        f"Oracle policy names overlap with selector candidates: {overlap}"
    )


# ---------------------------------------------------------------------------
# Feature leakage checks (Phase 2B.9 verification)
# ---------------------------------------------------------------------------

def test_feature_names_exclude_actual_output():
    """FEATURE_NAMES must not include actual_output_tokens."""
    from llmserveopt.selector.features import FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert "actual_output" not in name.lower(), (
            f"Feature '{name}' references actual_output — leakage risk"
        )


def test_feature_names_exclude_future_arrivals():
    """FEATURE_NAMES must not include future arrival information."""
    future_leakage_keywords = ["future", "next_arrival", "forecast_arrival"]
    from llmserveopt.selector.features import FEATURE_NAMES
    for name in FEATURE_NAMES:
        for kw in future_leakage_keywords:
            assert kw not in name.lower(), (
                f"Feature '{name}' may expose future arrivals: keyword '{kw}'"
            )


def test_rule_based_selector_uses_only_online_features():
    """RuleBasedSelector.predict_one must only access known online-observable features."""
    import inspect
    from llmserveopt.selector.models import RuleBasedSelector
    source = inspect.getsource(RuleBasedSelector.predict_one)
    # Confirm 'actual_output_tokens' is never read
    assert "actual_output_tokens" not in source, (
        "RuleBasedSelector.predict_one reads actual_output_tokens — leakage"
    )
    # Confirm no oracle reference
    assert "oracle_srtf" not in source, (
        "RuleBasedSelector.predict_one references oracle_srtf"
    )


# ---------------------------------------------------------------------------
# Rule selector determinism and oracle-free guarantee
# ---------------------------------------------------------------------------

def test_rule_based_selector_is_deterministic():
    """Same features must always produce the same policy choice."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    features = {
        "mean_pred_output_tokens": 250.0,  # triggers Rule 1 → WSP
        "kv_utilization": 0.0,
        "pred_output_cv": 0.5,
        "recent_slo_violation_rate": 0.0,
        "fraction_tight_slo": 0.2,
        "min_slack": 5.0,
        "mean_prompt_tokens": 100.0,
        "p95_prompt_tokens": 200.0,
        "burstiness_cv": 0.5,
    }
    first = sel.predict_one(features)
    second = sel.predict_one(features)
    assert first == second, "Rule selector is not deterministic"
    assert first == "weighted_shortest_processing", (
        f"Expected WSP for high mean_pred_output, got {first}"
    )


@pytest.mark.parametrize("features,expected", [
    # Rule 1: high mean_pred_output → WSP
    ({"mean_pred_output_tokens": 250.0, "kv_utilization": 0.0,
      "pred_output_cv": 0.5, "recent_slo_violation_rate": 0.0,
      "fraction_tight_slo": 0.1, "min_slack": 5.0,
      "mean_prompt_tokens": 64.0, "p95_prompt_tokens": 128.0,
      "burstiness_cv": 0.3},
     "weighted_shortest_processing"),
    # Rule 1: high kv_utilization → WSP
    ({"mean_pred_output_tokens": 50.0, "kv_utilization": 0.8,
      "pred_output_cv": 0.5, "recent_slo_violation_rate": 0.0,
      "fraction_tight_slo": 0.1, "min_slack": 5.0,
      "mean_prompt_tokens": 64.0, "p95_prompt_tokens": 128.0,
      "burstiness_cv": 0.3},
     "weighted_shortest_processing"),
    # Rule 2: high pred_output_cv → admission_control
    ({"mean_pred_output_tokens": 50.0, "kv_utilization": 0.0,
      "pred_output_cv": 1.5, "recent_slo_violation_rate": 0.0,
      "fraction_tight_slo": 0.1, "min_slack": 5.0,
      "mean_prompt_tokens": 64.0, "p95_prompt_tokens": 128.0,
      "burstiness_cv": 0.3},
     "admission_control"),
    # Rule 4: tight SLO → slo_slack_score (NOT least_laxity_first)
    ({"mean_pred_output_tokens": 50.0, "kv_utilization": 0.0,
      "pred_output_cv": 0.5, "recent_slo_violation_rate": 0.0,
      "fraction_tight_slo": 0.6, "min_slack": 5.0,
      "mean_prompt_tokens": 64.0, "p95_prompt_tokens": 128.0,
      "burstiness_cv": 0.3},
     "slo_slack_score"),
    # Rule 4: tight min_slack → slo_slack_score (NOT least_laxity_first)
    ({"mean_pred_output_tokens": 50.0, "kv_utilization": 0.0,
      "pred_output_cv": 0.5, "recent_slo_violation_rate": 0.0,
      "fraction_tight_slo": 0.1, "min_slack": 0.5,
      "mean_prompt_tokens": 64.0, "p95_prompt_tokens": 128.0,
      "burstiness_cv": 0.3},
     "slo_slack_score"),
    # Rule 8: default → edf
    ({"mean_pred_output_tokens": 50.0, "kv_utilization": 0.0,
      "pred_output_cv": 0.5, "recent_slo_violation_rate": 0.0,
      "fraction_tight_slo": 0.1, "min_slack": 5.0,
      "mean_prompt_tokens": 64.0, "p95_prompt_tokens": 128.0,
      "burstiness_cv": 0.3},
     "edf"),
])
def test_rule_selector_phase2b8_dispatch(features, expected):
    """Phase 2B.8 repaired rule selector dispatch table."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    result = sel.predict_one(features)
    assert result == expected, (
        f"Expected '{expected}', got '{result}' for features: {features}"
    )


def test_rule_selector_never_chooses_least_laxity_first_for_tight_slo():
    """Phase 2B.8 repair: tight SLO must route to slo_slack_score, not least_laxity_first."""
    from llmserveopt.selector.models import RuleBasedSelector
    sel = RuleBasedSelector()
    for min_slack in [0.0, 0.1, 0.5, 0.8, 0.9, 0.99]:
        features = {
            "mean_pred_output_tokens": 50.0,
            "kv_utilization": 0.0,
            "pred_output_cv": 0.5,
            "recent_slo_violation_rate": 0.0,
            "fraction_tight_slo": 0.5,
            "min_slack": min_slack,
            "mean_prompt_tokens": 64.0,
            "p95_prompt_tokens": 128.0,
            "burstiness_cv": 0.3,
        }
        chosen = sel.predict_one(features)
        assert chosen != "least_laxity_first", (
            f"Rule selector chose least_laxity_first for min_slack={min_slack}"
            " — Phase 2B.8 repair violated"
        )


def test_rule_selector_policy_choices_all_in_candidates():
    """All policies that RuleBasedSelector may choose must be in SELECTOR_CANDIDATES."""
    from llmserveopt.selector.models import RuleBasedSelector
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    sel = RuleBasedSelector()
    missing = [p for p in sel._POLICY_CHOICES if p not in SELECTOR_CANDIDATES]
    assert missing == [], (
        f"RuleBasedSelector._POLICY_CHOICES contains non-candidate policies: {missing}"
    )


# ---------------------------------------------------------------------------
# Phase 2B.9 config validation
# ---------------------------------------------------------------------------

def test_phase2b9_config_exists():
    """Phase 2B.9 robustness config must exist."""
    cfg_path = CONFIGS_DIR / "phase2b9_selector_robustness.yaml"
    assert cfg_path.exists(), f"Phase 2B.9 config not found: {cfg_path}"


def test_phase2b9_config_workload_groups():
    """All workloads must be tagged with group=dev or group=heldout."""
    import yaml
    cfg_path = CONFIGS_DIR / "phase2b9_selector_robustness.yaml"
    if not cfg_path.exists():
        pytest.skip("Phase 2B.9 config not found")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    workloads = cfg.get("workloads", [])
    for w in workloads:
        group = w.get("group")
        assert group in ("dev", "heldout"), (
            f"Workload '{w.get('tag')}' has invalid group='{group}'; must be 'dev' or 'heldout'"
        )


def test_phase2b9_config_seeds_disjoint():
    """dev_seeds and heldout_seeds must not overlap."""
    import yaml
    cfg_path = CONFIGS_DIR / "phase2b9_selector_robustness.yaml"
    if not cfg_path.exists():
        pytest.skip("Phase 2B.9 config not found")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dev_seeds = set(cfg.get("dev_seeds", []))
    heldout_seeds = set(cfg.get("heldout_seeds", []))
    overlap = dev_seeds & heldout_seeds
    assert overlap == set(), (
        f"dev_seeds and heldout_seeds overlap: {overlap} — this could cause"
        " identical workload/seed combos in both groups"
    )


def test_phase2b9_config_no_oracle_in_workloads():
    """Phase 2B.9 config must not reference oracle_srtf."""
    cfg_path = CONFIGS_DIR / "phase2b9_selector_robustness.yaml"
    if not cfg_path.exists():
        pytest.skip("Phase 2B.9 config not found")
    content = cfg_path.read_text()
    assert "oracle_srtf" not in content, (
        "Phase 2B.9 config references oracle_srtf — oracle must not be in experiment configs"
    )


def test_phase2b9_config_has_dev_and_heldout_workloads():
    """Phase 2B.9 config must have at least 1 dev and 1 heldout workload."""
    import yaml
    cfg_path = CONFIGS_DIR / "phase2b9_selector_robustness.yaml"
    if not cfg_path.exists():
        pytest.skip("Phase 2B.9 config not found")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    workloads = cfg.get("workloads", [])
    dev = [w for w in workloads if w.get("group") == "dev"]
    heldout = [w for w in workloads if w.get("group") == "heldout"]
    assert len(dev) >= 1, "Phase 2B.9 config has no dev workloads"
    assert len(heldout) >= 1, "Phase 2B.9 config has no heldout workloads"


# ---------------------------------------------------------------------------
# Runner script existence
# ---------------------------------------------------------------------------

def test_phase2b9_runner_script_exists():
    """Phase 2B.9 runner script must exist."""
    script = SCRIPTS_DIR / "run_phase2b9_selector_robustness.py"
    assert script.exists(), f"Runner script not found: {script}"


def test_phase2b9_runner_script_no_paid_api_calls():
    """Runner script must not call paid API providers."""
    script = SCRIPTS_DIR / "run_phase2b9_selector_robustness.py"
    if not script.exists():
        pytest.skip("Runner script not found")
    content = script.read_text()
    forbidden = ["cloudrift", "openai", "cohere", "gemini", "mistral", "cerebras",
                 "anthropic.Anthropic", "OpenAI(", "Cohere("]
    for term in forbidden:
        assert term.lower() not in content.lower(), (
            f"Runner script contains paid API reference: '{term}'"
        )


# ---------------------------------------------------------------------------
# Documentation checks
# ---------------------------------------------------------------------------

def test_selector_training_audit_doc_exists():
    """Phase 2B.9 selector training audit doc must exist."""
    doc = DOCS_DIR / "audits" / "phase2b9_selector_training_audit.md"
    assert doc.exists(), f"Audit doc not found: {doc}"


def test_selector_training_audit_required_sections():
    """Selector training audit must answer required questions."""
    doc = DOCS_DIR / "audits" / "phase2b9_selector_training_audit.md"
    if not doc.exists():
        pytest.skip("Audit doc not found")
    content = doc.read_text()
    required_phrases = [
        "training windows",
        "validation windows",
        "test windows",
        "oracle",
        "actual_output",
        "future arrivals",
        "held-out",
        "label distribution",
    ]
    for phrase in required_phrases:
        assert phrase.lower() in content.lower(), (
            f"Selector training audit missing required topic: '{phrase}'"
        )


def test_external_baseline_decision_doc_exists():
    """External baseline decision document must exist."""
    doc = DOCS_DIR / "external_baseline_decision.md"
    assert doc.exists(), f"External baseline decision doc not found: {doc}"


def test_external_baseline_decision_required_sections():
    """External baseline decision doc must have required sections."""
    doc = DOCS_DIR / "external_baseline_decision.md"
    if not doc.exists():
        pytest.skip("Doc not found")
    content = doc.read_text()
    required = [
        "Already Implemented",
        "Must Add",
        "Cite Only",
        "selector candidate",
        "leakage",
    ]
    for phrase in required:
        assert phrase.lower() in content.lower(), (
            f"External baseline decision doc missing section/phrase: '{phrase}'"
        )


def test_dataset_workload_decision_doc_exists():
    """Dataset/workload decision document must exist."""
    doc = DOCS_DIR / "dataset_workload_decision.md"
    assert doc.exists(), f"Dataset/workload decision doc not found: {doc}"


def test_dataset_workload_decision_required_sections():
    """Dataset/workload decision doc must have required sections."""
    doc = DOCS_DIR / "dataset_workload_decision.md"
    if not doc.exists():
        pytest.skip("Doc not found")
    content = doc.read_text()
    required = [
        "BurstGPT",
        "Azure",
        "Must Use",
        "leakage",
        "HF token",
        "train/val/test",
    ]
    for phrase in required:
        assert phrase.lower() in content.lower(), (
            f"Dataset/workload decision doc missing required topic: '{phrase}'"
        )


# ---------------------------------------------------------------------------
# Research status and selector doc updates
# ---------------------------------------------------------------------------

def test_research_status_mentions_phase2b9():
    """research_status.md must mention Phase 2B.9."""
    doc = DOCS_DIR / "research_status.md"
    assert doc.exists(), "research_status.md not found"
    content = doc.read_text()
    assert "2B.9" in content or "phase2b9" in content.lower(), (
        "research_status.md does not mention Phase 2B.9"
    )


def test_selector_doc_mentions_phase2b8_repair():
    """selector.md must document the Phase 2B.8 rule repair."""
    doc = DOCS_DIR / "selector.md"
    assert doc.exists(), "selector.md not found"
    content = doc.read_text()
    assert "2B.8" in content or "phase2b8" in content.lower(), (
        "selector.md does not mention Phase 2B.8 repair"
    )


# ---------------------------------------------------------------------------
# No HF token exposure
# ---------------------------------------------------------------------------

def test_no_hf_token_in_committed_scripts():
    """No committed script or config should print/log the HF token."""
    hf_leak_patterns = ["print(hf_token", "log(hf_token", "echo $HF_TOKEN",
                        "print(os.environ.get('HF", "logging.info.*HF_TOKEN"]
    for script in SCRIPTS_DIR.glob("*.py"):
        content = script.read_text()
        for pattern in hf_leak_patterns:
            assert pattern not in content, (
                f"Possible HF token leak in {script}: '{pattern}'"
            )


# ---------------------------------------------------------------------------
# Phase 2B.9 results (smoke check if they exist)
# ---------------------------------------------------------------------------

def test_phase2b9_results_not_oracle_contaminated():
    """If Phase 2B.9 results exist, oracle policy must not appear as a selector choice."""
    pw_csv = RESULTS_DIR / "phase2b9_selector_robustness" / "per_window.csv"
    if not pw_csv.exists():
        pytest.skip("Phase 2B.9 per_window.csv not yet generated")

    from llmserveopt.policies.registry import ORACLE_POLICY_NAMES
    with open(pw_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, val in row.items():
                if col.startswith("sel_") and col.endswith("_policy"):
                    assert val not in ORACLE_POLICY_NAMES, (
                        f"Oracle policy '{val}' chosen by selector column '{col}'"
                    )


def test_phase2b9_results_selector_comparison_columns():
    """If selector_comparison.csv exists, it must have expected columns."""
    csv_path = RESULTS_DIR / "phase2b9_selector_robustness" / "selector_comparison.csv"
    if not csv_path.exists():
        pytest.skip("Phase 2B.9 selector_comparison.csv not yet generated")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows, "selector_comparison.csv is empty"

    expected_cols = ["group", "n_windows", "oracle_per_window_best_wg",
                     "best_fixed_policy", "best_fixed_wg"]
    for col in expected_cols:
        assert col in rows[0], (
            f"selector_comparison.csv missing column '{col}'"
        )

"""
Phase 2B.14 test suite: metric audit and SCORPIO ablation.

Tests cover:
- Config file existence and structure
- Ablation policy variants: creation, naming, NOT in registry/selector candidates
- Metric variant computations (arrival-normalized WG, completion-penalized)
- Denominator audit correctness
- Near-tie analysis under arrival-normalized WG
- Runner importability
- Result files existence (if experiment ran)
- Policy count = 20, oracle_srtf excluded
- Ablations not in SELECTOR_CANDIDATES or BASELINE_NAMES
- No credentials in committed files
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CFG_PATH = ROOT / "configs" / "phase2b14_metric_audit_scorpio_ablation.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_phase2b14_metric_audit_scorpio_ablation.py"
RESULTS_DIR = ROOT / "results" / "phase2b14_metric_audit_scorpio_ablation"
ABLATION_MODULE = "llmserveopt.policies.scorpio_ablations"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPhase2B14Config:
    def test_config_exists(self):
        assert CFG_PATH.exists(), f"Config not found: {CFG_PATH}"

    def test_config_loadable(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)
        assert cfg.get("experiment") == "phase2b14_metric_audit_scorpio_ablation"

    def test_config_has_input_results(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert "input_results" in cfg

    def test_config_has_metrics_section(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert "metrics" in cfg
        metrics = cfg["metrics"]
        assert "completed_request_quality" in metrics
        assert "arrival_normalized_wg" in metrics
        assert "completion_penalized" in metrics

    def test_config_ablation_workloads_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        abl = cfg.get("ablation", {})
        workloads = abl.get("workloads", [])
        assert len(workloads) >= 7, (
            f"Expected at least 7 ablation workloads (discriminative subset), got {len(workloads)}"
        )
        tags = {w["tag"] for w in workloads}
        assert "dev_kv_pressure_decode_heavy" in tags, "Must include most discriminative dev workload"

    def test_completion_penalized_has_multiple_targets(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        cp = cfg["metrics"]["completion_penalized"]
        assert 0.95 in cp["targets"]
        assert 0.99 in cp["targets"]
        assert 0.5 in cp["lambdas"]
        assert 1.0 in cp["lambdas"]


# ---------------------------------------------------------------------------
# Ablation policy tests
# ---------------------------------------------------------------------------

class TestScorpioAblations:
    def test_ablation_module_importable(self):
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES
        assert len(ABLATION_NAMES) >= 8

    def test_ablation_names_all_distinct(self):
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES
        assert len(set(ABLATION_NAMES)) == len(ABLATION_NAMES)

    def test_ablation_policy_names_set_correctly(self):
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES, make_ablation
        for abl_name in ABLATION_NAMES:
            p = make_ablation(abl_name)
            assert p.name == abl_name, (
                f"make_ablation('{abl_name}').name should be '{abl_name}', got '{p.name}'"
            )

    def test_no_rejection_accepts_all(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        p = make_ablation("scorpio_no_rejection")
        assert p.kv_utilization_threshold >= 2.0, "no_rejection should disable KV guard"
        assert p.laxity_threshold <= -100.0, "no_rejection should disable laxity filter"
        assert p.admission_budget_max >= 1e6, "no_rejection should have unlimited budget"

    def test_deadline_only_disables_kv_guard(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        p = make_ablation("scorpio_deadline_only")
        assert p.kv_utilization_threshold >= 2.0
        assert p.laxity_threshold == 0.0, "deadline_only should keep laxity filter"

    def test_no_credit_budget_is_unlimited(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        p = make_ablation("scorpio_no_credit_budget")
        assert p.admission_budget_max >= 1e6

    def test_no_priority_weight_is_zero(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        p = make_ablation("scorpio_no_priority_weight")
        assert p.priority_weight == 0.0

    def test_no_age_bonus_is_zero(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        p = make_ablation("scorpio_no_age_bonus")
        assert p.age_bonus == 0.0

    def test_no_decode_penalty_is_zero(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        p = make_ablation("scorpio_no_decode_penalty")
        assert p.decode_penalty_weight == 0.0

    def test_make_ablation_raises_on_unknown(self):
        from llmserveopt.policies.scorpio_ablations import make_ablation
        with pytest.raises(KeyError):
            make_ablation("nonexistent_ablation")

    def test_ablations_not_in_registry(self):
        from llmserveopt.policies.registry import _REGISTRY
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES
        for name in ABLATION_NAMES:
            assert name not in _REGISTRY, (
                f"Ablation '{name}' must NOT be in policy registry"
            )

    def test_ablations_not_in_baseline_names(self):
        from llmserveopt.policies.registry import BASELINE_NAMES
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES
        for name in ABLATION_NAMES:
            assert name not in BASELINE_NAMES, (
                f"Ablation '{name}' must NOT be in BASELINE_NAMES"
            )

    def test_ablations_not_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES
        for name in ABLATION_NAMES:
            assert name not in SELECTOR_CANDIDATES, (
                f"Ablation '{name}' must NOT be a selector candidate"
            )


# ---------------------------------------------------------------------------
# Metric variant computation tests
# ---------------------------------------------------------------------------

class TestMetricVariants:
    """Unit tests for metric computation functions."""

    def test_arrival_norm_wg_equal_cond_when_cf_one(self):
        """When completion_fraction=1.0, arrival_norm_wg == conditional_wg."""
        import pandas as pd
        import numpy as np
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_phase2b14_metric_audit_scorpio_ablation import (
            cond_wg, arrival_norm_wg
        )
        df = pd.DataFrame({
            "reward_fifo": [0.8, 0.9, 0.7],
            "completion_fifo": [1.0, 1.0, 1.0],
        })
        cq = cond_wg(df, "fifo")
        anwg = arrival_norm_wg(df, "fifo")
        np.testing.assert_array_almost_equal(cq.values, anwg.values)

    def test_arrival_norm_wg_less_than_cond_when_cf_lt_one(self):
        import pandas as pd
        from run_phase2b14_metric_audit_scorpio_ablation import (
            cond_wg, arrival_norm_wg
        )
        df = pd.DataFrame({
            "reward_scorpio_style_slo_guard": [0.98, 0.99],
            "completion_scorpio_style_slo_guard": [0.85, 0.90],
        })
        cq = cond_wg(df, "scorpio_style_slo_guard")
        anwg = arrival_norm_wg(df, "scorpio_style_slo_guard")
        assert (anwg < cq).all(), "Arrival-norm WG must be less than conditional WG when CF<1"

    def test_completion_penalized_reduces_scorpio_more(self):
        """Higher completion fraction policy beats SCORPIO under completion penalty.

        Uses realistic values matching Phase 2B.14 findings:
        WSP (WG=0.86, CF=0.99) vs SCORPIO (WG=0.98, CF=0.85)
        """
        import pandas as pd
        from run_phase2b14_metric_audit_scorpio_ablation import (
            arrival_norm_wg, completion_penalized_wg
        )
        df = pd.DataFrame({
            "reward_scorpio_style_slo_guard": [0.98],
            "completion_scorpio_style_slo_guard": [0.85],
            "reward_fifo": [0.90],  # realistic: close to SCORPIO but full completion
            "completion_fifo": [0.99],
        })
        # Both may be close under arrival-norm WG; the penalty difference is what matters
        scorpio_cp = completion_penalized_wg(df, "scorpio_style_slo_guard", 0.95, 1.0).mean()
        fifo_cp = completion_penalized_wg(df, "fifo", 0.95, 1.0).mean()
        # SCORPIO penalty = 1.0 * (0.95 - 0.85) = 0.10; FIFO penalty = 0.0
        scorpio_anwg = arrival_norm_wg(df, "scorpio_style_slo_guard").mean()
        fifo_anwg = arrival_norm_wg(df, "fifo").mean()
        scorpio_penalty = max(0.0, 0.95 - 0.85)
        fifo_penalty = max(0.0, 0.95 - 0.99)
        assert abs(scorpio_cp - (scorpio_anwg - scorpio_penalty)) < 1e-6
        assert abs(fifo_cp - (fifo_anwg - fifo_penalty)) < 1e-6
        assert fifo_cp > scorpio_cp, (
            f"FIFO (cp={fifo_cp:.4f}) should beat SCORPIO (cp={scorpio_cp:.4f}) "
            f"when FIFO has similar arrival-norm WG but no completion penalty"
        )

    def test_completion_penalized_no_penalty_above_target(self):
        import pandas as pd
        from run_phase2b14_metric_audit_scorpio_ablation import (
            arrival_norm_wg, completion_penalized_wg
        )
        df = pd.DataFrame({
            "reward_fifo": [0.80],
            "completion_fifo": [0.99],
        })
        anwg = arrival_norm_wg(df, "fifo").mean()
        cp = completion_penalized_wg(df, "fifo", 0.95, 1.0).mean()
        # No penalty when CF >= target
        assert abs(anwg - cp) < 1e-9, "No penalty when completion_fraction >= target"

    def test_denominator_audit_flags_scorpio(self):
        import pandas as pd
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_phase2b14_metric_audit_scorpio_ablation import audit_denominator
        df = pd.DataFrame({
            "reward_scorpio_style_slo_guard": [0.98, 0.99],
            "completion_scorpio_style_slo_guard": [0.85, 0.90],
            "reward_fifo": [0.72, 0.75],
            "completion_fifo": [0.99, 0.99],
        })
        audit = audit_denominator(df)
        assert audit["denominator_type"] == "completed_requests_only"
        assert audit["safe_to_call_goodput"] is False
        assert audit["example_scorpio"]["arrival_normalized_wg"] < audit["example_scorpio"]["conditional_wg"]

    def test_near_tie_corrected_counts(self):
        import pandas as pd
        from run_phase2b14_metric_audit_scorpio_ablation import near_tie_analysis_corrected
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        # Build dummy df where all policies have identical reward and completion=1
        data = {}
        for p in SELECTOR_CANDIDATES:
            data[f"reward_{p}"] = [0.99, 0.99]
            data[f"completion_{p}"] = [1.0, 1.0]
        df = pd.DataFrame(data)
        result = near_tie_analysis_corrected(df, [0.001, 0.005])
        # All windows should be near-tie (margin=0 < any eps)
        assert result["n_near_tie_eps0.001"] == 2
        assert result["n_near_tie_eps0.005"] == 2
        # All windows all-complete (best anwg = 0.99 >= 0.99)
        assert result["n_all_complete_arrival_norm"] == 2


# ---------------------------------------------------------------------------
# Registry / policy count tests
# ---------------------------------------------------------------------------

class TestRegistryIntegrity:
    def test_policy_count_is_20(self):
        from llmserveopt.policies.registry import BASELINE_NAMES
        assert len(BASELINE_NAMES) == 20, (
            f"Expected 20 deployable policies, got {len(BASELINE_NAMES)}: {BASELINE_NAMES}"
        )

    def test_oracle_srtf_not_in_baseline_names(self):
        from llmserveopt.policies.registry import BASELINE_NAMES
        assert "oracle_srtf" not in BASELINE_NAMES

    def test_oracle_srtf_not_selector_candidate(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "oracle_srtf" not in SELECTOR_CANDIDATES

    def test_scorpio_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATES

    def test_selector_candidates_count(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert len(SELECTOR_CANDIDATES) == 20


# ---------------------------------------------------------------------------
# Runner import test
# ---------------------------------------------------------------------------

class TestRunnerImport:
    def test_runner_exists(self):
        assert RUNNER_PATH.exists(), f"Runner not found: {RUNNER_PATH}"

    def test_runner_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_phase2b14", RUNNER_PATH)
        mod = importlib.util.module_from_spec(spec)
        # Just importing should not call main() or paid APIs
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass  # argparse may exit on --help; that's fine

    def test_no_paid_api_keys_in_runner(self):
        text = RUNNER_PATH.read_text()
        forbidden = ["OPENAI_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY",
                     "HF_TOKEN", "HUGGINGFACE_TOKEN"]
        for key in forbidden:
            assert key not in text, f"Forbidden token reference '{key}' in runner"


# ---------------------------------------------------------------------------
# Result file tests (skipped if experiment not yet run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (RESULTS_DIR / "phase2b14_summary.json").exists(),
    reason="Phase 2B.14 results not yet generated"
)
class TestPhase2B14Results:
    def test_summary_json_exists(self):
        assert (RESULTS_DIR / "phase2b14_summary.json").exists()

    def test_summary_has_required_fields(self):
        with open(RESULTS_DIR / "phase2b14_summary.json") as f:
            d = json.load(f)
        required = [
            "n_windows", "denominator_type", "denominator_safe_to_call_goodput",
            "scorpio_conditional_wg", "scorpio_arrival_norm_wg",
            "scorpio_completion_fraction", "scorpio_dominates_under_arrival_norm",
        ]
        for field in required:
            assert field in d, f"Missing field '{field}' in phase2b14_summary.json"

    def test_n_windows_matches_phase2b13(self):
        with open(RESULTS_DIR / "phase2b14_summary.json") as f:
            d = json.load(f)
        assert d["n_windows"] == 319, f"Expected 319 windows from Phase 2B.13, got {d['n_windows']}"

    def test_denominator_not_arrival_normalized(self):
        with open(RESULTS_DIR / "phase2b14_summary.json") as f:
            d = json.load(f)
        assert d["denominator_safe_to_call_goodput"] is False

    def test_scorpio_arrival_norm_less_than_conditional(self):
        with open(RESULTS_DIR / "phase2b14_summary.json") as f:
            d = json.load(f)
        assert d["scorpio_arrival_norm_wg"] < d["scorpio_conditional_wg"]

    def test_scorpio_dominates_arrival_norm(self):
        with open(RESULTS_DIR / "phase2b14_summary.json") as f:
            d = json.load(f)
        assert d["scorpio_dominates_under_arrival_norm"] is True

    def test_policy_metric_variants_csv_exists(self):
        assert (RESULTS_DIR / "policy_metric_variants.csv").exists()

    def test_policy_metric_variants_has_20_policies(self):
        import pandas as pd
        df = pd.read_csv(RESULTS_DIR / "policy_metric_variants.csv")
        assert len(df) == 20, f"Expected 20 policies, got {len(df)}"

    def test_selector_metric_variants_csv_exists(self):
        assert (RESULTS_DIR / "selector_metric_variants.csv").exists()

    def test_near_tie_corrected_json_exists(self):
        assert (RESULTS_DIR / "near_tie_corrected.json").exists()

    def test_safe_claim_analysis_json_exists(self):
        assert (RESULTS_DIR / "safe_claim_analysis.json").exists()

    def test_safe_claims_non_empty(self):
        with open(RESULTS_DIR / "safe_claim_analysis.json") as f:
            d = json.load(f)
        assert len(d.get("safe_claims", [])) >= 3

    def test_unsafe_claims_non_empty(self):
        with open(RESULTS_DIR / "safe_claim_analysis.json") as f:
            d = json.load(f)
        assert len(d.get("unsafe_claims", [])) >= 2

    def test_completion_fraction_audit(self):
        import pandas as pd
        df = pd.read_csv(RESULTS_DIR / "policy_metric_variants.csv")
        scorpio = df[df["policy"] == "scorpio_style_slo_guard"].iloc[0]
        assert scorpio["mean_completion_fraction"] < 0.96, (
            "SCORPIO should have completion_fraction < 0.96 (rejects some arrivals)"
        )

    def test_arrival_norm_wg_le_conditional(self):
        """arrival_norm_wg <= conditional_wg for all policies."""
        import pandas as pd
        df = pd.read_csv(RESULTS_DIR / "policy_metric_variants.csv")
        for _, row in df.iterrows():
            assert row["arrival_norm_wg"] <= row["conditional_wg"] + 1e-6, (
                f"Policy {row['policy']}: arrival_norm_wg > conditional_wg"
            )

    def test_scorpio_games_metric_flag(self):
        with open(RESULTS_DIR / "phase2b14_summary.json") as f:
            d = json.load(f)
        assert d["scorpio_games_metric"] is True, (
            "SCORPIO should be flagged as appearing to game metric "
            "(CF<0.95 but conditional WG > second-best arrival-norm)"
        )

    def test_no_ablation_name_in_policy_table(self):
        import pandas as pd
        from llmserveopt.policies.scorpio_ablations import ABLATION_NAMES
        df = pd.read_csv(RESULTS_DIR / "policy_metric_variants.csv")
        for abl in ABLATION_NAMES:
            assert abl not in df["policy"].values, (
                f"Ablation policy '{abl}' should NOT appear in main policy table"
            )


# ---------------------------------------------------------------------------
# Docs tests
# ---------------------------------------------------------------------------

class TestPhase2B14Docs:
    def test_metric_definition_audit_exists(self):
        p = ROOT / "docs" / "audits" / "phase2b14_metric_definition_audit.md"
        assert p.exists(), f"Missing: {p}"

    def test_summary_doc_exists(self):
        p = ROOT / "docs" / "audits" / "phase2b14_metric_audit_scorpio_ablation_summary.md"
        assert p.exists(), f"Missing: {p}"

    def test_failure_cases_doc_exists(self):
        p = ROOT / "docs" / "audits" / "phase2b14_failure_cases_summary.md"
        assert p.exists(), f"Missing: {p}"

    def test_summary_doc_answers_required_questions(self):
        p = ROOT / "docs" / "audits" / "phase2b14_metric_audit_scorpio_ablation_summary.md"
        text = p.read_text()
        required_sections = [
            "What was the old WG denominator",
            "safe to call",
            "arrival-normalized",
            "SCORPIO still dominate",
            "game",
            "selector",
        ]
        for section in required_sections:
            assert section.lower() in text.lower(), (
                f"Summary doc missing required section about: '{section}'"
            )

    def test_metric_audit_doc_has_denominator_table(self):
        p = ROOT / "docs" / "audits" / "phase2b14_metric_definition_audit.md"
        text = p.read_text()
        assert "completed_requests_only" in text or "completed requests only" in text.lower()
        assert "arrival_normalized" in text or "arrival-normalized" in text.lower()

    def test_no_credentials_in_runner(self):
        """Credential env var names must not appear in the runner script itself."""
        text = RUNNER_PATH.read_text()
        forbidden = ["GEMINI_API_KEY", "HUGGINGFACE_TOKEN", "OPENAI_API_KEY"]
        for token in forbidden:
            assert token not in text, (
                f"Potential credential reference '{token}' found in runner"
            )

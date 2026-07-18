"""
Phase 2B.13 test suite: selector training and SCORPIO suspicion audit.

Tests cover:
- Config file existence and structure
- Extended diversity seeds and 6 new differentiated workloads
- Dev/heldout regression continuity from Phase 2B.11/2B.12
- Near-tie / regret / leakage audit helpers (unit tests)
- Runner importability (no paid APIs, no HF tokens)
- SCORPIO still in selector policy choices
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CFG_PATH = ROOT / "configs" / "phase2b13_selector_training_and_suspicion_audit.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_phase2b13_selector_training_and_suspicion_audit.py"


# ---------------------------------------------------------------------------
# Config file tests
# ---------------------------------------------------------------------------

class TestPhase2B13Config:
    def test_config_exists(self):
        assert CFG_PATH.exists(), f"Config not found: {CFG_PATH}"

    def test_config_loadable(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)
        assert cfg.get("experiment") == "phase2b13_selector_training_and_suspicion_audit"

    def test_config_has_workloads(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert "workloads" in cfg
        assert len(cfg["workloads"]) >= 25, (
            f"Expected at least 25 workloads (9 regression + diversity), "
            f"got {len(cfg['workloads'])}"
        )

    def test_config_has_dev_and_heldout_groups(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        workloads = cfg["workloads"]
        dev = [w for w in workloads if w.get("group") == "dev"]
        heldout = [w for w in workloads if w.get("group") == "heldout"]
        assert len(dev) == 4, f"Expected 4 dev workloads, got {len(dev)}"
        assert len(heldout) == 5, f"Expected 5 heldout workloads, got {len(heldout)}"

    def test_config_has_diversity_group(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        assert len(diversity) >= 20, (
            f"Expected at least 20 diversity workloads (14 Phase 2B.12 + 6 new), "
            f"got {len(diversity)}"
        )

    def test_new_differentiated_workloads_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        tags = {w["tag"] for w in diversity}
        new_tags = {
            "div_kv_extreme_tight_slo",
            "div_kv_extreme_decode_only",
            "div_high_overload_tight_priority",
            "div_kv_mixed_extreme_noise",
            "div_decode_saturation_bursty",
            "div_extreme_overload_short_tight",
        }
        assert new_tags.issubset(tags), f"Missing new differentiated workloads: {new_tags - tags}"

    def test_extended_diversity_seeds(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        seeds = cfg.get("diversity_seeds", [])
        assert seeds == [6, 7, 8, 9, 10, 11], f"Expected extended seeds [6..11], got {seeds}"

    def test_near_tie_thresholds_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        thresholds = cfg.get("near_tie_thresholds", [])
        assert 0.001 in thresholds
        assert 0.005 in thresholds
        assert 0.010 in thresholds

    def test_selector_training_split_config(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        st = cfg.get("selector_training", {})
        assert st.get("train_groups") == ["dev", "diversity"]
        assert st.get("test_groups") == ["heldout"]
        assert st.get("train_diversity_seeds") == [6, 7, 8, 9, 10]
        assert st.get("val_diversity_seeds") == [11]

    def test_diversity_seeds_no_overlap(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity_seeds = set(cfg["diversity_seeds"])
        dev_seeds = set(cfg["dev_seeds"])
        heldout_seeds = set(cfg["heldout_seeds"])
        assert not diversity_seeds & dev_seeds
        assert not diversity_seeds & heldout_seeds

    def test_regression_workload_tags_match_phase2b12(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        reg_tags = {w["tag"] for w in cfg["workloads"] if w.get("group") in ("dev", "heldout")}
        expected_tags = {
            "dev_overloaded_mixed_slo",
            "dev_high_prediction_noise",
            "dev_kv_pressure_decode_heavy",
            "dev_overloaded_prefill_heavy",
            "heldout_moderate_kv_pressure",
            "heldout_very_high_noise",
            "heldout_prefill_overloaded",
            "heldout_bursty_mixed_slo",
            "heldout_burstgpt_smoke",
        }
        assert reg_tags == expected_tags

    def test_phase2b12_reference_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        ref = cfg.get("phase2b12_reference", {})
        assert "rule_based_wg" in ref
        assert abs(ref["rule_based_wg"].get("overall", 0) - 0.9721) < 0.001

    def test_rf_feasibility_min_windows_200(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("rf_feasibility", {}).get("min_windows") == 200

    def test_no_cloudrift_or_paid_api_in_config(self):
        cfg_text = CFG_PATH.read_text()
        forbidden = ["cloudrift", "openai", "cohere", "gemini", "mistral", "cerebras"]
        for term in forbidden:
            assert term.lower() not in cfg_text.lower()


# ---------------------------------------------------------------------------
# Runner importability
# ---------------------------------------------------------------------------

class TestPhase2B13RunnerImports:
    def test_runner_exists(self):
        assert RUNNER_PATH.exists()

    def test_runner_imports_without_paid_apis(self):
        spec = importlib.util.spec_from_file_location("run_phase2b13", str(RUNNER_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in (
            "main", "compute_near_tie_stats", "filter_non_tie_rows",
            "compute_regret_weights", "leakage_audit", "AlwaysScorpioSelector",
            "evaluate_selector_on_rows", "train_rf_dt",
        ):
            assert hasattr(module, name), f"Missing export: {name}"

    def test_runner_no_hf_token_usage(self):
        runner_text = RUNNER_PATH.read_text()
        forbidden = ["HUGGINGFACE_TOKEN", "HF_TOKEN", "hf_hub_download", "from_pretrained"]
        for term in forbidden:
            assert term not in runner_text


# ---------------------------------------------------------------------------
# Near-tie / regret unit tests
# ---------------------------------------------------------------------------

class TestNearTieAnalysis:
    def _mod(self):
        spec = importlib.util.spec_from_file_location("run_phase2b13", str(RUNNER_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_compute_near_tie_stats(self):
        mod = self._mod()
        rows = [
            {"policy_margin": 0.0, "best_weighted_goodput": 1.0},
            {"policy_margin": 0.05, "best_weighted_goodput": 0.8},
            {"policy_margin": 0.0005, "best_weighted_goodput": 0.99},
        ]
        stats = mod.compute_near_tie_stats(rows, [0.001, 0.01])
        assert stats["n_total"] == 3
        assert stats["n_near_tie_eps0.001"] == 2
        assert stats["n_meaningful_eps0.001"] == 1

    def test_filter_non_tie_rows(self):
        mod = self._mod()
        rows = [
            {"policy_margin": 0.0},
            {"policy_margin": 0.05},
            {"policy_margin": 0.0005},
        ]
        filtered = mod.filter_non_tie_rows(rows, 0.001)
        assert len(filtered) == 1
        assert filtered[0]["policy_margin"] == 0.05

    def test_compute_regret_weights_normalised(self):
        mod = self._mod()
        rows = [{"policy_margin": 0.1}, {"policy_margin": 0.0}]
        w = mod.compute_regret_weights(rows, epsilon=0.001)
        assert abs(w.sum() - 1.0) < 1e-9
        assert w[0] > w[1]


class TestLeakageAudit:
    def test_leakage_audit_passes_clean_rows(self):
        spec = importlib.util.spec_from_file_location("run_phase2b13", str(RUNNER_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [{"feature_mode": "online_prefix", "best_policy": "edf"}]
        result = module.leakage_audit(rows)
        assert result["pass"] is True
        assert result["oracle_in_selector_candidates"] is False


class TestAlwaysScorpioSelector:
    def test_always_scorpio_predicts_scorpio(self):
        spec = importlib.util.spec_from_file_location("run_phase2b13", str(RUNNER_PATH))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sel = module.AlwaysScorpioSelector()
        rows = [{"feat_arrival_rate": 1.0}] * 3
        assert sel.predict(rows) == ["scorpio_style_slo_guard"] * 3


# ---------------------------------------------------------------------------
# Selector integration (static checks)
# ---------------------------------------------------------------------------

class TestPhase2B13SelectorIntegration:
    def test_scorpio_in_policy_choices(self):
        from llmserveopt.selector.models import RuleBasedSelector
        sel = RuleBasedSelector()
        assert "scorpio_style_slo_guard" in sel._POLICY_CHOICES

    def test_selector_candidates_count(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert len(SELECTOR_CANDIDATES) == 20

    def test_oracle_not_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "oracle_srtf" not in SELECTOR_CANDIDATES

    def test_burstgpt_files_exist(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        for w in cfg["workloads"]:
            if w.get("source") == "extended_jsonl":
                trace = ROOT / w["trace_path"]
                assert trace.exists(), f"BurstGPT trace not found: {trace}"


class TestPhase2B13SmokeOutputs:
    """Verify smoke run produces required summary artifacts."""

    REQUIRED_FILES = [
        "selector_comparison.csv",
        "label_distribution.csv",
        "label_distribution_non_tie.csv",
        "near_tie_summary.csv",
        "always_scorpio_comparison.csv",
        "chosen_policy_distribution.csv",
        "completion_admission_summary.csv",
        "objective_sensitivity.csv",
        "failure_cases.csv",
        "leakage_audit.json",
        "rf_dt_training_summary.json",
        "metadata.json",
    ]

    def test_smoke_output_files_exist(self):
        out = ROOT / "results" / "phase2b13_selector_training_and_suspicion_audit"
        for fname in self.REQUIRED_FILES:
            assert (out / fname).exists(), f"Missing smoke output: {fname}"

    def test_failure_cases_csv_has_rows(self):
        import csv
        path = ROOT / "results" / "phase2b13_selector_training_and_suspicion_audit" / "failure_cases.csv"
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        assert "failure_id" in rows[0]


class TestPhase2B13ReportStatus:
    def test_report_detects_phase2b13(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "report_research_status",
            str(ROOT / "scripts" / "report_research_status.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        status = mod.gather_status()
        assert status["phase2b13"]["artifacts_present"] is True


class TestPhase2B13Docs:
    def test_summary_doc_exists(self):
        path = ROOT / "docs" / "audits" / "phase2b13_selector_training_and_suspicion_audit_summary.md"
        assert path.exists(), "Phase 2B.13 summary audit doc missing"

    def test_failure_cases_doc_exists(self):
        path = ROOT / "docs" / "audits" / "phase2b13_failure_cases_summary.md"
        assert path.exists(), "Phase 2B.13 failure cases doc missing"

    def test_summary_doc_mentions_always_scorpio(self):
        text = (ROOT / "docs" / "audits" / "phase2b13_selector_training_and_suspicion_audit_summary.md").read_text()
        assert "always-SCORPIO" in text or "always_scorpio" in text

    def test_summary_doc_mentions_near_tie(self):
        text = (ROOT / "docs" / "audits" / "phase2b13_selector_training_and_suspicion_audit_summary.md").read_text()
        assert "near-tie" in text.lower() or "near_tie" in text

"""
Phase 2B.12 test suite: workload diversity for selector label analysis.

Tests cover:
- Config file existence and structure
- Diversity workload group presence (14 new workloads)
- Dev/heldout regression continuity from Phase 2B.9/2B.11
- Runner importability (no paid APIs, no HF tokens)
- check_rf_feasibility logic (unit tests)
- label_diversity_summary logic (unit tests)
- per_workload_label_table (unit tests)
- SCORPIO still in selector policy choices
- Diversity seeds do not overlap with dev/heldout seeds
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CFG_PATH = ROOT / "configs" / "phase2b12_workload_diversity_selector_labels.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_phase2b12_workload_diversity_selector_labels.py"


# ---------------------------------------------------------------------------
# Config file tests
# ---------------------------------------------------------------------------

class TestPhase2B12Config:
    def test_config_exists(self):
        assert CFG_PATH.exists(), f"Config not found: {CFG_PATH}"

    def test_config_loadable(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)
        assert cfg.get("experiment") == "phase2b12_workload_diversity_selector_labels"

    def test_config_has_workloads(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert "workloads" in cfg
        assert len(cfg["workloads"]) >= 20, (
            f"Expected at least 20 workloads (9 regression + diversity), "
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
        assert len(diversity) >= 10, (
            f"Expected at least 10 diversity workloads, got {len(diversity)}"
        )

    def test_diversity_workloads_have_unique_tags(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        tags = [w["tag"] for w in diversity]
        assert len(tags) == len(set(tags)), "Duplicate tags in diversity group"

    def test_diversity_workloads_start_with_div_prefix(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        for w in diversity:
            assert w["tag"].startswith("div_"), (
                f"Diversity workload tag should start with 'div_': {w['tag']}"
            )

    def test_config_has_diversity_seeds(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert "diversity_seeds" in cfg
        diversity_seeds = cfg["diversity_seeds"]
        dev_seeds = cfg["dev_seeds"]
        heldout_seeds = cfg["heldout_seeds"]
        # No overlap
        assert not set(diversity_seeds) & set(dev_seeds), "Diversity seeds overlap with dev seeds"
        assert not set(diversity_seeds) & set(heldout_seeds), "Diversity seeds overlap with heldout"

    def test_config_has_rf_feasibility_thresholds(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        rf = cfg.get("rf_feasibility", {})
        assert "min_windows" in rf
        assert "min_policies_winning" in rf
        assert "min_windows_per_policy" in rf
        assert "max_single_policy_fraction" in rf

    def test_regression_workload_tags_match_phase2b11(self):
        """Regression workloads must match Phase 2B.11 for continuity."""
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
        assert reg_tags == expected_tags, (
            f"Regression tags mismatch.\nExpected: {expected_tags}\nGot: {reg_tags}"
        )

    def test_diversity_prefill_heavy_workload_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        prefill_tags = [w["tag"] for w in diversity if "prefill" in w["tag"]]
        assert len(prefill_tags) >= 1, "Expected at least 1 prefill-heavy diversity workload"

    def test_diversity_decode_heavy_workload_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        decode_tags = [w["tag"] for w in diversity if "decode" in w["tag"]]
        assert len(decode_tags) >= 1, "Expected at least 1 decode-heavy diversity workload"

    def test_diversity_burstgpt_workloads_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        diversity = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        burstgpt = [w for w in diversity if w.get("source") == "extended_jsonl"]
        assert len(burstgpt) >= 1, "Expected at least 1 BurstGPT diversity workload"

    def test_phase2b11_reference_present(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        ref = cfg.get("phase2b11_reference", {})
        assert "rule_based_wg" in ref
        assert "scorpio_fixed_wg" in ref
        # Overall rule_based_wg should match Phase 2B.11 result
        assert abs(ref["rule_based_wg"].get("overall", 0) - 0.9518) < 0.001

    def test_no_cloudrift_or_paid_api_in_config(self):
        """Config must not reference paid API providers."""
        cfg_text = CFG_PATH.read_text()
        forbidden = ["cloudrift", "openai", "cohere", "gemini", "mistral", "cerebras"]
        for term in forbidden:
            assert term.lower() not in cfg_text.lower(), (
                f"Forbidden paid API provider '{term}' found in config"
            )


# ---------------------------------------------------------------------------
# Runner importability
# ---------------------------------------------------------------------------

class TestPhase2B12RunnerImports:
    def test_runner_exists(self):
        assert RUNNER_PATH.exists(), f"Runner not found: {RUNNER_PATH}"

    def test_runner_imports_without_paid_apis(self):
        """Runner must be importable without triggering paid API calls."""
        spec = importlib.util.spec_from_file_location(
            "run_phase2b12", str(RUNNER_PATH)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main")
        assert hasattr(module, "build_rows_for_group")
        assert hasattr(module, "compute_label_distribution")
        assert hasattr(module, "check_rf_feasibility")
        assert hasattr(module, "label_diversity_summary")
        assert hasattr(module, "per_workload_label_table")

    def test_runner_no_hf_token_usage(self):
        """Runner source must not attempt to use HuggingFace tokens."""
        runner_text = RUNNER_PATH.read_text()
        forbidden = ["HUGGINGFACE_TOKEN", "HF_TOKEN", "hf_hub_download", "from_pretrained"]
        for term in forbidden:
            assert term not in runner_text, (
                f"Forbidden HF token usage '{term}' found in runner"
            )


# ---------------------------------------------------------------------------
# check_rf_feasibility unit tests
# ---------------------------------------------------------------------------

class TestCheckRFFeasibility:
    def _run(self, dist, min_w=200, min_pol=3, min_win_pp=10, max_frac=0.85):
        from run_phase2b12_workload_diversity_selector_labels import check_rf_feasibility
        return check_rf_feasibility(dist, min_w, min_pol, min_win_pp, max_frac)

    def test_all_criteria_met(self):
        dist = {
            "scorpio_style_slo_guard": 100,
            "edf": 50,
            "weighted_shortest_processing": 30,
            "slo_slack_score": 20,
        }
        feasible, details = self._run(dist, min_w=200)
        assert feasible
        assert details["n_policies_with_enough_windows"] == 4

    def test_fails_window_count(self):
        dist = {"scorpio_style_slo_guard": 50, "edf": 30, "slo_slack_score": 20}
        feasible, details = self._run(dist, min_w=200)
        assert not feasible
        assert not details["passes_window_count"]
        assert details["total_windows"] == 100

    def test_fails_policy_spread(self):
        dist = {"scorpio_style_slo_guard": 180, "edf": 20}
        feasible, details = self._run(dist, min_w=200, min_pol=3, min_win_pp=10)
        assert not feasible
        assert not details["passes_policy_spread"]

    def test_fails_concentration(self):
        dist = {
            "scorpio_style_slo_guard": 170,
            "edf": 15,
            "slo_slack_score": 15,
        }
        feasible, details = self._run(dist, min_w=200, max_frac=0.85)
        assert not feasible
        assert not details["passes_concentration"]
        assert abs(details["top_policy_fraction"] - 170 / 200) < 0.001

    def test_boundary_exactly_85_percent(self):
        dist = {
            "scorpio_style_slo_guard": 170,
            "edf": 10,
            "slo_slack_score": 10,
            "wsp": 10,
        }
        feasible, _ = self._run(dist, min_w=200, max_frac=0.85)
        assert not feasible  # 170/200 = 0.85 is NOT strictly < 0.85

    def test_scorpio_heavy_not_feasible(self):
        """60-window suite where SCORPIO wins all → not feasible."""
        dist = {"scorpio_style_slo_guard": 60}
        feasible, details = self._run(dist, min_w=200)
        assert not feasible
        assert not details["passes_window_count"]
        assert not details["passes_policy_spread"]

    def test_empty_distribution(self):
        feasible, details = self._run({})
        assert not feasible
        assert details["total_windows"] == 0


# ---------------------------------------------------------------------------
# label_diversity_summary unit tests
# ---------------------------------------------------------------------------

class TestLabelDiversitySummary:
    def _run(self, rows):
        from run_phase2b12_workload_diversity_selector_labels import label_diversity_summary
        return label_diversity_summary(rows, "test")

    def _make_rows(self, labels):
        return [{"best_policy": lbl} for lbl in labels]

    def test_single_dominant_label(self):
        rows = self._make_rows(["scorpio_style_slo_guard"] * 10)
        summary = self._run(rows)
        assert summary["top_label"] == "scorpio_style_slo_guard"
        assert summary["top_label_fraction"] == 1.0
        assert summary["n_distinct_labels"] == 1

    def test_diverse_labels(self):
        rows = self._make_rows(
            ["edf"] * 5 + ["wsp"] * 5 + ["sarathi"] * 5 + ["scorpio"] * 5
        )
        summary = self._run(rows)
        assert summary["n_distinct_labels"] == 4
        assert summary["top_label_fraction"] == 0.25

    def test_empty_rows(self):
        summary = self._run([])
        assert summary["n_windows"] == 0
        assert summary["n_distinct_labels"] == 0


# ---------------------------------------------------------------------------
# per_workload_label_table unit tests
# ---------------------------------------------------------------------------

class TestPerWorkloadLabelTable:
    def test_groups_by_workload_tag(self):
        from run_phase2b12_workload_diversity_selector_labels import per_workload_label_table
        rows = [
            {"trace_id": "wl_a_s0", "best_policy": "edf", "best_weighted_goodput": 0.9,
             "reward_edf": 0.9, "reward_scorpio_style_slo_guard": 0.8},
            {"trace_id": "wl_a_s1", "best_policy": "edf", "best_weighted_goodput": 0.95,
             "reward_edf": 0.95, "reward_scorpio_style_slo_guard": 0.85},
            {"trace_id": "wl_b_s0", "best_policy": "scorpio_style_slo_guard",
             "best_weighted_goodput": 0.99,
             "reward_edf": 0.7, "reward_scorpio_style_slo_guard": 0.99},
        ]
        table = per_workload_label_table(rows)
        assert "wl_a" in table
        assert "wl_b" in table
        assert table["wl_a"]["n_windows"] == 2
        assert table["wl_b"]["n_windows"] == 1

    def test_label_distribution_per_workload(self):
        from run_phase2b12_workload_diversity_selector_labels import per_workload_label_table
        rows = [
            {"trace_id": "wl_a_s0", "best_policy": "edf",
             "best_weighted_goodput": 0.9, "reward_edf": 0.9},
            {"trace_id": "wl_a_s1", "best_policy": "edf",
             "best_weighted_goodput": 0.9, "reward_edf": 0.9},
            {"trace_id": "wl_a_s2", "best_policy": "slo_slack_score",
             "best_weighted_goodput": 0.95, "reward_edf": 0.8,
             "reward_slo_slack_score": 0.95},
        ]
        table = per_workload_label_table(rows)
        label_dist = table["wl_a"]["label_distribution"]
        assert label_dist.get("edf", 0) == 2
        assert label_dist.get("slo_slack_score", 0) == 1


# ---------------------------------------------------------------------------
# Selector integration (static checks)
# ---------------------------------------------------------------------------

class TestPhase2B12SelectorIntegration:
    def test_scorpio_in_policy_choices(self):
        from llmserveopt.selector.models import RuleBasedSelector
        sel = RuleBasedSelector()
        assert "scorpio_style_slo_guard" in sel._POLICY_CHOICES

    def test_selector_candidates_count(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert len(SELECTOR_CANDIDATES) == 20

    def test_scorpio_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATES

    def test_oracle_not_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "oracle_srtf" not in SELECTOR_CANDIDATES

    def test_design_doc_exists(self):
        design_doc = ROOT / "docs" / "audits" / "phase2b12_workload_diversity_design.md"
        assert design_doc.exists()

    def test_burstgpt_files_exist(self):
        """Verify BurstGPT trace files referenced in config exist."""
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        for w in cfg["workloads"]:
            if w.get("source") == "extended_jsonl":
                trace_path = ROOT / w["trace_path"]
                assert trace_path.exists(), f"BurstGPT file not found: {trace_path}"

"""
Phase 2B.15 tests: corrected-objective selector retraining.

Covers:
- Config existence and required fields
- Metric computation (arrival-norm WG, completion-penalized WG)
- Relabeling under arrival-norm WG
- Near-tie analysis under arrival-norm WG
- Selector classes (AlwaysWSP, KNNAnwg, PerPolicyRegressionAnwg, SafeFallbackWsp)
- Train/val/test split correctness
- Registry integrity (no new policies or oracle leakage)
- Result files (skipped if experiment not yet run)
- Documentation existence
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_RESULTS_DIR = ROOT / "results" / "phase2b15_corrected_objective_selector_retraining"
_RESULTS_EXIST = _RESULTS_DIR.exists() and any(_RESULTS_DIR.iterdir())
_SKIP_RESULTS = pytest.mark.skipif(
    not _RESULTS_EXIST,
    reason="Phase 2B.15 results not yet generated — run the script first",
)

_B14_RESULTS_DIR = ROOT / "results" / "phase2b14_metric_audit_scorpio_ablation"
_B14_EXISTS = _B14_RESULTS_DIR.exists() and (_B14_RESULTS_DIR / "ablation_gap_analysis.json").exists()
_SKIP_B14 = pytest.mark.skipif(
    not _B14_EXISTS,
    reason="Phase 2B.14 ablation results not found",
)


# ============================================================================
# Config tests
# ============================================================================

class TestConfig:
    _cfg_path = ROOT / "configs" / "phase2b15_corrected_objective_selector_retraining.yaml"

    def test_config_exists(self):
        assert self._cfg_path.exists(), f"Config not found: {self._cfg_path}"

    def test_config_loads(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_experiment_key(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["experiment"] == "phase2b15_corrected_objective_selector_retraining"

    def test_input_dir_references_b13(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "phase2b13" in cfg["input_dir"]

    def test_phase2b14_input_dir(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "phase2b14_input_dir" in cfg
        assert "phase2b14" in cfg["phase2b14_input_dir"]

    def test_output_dir(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "phase2b15" in cfg["output_dir"]

    def test_near_tie_thresholds(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        thresholds = cfg["near_tie_thresholds"]
        assert 0.001 in thresholds
        assert 0.005 in thresholds
        assert 0.010 in thresholds

    def test_safe_fallback_default_is_wsp(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        sf_cfg = cfg.get("safe_fallback", {})
        assert sf_cfg.get("default_policy") == "weighted_shortest_processing"

    def test_knn_config(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        knn = cfg.get("knn", {})
        assert knn.get("k", 0) >= 3

    def test_selector_training_split_seeds(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        st = cfg.get("selector_training", {})
        assert set(st["train_diversity_seeds"]) == {6, 7, 8, 9, 10}
        assert set(st["val_diversity_seeds"]) == {11}

    def test_metric_variants_defined(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        names = {v["name"] for v in cfg.get("metric_variants", [])}
        assert "arrival_normalized_wg" in names
        assert "completed_request_quality" in names

    def test_completion_penalized_configs(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        cp = cfg.get("completion_penalized", {})
        configs = cp.get("configs", [])
        keys = {c["key"] for c in configs}
        assert "cp_wg_t095_l05" in keys
        assert "cp_wg_t099_l10" in keys


# ============================================================================
# Metric computation tests
# ============================================================================

def _make_row(cond_wg: float, cf: float, policy: str = "scorpio_style_slo_guard") -> Dict:
    return {
        f"reward_{policy}": cond_wg,
        f"completion_{policy}": cf,
    }


class TestMetricComputation:

    def _import_helpers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_anwg_identity_when_cf_one(self):
        mod = self._import_helpers()
        row = _make_row(0.8, 1.0)
        assert abs(mod._anwg(row, "scorpio_style_slo_guard") - 0.8) < 1e-6

    def test_anwg_scales_with_cf(self):
        mod = self._import_helpers()
        row = _make_row(1.0, 0.9)
        assert abs(mod._anwg(row, "scorpio_style_slo_guard") - 0.9) < 1e-6

    def test_anwg_scorpio_reduced(self):
        mod = self._import_helpers()
        # SCORPIO pattern: high cond_wg, low CF
        row = _make_row(0.9846, 0.899)
        anwg = mod._anwg(row, "scorpio_style_slo_guard")
        assert abs(anwg - 0.8846954) < 0.001

    def test_cp_wg_no_penalty_when_cf_exceeds_target(self):
        mod = self._import_helpers()
        row = _make_row(0.95, 0.99)
        cp = mod._cp_wg(row, "scorpio_style_slo_guard", target=0.95, lam=1.0)
        # CF=0.99 > target=0.95 → no penalty
        assert abs(cp - mod._anwg(row, "scorpio_style_slo_guard")) < 1e-6

    def test_cp_wg_penalty_when_cf_below_target(self):
        mod = self._import_helpers()
        row = _make_row(0.9846, 0.899)
        anwg = mod._anwg(row, "scorpio_style_slo_guard")
        cp = mod._cp_wg(row, "scorpio_style_slo_guard", target=0.95, lam=1.0)
        # penalty = 1.0 * (0.95 - 0.899) = 0.051
        expected_penalty = 1.0 * (0.95 - 0.899)
        assert abs(cp - (anwg - expected_penalty)) < 1e-6

    def test_compute_all_metrics_keys(self):
        mod = self._import_helpers()
        row = _make_row(0.95, 0.95)
        metrics = mod.compute_all_metrics(row, "scorpio_style_slo_guard")
        assert "completed_request_quality" in metrics
        assert "arrival_normalized_wg" in metrics
        assert "cp_wg_t095_l05" in metrics
        assert "cp_wg_t099_l05" in metrics
        assert "cp_wg_t099_l10" in metrics

    def test_anwg_missing_completion_defaults_to_one(self):
        mod = self._import_helpers()
        row = {"reward_fifo": 0.85}  # no completion_fifo key
        assert abs(mod._anwg(row, "fifo") - 0.85) < 1e-6

    def test_wsp_beats_scorpio_under_cp_wg(self):
        """WSP beats SCORPIO under cp_wg when SCORPIO's CF is significantly below target."""
        mod = self._import_helpers()
        # SCORPIO: high cond_wg but CF=0.80 → anwg=0.76, penalty=0.5*(0.95-0.80)=0.075, cp=0.685
        scorpio_row = _make_row(0.95, 0.80)
        # WSP: CF=0.99 (no penalty) → anwg=0.8415, cp=0.8415
        wsp_row = _make_row(0.85, 0.99, policy="weighted_shortest_processing")
        cp_scorpio = mod._cp_wg(scorpio_row, "scorpio_style_slo_guard", 0.95, 0.5)
        cp_wsp = mod._cp_wg(wsp_row, "weighted_shortest_processing", 0.95, 0.5)
        assert cp_wsp > cp_scorpio, f"Expected WSP {cp_wsp:.4f} > SCORPIO {cp_scorpio:.4f}"


# ============================================================================
# Relabeling tests
# ============================================================================

class TestRelabeling:

    def _import_helpers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_multi_row(self, policy_values: Dict[str, tuple]) -> Dict:
        """policy_values: {policy: (cond_wg, cf)}"""
        row = {}
        for p, (wg, cf) in policy_values.items():
            row[f"reward_{p}"] = wg
            row[f"completion_{p}"] = cf
        return row

    def test_best_policy_anwg_selected_correctly(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        row = {}
        for p in SELECTOR_CANDIDATES:
            row[f"reward_{p}"] = 0.5
            row[f"completion_{p}"] = 1.0
        row["reward_scorpio_style_slo_guard"] = 0.9846
        row["completion_scorpio_style_slo_guard"] = 0.899
        row["reward_weighted_shortest_processing"] = 0.85
        row["completion_weighted_shortest_processing"] = 0.99
        rows = mod.relabel_rows([row])
        anwg_s = 0.9846 * 0.899
        anwg_w = 0.85 * 0.99
        expected = "scorpio_style_slo_guard" if anwg_s > anwg_w else "weighted_shortest_processing"
        assert rows[0]["best_policy_anwg"] == expected

    def test_fifo_wins_when_cf_one_vs_scorpio_near_one(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        row = {}
        for p in SELECTOR_CANDIDATES:
            row[f"reward_{p}"] = 1.0
            row[f"completion_{p}"] = 1.0
        # SCORPIO: cond_wg=1.0, CF=0.999 → anwg=0.999
        row["completion_scorpio_style_slo_guard"] = 0.999
        rows = mod.relabel_rows([row])
        # Some policy with CF=1.0 should win (FIFO, EDF, etc.) — not SCORPIO
        assert rows[0]["best_policy_anwg"] != "scorpio_style_slo_guard"

    def test_policy_margin_anwg_is_non_negative(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        row = {}
        for p in SELECTOR_CANDIDATES:
            row[f"reward_{p}"] = float(np.random.rand())
            row[f"completion_{p}"] = float(0.8 + 0.2 * np.random.rand())
        rows = mod.relabel_rows([row])
        assert rows[0]["policy_margin_anwg"] >= 0.0

    def test_near_tie_stat_all_complete(self):
        mod = self._import_helpers()
        row = {"best_anwg": 1.0, "policy_margin_anwg": 0.0}
        stats = mod.near_tie_stats([row] * 100, [0.001, 0.005, 0.010])
        assert stats["all_complete_fraction_anwg"] == 1.0
        assert stats["n_meaningful_eps0.001"] == 0

    def test_near_tie_stat_discriminative(self):
        mod = self._import_helpers()
        row_d = {"best_anwg": 0.7, "policy_margin_anwg": 0.2}
        row_t = {"best_anwg": 1.0, "policy_margin_anwg": 0.0}
        stats = mod.near_tie_stats([row_d] * 10 + [row_t] * 90, [0.001, 0.005, 0.010])
        assert stats["n_meaningful_eps0.001"] == 10
        assert stats["all_complete_fraction_anwg"] == pytest.approx(0.9, abs=0.01)

    def test_filter_meaningful_removes_near_ties(self):
        mod = self._import_helpers()
        rows = [
            {"policy_margin_anwg": 0.0001},
            {"policy_margin_anwg": 0.01},
            {"policy_margin_anwg": 0.1},
        ]
        filtered = mod.filter_meaningful(rows, eps=0.005)
        assert len(filtered) == 2


# ============================================================================
# Selector class tests
# ============================================================================

class TestSelectorClasses:

    def _import_helpers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_always_wsp_predicts_wsp(self):
        mod = self._import_helpers()
        sel = mod.AlwaysWSPSelector()
        rows = [{}, {}, {}]
        preds = sel.predict(rows)
        assert all(p == "weighted_shortest_processing" for p in preds)

    def test_always_scorpio_predicts_scorpio(self):
        mod = self._import_helpers()
        sel = mod.AlwaysScorpioSelector()
        rows = [{}, {}, {}]
        preds = sel.predict(rows)
        assert all(p == "scorpio_style_slo_guard" for p in preds)

    def test_safe_fallback_wsp_uses_base_when_better(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        base = mod.AlwaysScorpioSelector()
        sf = mod.SafeFallbackWspSelector(base, margin=0.01)
        # Row where SCORPIO anwg >> WSP anwg → base selector wins
        row = {f"reward_{p}": 0.5 for p in SELECTOR_CANDIDATES}
        for p in SELECTOR_CANDIDATES:
            row[f"completion_{p}"] = 1.0
        row["reward_scorpio_style_slo_guard"] = 0.95
        row["completion_scorpio_style_slo_guard"] = 0.95
        row["reward_weighted_shortest_processing"] = 0.50
        row["completion_weighted_shortest_processing"] = 1.0
        preds = sf.predict([row])
        # SCORPIO anwg=0.9025 >> WSP anwg=0.5 → margin 0.4025 > 0.01 → use SCORPIO
        assert preds[0] == "scorpio_style_slo_guard"

    def test_safe_fallback_wsp_falls_back_when_base_not_better(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        base = mod.AlwaysScorpioSelector()
        sf = mod.SafeFallbackWspSelector(base, margin=0.05)
        # Row where SCORPIO anwg is below WSP anwg → fallback to WSP
        row = {f"reward_{p}": 0.5 for p in SELECTOR_CANDIDATES}
        for p in SELECTOR_CANDIDATES:
            row[f"completion_{p}"] = 1.0
        row["reward_scorpio_style_slo_guard"] = 0.85
        row["completion_scorpio_style_slo_guard"] = 0.90
        row["reward_weighted_shortest_processing"] = 0.84
        row["completion_weighted_shortest_processing"] = 0.99
        # SCORPIO anwg=0.765, WSP anwg=0.8316 → WSP is HIGHER → fallback to WSP
        preds = sf.predict([row])
        assert preds[0] == "weighted_shortest_processing"

    def test_knn_anwg_returns_valid_policy(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        from llmserveopt.selector.features import FEATURE_NAMES
        n_train = 20
        train_rows = []
        for _ in range(n_train):
            r = {f"feat_{f}": float(np.random.rand()) for f in FEATURE_NAMES}
            for p in SELECTOR_CANDIDATES:
                r[f"reward_{p}"] = float(np.random.rand())
                r[f"completion_{p}"] = float(0.8 + 0.2 * np.random.rand())
            train_rows.append(r)
        test_row = {f"feat_{f}": float(np.random.rand()) for f in FEATURE_NAMES}
        knn = mod.KNNAnwgSelector(k=3)
        knn.fit(train_rows)
        preds = knn.predict([test_row])
        assert len(preds) == 1
        assert preds[0] in SELECTOR_CANDIDATES

    def test_regression_anwg_returns_valid_policy(self):
        mod = self._import_helpers()
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        from llmserveopt.selector.features import FEATURE_NAMES
        n = 30
        train_rows = []
        for _ in range(n):
            r = {f"feat_{f}": float(np.random.rand()) for f in FEATURE_NAMES}
            for p in SELECTOR_CANDIDATES:
                r[f"reward_{p}"] = float(np.random.rand())
                r[f"completion_{p}"] = float(0.8 + 0.2 * np.random.rand())
            train_rows.append(r)
        sel = mod.PerPolicyRegressionAnwgSelector(n_estimators=5, max_depth=3)
        sel.fit(train_rows)
        test_row = {f"feat_{f}": float(np.random.rand()) for f in FEATURE_NAMES}
        preds = sel.predict([test_row])
        assert preds[0] in SELECTOR_CANDIDATES

    def test_safe_fallback_wsp_name_includes_margin(self):
        mod = self._import_helpers()
        base = mod.AlwaysScorpioSelector()
        sf = mod.SafeFallbackWspSelector(base, margin=0.005)
        assert "0.005" in sf.name
        assert "wsp" in sf.name.lower()


# ============================================================================
# Train/val/test split tests
# ============================================================================

class TestSplit:

    def _import_helpers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_dev_goes_to_train(self):
        mod = self._import_helpers()
        rows = [{"trace_id": f"dev_overloaded_mixed_slo_s{s}"} for s in [0, 1, 2]]
        train, val, test = mod.split_rows(rows, [6, 7, 8, 9, 10], [11])
        assert len(train) == 3
        assert len(val) == 0
        assert len(test) == 0

    def test_heldout_goes_to_test(self):
        mod = self._import_helpers()
        rows = [{"trace_id": f"heldout_moderate_kv_pressure_s{s}"} for s in [3, 4, 5]]
        train, val, test = mod.split_rows(rows, [6, 7, 8, 9, 10], [11])
        assert len(test) == 3
        assert len(train) == 0

    def test_diversity_seed11_goes_to_val(self):
        mod = self._import_helpers()
        rows = [{"trace_id": "div_bursty_moderate_s11"}]
        train, val, test = mod.split_rows(rows, [6, 7, 8, 9, 10], [11])
        assert len(val) == 1

    def test_diversity_seeds_6_to_10_go_to_train(self):
        mod = self._import_helpers()
        rows = [{"trace_id": f"div_bursty_moderate_s{s}"} for s in [6, 7, 8, 9, 10]]
        train, val, test = mod.split_rows(rows, [6, 7, 8, 9, 10], [11])
        assert len(train) == 5

    def test_no_overlap_between_splits(self):
        mod = self._import_helpers()
        trace_ids = [
            "dev_overloaded_mixed_slo_s0",
            "dev_overloaded_mixed_slo_s1",
            "div_bursty_moderate_s6",
            "div_bursty_moderate_s11",
            "heldout_moderate_kv_pressure_s3",
        ]
        rows = [{"trace_id": t} for t in trace_ids]
        train, val, test = mod.split_rows(rows, [6, 7, 8, 9, 10], [11])
        train_ids = {r["trace_id"] for r in train}
        val_ids = {r["trace_id"] for r in val}
        test_ids = {r["trace_id"] for r in test}
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)


# ============================================================================
# Registry integrity tests
# ============================================================================

class TestRegistryIntegrity:

    def test_scorpio_deadline_only_not_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "scorpio_deadline_only" not in SELECTOR_CANDIDATES

    def test_scorpio_deadline_only_not_in_registry(self):
        from llmserveopt.policies.registry import make_policy
        try:
            make_policy("scorpio_deadline_only")
            assert False, "scorpio_deadline_only should not be in registry"
        except (KeyError, ValueError):
            pass

    def test_oracle_srtf_not_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "oracle_srtf" not in SELECTOR_CANDIDATES

    def test_always_wsp_is_deployable(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "weighted_shortest_processing" in SELECTOR_CANDIDATES

    def test_scorpio_in_selector_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATES

    def test_no_new_policies_added_to_registry(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert len(SELECTOR_CANDIDATES) == 20

    def test_phase2b15_selectors_are_not_in_registry(self):
        """New selectors (rf_anwg, knn_anwg, etc.) are Python objects, not in the policy registry."""
        from llmserveopt.policies.registry import make_policy
        for name in ["rf_anwg", "knn_anwg", "regression_anwg", "always_wsp"]:
            try:
                make_policy(name)
                assert False, f"{name} should not be in policy registry"
            except (KeyError, ValueError):
                pass

    def test_wsp_policy_is_registered(self):
        from llmserveopt.policies.registry import make_policy
        p = make_policy("weighted_shortest_processing")
        assert p is not None


# ============================================================================
# Runner import test
# ============================================================================

class TestRunnerImport:

    def test_runner_imports_without_error(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "AlwaysWSPSelector")
        assert hasattr(mod, "SafeFallbackWspSelector")
        assert hasattr(mod, "KNNAnwgSelector")
        assert hasattr(mod, "PerPolicyRegressionAnwgSelector")
        assert hasattr(mod, "relabel_rows")
        assert hasattr(mod, "near_tie_stats")
        assert hasattr(mod, "split_rows")
        assert hasattr(mod, "deadline_only_comparison")

    def test_metric_keys_constant(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert "completed_request_quality" in mod.METRIC_KEYS
        assert "arrival_normalized_wg" in mod.METRIC_KEYS
        assert len(mod.METRIC_KEYS) == 5

    def test_scorpio_wsp_constants(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.SCORPIO == "scorpio_style_slo_guard"
        assert mod.WSP == "weighted_shortest_processing"


# ============================================================================
# Phase 2B.14 ablation data tests
# ============================================================================

class TestDeadlineOnlyComparison:

    def _import_helpers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b15",
            ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @_SKIP_B14
    def test_deadline_only_comparison_runs(self):
        mod = self._import_helpers()
        result = mod.deadline_only_comparison(
            _B14_RESULTS_DIR / "ablation_gap_analysis.json",
            gap_threshold_anwg=0.005,
            gap_threshold_cq=0.010,
        )
        assert "recommendation_promote" in result
        assert "gap_anwg" in result
        assert "gap_cq" in result

    @_SKIP_B14
    def test_deadline_only_anwg_gap_small(self):
        mod = self._import_helpers()
        result = mod.deadline_only_comparison(_B14_RESULTS_DIR / "ablation_gap_analysis.json")
        # Gap should be small (we know from Phase 2B.14 it is -0.0017)
        assert abs(result["gap_anwg"]) < 0.01

    @_SKIP_B14
    def test_deadline_only_passes_anwg_threshold(self):
        mod = self._import_helpers()
        result = mod.deadline_only_comparison(
            _B14_RESULTS_DIR / "ablation_gap_analysis.json",
            gap_threshold_anwg=0.005,
            gap_threshold_cq=0.010,
        )
        assert result["passes_anwg_threshold"] is True

    def test_missing_ablation_file_returns_error(self):
        mod = self._import_helpers()
        result = mod.deadline_only_comparison(Path("/nonexistent/file.json"))
        assert "error" in result


# ============================================================================
# Result file tests (skipped if experiment not run)
# ============================================================================

class TestResultFiles:

    @_SKIP_RESULTS
    def test_summary_json_exists(self):
        assert (_RESULTS_DIR / "phase2b15_summary.json").exists()

    @_SKIP_RESULTS
    def test_summary_json_valid(self):
        with open(_RESULTS_DIR / "phase2b15_summary.json") as f:
            d = json.load(f)
        assert d["experiment"] == "phase2b15_corrected_objective_selector_retraining"
        assert d["n_windows_total"] == 319

    @_SKIP_RESULTS
    def test_label_distribution_json_exists(self):
        assert (_RESULTS_DIR / "label_distribution.json").exists()

    @_SKIP_RESULTS
    def test_label_distribution_tracks_changes(self):
        with open(_RESULTS_DIR / "label_distribution.json") as f:
            d = json.load(f)
        assert d["n_total"] == 319
        assert d["n_label_changes"] > 0
        assert "label_dist_arrival_norm_wg" in d
        assert "label_dist_conditional_wg" in d

    @_SKIP_RESULTS
    def test_near_tie_analysis_exists(self):
        assert (_RESULTS_DIR / "near_tie_analysis_anwg.json").exists()

    @_SKIP_RESULTS
    def test_near_tie_meaningful_count(self):
        with open(_RESULTS_DIR / "near_tie_analysis_anwg.json") as f:
            d = json.load(f)
        # Under arrival-norm WG, expect ~84-97 meaningful windows at eps=0.010
        meaningful = d.get("n_meaningful_eps0.010", 0)
        assert 50 <= meaningful <= 150, f"Unexpected meaningful count: {meaningful}"

    @_SKIP_RESULTS
    def test_policy_metric_table_exists(self):
        assert (_RESULTS_DIR / "policy_metric_table.csv").exists()

    @_SKIP_RESULTS
    def test_policy_metric_table_has_20_rows(self):
        import pandas as pd
        df = pd.read_csv(_RESULTS_DIR / "policy_metric_table.csv")
        assert len(df) == 20

    @_SKIP_RESULTS
    def test_selector_comparison_csv_exists(self):
        assert (_RESULTS_DIR / "selector_comparison_test.csv").exists()

    @_SKIP_RESULTS
    def test_selector_comparison_has_all_selectors(self):
        import pandas as pd
        df = pd.read_csv(_RESULTS_DIR / "selector_comparison_test.csv")
        selectors = df["selector"].tolist()
        assert any("rf_anwg" in s for s in selectors)
        assert any("always_wsp" in s for s in selectors)
        assert any("always_scorpio" in s for s in selectors)
        assert any("wsp" in s for s in selectors)

    @_SKIP_RESULTS
    def test_rf_anwg_beats_b13_rf_on_test(self):
        """rf_anwg should outperform Phase 2B.13 RF under arrival-norm WG."""
        import pandas as pd
        df = pd.read_csv(_RESULTS_DIR / "selector_comparison_test.csv")
        rf_anwg_row = df[df["selector"] == "rf_anwg"]
        b13_rf_row = df[df["selector"] == "b13_random_forest"]
        if rf_anwg_row.empty or b13_rf_row.empty:
            pytest.skip("rf_anwg or b13_random_forest not in comparison table")
        rf_anwg_wg = float(rf_anwg_row["mean_arrival_normalized_wg"].iloc[0])
        b13_rf_wg = float(b13_rf_row["mean_arrival_normalized_wg"].iloc[0])
        assert rf_anwg_wg >= b13_rf_wg - 0.001, (
            f"rf_anwg ({rf_anwg_wg:.4f}) should be ≥ b13_rf ({b13_rf_wg:.4f}) under corrected metric"
        )

    @_SKIP_RESULTS
    def test_deadline_only_comparison_json_exists(self):
        assert (_RESULTS_DIR / "deadline_only_comparison.json").exists()

    @_SKIP_RESULTS
    def test_deadline_only_comparison_has_required_fields(self):
        with open(_RESULTS_DIR / "deadline_only_comparison.json") as f:
            d = json.load(f)
        required = [
            "scorpio_reference_anwg",
            "scorpio_deadline_only_anwg",
            "gap_anwg",
            "gap_cq",
            "recommendation_promote",
            "rationale",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    @_SKIP_RESULTS
    def test_b15_selector_evals_json_exists(self):
        assert (_RESULTS_DIR / "b15_selector_evals.json").exists()

    @_SKIP_RESULTS
    def test_rf_anwg_feature_importances_exist(self):
        assert (_RESULTS_DIR / "rf_anwg_feature_importances.json").exists()

    @_SKIP_RESULTS
    def test_feature_importance_top_feature_is_output_tokens(self):
        with open(_RESULTS_DIR / "rf_anwg_feature_importances.json") as f:
            fi = json.load(f)
        top_feature = next(iter(fi))
        assert "output" in top_feature or "pred" in top_feature, (
            f"Expected output-related feature at top, got: {top_feature}"
        )


# ============================================================================
# Documentation tests
# ============================================================================

class TestDocumentation:

    def test_phase2b15_summary_doc_exists(self):
        p = ROOT / "docs" / "audits" / "phase2b15_corrected_objective_selector_summary.md"
        assert p.exists()

    def test_phase2b15_failure_cases_doc_exists(self):
        p = ROOT / "docs" / "audits" / "phase2b15_failure_cases_summary.md"
        assert p.exists()

    def test_research_status_updated(self):
        p = ROOT / "docs" / "research_status.md"
        assert p.exists()
        content = p.read_text()
        assert "phase2b15" in content.lower() or "2B.15" in content

    def test_result_claims_updated(self):
        p = ROOT / "docs" / "result_claims.md"
        assert p.exists()
        content = p.read_text()
        assert "2B.15" in content

    def test_config_exists(self):
        p = ROOT / "configs" / "phase2b15_corrected_objective_selector_retraining.yaml"
        assert p.exists()

    def test_runner_script_exists(self):
        p = ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py"
        assert p.exists()

    def test_phase2b15_summary_doc_contains_key_findings(self):
        p = ROOT / "docs" / "audits" / "phase2b15_corrected_objective_selector_summary.md"
        content = p.read_text()
        assert "rf_anwg" in content
        assert "safe_fallback_wsp" in content or "safe_fallback" in content
        assert "always_wsp" in content or "always-WSP" in content

    def test_failure_cases_doc_contains_fail_021(self):
        p = ROOT / "docs" / "audits" / "phase2b15_failure_cases_summary.md"
        content = p.read_text()
        assert "fail_021" in content

    def test_no_api_credentials_in_runner(self):
        p = ROOT / "scripts" / "run_phase2b15_corrected_objective_selector_retraining.py"
        content = p.read_text()
        assert "GEMINI_API_KEY" not in content
        assert "HUGGINGFACE_TOKEN" not in content
        assert "OPENAI_API_KEY" not in content
        assert "COHERE_API_KEY" not in content

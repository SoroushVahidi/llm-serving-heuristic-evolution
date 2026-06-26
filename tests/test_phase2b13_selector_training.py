"""
Phase 2B.13 test suite: selector training on diversified workload suite.

Tests cover:
- Config file existence, structure, and extended fields
- Workload group completeness: 25 workloads (4 dev + 5 heldout + 16 diversity)
- Diversity seeds extended to [6,7,8,9,10,11] (was [6,7,8,9])
- Two new high-differentiation workloads: div_overloaded_all_loose_slo,
  div_kv_saturated_medium_slo
- Selector training split config (train/val/test)
- Rule selector repair spec documented in config
- Phase 2B.12 reference block present
- Runner importability and function signatures
- RepairedRuleBasedSelector Rule 5 repair: prefill → AC
- RepairedRuleBasedSelector other rules unchanged vs base
- check_rf_feasibility thresholds (200 windows, ≥3 policies, ≤85%)
- split_rows_for_training partitioning correctness
- train_selectors: returns None when sklearn missing or no data
- evaluate_ml_selector: accuracy, WG, dist computation
- No oracle_srtf or paid APIs in selector candidates
- Security: runner source has no API key strings
- BurstGPT trace files exist
- New diversity workloads have sensible parameters (arrival_rate, SLO classes)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CFG_PATH = ROOT / "configs" / "phase2b13_selector_training_after_diversity.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_phase2b13_selector_training_after_diversity.py"


# ---------------------------------------------------------------------------
# Config structure
# ---------------------------------------------------------------------------

class TestPhase2B13Config:
    @pytest.fixture
    def cfg(self):
        import yaml
        with open(CFG_PATH) as f:
            return yaml.safe_load(f)

    def test_config_exists(self):
        assert CFG_PATH.exists(), f"Config not found: {CFG_PATH}"

    def test_config_loadable(self, cfg):
        assert isinstance(cfg, dict)
        assert cfg.get("experiment") == "phase2b13_selector_training_after_diversity"

    def test_config_output_dir(self, cfg):
        assert "output_dir" in cfg
        assert "phase2b13" in cfg["output_dir"]

    def test_config_25_workloads(self, cfg):
        wloads = cfg.get("workloads", [])
        assert len(wloads) == 25, (
            f"Expected 25 workloads (4 dev + 5 heldout + 16 diversity), got {len(wloads)}"
        )

    def test_config_dev_group(self, cfg):
        dev = [w for w in cfg["workloads"] if w.get("group") == "dev"]
        assert len(dev) == 4, f"Expected 4 dev workloads, got {len(dev)}"

    def test_config_heldout_group(self, cfg):
        held = [w for w in cfg["workloads"] if w.get("group") == "heldout"]
        assert len(held) == 5, f"Expected 5 heldout workloads, got {len(held)}"

    def test_config_diversity_group(self, cfg):
        div = [w for w in cfg["workloads"] if w.get("group") == "diversity"]
        assert len(div) == 16, f"Expected 16 diversity workloads, got {len(div)}"

    def test_config_diversity_seeds_extended(self, cfg):
        seeds = cfg.get("diversity_seeds", [])
        assert sorted(seeds) == [6, 7, 8, 9, 10, 11], (
            f"diversity_seeds should be [6,7,8,9,10,11], got {seeds}"
        )

    def test_config_dev_seeds_unchanged(self, cfg):
        seeds = cfg.get("dev_seeds", [])
        assert sorted(seeds) == [0, 1, 2], f"dev_seeds should be [0,1,2], got {seeds}"

    def test_config_heldout_seeds_unchanged(self, cfg):
        seeds = cfg.get("heldout_seeds", [])
        assert sorted(seeds) == [3, 4, 5], f"heldout_seeds should be [3,4,5], got {seeds}"

    def test_config_selector_training_block(self, cfg):
        st = cfg.get("selector_training", {})
        assert "train_diversity_seeds" in st
        assert "val_diversity_seeds" in st
        assert set(st["train_diversity_seeds"]).isdisjoint(set(st["val_diversity_seeds"])), (
            "train and val diversity seeds must not overlap"
        )

    def test_config_train_val_seed_coverage(self, cfg):
        st = cfg.get("selector_training", {})
        train_s = set(st.get("train_diversity_seeds", []))
        val_s = set(st.get("val_diversity_seeds", []))
        diversity_s = set(cfg.get("diversity_seeds", []))
        assert train_s | val_s == diversity_s, (
            "train + val diversity seeds should equal all diversity seeds"
        )

    def test_config_rf_feasibility_block(self, cfg):
        rf = cfg.get("rf_feasibility", {})
        assert rf.get("min_windows", 0) == 200
        assert rf.get("min_policies_winning", 0) == 3
        assert rf.get("min_windows_per_policy", 0) == 10
        assert rf.get("max_single_policy_fraction", 1.0) == 0.85

    def test_config_phase2b12_reference_block(self, cfg):
        ref = cfg.get("phase2b12_reference", {})
        assert "rule_based_wg" in ref
        assert "best_fixed_wg" in ref
        assert "n_windows" in ref
        assert ref.get("n_windows") == 172

    def test_config_rule_selector_repair_block(self, cfg):
        rr = cfg.get("rule_selector_repair", {})
        assert rr.get("apply") is True
        changes = rr.get("changes", [])
        assert len(changes) >= 1
        change = changes[0]
        assert change["old_policy"] == "sarathi_style"
        assert change["new_policy"] == "admission_control"

    def test_config_new_workloads_present(self, cfg):
        tags = {w["tag"] for w in cfg["workloads"]}
        assert "div_overloaded_all_loose_slo" in tags, (
            "New Phase 2B.13 workload div_overloaded_all_loose_slo missing"
        )
        assert "div_kv_saturated_medium_slo" in tags, (
            "New Phase 2B.13 workload div_kv_saturated_medium_slo missing"
        )

    def test_config_new_workload_groups(self, cfg):
        new_wloads = [w for w in cfg["workloads"]
                      if w["tag"] in {"div_overloaded_all_loose_slo", "div_kv_saturated_medium_slo"}]
        for w in new_wloads:
            assert w.get("group") == "diversity"

    def test_config_new_workload_parameters(self, cfg):
        wmap = {w["tag"]: w for w in cfg["workloads"]}
        overloaded = wmap["div_overloaded_all_loose_slo"]
        assert overloaded["arrival_rate"] >= 60.0, "overloaded workload should be high rate"
        slo_classes = overloaded.get("slo_classes", [])
        tight_classes = [c for c in slo_classes if c.get("class_id") == "tight"]
        assert len(tight_classes) == 0, (
            "div_overloaded_all_loose_slo should have NO tight SLO classes"
        )

    def test_config_kv_saturated_long_outputs(self, cfg):
        wmap = {w["tag"]: w for w in cfg["workloads"]}
        kv_sat = wmap["div_kv_saturated_medium_slo"]
        assert kv_sat["output_mean"] >= 200, "kv_saturated workload should have long outputs"

    def test_config_no_duplicate_tags(self, cfg):
        tags = [w["tag"] for w in cfg["workloads"]]
        assert len(tags) == len(set(tags)), f"Duplicate workload tags: {[t for t in tags if tags.count(t) > 1]}"

    def test_config_diversity_seeds_disjoint_from_regression(self, cfg):
        dev_s = set(cfg.get("dev_seeds", []))
        held_s = set(cfg.get("heldout_seeds", []))
        div_s = set(cfg.get("diversity_seeds", []))
        assert dev_s.isdisjoint(div_s), "dev seeds and diversity seeds overlap"
        assert held_s.isdisjoint(div_s), "heldout seeds and diversity seeds overlap"
        assert dev_s.isdisjoint(held_s), "dev and heldout seeds overlap"

    def test_config_simulator_settings(self, cfg):
        sim = cfg.get("simulator", {})
        assert sim.get("drain_steps", 0) == 20000
        assert sim.get("step_size", 0) == 0.001

    def test_config_gpu_settings(self, cfg):
        gpus = cfg.get("gpus", [])
        assert len(gpus) >= 1
        g = gpus[0]
        assert g.get("max_active_sequences") == 4
        assert g.get("max_kv_tokens") == 32768


# ---------------------------------------------------------------------------
# Runner import
# ---------------------------------------------------------------------------

class TestPhase2B13RunnerImport:
    def test_runner_exists(self):
        assert RUNNER_PATH.exists(), f"Runner not found: {RUNNER_PATH}"

    def test_runner_importable(self):
        import run_phase2b13_selector_training_after_diversity as m  # noqa: F401
        assert m is not None

    def test_runner_exports_repaired_rule_selector(self):
        from run_phase2b13_selector_training_after_diversity import RepairedRuleBasedSelector
        assert RepairedRuleBasedSelector is not None

    def test_runner_exports_split_rows_for_training(self):
        from run_phase2b13_selector_training_after_diversity import split_rows_for_training
        assert callable(split_rows_for_training)

    def test_runner_exports_train_selectors(self):
        from run_phase2b13_selector_training_after_diversity import train_selectors
        assert callable(train_selectors)

    def test_runner_exports_evaluate_ml_selector(self):
        from run_phase2b13_selector_training_after_diversity import evaluate_ml_selector
        assert callable(evaluate_ml_selector)

    def test_runner_no_api_keys_in_source(self):
        src = RUNNER_PATH.read_text()
        forbidden = ["sk-", "AIzaSy", "hf_", "Bearer ey"]
        for f in forbidden:
            assert f not in src, f"Potential API key string '{f}' found in runner source"

    def test_runner_no_paid_api_calls(self):
        src = RUNNER_PATH.read_text()
        for banned in ["openai.com", "api.cohere.ai", "generativelanguage.googleapis.com",
                       "api.mistral.ai", "api.cerebras.ai", "cloudrift"]:
            assert banned not in src.lower(), f"Banned API call to '{banned}' in runner"


# ---------------------------------------------------------------------------
# RepairedRuleBasedSelector — Rule 5 fix
# ---------------------------------------------------------------------------

class TestRepairedRuleBasedSelector:
    @pytest.fixture
    def selector(self):
        from run_phase2b13_selector_training_after_diversity import RepairedRuleBasedSelector
        return RepairedRuleBasedSelector()

    @pytest.fixture
    def base_selector(self):
        from llmserveopt.selector.models import RuleBasedSelector
        return RuleBasedSelector()

    def _feats(self, **kwargs):
        defaults = {
            "fraction_tight_slo": 0.1,
            "min_slack": 5.0,
            "recent_slo_violation_rate": 0.0,
            "kv_utilization": 0.3,
            "mean_prompt_tokens": 100.0,
            "p95_prompt_tokens": 200.0,
            "mean_pred_output_tokens": 50.0,
            "pred_output_cv": 0.3,
            "burstiness_cv": 0.5,
        }
        defaults.update(kwargs)
        return defaults

    def test_prefill_heavy_mean_prompt_maps_to_ac(self, selector):
        f = self._feats(mean_prompt_tokens=600.0, p95_prompt_tokens=800.0)
        assert selector.predict_one(f) == "admission_control"

    def test_prefill_heavy_p95_maps_to_ac(self, selector):
        f = self._feats(mean_prompt_tokens=200.0, p95_prompt_tokens=1100.0)
        assert selector.predict_one(f) == "admission_control"

    def test_not_prefill_heavy_does_not_dispatch_sarathi(self, selector):
        f = self._feats(mean_prompt_tokens=100.0, p95_prompt_tokens=200.0)
        result = selector.predict_one(f)
        assert result != "sarathi_style", (
            "Repaired selector should never dispatch sarathi_style"
        )

    def test_sarathi_not_in_policy_choices(self, selector):
        assert "sarathi_style" not in selector._POLICY_CHOICES

    def test_ac_in_policy_choices(self, selector):
        assert "admission_control" in selector._POLICY_CHOICES

    def test_rule0_scorpio_for_tight_slo_with_violations(self, selector):
        f = self._feats(fraction_tight_slo=0.5, min_slack=0.8,
                        recent_slo_violation_rate=0.3)
        assert selector.predict_one(f) == "scorpio_style_slo_guard"

    def test_rule1_wsp_for_long_outputs(self, selector):
        f = self._feats(mean_pred_output_tokens=250.0, kv_utilization=0.3)
        assert selector.predict_one(f) == "weighted_shortest_processing"

    def test_rule1_wsp_for_high_kv(self, selector):
        f = self._feats(kv_utilization=0.8)
        assert selector.predict_one(f) == "weighted_shortest_processing"

    def test_rule2a_scorpio_for_very_high_cv(self, selector):
        f = self._feats(pred_output_cv=2.5)
        assert selector.predict_one(f) == "scorpio_style_slo_guard"

    def test_rule2b_ac_for_high_cv(self, selector):
        f = self._feats(pred_output_cv=1.5,
                        mean_prompt_tokens=100.0, p95_prompt_tokens=200.0)
        assert selector.predict_one(f) == "admission_control"

    def test_rule3_scorpio_for_high_violation_rate(self, selector):
        f = self._feats(recent_slo_violation_rate=0.35)
        assert selector.predict_one(f) == "scorpio_style_slo_guard"

    def test_rule4_slo_slack_for_tight_slo(self, selector):
        f = self._feats(fraction_tight_slo=0.5, pred_output_cv=0.3)
        assert selector.predict_one(f) == "slo_slack_score"

    def test_rule6_estST_for_short_uniform_outputs(self, selector):
        f = self._feats(mean_pred_output_tokens=32.0, pred_output_cv=0.3,
                        burstiness_cv=0.5, mean_prompt_tokens=100.0,
                        p95_prompt_tokens=200.0)
        assert selector.predict_one(f) == "estimated_service_time_first"

    def test_rule8_default_edf(self, selector):
        f = self._feats(mean_pred_output_tokens=70.0, pred_output_cv=0.6,
                        burstiness_cv=0.5, mean_prompt_tokens=100.0,
                        p95_prompt_tokens=200.0)
        assert selector.predict_one(f) == "edf"

    def test_repaired_and_base_agree_on_non_prefill_heavy(self, selector, base_selector):
        f = self._feats()  # not prefill-heavy, low CV, not tight
        assert selector.predict_one(f) == base_selector.predict_one(f)

    def test_repaired_differs_from_base_on_prefill_heavy(self, selector, base_selector):
        f = self._feats(mean_prompt_tokens=700.0, p95_prompt_tokens=1500.0)
        base_pred = base_selector.predict_one(f)
        repaired_pred = selector.predict_one(f)
        assert repaired_pred == "admission_control"
        assert base_pred == "sarathi_style"

    def test_predict_list(self, selector):
        feats = [
            self._feats(mean_prompt_tokens=700.0),
            self._feats(),
        ]
        preds = selector.predict(feats)
        assert len(preds) == 2
        assert preds[0] == "admission_control"


# ---------------------------------------------------------------------------
# split_rows_for_training
# ---------------------------------------------------------------------------

class TestSplitRowsForTraining:
    @pytest.fixture
    def split_fn(self):
        from run_phase2b13_selector_training_after_diversity import split_rows_for_training
        return split_rows_for_training

    def _make_row(self, trace_id, seed):
        return {"trace_id": f"workload_a_s{seed}", "best_policy": "edf"}

    def test_dev_rows_go_to_train(self, split_fn):
        dev = [{"trace_id": "dev_wl_s0", "best_policy": "edf"}]
        div = []
        held = [{"trace_id": "held_wl_s3", "best_policy": "scorpio_style_slo_guard"}]
        train, val, test = split_fn(dev, div, held, [6, 7], [11])
        assert {"trace_id": "dev_wl_s0", "best_policy": "edf"} in train
        assert test == held

    def test_diversity_seed_partitioned_correctly(self, split_fn):
        dev = []
        div = [
            {"trace_id": "div_wl_s6", "best_policy": "edf"},
            {"trace_id": "div_wl_s10", "best_policy": "admission_control"},
            {"trace_id": "div_wl_s11", "best_policy": "best_fit"},
        ]
        held = []
        train, val, test = split_fn(dev, div, held, [6, 10], [11])
        train_ids = {r["trace_id"] for r in train}
        val_ids = {r["trace_id"] for r in val}
        assert "div_wl_s6" in train_ids
        assert "div_wl_s10" in train_ids
        assert "div_wl_s11" in val_ids
        assert "div_wl_s11" not in train_ids

    def test_heldout_always_in_test(self, split_fn):
        dev = []
        div = []
        held = [{"trace_id": "held_wl_s3", "best_policy": "scorpio_style_slo_guard"},
                {"trace_id": "held_wl_s4", "best_policy": "edf"}]
        train, val, test = split_fn(dev, div, held, [6], [11])
        assert test == held

    def test_no_overlap_between_train_val_test(self, split_fn):
        dev = [{"trace_id": "dev_wl_s0", "best_policy": "edf"}]
        div = [
            {"trace_id": "div_wl_s6", "best_policy": "edf"},
            {"trace_id": "div_wl_s11", "best_policy": "best_fit"},
        ]
        held = [{"trace_id": "held_wl_s3", "best_policy": "scorpio_style_slo_guard"}]
        train, val, test = split_fn(dev, div, held, [6], [11])
        train_ids = {r["trace_id"] for r in train}
        val_ids = {r["trace_id"] for r in val}
        test_ids = {r["trace_id"] for r in test}
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_empty_inputs_return_empty(self, split_fn):
        train, val, test = split_fn([], [], [], [6, 7], [11])
        assert train == []
        assert val == []
        assert test == []


# ---------------------------------------------------------------------------
# train_selectors
# ---------------------------------------------------------------------------

class TestTrainSelectors:
    @pytest.fixture
    def train_fn(self):
        from run_phase2b13_selector_training_after_diversity import train_selectors
        return train_fn

    @pytest.fixture
    def train_fn(self):
        from run_phase2b13_selector_training_after_diversity import train_selectors
        return train_selectors

    def _make_row(self, best_policy="edf", **feat_overrides):
        from llmserveopt.selector.features import FEATURE_NAMES
        row = {"best_policy": best_policy, "trace_id": "wl_s6"}
        for fname in FEATURE_NAMES:
            row[f"feat_{fname}"] = 0.0
        row.update(feat_overrides)
        return row

    def test_returns_none_on_empty_rows(self, train_fn):
        rf, dt, status = train_fn([])
        assert rf is None
        assert dt is None
        assert status != "ok"

    def test_trains_successfully_with_valid_rows(self, train_fn):
        pytest.importorskip("sklearn")
        rows = [self._make_row(best_policy=p) for p in (["edf"] * 30 + ["admission_control"] * 20)]
        rf, dt, status = train_fn(rows)
        assert status == "ok"
        assert rf is not None
        assert dt is not None

    def test_trained_rf_predicts(self, train_fn):
        pytest.importorskip("sklearn")
        rows = [self._make_row(best_policy=p) for p in (["edf"] * 30 + ["admission_control"] * 20)]
        rf, dt, status = train_fn(rows)
        assert rf is not None
        preds = rf.predict(rows[:5])
        assert len(preds) == 5
        for p in preds:
            assert isinstance(p, str)

    def test_trained_dt_predicts(self, train_fn):
        pytest.importorskip("sklearn")
        rows = [self._make_row(best_policy=p) for p in (["edf"] * 30 + ["admission_control"] * 20)]
        rf, dt, status = train_fn(rows)
        assert dt is not None
        preds = dt.predict(rows[:5])
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# evaluate_ml_selector
# ---------------------------------------------------------------------------

class TestEvaluateMlSelector:
    @pytest.fixture
    def eval_fn(self):
        from run_phase2b13_selector_training_after_diversity import evaluate_ml_selector
        return evaluate_ml_selector

    def _make_selector(self, fixed_pred):
        class FixedSelector:
            def predict(self, rows):
                return [fixed_pred] * len(rows)
        return FixedSelector()

    def _make_row(self, best_policy, pred_policy, wg=0.95, best_wg=1.0):
        return {
            "best_policy": best_policy,
            "best_weighted_goodput": best_wg,
            f"reward_{pred_policy}": wg,
            f"reward_{best_policy}": best_wg,
        }

    def test_returns_n_windows(self, eval_fn):
        sel = self._make_selector("edf")
        rows = [self._make_row("edf", "edf")] * 5
        result = eval_fn(sel, rows, "test_sel")
        assert result["n_windows"] == 5

    def test_perfect_accuracy(self, eval_fn):
        sel = self._make_selector("edf")
        rows = [self._make_row("edf", "edf", wg=0.95, best_wg=0.95)] * 10
        result = eval_fn(sel, rows, "test_sel")
        assert result["accuracy"] == 1.0

    def test_zero_accuracy(self, eval_fn):
        sel = self._make_selector("edf")
        rows = [self._make_row("admission_control", "edf", wg=0.95, best_wg=1.0)] * 10
        result = eval_fn(sel, rows, "test_sel")
        assert result["accuracy"] == 0.0

    def test_mean_wg_computed(self, eval_fn):
        # Selector predicts "edf" but oracle is "admission_control"
        # reward_edf = 0.90, reward_admission_control = 1.0
        sel = self._make_selector("edf")
        rows = [self._make_row("admission_control", "edf", wg=0.90, best_wg=1.0)] * 4
        result = eval_fn(sel, rows, "test_sel")
        assert abs(result["mean_wg"] - 0.90) < 1e-4

    def test_empty_rows_returns_zero_n(self, eval_fn):
        sel = self._make_selector("edf")
        result = eval_fn(sel, [], "test_sel")
        assert result["n"] == 0

    def test_chosen_policy_dist(self, eval_fn):
        sel = self._make_selector("edf")
        rows = [self._make_row("edf", "edf", wg=0.9, best_wg=1.0)] * 3
        result = eval_fn(sel, rows, "test_sel")
        assert result["chosen_policy_dist"].get("edf", 0) == 3


# ---------------------------------------------------------------------------
# Selector candidates / oracle exclusion
# ---------------------------------------------------------------------------

class TestSelectorCandidates:
    def test_oracle_srtf_not_in_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "oracle_srtf" not in SELECTOR_CANDIDATES

    def test_scorpio_in_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "scorpio_style_slo_guard" in SELECTOR_CANDIDATES

    def test_admission_control_in_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert "admission_control" in SELECTOR_CANDIDATES

    def test_20_deployable_candidates(self):
        from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
        assert len(SELECTOR_CANDIDATES) == 20, (
            f"Expected 20 deployable candidates, got {len(SELECTOR_CANDIDATES)}"
        )


# ---------------------------------------------------------------------------
# BurstGPT trace files
# ---------------------------------------------------------------------------

class TestBurstGPTTraces:
    TRACES = [
        "data/processed/burstgpt/burstgpt_scaled_moderate_10k.jsonl",
        "data/processed/burstgpt/burstgpt_scaled_high_10k.jsonl",
        "data/processed/burstgpt/burstgpt_natural_10k.jsonl",
    ]

    def test_trace_files_exist(self):
        missing = [p for p in self.TRACES if not (ROOT / p).exists()]
        assert not missing, f"Missing BurstGPT trace files: {missing}"

    def test_phase2b13_config_burstgpt_paths_exist(self):
        import yaml
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        burstgpt_wloads = [w for w in cfg["workloads"] if w.get("source") == "extended_jsonl"]
        for w in burstgpt_wloads:
            path = ROOT / w["trace_path"]
            assert path.exists(), f"BurstGPT trace not found: {w['trace_path']}"


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

class TestSecurityConstraints:
    def test_runner_source_no_api_keys(self):
        src = RUNNER_PATH.read_text()
        for token_prefix in ["sk-ant-", "sk-", "AIzaSy", "hf_", "co_"]:
            assert token_prefix not in src

    def test_runner_source_no_huggingface_imports(self):
        src = RUNNER_PATH.read_text()
        assert "from_pretrained" not in src
        assert "AutoModel" not in src
        assert "HfApi" not in src

    def test_runner_source_no_paid_endpoints(self):
        src = RUNNER_PATH.read_text()
        banned = ["api.openai.com", "api.cohere.ai", "api.mistral.ai",
                  "api.cerebras.ai", "api.cloudrift", "generativelanguage.googleapis"]
        for b in banned:
            assert b not in src

    def test_config_no_api_keys(self):
        text = CFG_PATH.read_text()
        for token_prefix in ["sk-", "hf_", "AIzaSy"]:
            assert token_prefix not in text

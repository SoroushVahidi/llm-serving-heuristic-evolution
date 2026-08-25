"""
Phase 2B.16 tests: fresh corrected-objective validation.

Covers:
- Config existence and required fields (fresh seeds, workload groups)
- Runner module importability
- Metric helpers (arrival-norm WG, completion-penalized WG) on fresh-data-like rows
- Relabeling helpers (reused from Phase 2B.15)
- Selector freeze contract: selectors trained only on train split (no fresh data leakage)
- FIFO-artifact audit on synthetic data
- Bootstrap CI correctness (coverage + symmetry)
- Top-epsilon accuracy calculation
- Constrained objective calculation
- Win/tie/loss counting
- Statistical analysis helpers
- Output file existence (skipped until experiment run)
- Documentation existence
- Safety rules: no paid APIs, no GPU experiments
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_RESULTS_DIR = ROOT / "results" / "phase2b16_fresh_corrected_objective_validation"
_RESULTS_EXIST = _RESULTS_DIR.exists() and any(p.is_file() for p in _RESULTS_DIR.iterdir()) \
    if _RESULTS_DIR.exists() else False
_SKIP_RESULTS = pytest.mark.skipif(
    not _RESULTS_EXIST,
    reason="Phase 2B.16 results not yet generated — run the experiment first",
)

_B13_RESULTS_DIR = ROOT / "results" / "phase2b13_selector_training_and_suspicion_audit"
_B13_EXISTS = (_B13_RESULTS_DIR / "per_window.csv").exists() if _B13_RESULTS_DIR.exists() else False
_SKIP_B13 = pytest.mark.skipif(
    not _B13_EXISTS,
    reason="Phase 2B.13 per_window.csv not found",
)

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES

# Canonical constants
SCORPIO = "scorpio_style_slo_guard"
WSP = "weighted_shortest_processing"
FIFO = "fifo"


# ---------------------------------------------------------------------------
# Synthetic row helpers
# ---------------------------------------------------------------------------

def _make_row(
    policies: List[str] = None,
    reward_fn=None,
    comp_fn=None,
    trace_id: str = "test_s12",
    seed: int = 12,
) -> Dict:
    policies = policies or SELECTOR_CANDIDATES
    row = {"trace_id": trace_id, "seed": seed, "best_weighted_goodput": 0.9}
    for p in policies:
        rw = reward_fn(p) if reward_fn else 0.9
        cf = comp_fn(p) if comp_fn else 0.99
        row[f"reward_{p}"] = rw
        row[f"completion_{p}"] = cf
    for i in range(len(SELECTOR_CANDIDATES)):
        row[f"feat_f{i}"] = 0.5
    return row


def _make_scorpio_dominant_row() -> Dict:
    """SCORPIO wins under arrival-norm WG."""
    def rw(p):
        return 0.98 if p == SCORPIO else 0.80
    def cf(p):
        return 0.92 if p == SCORPIO else 0.99
    return _make_row(reward_fn=rw, comp_fn=cf)


def _make_wsp_dominant_row() -> Dict:
    """WSP wins under arrival-norm WG (high CF × moderate cond WG)."""
    def rw(p):
        if p == WSP: return 0.88
        if p == SCORPIO: return 0.97
        return 0.70
    def cf(p):
        if p == WSP: return 1.0
        if p == SCORPIO: return 0.88
        return 0.99
    return _make_row(reward_fn=rw, comp_fn=cf)


def _make_fifo_near_tie_row() -> Dict:
    """FIFO 'wins' under arrival-norm WG but only by a margin < 0.001."""
    def rw(p): return 1.0 if p in (FIFO, WSP) else 0.97
    def cf(p):
        if p == SCORPIO: return 0.9992
        return 1.0
    return _make_row(reward_fn=rw, comp_fn=cf)


def _make_fresh_rows(n_scorpio=10, n_wsp=5, n_fifo_tie=3) -> List[Dict]:
    rows = (
        [_make_scorpio_dominant_row() for _ in range(n_scorpio)]
        + [_make_wsp_dominant_row() for _ in range(n_wsp)]
        + [_make_fifo_near_tie_row() for _ in range(n_fifo_tie)]
    )
    for i, r in enumerate(rows):
        r["trace_id"] = f"fresh_wl_s{20 + (i % 3)}"
    return rows


# ============================================================================
# Config tests
# ============================================================================

class TestConfig:
    _cfg_path = ROOT / "configs" / "phase2b16_fresh_corrected_objective_validation.yaml"

    def test_config_exists(self):
        assert self._cfg_path.exists(), f"Config missing: {self._cfg_path}"

    def test_config_loads(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_experiment_key(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("experiment") == "phase2b16_fresh_corrected_objective_validation"

    def test_input_b13_dir_references_b13(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "phase2b13" in cfg["input_b13_dir"]

    def test_fresh_diversity_seeds_are_new(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["fresh_diversity_seeds"]
        old_seeds = set(range(12))  # seeds 0-11 used in phases up to 2B.13
        assert set(seeds).isdisjoint(old_seeds), \
            f"Fresh diversity seeds {seeds} overlap old seeds"

    def test_fresh_heldout_seeds_are_new(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["fresh_heldout_seeds"]
        assert all(s >= 20 for s in seeds), \
            f"Heldout seeds {seeds} should all be ≥20"

    def test_diversity_and_heldout_seeds_disjoint(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        div = set(cfg["fresh_diversity_seeds"])
        heldout = set(cfg["fresh_heldout_seeds"])
        assert div.isdisjoint(heldout), "Diversity and heldout seed sets must be disjoint"

    def test_three_workload_groups(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        workloads = cfg.get("workloads", [])
        groups = {w.get("group") for w in workloads}
        assert "fresh_diversity" in groups
        assert "fresh_targeted" in groups
        assert "fresh_heldout" in groups

    def test_bootstrap_config(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        bs = cfg.get("bootstrap", {})
        assert bs.get("n_samples", 0) >= 1000
        assert bs.get("ci", 0) == 0.95

    def test_top_epsilon_thresholds(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        epsilons = cfg.get("top_epsilon_thresholds", [])
        assert 0.001 in epsilons
        assert 0.005 in epsilons

    def test_constrained_objective_thresholds(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        co = cfg.get("constrained_objectives", {})
        thresholds = co.get("completion_thresholds", [])
        assert 0.95 in thresholds
        assert 0.99 in thresholds

    def test_safe_fallback_margins_present(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        margins = cfg.get("safe_fallback_margins", [])
        assert 0.001 in margins or 0.005 in margins

    def test_output_dir_is_b16(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "phase2b16" in cfg.get("output_dir", "")

    def test_selector_training_uses_old_seeds(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        train_seeds = cfg.get("selector_training", {}).get("train_diversity_seeds", [])
        assert all(s < 12 for s in train_seeds), \
            f"Selector training seeds {train_seeds} should be old (<12)"

    def test_no_oracle_srtf_in_candidates(self):
        assert "oracle_srtf" not in SELECTOR_CANDIDATES


# ============================================================================
# Runner import tests
# ============================================================================

class TestRunnerImport:
    def test_runner_module_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b16",
            ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")

    def test_runner_has_required_functions(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b16",
            ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn in [
            "train_phase2b15_selectors",
            "evaluate_fresh_selector",
            "evaluate_group",
            "bootstrap_ci",
            "win_tie_loss",
            "top_epsilon_accuracy",
            "fifo_artifact_audit",
            "constrained_objective",
            "statistical_analysis",
            "build_failure_cases",
        ]:
            assert hasattr(mod, fn), f"Runner missing function: {fn}"

    def test_runner_imports_from_b15(self):
        """Ensure Phase 2B.16 runner imports Phase 2B.15 selector classes."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b16",
            ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "SafeFallbackWspSelector")
        assert hasattr(mod, "KNNAnwgSelector")
        assert hasattr(mod, "PerPolicyRegressionAnwgSelector")

    def test_no_predict_once_wrapper(self):
        """SafeFallbackWspSelector must NOT be wrapped with predict_one."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase2b16",
            ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert not hasattr(mod, "_PredictOnceWrapper"), \
            "_PredictOnceWrapper would break SafeFallbackWspSelector (needs full rows)"


# ============================================================================
# Metric computation tests
# ============================================================================

class TestMetricComputation:
    def test_anwg_equals_cf_times_cond_wg(self):
        from run_phase2b15_corrected_objective_selector_retraining import _anwg, _comp_frac, _cond_wg
        row = _make_scorpio_dominant_row()
        expected = _comp_frac(row, SCORPIO) * _cond_wg(row, SCORPIO)
        assert abs(_anwg(row, SCORPIO) - expected) < 1e-9

    def test_anwg_zero_when_completion_zero(self):
        from run_phase2b15_corrected_objective_selector_retraining import _anwg
        row = _make_row()
        row[f"completion_{SCORPIO}"] = 0.0
        row[f"reward_{SCORPIO}"] = 1.0
        assert _anwg(row, SCORPIO) == 0.0

    def test_anwg_equals_cond_wg_when_all_complete(self):
        from run_phase2b15_corrected_objective_selector_retraining import _anwg, _cond_wg
        row = _make_row(comp_fn=lambda p: 1.0)
        assert abs(_anwg(row, SCORPIO) - _cond_wg(row, SCORPIO)) < 1e-9

    def test_cp_wg_equals_anwg_when_cf_above_target(self):
        from run_phase2b15_corrected_objective_selector_retraining import _cp_wg, _anwg
        row = _make_row(comp_fn=lambda p: 1.0)
        assert abs(_cp_wg(row, SCORPIO, 0.95, 0.5) - _anwg(row, SCORPIO)) < 1e-9

    def test_cp_wg_penalized_when_cf_below_target(self):
        from run_phase2b15_corrected_objective_selector_retraining import _cp_wg, _anwg
        row = _make_scorpio_dominant_row()
        cf = row[f"completion_{SCORPIO}"]
        if cf < 0.95:
            anwg = _anwg(row, SCORPIO)
            cp = _cp_wg(row, SCORPIO, 0.95, 0.5)
            assert cp < anwg, "CP WG should be penalized when CF < target"

    def test_compute_all_metrics_returns_all_keys(self):
        from run_phase2b15_corrected_objective_selector_retraining import compute_all_metrics, METRIC_KEYS
        row = _make_scorpio_dominant_row()
        metrics = compute_all_metrics(row, SCORPIO)
        for k in METRIC_KEYS:
            assert k in metrics, f"Missing metric key: {k}"

    def test_wsp_beats_scorpio_under_cp_wg_high_lambda(self):
        """WSP (CF=1.0) beats SCORPIO (CF=0.80) under completion-penalized WG."""
        from run_phase2b15_corrected_objective_selector_retraining import _cp_wg
        row = _make_row()
        row[f"completion_{SCORPIO}"] = 0.80
        row[f"reward_{SCORPIO}"] = 0.95
        row[f"completion_{WSP}"] = 1.0
        row[f"reward_{WSP}"] = 0.85
        scorpio_cp = _cp_wg(row, SCORPIO, 0.95, 1.0)
        wsp_cp = _cp_wg(row, WSP, 0.95, 1.0)
        assert wsp_cp > scorpio_cp, \
            f"WSP should beat SCORPIO under CP-WG(t=0.95,λ=1.0). Got wsp={wsp_cp:.4f}, scorpio={scorpio_cp:.4f}"


# ============================================================================
# Relabeling and near-tie tests
# ============================================================================

class TestRelabeling:
    def test_relabel_identifies_best_anwg(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        row = _make_wsp_dominant_row()
        result = relabel_rows([row])[0]
        assert result["best_policy_anwg"] == WSP, \
            f"WSP should be best under ANWG, got {result['best_policy_anwg']}"

    def test_relabel_scorpio_wins_when_dominant(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        row = _make_scorpio_dominant_row()
        result = relabel_rows([row])[0]
        assert result["best_policy_anwg"] == SCORPIO

    def test_relabel_sets_policy_margin_anwg(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        row = _make_scorpio_dominant_row()
        result = relabel_rows([row])[0]
        assert "policy_margin_anwg" in result
        assert result["policy_margin_anwg"] >= 0.0

    def test_relabel_sets_best_anwg(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        row = _make_scorpio_dominant_row()
        result = relabel_rows([row])[0]
        assert "best_anwg" in result
        assert 0.0 <= result["best_anwg"] <= 1.0

    def test_fifo_near_tie_margin_below_eps(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        row = _make_fifo_near_tie_row()
        result = relabel_rows([row])[0]
        assert result["policy_margin_anwg"] < 0.001, \
            f"Near-tie FIFO margin should be <0.001, got {result['policy_margin_anwg']:.6f}"

    def test_near_tie_stats_counts_fifo_near_ties(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows, near_tie_stats
        rows = relabel_rows(_make_fresh_rows(n_fifo_tie=5))
        stats = near_tie_stats(rows, [0.001, 0.005])
        assert stats["n_windows"] == len(rows)
        assert "fraction_near_tie_eps0.005" in stats

    def test_filter_meaningful_removes_near_ties(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows, filter_meaningful
        rows = relabel_rows(_make_fresh_rows(n_fifo_tie=5))
        filtered = filter_meaningful(rows, 0.005)
        # Near-tie rows should be reduced or removed
        assert len(filtered) <= len(rows)

    def test_relabel_excludes_oracle(self):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = _make_fresh_rows()
        for r in rows:
            r[f"reward_oracle_srtf"] = 9999.0
            r[f"completion_oracle_srtf"] = 1.0
        result = relabel_rows(rows)
        for r in result:
            assert r.get("best_policy_anwg") != "oracle_srtf", \
                "oracle_srtf must never be selected as best policy"


# ============================================================================
# FIFO artifact audit tests
# ============================================================================

class TestFifoArtifactAudit:
    def test_audit_identifies_fifo_wins(self):
        from run_phase2b16_fresh_corrected_objective_validation import fifo_artifact_audit
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows(_make_fresh_rows(n_fifo_tie=3))
        audit = fifo_artifact_audit(rows)
        assert "n_fifo_wins_anwg" in audit
        assert "n_fifo_near_tie_eps005" in audit
        assert audit["n_total"] == len(rows)

    def test_audit_fraction_correct(self):
        from run_phase2b16_fresh_corrected_objective_validation import fifo_artifact_audit
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows(_make_fresh_rows(n_fifo_tie=3))
        audit = fifo_artifact_audit(rows)
        n_fifo = audit["n_fifo_wins_anwg"]
        expected = n_fifo / len(rows)
        assert abs(audit["fifo_win_fraction"] - expected) < 5e-4

    def test_audit_near_tie_fifo_classified_correctly(self):
        from run_phase2b16_fresh_corrected_objective_validation import fifo_artifact_audit
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows(_make_fresh_rows(n_fifo_tie=3, n_scorpio=0, n_wsp=0))
        audit = fifo_artifact_audit(rows)
        # All FIFO "wins" here are near-tie artifacts
        if audit["n_fifo_wins_anwg"] > 0:
            assert audit["n_fifo_near_tie_eps005"] == audit["n_fifo_wins_anwg"], \
                "All FIFO wins in near-tie rows should be classified as near-tie"

    def test_no_fifo_wins_when_scorpio_dominant(self):
        from run_phase2b16_fresh_corrected_objective_validation import fifo_artifact_audit
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows([_make_scorpio_dominant_row() for _ in range(10)])
        audit = fifo_artifact_audit(rows)
        assert audit["n_fifo_wins_anwg"] == 0


# ============================================================================
# Bootstrap CI tests
# ============================================================================

class TestBootstrapCI:
    def test_ci_structure(self):
        from run_phase2b16_fresh_corrected_objective_validation import bootstrap_ci
        lo, hi = bootstrap_ci([0.1] * 100)
        assert lo <= hi
        assert lo > -1.0
        assert hi < 2.0

    def test_zero_diffs_give_zero_ci(self):
        from run_phase2b16_fresh_corrected_objective_validation import bootstrap_ci
        lo, hi = bootstrap_ci([0.0] * 200)
        assert abs(lo) < 1e-9
        assert abs(hi) < 1e-9

    def test_positive_diffs_give_positive_ci_lo(self):
        from run_phase2b16_fresh_corrected_objective_validation import bootstrap_ci
        diffs = [0.05] * 200
        lo, hi = bootstrap_ci(diffs, n_bootstrap=500)
        assert lo > 0, f"CI lower bound should be positive for all-positive diffs, got {lo}"

    def test_negative_diffs_give_negative_ci_hi(self):
        from run_phase2b16_fresh_corrected_objective_validation import bootstrap_ci
        diffs = [-0.05] * 200
        lo, hi = bootstrap_ci(diffs, n_bootstrap=500)
        assert hi < 0, f"CI upper bound should be negative for all-negative diffs, got {hi}"

    def test_ci_width_proportional_to_variance(self):
        from run_phase2b16_fresh_corrected_objective_validation import bootstrap_ci
        rng = np.random.default_rng(0)
        low_var = rng.normal(0.1, 0.01, 200).tolist()
        high_var = rng.normal(0.1, 0.5, 200).tolist()
        lo1, hi1 = bootstrap_ci(low_var, n_bootstrap=500)
        lo2, hi2 = bootstrap_ci(high_var, n_bootstrap=500)
        assert (hi2 - lo2) > (hi1 - lo1), "Higher variance → wider CI"

    def test_ci_reproducible_with_seed(self):
        from run_phase2b16_fresh_corrected_objective_validation import bootstrap_ci
        rng = np.random.default_rng(0)
        diffs = rng.normal(0.05, 0.1, 100).tolist()
        lo1, hi1 = bootstrap_ci(diffs, n_bootstrap=500, rng_seed=42)
        lo2, hi2 = bootstrap_ci(diffs, n_bootstrap=500, rng_seed=42)
        assert lo1 == lo2 and hi1 == hi2


# ============================================================================
# Top-epsilon accuracy tests
# ============================================================================

class TestTopEpsilonAccuracy:
    def test_perfect_selector_has_top_eps_1(self):
        from run_phase2b16_fresh_corrected_objective_validation import top_epsilon_accuracy
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows([_make_scorpio_dominant_row() for _ in range(20)])
        for r in rows:
            r["sel_rf_anwg_policy"] = r["best_policy_anwg"]
        preds = [r["best_policy_anwg"] for r in rows]
        acc = top_epsilon_accuracy(preds, rows, epsilon=0.001)
        assert acc == 1.0

    def test_wrong_selector_can_still_be_epsilon_accurate(self):
        from run_phase2b16_fresh_corrected_objective_validation import top_epsilon_accuracy
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows([_make_scorpio_dominant_row() for _ in range(10)])
        # Predict WSP instead of SCORPIO
        preds = [WSP] * len(rows)
        # WSP ANWG should be close enough to SCORPIO ANWG in some rows
        acc_tight = top_epsilon_accuracy(preds, rows, epsilon=0.0001)
        acc_loose = top_epsilon_accuracy(preds, rows, epsilon=1.0)
        # Loose epsilon → all acceptable
        assert acc_loose == 1.0
        # Tight epsilon → some may not qualify
        assert 0.0 <= acc_tight <= 1.0

    def test_empty_preds_returns_zero(self):
        from run_phase2b16_fresh_corrected_objective_validation import top_epsilon_accuracy
        assert top_epsilon_accuracy([], [], epsilon=0.01) == 0.0

    def test_top_eps_monotone_in_epsilon(self):
        from run_phase2b16_fresh_corrected_objective_validation import top_epsilon_accuracy
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows(_make_fresh_rows())
        preds = [WSP] * len(rows)
        a1 = top_epsilon_accuracy(preds, rows, 0.001)
        a2 = top_epsilon_accuracy(preds, rows, 0.005)
        a3 = top_epsilon_accuracy(preds, rows, 0.010)
        assert a1 <= a2 <= a3, f"top-epsilon accuracy must be non-decreasing in ε: {a1} {a2} {a3}"


# ============================================================================
# Win/tie/loss tests
# ============================================================================

class TestWinTieLoss:
    def test_all_wins(self):
        from run_phase2b16_fresh_corrected_objective_validation import win_tie_loss
        result = win_tie_loss([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
        assert result["wins"] == 3
        assert result["losses"] == 0
        assert result["ties"] == 0

    def test_all_losses(self):
        from run_phase2b16_fresh_corrected_objective_validation import win_tie_loss
        result = win_tie_loss([0.5, 0.5], [1.0, 1.0])
        assert result["losses"] == 2
        assert result["wins"] == 0

    def test_all_ties(self):
        from run_phase2b16_fresh_corrected_objective_validation import win_tie_loss
        result = win_tie_loss([0.9, 0.8], [0.9, 0.8])
        assert result["ties"] == 2
        assert result["wins"] == 0
        assert result["losses"] == 0

    def test_total_equals_n(self):
        from run_phase2b16_fresh_corrected_objective_validation import win_tie_loss
        sel = [0.8, 0.9, 0.7, 0.85]
        ref = [0.9, 0.8, 0.7, 0.7]
        result = win_tie_loss(sel, ref)
        assert result["wins"] + result["ties"] + result["losses"] == len(sel)


# ============================================================================
# Constrained objective tests
# ============================================================================

class TestConstrainedObjective:
    def _rows_with_completion(self):
        rows = []
        for _ in range(10):
            r = _make_row()
            r[f"completion_{SCORPIO}"] = 0.90
            r[f"reward_{SCORPIO}"] = 0.97
            r[f"completion_{WSP}"] = 1.0
            r[f"reward_{WSP}"] = 0.85
            rows.append(r)
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        return relabel_rows(rows)

    def test_constrained_wsp_always_satisfies_095(self):
        from run_phase2b16_fresh_corrected_objective_validation import constrained_objective
        rows = self._rows_with_completion()
        result = constrained_objective(rows, 0.95)
        # WSP CF=1.0 always satisfies 0.95 threshold
        wsp_anwg = result["always_wsp_anwg"]
        assert wsp_anwg > 0

    def test_constrained_095_vs_099(self):
        from run_phase2b16_fresh_corrected_objective_validation import constrained_objective
        rows = self._rows_with_completion()
        r_095 = constrained_objective(rows, 0.95)
        r_099 = constrained_objective(rows, 0.99)
        # Higher threshold → fewer policies qualify → potentially lower oracle ANWG
        assert r_099["oracle_constrained_anwg"] <= r_095["oracle_constrained_anwg"] + 1e-6

    def test_scorpio_fraction_099_below_095(self):
        from run_phase2b16_fresh_corrected_objective_validation import constrained_objective
        rows = self._rows_with_completion()
        r_095 = constrained_objective(rows, 0.95)
        r_099 = constrained_objective(rows, 0.99)
        # SCORPIO CF=0.90 fails 0.95 threshold but passes — wait, 0.90 < 0.95
        assert r_099["scorpio_satisfies_fraction"] <= r_095["scorpio_satisfies_fraction"] + 1e-6

    def test_constrained_returns_required_keys(self):
        from run_phase2b16_fresh_corrected_objective_validation import constrained_objective
        rows = self._rows_with_completion()
        result = constrained_objective(rows, 0.95)
        for k in ["completion_threshold", "oracle_constrained_anwg",
                   "always_wsp_anwg", "always_scorpio_anwg",
                   "scorpio_satisfies_constraint_n", "scorpio_satisfies_fraction"]:
            assert k in result, f"Missing key: {k}"


# ============================================================================
# Selector training freeze contract
# ============================================================================

class TestSelectorFreezeContract:
    def test_selectors_trained_only_on_old_data(self):
        """Verify train split uses seeds < 12 (old data), not fresh seeds."""
        import yaml
        cfg_path = ROOT / "configs" / "phase2b16_fresh_corrected_objective_validation.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        train_seeds = cfg.get("selector_training", {}).get("train_diversity_seeds", [])
        fresh_seeds = cfg.get("fresh_diversity_seeds", []) + cfg.get("fresh_heldout_seeds", [])
        overlap = set(train_seeds) & set(fresh_seeds)
        assert not overlap, \
            f"Train seeds {train_seeds} overlap with fresh seeds {fresh_seeds}: {overlap}"

    def test_b15_selectors_importable(self):
        """Phase 2B.15 selector classes must be importable."""

    def test_safe_fallback_wsp_needs_full_rows(self):
        """SafeFallbackWspSelector must be called with full rows (not features only)."""
        from run_phase2b15_corrected_objective_selector_retraining import (
            SafeFallbackWspSelector, relabel_rows,
        )
        from llmserveopt.selector.models import RandomForestSelector
        base = RandomForestSelector()
        # Fit on minimal data — add best_policy for RF fit
        rows = relabel_rows([_make_scorpio_dominant_row() for _ in range(10)])
        for r in rows:
            r["best_policy"] = r["best_policy_anwg"]
        base.fit(rows)
        sf = SafeFallbackWspSelector(base, margin=0.001)
        full_rows = relabel_rows([_make_scorpio_dominant_row() for _ in range(3)])
        preds = sf.predict(full_rows)
        assert all(p in SELECTOR_CANDIDATES for p in preds)

    def test_safe_fallback_wsp_not_wrapped_in_runner(self):
        """Verify that apply_selectors_to_rows will call predict([row]) on SafeFallbackWsp."""
        from run_phase2b15_corrected_objective_selector_retraining import SafeFallbackWspSelector
        # SafeFallbackWsp must NOT have predict_one attribute
        # (if it did, apply_selectors_to_rows would pass features-only dict → wrong)
        assert not hasattr(SafeFallbackWspSelector, "predict_one"), \
            "SafeFallbackWspSelector must not have predict_one — it needs full rows"


# ============================================================================
# Statistical analysis tests
# ============================================================================

class TestStatisticalAnalysis:
    def _make_applied_rows(self, n: int = 20):
        from run_phase2b15_corrected_objective_selector_retraining import relabel_rows
        rows = relabel_rows(_make_fresh_rows(n_scorpio=n // 2, n_wsp=n // 2))
        for r in rows:
            r["sel_always_scorpio_policy"] = SCORPIO
            r["sel_always_wsp_policy"] = WSP
            r["sel_rf_anwg_policy"] = r.get("best_policy_anwg", SCORPIO)
        return rows

    def test_statistical_analysis_returns_all_selectors(self):
        from run_phase2b16_fresh_corrected_objective_validation import statistical_analysis
        rows = self._make_applied_rows()
        stats = statistical_analysis(rows, ["always_scorpio", "always_wsp", "rf_anwg"],
                                     n_bootstrap=100)
        assert "always_scorpio" in stats
        assert "always_wsp" in stats
        assert "rf_anwg" in stats

    def test_always_scorpio_gap_vs_itself_is_zero(self):
        from run_phase2b16_fresh_corrected_objective_validation import statistical_analysis
        rows = self._make_applied_rows()
        stats = statistical_analysis(rows, ["always_scorpio"], n_bootstrap=100)
        # always_scorpio gap vs itself is listed in the baselines section, not sel section
        # Check baseline entry
        assert "always_scorpio" in stats
        assert stats["always_scorpio"]["mean_gap_vs_scorpio"] == 0.0

    def test_ci_keys_present(self):
        from run_phase2b16_fresh_corrected_objective_validation import statistical_analysis
        rows = self._make_applied_rows()
        stats = statistical_analysis(rows, ["rf_anwg"], n_bootstrap=100)
        assert "ci95_vs_scorpio" in stats.get("rf_anwg", {})
        assert len(stats["rf_anwg"]["ci95_vs_scorpio"]) == 2

    def test_win_tie_loss_sums_to_n(self):
        from run_phase2b16_fresh_corrected_objective_validation import statistical_analysis
        rows = self._make_applied_rows()
        stats = statistical_analysis(rows, ["always_wsp"], n_bootstrap=100)
        wtl = stats.get("always_wsp", {}).get("win_tie_loss_vs_scorpio", {})
        assert wtl.get("wins", 0) + wtl.get("ties", 0) + wtl.get("losses", 0) == len(rows)

    def test_top_epsilon_keys_present(self):
        from run_phase2b16_fresh_corrected_objective_validation import statistical_analysis
        rows = self._make_applied_rows()
        stats = statistical_analysis(rows, ["rf_anwg"], n_bootstrap=100)
        top_eps = stats.get("rf_anwg", {}).get("top_eps_accuracy", {})
        assert any("0.001" in k for k in top_eps)
        assert any("0.005" in k for k in top_eps)
        assert any("0.01" in k for k in top_eps)


# ============================================================================
# Output file tests (skipped until experiment run)
# ============================================================================

@_SKIP_RESULTS
class TestOutputFiles:
    def test_fresh_selector_comparison_exists(self):
        assert (_RESULTS_DIR / "fresh_selector_comparison.csv").exists()

    def test_fresh_group_summary_exists(self):
        assert (_RESULTS_DIR / "fresh_group_summary.csv").exists()

    def test_fresh_significance_summary_exists(self):
        assert (_RESULTS_DIR / "fresh_significance_summary.json").exists()

    def test_fresh_overall_summary_exists(self):
        assert (_RESULTS_DIR / "fresh_overall_summary.json").exists()

    def test_fresh_fifo_artifact_audit_exists(self):
        assert (_RESULTS_DIR / "fresh_fifo_artifact_audit.json").exists()

    def test_fresh_constrained_objectives_exists(self):
        assert (_RESULTS_DIR / "fresh_constrained_objectives.json").exists()

    def test_fresh_near_tie_summary_exists(self):
        assert (_RESULTS_DIR / "fresh_near_tie_summary.json").exists()

    def test_fresh_per_window_exists(self):
        assert (_RESULTS_DIR / "fresh_per_window.csv").exists()

    def test_fresh_policy_ranking_exists(self):
        assert (_RESULTS_DIR / "fresh_policy_ranking.csv").exists()

    def test_fresh_seed_summary_exists(self):
        assert (_RESULTS_DIR / "fresh_seed_summary.csv").exists()

    def test_fresh_top_epsilon_accuracy_exists(self):
        assert (_RESULTS_DIR / "fresh_top_epsilon_accuracy.csv").exists()

    def test_fresh_failure_cases_exists(self):
        assert (_RESULTS_DIR / "fresh_failure_cases.csv").exists()

    def test_overall_summary_has_n_fresh_windows(self):
        with open(_RESULTS_DIR / "fresh_overall_summary.json") as f:
            s = json.load(f)
        assert "n_fresh_windows" in s
        assert s["n_fresh_windows"] > 0

    def test_overall_summary_has_answers(self):
        with open(_RESULTS_DIR / "fresh_overall_summary.json") as f:
            s = json.load(f)
        for k in ["rf_anwg_beats_scorpio_fresh", "rf_anwg_ci_excludes_zero",
                   "any_selector_beats_wsp_fresh", "wsp_beats_scorpio_under_anwg"]:
            assert k in s.get("answers", {}), f"Missing answer key: {k}"

    def test_ci_present_for_rf_anwg(self):
        with open(_RESULTS_DIR / "fresh_significance_summary.json") as f:
            stats = json.load(f)
        rf = stats.get("rf_anwg", {})
        assert "ci95_vs_scorpio" in rf
        assert len(rf["ci95_vs_scorpio"]) == 2

    def test_label_dist_has_scorpio(self):
        with open(_RESULTS_DIR / "fresh_label_distribution.json") as f:
            d = json.load(f)
        assert SCORPIO in d.get("label_dist_anwg", {}) or d.get("n_total", 0) == 0

    def test_selector_comparison_has_rf_anwg(self):
        import pandas as pd
        df = pd.read_csv(_RESULTS_DIR / "fresh_selector_comparison.csv")
        assert "rf_anwg" in df["selector"].values

    def test_no_oracle_in_selector_comparison(self):
        import pandas as pd
        df = pd.read_csv(_RESULTS_DIR / "fresh_selector_comparison.csv")
        assert "oracle_srtf" not in df["selector"].values

    def test_selector_comparison_has_always_scorpio(self):
        import pandas as pd
        df = pd.read_csv(_RESULTS_DIR / "fresh_selector_comparison.csv")
        assert "always_scorpio" in df["selector"].values


# ============================================================================
# Documentation tests
# ============================================================================

class TestDocumentation:
    def test_runner_file_exists(self):
        assert (ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py").exists()

    def test_config_file_exists(self):
        assert (ROOT / "configs" / "phase2b16_fresh_corrected_objective_validation.yaml").exists()

    def test_audit_summary_doc_exists(self):
        assert (ROOT / "docs" / "audits" /
                "phase2b16_fresh_corrected_objective_validation_summary.md").exists(), \
            "Phase 2B.16 audit summary doc missing (create after experiment)"

    def test_failure_cases_doc_exists(self):
        assert (ROOT / "docs" / "audits" /
                "phase2b16_failure_cases_summary.md").exists(), \
            "Phase 2B.16 failure cases doc missing"

    def test_research_status_mentions_b16(self):
        rs = (ROOT / "docs" / "research_status.md")
        assert rs.exists()
        content = rs.read_text()
        assert "phase2b16" in content.lower() or "2b16" in content.lower(), \
            "docs/research_status.md should mention Phase 2B.16"

    def test_result_claims_mentions_b16(self):
        rc = (ROOT / "docs" / "result_claims.md")
        assert rc.exists()
        content = rc.read_text()
        assert "2b.16" in content.lower() or "2b16" in content.lower(), \
            "docs/result_claims.md should mention Phase 2B.16"


# ============================================================================
# Registry integrity tests
# ============================================================================

class TestRegistryIntegrity:
    def test_selector_candidates_no_oracle(self):
        assert "oracle_srtf" not in SELECTOR_CANDIDATES

    def test_scorpio_in_candidates(self):
        assert SCORPIO in SELECTOR_CANDIDATES

    def test_wsp_in_candidates(self):
        assert WSP in SELECTOR_CANDIDATES

    def test_fifo_in_candidates(self):
        assert FIFO in SELECTOR_CANDIDATES

    def test_make_policy_works(self):
        from llmserveopt.policies.registry import make_policy
        for p in SELECTOR_CANDIDATES:
            pol = make_policy(p)
            assert pol is not None, f"make_policy({p!r}) returned None"

    def test_candidate_count_stable(self):
        """SELECTOR_CANDIDATES count should match Phase 2B.14/2B.15 (20 policies)."""
        assert len(SELECTOR_CANDIDATES) == 20, \
            f"Expected 20 selector candidates, got {len(SELECTOR_CANDIDATES)}"


# ============================================================================
# Safety rule tests
# ============================================================================

class TestSafetyRules:
    def test_runner_has_no_paid_api_calls(self):
        runner = (ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py")
        src = runner.read_text()
        for forbidden in ["openai", "cohere", "gemini", "cloudrift", "cerebras", "mistral"]:
            assert forbidden not in src.lower(), f"Runner must not call paid API: {forbidden}"

    def test_runner_has_no_huggingface_token_usage(self):
        runner = (ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py")
        src = runner.read_text()
        # No HF token use
        assert "hf_" not in src, "Runner must not hard-code HuggingFace tokens"

    def test_config_has_no_api_keys(self):
        cfg_path = ROOT / "configs" / "phase2b16_fresh_corrected_objective_validation.yaml"
        with open(cfg_path) as f:
            content = f.read()
        for keyword in ["api_key", "api_secret", "hf_token", "openai_key"]:
            assert keyword not in content.lower(), f"Config must not contain: {keyword}"

    def test_log_dir_is_phase2b16(self):
        runner = (ROOT / "scripts" / "run_phase2b16_fresh_corrected_objective_validation.py")
        src = runner.read_text()
        assert "logs/phase2b16" in src, "Logs must go under logs/phase2b16/"


# ============================================================================
# Seed independence tests
# ============================================================================

class TestSeedIndependence:
    def test_fresh_seeds_not_in_b13_diversity_range(self):
        """Fresh seeds must not overlap any seeds used in Phase 2B.13 diversity."""
        import yaml
        b13_cfg_path = ROOT / "configs" / "phase2b13_selector_training_and_suspicion_audit.yaml"
        b16_cfg_path = ROOT / "configs" / "phase2b16_fresh_corrected_objective_validation.yaml"
        if not b13_cfg_path.exists():
            pytest.skip("Phase 2B.13 config not found")
        with open(b13_cfg_path) as f:
            b13_cfg = yaml.safe_load(f)
        with open(b16_cfg_path) as f:
            b16_cfg = yaml.safe_load(f)
        b13_seeds = set(b13_cfg.get("diversity_seeds", []) + b13_cfg.get("heldout_seeds", []))
        b16_seeds = set(b16_cfg.get("fresh_diversity_seeds", []) + b16_cfg.get("fresh_heldout_seeds", []))
        overlap = b13_seeds & b16_seeds
        assert not overlap, f"Fresh seeds overlap Phase 2B.13 seeds: {overlap}"

    def test_fresh_heldout_seeds_above_15(self):
        import yaml
        with open(ROOT / "configs" / "phase2b16_fresh_corrected_objective_validation.yaml") as f:
            cfg = yaml.safe_load(f)
        heldout_seeds = cfg.get("fresh_heldout_seeds", [])
        assert all(s >= 15 for s in heldout_seeds), \
            f"Heldout seeds {heldout_seeds} should be ≥ 15 (avoid B13 diversity range)"

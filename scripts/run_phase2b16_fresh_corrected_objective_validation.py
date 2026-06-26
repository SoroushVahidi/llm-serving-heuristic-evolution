#!/usr/bin/env python3
"""
Phase 2B.16: Fresh Corrected Objective Validation.

Tests whether Phase 2B.15 selector gains survive evaluation on fresh, unseen
simulation windows.  Selectors are retrained from Phase 2B.13 training data,
frozen before any fresh window is evaluated, then applied to entirely new
workload seeds and regimes.

Steps
-----
  Phase A — Selector training (from old data only, ~5s)
    1. Load Phase 2B.13 per_window.csv.
    2. Apply Phase 2B.13 train split (dev s0-2, diversity s6-10).
    3. Retrain Phase 2B.15 selectors under arrival-normalized WG:
         rf_anwg, rf_anwg_regret, dt_anwg, dt_anwg_regret,
         knn_anwg, regression_anwg, safe_fallback_wsp_{0.001,0.005,0.010}.
    4. Freeze selectors — no further training after this point.

  Phase B — Fresh simulation (~50-90 min)
    5. Run all 20 deployable policies on each fresh workload window.
    6. Compute per-window WG, completion fraction, SLO violation.
    7. Save raw per_window data.

  Phase C — Selector evaluation (~5s)
    8. Apply all frozen Phase 2B.15 selectors to fresh windows.
    9. Compute arrival_normalized_wg and 4 completion-penalized variants.
    10. Compute relabeling under arrival-norm WG.

  Phase D — Statistical analysis (~10s)
    11. Bootstrap 95% CI for key gaps (selector vs always-SCORPIO, vs always-WSP).
    12. Win/tie/loss counts for main comparisons.
    13. FIFO-artifact audit (identify near-tie FIFO "wins").
    14. Top-epsilon accuracy at ε=0.001, 0.005, 0.010.
    15. Constrained objective: best policy satisfying CF ≥ 0.95 / 0.99.
    16. Group-level and seed-level analysis.

  Phase E — Output and reporting (~5s)
    17. Write all output files.
    18. Generate failure cases report.
    19. Answer required validation questions.

Usage
-----
python scripts/run_phase2b16_fresh_corrected_objective_validation.py \\
    --config configs/phase2b16_fresh_corrected_objective_validation.yaml \\
    [--output results/phase2b16_fresh_corrected_objective_validation] \\
    [--log-file logs/phase2b16/phase2b16_fresh_validation.log] \\
    [--skip-simulation]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FeatureMode, FEATURE_NAMES
from llmserveopt.selector.models import (
    DecisionTreeSelector,
    RandomForestSelector,
    RuleBasedSelector,
)

from run_phase2b9_selector_robustness import (
    apply_selectors_to_rows,
    build_gpu_configs,
    compute_fixed_baseline_wgs,
    load_config,
    load_or_generate_trace,
    write_per_window_csv,
)
from run_phase2b12_workload_diversity_selector_labels import build_rows_for_group
from run_phase2b15_corrected_objective_selector_retraining import (
    AlwaysScorpioSelector,
    AlwaysWSPSelector,
    KNNAnwgSelector,
    PerPolicyRegressionAnwgSelector,
    SafeFallbackWspSelector,
    METRIC_KEYS,
    SCORPIO,
    WSP,
    _anwg,
    _comp_frac,
    _cond_wg,
    _cp_wg,
    compute_all_metrics,
    compute_anwg_regret_weights,
    filter_meaningful,
    near_tie_stats,
    relabel_rows,
    split_rows,
)

try:
    from llmserveopt.simulator.service_model_factory import build_service_model_from_config
    _HAS_SERVICE_MODEL = True
except ImportError:
    _HAS_SERVICE_MODEL = False

ORACLE_POLICY = "oracle_srtf"


# ---------------------------------------------------------------------------
# Phase A: Selector training from old Phase 2B.13 data
# ---------------------------------------------------------------------------

def _feature_matrix(rows: List[Dict]) -> np.ndarray:
    return np.array(
        [[float(r.get(f"feat_{n}", 0.0) or 0.0) for n in FEATURE_NAMES] for r in rows],
        dtype=float,
    )


def _anwg_labels(rows: List[Dict]) -> List[Dict]:
    """Override best_policy with arrival-norm WG best (for sklearn fit)."""
    return [{**r, "best_policy": r.get("best_policy_anwg", r.get("best_policy", SCORPIO))} for r in rows]


def train_phase2b15_selectors(
    b13_per_window_path: Path,
    train_div_seeds: List[int],
    near_tie_eps: float,
    rw_eps: float,
    sf_margins: List[float],
    knn_k: int,
) -> Dict[str, Any]:
    """Load Phase 2B.13 data, retrain Phase 2B.15 selectors on TRAIN SPLIT ONLY."""
    if not b13_per_window_path.exists():
        raise FileNotFoundError(
            f"Phase 2B.13 per_window.csv not found at {b13_per_window_path}. "
            "Run Phase 2B.13 first."
        )
    df = pd.read_csv(b13_per_window_path)
    all_rows = relabel_rows(df.to_dict(orient="records"))
    logging.info("Loaded %d rows from Phase 2B.13 per_window.csv", len(all_rows))

    # Apply TRAIN split (dev+diversity seeds 6-10 only)
    train_rows, val_rows, _test_rows = split_rows(all_rows, train_div_seeds, [11])
    logging.info("Train split: %d windows (dev+div s6-10)", len(train_rows))

    train_anwg = _anwg_labels(train_rows)
    X_tr = _feature_matrix(train_anwg)
    y_tr = [r["best_policy"] for r in train_anwg]
    rw_all = compute_anwg_regret_weights(train_rows, rw_eps)

    # RF (arrival-norm labels)
    rf_anwg = RandomForestSelector(n_estimators=200, max_depth=10, random_state=42)
    rf_anwg.name = "rf_anwg"
    rf_anwg.fit(train_anwg)

    # RF regret-weighted
    rf_anwg_rw = RandomForestSelector(n_estimators=200, max_depth=10, random_state=42)
    rf_anwg_rw.name = "rf_anwg_regret"
    rf_anwg_rw._clf.fit(X_tr, y_tr, sample_weight=rw_all)

    # DT
    dt_anwg = DecisionTreeSelector(max_depth=8, min_samples_leaf=5, random_state=42)
    dt_anwg.name = "dt_anwg"
    dt_anwg.fit(train_anwg)

    # DT regret-weighted
    dt_anwg_rw = DecisionTreeSelector(max_depth=8, min_samples_leaf=5, random_state=42)
    dt_anwg_rw.name = "dt_anwg_regret"
    dt_anwg_rw._clf.fit(X_tr, y_tr, sample_weight=rw_all)

    # KNN
    knn_anwg = KNNAnwgSelector(k=knn_k)
    knn_anwg.fit(train_rows)

    # PerPolicyRegression
    reg_anwg = PerPolicyRegressionAnwgSelector()
    reg_anwg.fit(train_rows)

    # Safe-fallback-WSP selectors (base = rf_anwg)
    sf_selectors = [SafeFallbackWspSelector(rf_anwg, m) for m in sf_margins]

    # Rule-based (from Phase 2B.13, deterministic)
    rule_based = RuleBasedSelector()
    rule_based.name = "rule_based"

    selectors = {
        "always_scorpio": AlwaysScorpioSelector(),
        "always_wsp": AlwaysWSPSelector(),
        "rule_based": rule_based,
        "rf_anwg": rf_anwg,
        "rf_anwg_regret": rf_anwg_rw,
        "dt_anwg": dt_anwg,
        "dt_anwg_regret": dt_anwg_rw,
        "knn_anwg": knn_anwg,
        "regression_anwg": reg_anwg,
    }
    for sf in sf_selectors:
        selectors[sf.name] = sf

    logging.info("Trained and frozen %d selectors from Phase 2B.13 train split", len(selectors))
    return selectors


# ---------------------------------------------------------------------------
# Phase C: Corrected metric evaluation on fresh windows
# ---------------------------------------------------------------------------

def evaluate_fresh_selector(
    sel_key: str,
    rows: List[Dict],
    anwg_label_key: str = "best_policy_anwg",
) -> Dict:
    """Evaluate one selector on fresh rows using all metric variants."""
    policy_col = f"sel_{sel_key}_policy"
    preds = [r.get(policy_col, SCORPIO) or SCORPIO for r in rows]
    n = len(rows)
    if n == 0:
        return {"selector": sel_key, "n_windows": 0}

    labels = [r.get(anwg_label_key, "") for r in rows]
    correct = sum(p == l for p, l in zip(preds, labels))

    metric_vals: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for pred, row in zip(preds, rows):
        m = compute_all_metrics(row, pred)
        for k in METRIC_KEYS:
            metric_vals[k].append(m[k])

    scorpio_anwgs = [_anwg(r, SCORPIO) for r in rows]
    wsp_anwgs = [_anwg(r, WSP) for r in rows]
    oracle_anwgs = [float(r.get("best_anwg", 0.0)) for r in rows]
    mean_scorpio = float(np.mean(scorpio_anwgs))
    mean_wsp = float(np.mean(wsp_anwgs))
    mean_oracle = float(np.mean(oracle_anwgs))
    mean_anwg = float(np.mean(metric_vals["arrival_normalized_wg"]))

    return {
        "selector": sel_key,
        "n_windows": n,
        "label_accuracy_anwg": round(correct / n, 4),
        "chosen_policy_dist": dict(Counter(preds)),
        "collapses_to_scorpio": dict(Counter(preds)).get(SCORPIO, 0) == n,
        **{f"mean_{k}": round(float(np.mean(v)), 4) for k, v in metric_vals.items()},
        "gap_vs_always_scorpio_anwg": round(mean_anwg - mean_scorpio, 4),
        "gap_vs_always_wsp_anwg": round(mean_anwg - mean_wsp, 4),
        "gap_vs_oracle_anwg": round(mean_anwg - mean_oracle, 4),
        "always_scorpio_anwg": round(mean_scorpio, 4),
        "always_wsp_anwg": round(mean_wsp, 4),
        "oracle_anwg": round(mean_oracle, 4),
    }


def evaluate_group(rows: List[Dict], sel_keys: List[str], group_name: str) -> List[Dict]:
    return [
        {"group": group_name, **evaluate_fresh_selector(k, rows)}
        for k in sel_keys
    ]


# ---------------------------------------------------------------------------
# Phase D: Statistical analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(
    paired_diffs: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    arr = np.array(paired_diffs)
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan")
    boots = rng.choice(arr, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


def win_tie_loss(
    sel_anwgs: List[float],
    ref_anwgs: List[float],
    eps: float = 1e-6,
) -> Dict[str, int]:
    wins = sum(s > r + eps for s, r in zip(sel_anwgs, ref_anwgs))
    losses = sum(s < r - eps for s, r in zip(sel_anwgs, ref_anwgs))
    ties = len(sel_anwgs) - wins - losses
    return {"wins": wins, "ties": ties, "losses": losses}


def top_epsilon_accuracy(
    preds: List[str],
    rows: List[Dict],
    epsilon: float,
) -> float:
    if not preds:
        return 0.0
    count = 0
    for pred, row in zip(preds, rows):
        pred_anwg = _anwg(row, pred)
        best_anwg = float(row.get("best_anwg", 0.0))
        if best_anwg - pred_anwg <= epsilon:
            count += 1
    return round(count / len(preds), 4)


def fifo_artifact_audit(rows: List[Dict]) -> Dict:
    """Classify FIFO wins as genuine or near-tie artifact."""
    fifo_wins = [r for r in rows if r.get("best_policy_anwg") == "fifo"]
    n_fifo = len(fifo_wins)
    n_near_tie_001 = sum(
        1 for r in fifo_wins if float(r.get("policy_margin_anwg", 0.0)) < 0.001
    )
    n_near_tie_005 = sum(
        1 for r in fifo_wins if float(r.get("policy_margin_anwg", 0.0)) < 0.005
    )
    n_near_tie_010 = sum(
        1 for r in fifo_wins if float(r.get("policy_margin_anwg", 0.0)) < 0.010
    )
    return {
        "n_fifo_wins_anwg": n_fifo,
        "n_total": len(rows),
        "fifo_win_fraction": round(n_fifo / len(rows), 4) if rows else 0.0,
        "n_fifo_near_tie_eps001": n_near_tie_001,
        "n_fifo_near_tie_eps005": n_near_tie_005,
        "n_fifo_near_tie_eps010": n_near_tie_010,
        "fraction_fifo_wins_are_near_tie_eps005": (
            round(n_near_tie_005 / n_fifo, 4) if n_fifo > 0 else float("nan")
        ),
        "n_fifo_genuine_wins_eps005": n_fifo - n_near_tie_005,
        "note": (
            "FIFO 'wins' under arrival-norm WG occur when all policies achieve cond_WG=1.0 "
            "but SCORPIO has CF<1.0. If all FIFO wins are near-ties (margin<0.005), "
            "they are metric artifacts, not genuine FIFO advantages."
        ),
    }


def constrained_objective(
    rows: List[Dict],
    completion_threshold: float,
) -> Dict:
    """Best policy satisfying CF >= threshold per window (oracle constrained)."""
    best_constrained_wgs: List[float] = []
    best_constrained_policies: List[str] = []
    for row in rows:
        candidates = []
        for p in SELECTOR_CANDIDATES:
            cf = _comp_frac(row, p)
            wg = _anwg(row, p)
            if cf >= completion_threshold:
                candidates.append((p, wg, cf))
        if candidates:
            # Among satisfying policies, pick highest ANWG
            best_p, best_wg, _ = max(candidates, key=lambda x: x[1])
        else:
            # No policy satisfies threshold: pick highest CF then highest ANWG
            all_cfwg = [(p, _comp_frac(row, p), _anwg(row, p)) for p in SELECTOR_CANDIDATES]
            all_cfwg.sort(key=lambda x: (-x[1], -x[2]))
            best_p, _, best_wg = all_cfwg[0]
        best_constrained_wgs.append(best_wg)
        best_constrained_policies.append(best_p)

    # WSP constrained performance (WSP has CF≈1.0, always satisfies threshold)
    wsp_anwgs = [_anwg(r, WSP) for r in rows]
    scorpio_anwgs = [_anwg(r, SCORPIO) for r in rows]
    scorpio_cfs = [_comp_frac(r, SCORPIO) for r in rows]
    scorpio_satisfies = sum(1 for cf in scorpio_cfs if cf >= completion_threshold)

    return {
        "completion_threshold": completion_threshold,
        "oracle_constrained_anwg": round(float(np.mean(best_constrained_wgs)), 4),
        "constrained_policy_dist": dict(Counter(best_constrained_policies)),
        "always_wsp_anwg": round(float(np.mean(wsp_anwgs)), 4),
        "always_scorpio_anwg": round(float(np.mean(scorpio_anwgs)), 4),
        "scorpio_satisfies_constraint_n": scorpio_satisfies,
        "scorpio_satisfies_fraction": round(scorpio_satisfies / len(rows), 4) if rows else 0.0,
        "note": (
            f"Oracle constrained: best deployable policy with CF≥{completion_threshold} per window. "
            "WSP always satisfies constraint (CF≈1.0). "
            f"SCORPIO satisfies {scorpio_satisfies}/{len(rows)} windows."
        ),
    }


def statistical_analysis(
    rows: List[Dict],
    sel_keys: List[str],
    n_bootstrap: int = 2000,
    rng_seed: int = 42,
) -> Dict:
    """Bootstrap CIs and win/tie/loss for key selectors vs baselines."""
    scorpio_anwgs = np.array([_anwg(r, SCORPIO) for r in rows])
    wsp_anwgs = np.array([_anwg(r, WSP) for r in rows])
    oracle_anwgs = np.array([float(r.get("best_anwg", 0.0)) for r in rows])

    results = {}
    for sk in sel_keys:
        policy_col = f"sel_{sk}_policy"
        preds = [r.get(policy_col, SCORPIO) or SCORPIO for r in rows]
        sel_anwgs = np.array([_anwg(r, p) for p, r in zip(preds, rows)])

        diff_vs_scorpio = (sel_anwgs - scorpio_anwgs).tolist()
        diff_vs_wsp = (sel_anwgs - wsp_anwgs).tolist()
        diff_vs_oracle = (sel_anwgs - oracle_anwgs).tolist()

        ci_lo_scorpio, ci_hi_scorpio = bootstrap_ci(diff_vs_scorpio, n_bootstrap, rng_seed=rng_seed)
        ci_lo_wsp, ci_hi_wsp = bootstrap_ci(diff_vs_wsp, n_bootstrap, rng_seed=rng_seed)

        wtl_scorpio = win_tie_loss(sel_anwgs.tolist(), scorpio_anwgs.tolist())
        wtl_wsp = win_tie_loss(sel_anwgs.tolist(), wsp_anwgs.tolist())

        results[sk] = {
            "mean_anwg": round(float(sel_anwgs.mean()), 4),
            "mean_gap_vs_scorpio": round(float(np.mean(diff_vs_scorpio)), 4),
            "mean_gap_vs_wsp": round(float(np.mean(diff_vs_wsp)), 4),
            "mean_gap_vs_oracle": round(float(np.mean(diff_vs_oracle)), 4),
            "ci95_vs_scorpio": [round(ci_lo_scorpio, 4), round(ci_hi_scorpio, 4)],
            "ci95_vs_wsp": [round(ci_lo_wsp, 4), round(ci_hi_wsp, 4)],
            "ci_includes_zero_vs_scorpio": ci_lo_scorpio <= 0 <= ci_hi_scorpio,
            "ci_includes_zero_vs_wsp": ci_lo_wsp <= 0 <= ci_hi_wsp,
            "win_tie_loss_vs_scorpio": wtl_scorpio,
            "win_tie_loss_vs_wsp": wtl_wsp,
            "top_eps_accuracy": {
                str(eps): top_epsilon_accuracy(preds, rows, eps)
                for eps in [0.001, 0.005, 0.010]
            },
        }

    # Baseline stats (only add if not already computed as a sel_key, to avoid overwriting)
    if "always_scorpio" not in results:
        results["always_scorpio"] = {
            "mean_anwg": round(float(scorpio_anwgs.mean()), 4),
            "mean_gap_vs_scorpio": 0.0,
            "mean_gap_vs_wsp": round(float((scorpio_anwgs - wsp_anwgs).mean()), 4),
        }
    if "always_wsp" not in results:
        results["always_wsp"] = {
            "mean_anwg": round(float(wsp_anwgs.mean()), 4),
            "mean_gap_vs_scorpio": round(float((wsp_anwgs - scorpio_anwgs).mean()), 4),
            "mean_gap_vs_wsp": 0.0,
        }
    if "oracle_per_window" not in results:
        results["oracle_per_window"] = {
            "mean_anwg": round(float(oracle_anwgs.mean()), 4),
        }
    return results


def build_failure_cases(
    fresh_rows: List[Dict],
    stats: Dict,
    nt_stats: Dict,
    fifo_audit: Dict,
) -> List[Dict]:
    cases = []
    rf_stats = stats.get("rf_anwg", {})
    rf_gap = rf_stats.get("mean_gap_vs_scorpio", float("nan"))
    rf_ci = rf_stats.get("ci95_vs_scorpio", [float("nan"), float("nan")])
    sf_gap = stats.get("safe_fallback_wsp_margin0.001", {}).get("mean_gap_vs_scorpio", float("nan"))

    if not np.isnan(rf_gap) and rf_gap <= 0.0:
        cases.append({
            "failure_id": "fail_027",
            "pattern": "rf_anwg does not beat always-SCORPIO on fresh validation",
            "status": "confirmed" if rf_gap <= 0 else "resolved",
            "detail": f"rf_anwg gap={rf_gap:+.4f} always-SCORPIO",
        })

    if rf_ci[0] <= 0 <= rf_ci[1]:
        cases.append({
            "failure_id": "fail_028",
            "pattern": "rf_anwg vs SCORPIO CI includes zero — not statistically reliable",
            "status": "confirmed",
            "detail": f"95% CI={rf_ci} (includes zero)",
        })

    n_fifo = fifo_audit.get("n_fifo_wins_anwg", 0)
    n_fifo_tied = fifo_audit.get("n_fifo_near_tie_eps005", 0)
    if n_fifo > 0 and n_fifo_tied == n_fifo:
        cases.append({
            "failure_id": "fail_029",
            "pattern": "All FIFO label wins under arrival-norm WG are near-tie artifacts",
            "status": "confirmed",
            "detail": f"n_fifo={n_fifo} all have margin<0.005",
        })

    # Near-tie domination
    nt_frac = nt_stats.get("fraction_near_tie_eps0.005", 0.0)
    if nt_frac > 0.6:
        cases.append({
            "failure_id": "fail_030",
            "pattern": "Near-tie labels dominate fresh validation (>60% at eps=0.005)",
            "status": "confirmed",
            "detail": f"near-tie fraction={nt_frac:.3f}",
        })

    # WSP beats best selector under ANWG
    best_sel_gap_vs_wsp = max(
        (v.get("mean_gap_vs_wsp", float("-inf")) for k, v in stats.items()
         if k not in ("always_scorpio", "always_wsp", "oracle_per_window")),
        default=float("-inf"),
    )
    if not np.isnan(best_sel_gap_vs_wsp) and best_sel_gap_vs_wsp < 0:
        cases.append({
            "failure_id": "fail_031",
            "pattern": "No selector beats always-WSP under arrival-norm WG on fresh validation",
            "status": "confirmed",
            "detail": f"best selector gap_vs_wsp={best_sel_gap_vs_wsp:+.4f}",
        })

    return cases


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _flatten_dict(d: Dict, prefix: str = "") -> Dict:
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=full_key + "_"))
        else:
            out[full_key] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2B.16 Fresh Corrected Objective Validation")
    p.add_argument("--config",
                   default="configs/phase2b16_fresh_corrected_objective_validation.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--skip-simulation", action="store_true",
                   help="Skip Phase B (simulation); load existing fresh_per_window.csv")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output or cfg.get("output_dir",
                   "results/phase2b16_fresh_corrected_objective_validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or "logs/phase2b16/phase2b16_fresh_validation.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    log = logging.getLogger(__name__)
    t0 = time.time()
    log.info("Phase 2B.16 start — output: %s", out_dir)

    # -------------------------------------------------------------------------
    # Phase A: Train selectors from old Phase 2B.13 data
    # -------------------------------------------------------------------------
    log.info("[Phase A] Training Phase 2B.15 selectors from Phase 2B.13 data …")
    b13_dir = Path(cfg["input_b13_dir"])
    sel_tr_cfg = cfg.get("selector_training", {})
    train_div_seeds = sel_tr_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10])
    near_tie_eps = sel_tr_cfg.get("near_tie_filter_epsilon", 0.005)
    rw_eps = sel_tr_cfg.get("regret_weight_epsilon", 0.001)
    sf_margins = cfg.get("safe_fallback_margins", [0.001, 0.005, 0.010])
    knn_k = cfg.get("knn", {}).get("k", 5)

    t_train = time.time()
    selectors = train_phase2b15_selectors(
        b13_per_window_path=b13_dir / "per_window.csv",
        train_div_seeds=train_div_seeds,
        near_tie_eps=near_tie_eps,
        rw_eps=rw_eps,
        sf_margins=sf_margins,
        knn_k=knn_k,
    )
    log.info("Phase A done in %.1fs — %d selectors frozen", time.time() - t_train, len(selectors))

    # Pass selectors directly: apply_selectors_to_rows uses predict_one (RuleBasedSelector)
    # or predict([row]) (all Phase 2B.15 selectors, including SafeFallbackWspSelector
    # which needs full rows with reward_* / completion_* keys).
    models = selectors

    # -------------------------------------------------------------------------
    # Phase B: Fresh simulation
    # -------------------------------------------------------------------------
    fresh_pw_path = out_dir / "fresh_per_window.csv"

    if args.skip_simulation and fresh_pw_path.exists():
        log.info("[Phase B] Skipping simulation — loading %s", fresh_pw_path)
        df_fresh = pd.read_csv(fresh_pw_path)
        fresh_rows = df_fresh.to_dict(orient="records")
        log.info("Loaded %d fresh rows", len(fresh_rows))
    else:
        log.info("[Phase B] Running fresh simulation …")
        if not _HAS_SERVICE_MODEL:
            log.error("service_model_factory not available — cannot run simulation")
            sys.exit(1)

        service_model = build_service_model_from_config(cfg.get("service_model", {}))
        gpu_configs = build_gpu_configs(cfg)
        drain_steps = cfg.get("simulator", {}).get("drain_steps", 20000)
        window_size = cfg.get("window_size", 200)
        min_partial = cfg.get("min_partial_window", 50)
        feature_mode = FeatureMode[cfg.get("feature_mode", "online_prefix").upper()]
        verbose = args.verbose

        # Build group → workloads/seeds mapping
        workload_defs = cfg.get("workloads", [])
        groups = defaultdict(list)
        for w in workload_defs:
            groups[w.get("group", "unknown")].append(w)

        seed_map = {
            "fresh_diversity": cfg.get("fresh_diversity_seeds", [12, 13, 14, 15]),
            "fresh_targeted": cfg.get("fresh_diversity_seeds", [12, 13, 14, 15]),
            "fresh_heldout": cfg.get("fresh_heldout_seeds", [20, 21, 22]),
        }

        t_sim = time.time()
        all_fresh_rows: List[Dict] = []
        for group_name, wdefs in sorted(groups.items()):
            seeds = seed_map.get(group_name, [12, 13, 14])
            log.info("  Simulating group=%s (%d workloads × %d seeds)", group_name, len(wdefs), len(seeds))
            rows = build_rows_for_group(
                workloads=wdefs,
                seeds=seeds,
                gpu_configs=gpu_configs,
                service_model=service_model,
                drain_steps=drain_steps,
                window_size=window_size,
                min_partial=min_partial,
                feature_mode=feature_mode,
                verbose=verbose,
            )
            all_fresh_rows.extend(rows)
            log.info("  Group %s: %d windows", group_name, len(rows))

        log.info("Simulation done in %.1fs — %d fresh windows", time.time() - t_sim, len(all_fresh_rows))

        # Apply selectors
        log.info("Applying %d frozen selectors to %d fresh windows …", len(models), len(all_fresh_rows))
        all_fresh_rows = apply_selectors_to_rows(all_fresh_rows, models)

        # Save raw per_window
        write_per_window_csv(all_fresh_rows, fresh_pw_path)
        fresh_rows = all_fresh_rows

    # -------------------------------------------------------------------------
    # Phase C: Corrected metric evaluation
    # -------------------------------------------------------------------------
    log.info("[Phase C] Relabeling and computing corrected metrics …")
    fresh_rows = relabel_rows(fresh_rows)
    n_fresh = len(fresh_rows)
    log.info("Relabeled %d fresh windows under arrival-norm WG", n_fresh)

    nt_thresholds = cfg.get("near_tie_thresholds", [0.001, 0.005, 0.010])
    nt_stats_all = near_tie_stats(fresh_rows, nt_thresholds)
    log.info("Near-tie (eps=0.005): n_meaningful=%d / %d (%.1f%%)",
             nt_stats_all.get("n_meaningful_eps0.005", 0), n_fresh,
             100 * nt_stats_all.get("n_meaningful_eps0.005", 0) / n_fresh if n_fresh else 0)

    label_dist_anwg = dict(Counter(r.get("best_policy_anwg", "?") for r in fresh_rows))
    label_dist_cond = dict(Counter(r.get("best_policy", "?") for r in fresh_rows))
    n_changes = sum(
        1 for r in fresh_rows
        if r.get("best_policy_anwg") != r.get("best_policy")
    )
    log.info("Label changes (cond→anwg): %d/%d (%.1f%%)", n_changes, n_fresh,
             100 * n_changes / n_fresh if n_fresh else 0)

    # Group breakdown
    group_rows: Dict[str, List[Dict]] = defaultdict(list)
    for r in fresh_rows:
        tid = r.get("trace_id", "?")
        grp = (
            "fresh_heldout" if "heldout" in tid else
            "fresh_targeted" if "tgt" in tid else
            "fresh_diversity"
        )
        group_rows[grp].append(r)
    for grp, rows in group_rows.items():
        log.info("  Group %s: %d windows", grp, len(rows))

    # FIFO artifact audit
    fifo_audit = fifo_artifact_audit(fresh_rows)
    _write_json(out_dir / "fresh_fifo_artifact_audit.json", fifo_audit)
    log.info("FIFO artifact audit: n_fifo=%d fraction=%.3f near_tie_eps005=%d",
             fifo_audit["n_fifo_wins_anwg"], fifo_audit["fifo_win_fraction"],
             fifo_audit["n_fifo_near_tie_eps005"])

    # Constrained objectives
    constrained_results = {}
    for thresh in cfg.get("constrained_objectives", {}).get("completion_thresholds", [0.95, 0.99]):
        constrained_results[thresh] = constrained_objective(fresh_rows, thresh)
    _write_json(out_dir / "fresh_constrained_objectives.json", constrained_results)

    # Selector evaluation per group
    sel_keys = list(models.keys())
    all_group_evals: List[Dict] = []
    for grp, rows in group_rows.items():
        all_group_evals.extend(evaluate_group(rows, sel_keys, grp))
    all_group_evals.extend(evaluate_group(fresh_rows, sel_keys, "overall"))
    _write_csv(out_dir / "fresh_group_summary.csv", [_flatten_dict(e) for e in all_group_evals])

    # Meaningful-only evaluation (eps=0.005)
    meaningful_rows = filter_meaningful(fresh_rows, 0.005)
    log.info("Meaningful windows (eps=0.005): %d / %d", len(meaningful_rows), n_fresh)
    meaningful_evals = evaluate_group(meaningful_rows, sel_keys, "meaningful_eps0.005")
    _write_csv(out_dir / "fresh_meaningful_summary.csv", [_flatten_dict(e) for e in meaningful_evals])

    # Overall selector evaluation
    overall_evals = [evaluate_fresh_selector(k, fresh_rows) for k in sel_keys]
    _write_csv(out_dir / "fresh_selector_comparison.csv", [_flatten_dict(e) for e in overall_evals])

    # Policy metric table
    policy_rows = []
    for p in SELECTOR_CANDIDATES:
        if f"reward_{p}" not in fresh_rows[0]:
            continue
        cf_vals = [_comp_frac(r, p) for r in fresh_rows]
        m = {k: [] for k in METRIC_KEYS}
        for row in fresh_rows:
            for k in METRIC_KEYS:
                m[k].append(compute_all_metrics(row, p)[k])
        policy_rows.append({"policy": p, "mean_completion_fraction": round(float(np.mean(cf_vals)), 4),
                             **{f"mean_{k}": round(float(np.mean(v)), 4) for k, v in m.items()}})
    policy_rows.sort(key=lambda x: -x["mean_arrival_normalized_wg"])
    _write_csv(out_dir / "fresh_policy_ranking.csv", policy_rows)

    # Label distribution CSV
    _write_json(out_dir / "fresh_label_distribution.json", {
        "n_total": n_fresh,
        "n_label_changes": n_changes,
        "label_dist_anwg": label_dist_anwg,
        "label_dist_cond": label_dist_cond,
    })

    # Near-tie summary
    _write_json(out_dir / "fresh_near_tie_summary.json", nt_stats_all)

    # -------------------------------------------------------------------------
    # Phase D: Statistical analysis
    # -------------------------------------------------------------------------
    log.info("[Phase D] Statistical analysis …")
    bs_cfg = cfg.get("bootstrap", {})
    n_bs = bs_cfg.get("n_samples", 2000)
    bs_seed = bs_cfg.get("random_seed", 42)

    # Ensure selectors are applied (already done via apply_selectors_to_rows)
    stats = statistical_analysis(fresh_rows, sel_keys, n_bootstrap=n_bs, rng_seed=bs_seed)
    _write_json(out_dir / "fresh_significance_summary.json", stats)

    # Log key results
    log.info("=== FRESH VALIDATION RESULTS (arrival-norm WG) ===")
    always_scorpio_anwg = stats.get("always_scorpio", {}).get("mean_anwg", 0.0)
    always_wsp_anwg = stats.get("always_wsp", {}).get("mean_anwg", 0.0)
    oracle_anwg = stats.get("oracle_per_window", {}).get("mean_anwg", 0.0)
    log.info("  always-SCORPIO: %.4f", always_scorpio_anwg)
    log.info("  always-WSP:     %.4f", always_wsp_anwg)
    log.info("  oracle:         %.4f", oracle_anwg)
    for sk in ["rf_anwg", "knn_anwg", "safe_fallback_wsp_margin0.001", "rule_based"]:
        if sk in stats:
            s = stats[sk]
            ci = s.get("ci95_vs_scorpio", [float("nan"), float("nan")])
            log.info("  %-40s anwg=%.4f gap_vs_scorpio=%+.4f CI=[%.4f, %.4f] ci0=%s",
                     sk, s.get("mean_anwg", 0), s.get("mean_gap_vs_scorpio", 0),
                     ci[0], ci[1],
                     "YES" if s.get("ci_includes_zero_vs_scorpio") else "no")

    # Top-epsilon accuracy table
    top_eps_table = []
    for sk in sel_keys:
        if sk in stats:
            s = stats[sk]
            row = {"selector": sk}
            for eps, acc in s.get("top_eps_accuracy", {}).items():
                row[f"top_eps_{eps}"] = acc
            top_eps_table.append(row)
    _write_csv(out_dir / "fresh_top_epsilon_accuracy.csv", top_eps_table)

    # Failure cases
    failure_cases = build_failure_cases(fresh_rows, stats, nt_stats_all, fifo_audit)
    _write_csv(out_dir / "fresh_failure_cases.csv", failure_cases)
    log.info("Failure cases: %d", len(failure_cases))

    # Seed-level analysis
    seed_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in fresh_rows:
        tid = r.get("trace_id", "?")
        seed = tid.rsplit("_s", 1)[1] if "_s" in tid else "?"
        seed_groups[seed].append(r)
    seed_rows = []
    for seed, s_rows in sorted(seed_groups.items()):
        s_anwg_map = {sk: float(np.mean([_anwg(r, (r.get(f"sel_{sk}_policy") or SCORPIO))
                                          for r in s_rows])) for sk in sel_keys}
        row = {
            "seed": seed,
            "n_windows": len(s_rows),
            "always_scorpio_anwg": round(float(np.mean([_anwg(r, SCORPIO) for r in s_rows])), 4),
            "always_wsp_anwg": round(float(np.mean([_anwg(r, WSP) for r in s_rows])), 4),
        }
        for sk in ["rf_anwg", "knn_anwg", "safe_fallback_wsp_margin0.001", "rule_based"]:
            row[f"{sk}_anwg"] = round(s_anwg_map.get(sk, float("nan")), 4)
        seed_rows.append(row)
    _write_csv(out_dir / "fresh_seed_summary.csv", seed_rows)

    # -------------------------------------------------------------------------
    # Phase E: Overall summary
    # -------------------------------------------------------------------------
    log.info("[Phase E] Writing summary …")

    # Best selector on fresh validation
    best_sk_anwg = max(
        (k for k in sel_keys if k in stats and "mean_anwg" in stats[k]),
        key=lambda k: stats[k].get("mean_anwg", 0.0),
        default="none",
    )
    best_anwg = stats.get(best_sk_anwg, {}).get("mean_anwg", float("nan"))
    rf_anwg_val = stats.get("rf_anwg", {}).get("mean_anwg", float("nan"))
    rf_ci = stats.get("rf_anwg", {}).get("ci95_vs_scorpio", [float("nan"), float("nan")])

    # Answer Phase 2B.16 questions
    rf_beats_scorpio = rf_anwg_val > always_scorpio_anwg if not np.isnan(rf_anwg_val) else False
    rf_ci_excludes_zero = not stats.get("rf_anwg", {}).get("ci_includes_zero_vs_scorpio", True)
    any_beats_wsp = any(
        stats.get(k, {}).get("mean_gap_vs_wsp", float("-inf")) > 0.0
        for k in sel_keys if k != "always_wsp"
    )
    wsp_beats_scorpio_under_anwg = always_wsp_anwg > always_scorpio_anwg

    summary = {
        "experiment": "phase2b16_fresh_corrected_objective_validation",
        "date": "2026-06-26",
        "branch": "phase2b16-fresh-corrected-objective-validation",
        "n_fresh_windows": n_fresh,
        "n_label_changes_cond_to_anwg": n_changes,
        "label_change_fraction": round(n_changes / n_fresh, 4) if n_fresh else 0.0,
        "label_dist_anwg": label_dist_anwg,
        "label_dist_cond": label_dist_cond,
        "near_tie_stats": nt_stats_all,
        "fifo_artifact_audit": fifo_audit,
        "baselines": {
            "always_scorpio_anwg": round(always_scorpio_anwg, 4),
            "always_wsp_anwg": round(always_wsp_anwg, 4),
            "oracle_anwg": round(oracle_anwg, 4),
        },
        "rf_anwg_fresh": {
            "mean_anwg": round(rf_anwg_val, 4),
            "gap_vs_scorpio": round(rf_anwg_val - always_scorpio_anwg, 4) if not np.isnan(rf_anwg_val) else None,
            "ci95_vs_scorpio": rf_ci,
            "ci_excludes_zero": rf_ci_excludes_zero,
        },
        "best_selector_fresh": {
            "selector": best_sk_anwg,
            "mean_anwg": round(best_anwg, 4),
        },
        "answers": {
            "rf_anwg_beats_scorpio_fresh": rf_beats_scorpio,
            "rf_anwg_ci_excludes_zero": rf_ci_excludes_zero,
            "any_selector_beats_wsp_fresh": any_beats_wsp,
            "wsp_beats_scorpio_under_anwg": wsp_beats_scorpio_under_anwg,
            "scorpio_best_under_anwg": always_scorpio_anwg >= always_wsp_anwg,
        },
        "constrained_objectives": constrained_results,
        "n_failure_cases": len(failure_cases),
        "selector_stats_summary": {
            sk: {
                "mean_anwg": stats.get(sk, {}).get("mean_anwg", float("nan")),
                "gap_vs_scorpio": stats.get(sk, {}).get("mean_gap_vs_scorpio", float("nan")),
                "ci95": stats.get(sk, {}).get("ci95_vs_scorpio"),
                "ci_includes_zero": stats.get(sk, {}).get("ci_includes_zero_vs_scorpio"),
            }
            for sk in sel_keys
            if sk in stats
        },
    }
    _write_json(out_dir / "fresh_overall_summary.json", summary)

    elapsed = time.time() - t0
    log.info("Phase 2B.16 complete in %.1fs — output: %s", elapsed, out_dir)

    print(f"\n=== Phase 2B.16 Fresh Validation Complete ({elapsed:.1f}s) ===")
    print(f"Fresh windows: {n_fresh}")
    print(f"Label changes (cond→anwg): {n_changes}/{n_fresh} ({100*n_changes/n_fresh:.1f}%)")
    print(f"Near-tie (eps=0.005): {nt_stats_all.get('fraction_near_tie_eps0.005', 0):.3f}")
    print(f"Meaningful (eps=0.005): {nt_stats_all.get('n_meaningful_eps0.005', 0)}")
    print(f"FIFO wins: {fifo_audit['n_fifo_wins_anwg']} ({fifo_audit['fifo_win_fraction']:.3f}), near-tie: {fifo_audit['n_fifo_near_tie_eps005']}")
    print(f"Baselines: always-SCORPIO={always_scorpio_anwg:.4f} | always-WSP={always_wsp_anwg:.4f} | oracle={oracle_anwg:.4f}")
    print(f"rf_anwg: {rf_anwg_val:.4f} gap_vs_scorpio={rf_anwg_val-always_scorpio_anwg:+.4f} CI={rf_ci} CI_excludes0={rf_ci_excludes_zero}")
    print(f"Best selector: {best_sk_anwg} ({best_anwg:.4f})")
    print(f"Failure cases: {len(failure_cases)}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()

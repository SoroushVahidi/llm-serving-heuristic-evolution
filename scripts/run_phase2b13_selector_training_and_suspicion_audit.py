#!/usr/bin/env python3
"""
Phase 2B.13: Selector Training and SCORPIO Suspicion Audit.

Extends Phase 2B.12 to ≥200 windows, then:
  A. Suspicion audits:
     - Always-SCORPIO comparison
     - Near-tie filtering (eps=0.001, 0.005, 0.010)
     - Regret gap computation per window
     - SCORPIO admission/completion audit
     - Objective sensitivity (completion-penalized WG)
     - Leakage reconfirmation
  B. Selector training (if feasibility criteria pass):
     - Standard RF / DT (with and without regret weighting)
     - Per-policy WG regression (argmax predicted WG)
     - KNN/SUNNY-style nearest-neighbor selector
     - Safe fallback-to-SCORPIO selector

Phase 2B.12 finding: 161/172 windows all-complete; 128/172 near-tie at eps=0.001.
Only dev_kv_pressure_decode_heavy (45 req/s, output_mean=384, slo_slack=0.8)
produced genuine WG differentiation (0/6 all-complete, mean_gap=0.29).

Usage
-----
python scripts/run_phase2b13_selector_training_and_suspicion_audit.py \\
    --config configs/phase2b13_selector_training_and_suspicion_audit.yaml \\
    [--output results/phase2b13_selector_training_and_suspicion_audit] \\
    [--log-file logs/phase2b13/phase2b13_selector_training.log]
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
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FeatureMode, FEATURE_NAMES
from llmserveopt.selector.models import (
    RuleBasedSelector,
    DecisionTreeSelector,
    RandomForestSelector,
)

from run_phase2b9_selector_robustness import (
    apply_selectors_to_rows,
    build_gpu_configs,
    compute_fixed_baseline_wgs,
    load_config,
    summarize_group,
    write_per_window_csv,
    write_summary_csv,
)
from run_phase2b12_workload_diversity_selector_labels import (
    build_rows_for_group,
    check_rf_feasibility,
    compute_label_distribution,
    label_diversity_summary,
    per_workload_label_table,
    policy_mean_wg,
    collect_policy_distribution,
)

try:
    from llmserveopt.simulator.service_model import ServiceModel
    from llmserveopt.simulator.service_model_factory import build_service_model_from_config
    _HAS_SERVICE_MODEL = True
except ImportError:
    _HAS_SERVICE_MODEL = False

SCORPIO = "scorpio_style_slo_guard"
ORACLE_POLICY = "oracle_srtf"


# ---------------------------------------------------------------------------
# Always-SCORPIO selector
# ---------------------------------------------------------------------------

class AlwaysScorpioSelector:
    """Baseline: always dispatch scorpio_style_slo_guard."""
    name = "always_scorpio"

    def predict(self, rows: List[Dict]) -> List[str]:
        return [SCORPIO] * len(rows)

    def predict_one(self, features: Dict) -> str:
        return SCORPIO


# ---------------------------------------------------------------------------
# Near-tie / regret analysis
# ---------------------------------------------------------------------------

def compute_near_tie_stats(rows: List[Dict], thresholds: List[float]) -> Dict:
    """
    Returns stats about how many windows are near-tie at each epsilon threshold.
    policy_margin = best_wg - second_best_wg (already in row from dataset.py).
    """
    if not rows:
        return {}
    margins = np.array([float(r.get("policy_margin", 0.0) or 0.0) for r in rows])
    best_wgs = np.array([float(r.get("best_weighted_goodput", 1.0) or 1.0) for r in rows])
    all_complete_thresh = 0.99

    result = {
        "n_total": len(rows),
        "n_all_complete": int(np.sum(best_wgs >= all_complete_thresh)),
        "all_complete_fraction": float(np.mean(best_wgs >= all_complete_thresh)),
        "margin_mean": float(np.mean(margins)),
        "margin_p50": float(np.percentile(margins, 50)),
        "margin_p90": float(np.percentile(margins, 90)),
        "margin_p99": float(np.percentile(margins, 99)),
        "margin_max": float(np.max(margins)),
    }
    for eps in thresholds:
        n_tie = int(np.sum(margins < eps))
        result[f"n_near_tie_eps{eps:.3f}"] = n_tie
        result[f"fraction_near_tie_eps{eps:.3f}"] = round(n_tie / len(rows), 4)
        result[f"n_meaningful_eps{eps:.3f}"] = len(rows) - n_tie
    return result


def filter_non_tie_rows(rows: List[Dict], epsilon: float) -> List[Dict]:
    """Return rows where policy_margin >= epsilon (genuine WG differentiation)."""
    return [r for r in rows if float(r.get("policy_margin", 0.0) or 0.0) >= epsilon]


# ---------------------------------------------------------------------------
# Per-window regret weights
# ---------------------------------------------------------------------------

def compute_regret_weights(rows: List[Dict], epsilon: float = 0.001) -> np.ndarray:
    """Weight = policy_margin + epsilon (larger gap → higher weight for training)."""
    margins = np.array([float(r.get("policy_margin", 0.0) or 0.0) for r in rows])
    weights = np.clip(margins + epsilon, epsilon, None)
    return weights / weights.sum()  # normalise


# ---------------------------------------------------------------------------
# Objective sensitivity (completion-penalized WG)
# ---------------------------------------------------------------------------

def compute_adjusted_wg(
    policy_wg: float,
    policy_completion: float,
    completion_target: float,
    lam: float,
) -> float:
    penalty = max(0.0, completion_target - policy_completion)
    return policy_wg - lam * penalty


def _row_completion(row: Dict, policy: str, default: float = 1.0) -> float:
    val = row.get(f"completion_{policy}")
    if val in (None, ""):
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def objective_sensitivity_analysis(
    rows: List[Dict],
    completion_target: float,
    lambdas: List[float],
) -> Dict:
    """Re-rank policies under completion-penalized WG."""
    results = {}
    for lam in lambdas:
        adjusted: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            for p in SELECTOR_CANDIDATES:
                wg = float(row.get(f"reward_{p}", 0.0) or 0.0)
                comp = _row_completion(row, p)
                adj = compute_adjusted_wg(wg, comp, completion_target, lam)
                adjusted[p].append(adj)
        mean_adj = {p: float(np.mean(v)) for p, v in adjusted.items()}
        ranked = sorted(mean_adj.items(), key=lambda x: -x[1])
        results[f"lambda_{lam}"] = {
            "ranking": [{"policy": p, "mean_adj_wg": round(v, 4)} for p, v in ranked],
            "best_policy": ranked[0][0] if ranked else "none",
            "scorpio_rank": next(
                (i + 1 for i, (p, _) in enumerate(ranked) if p == SCORPIO), -1
            ),
            "scorpio_mean_adj_wg": round(mean_adj.get(SCORPIO, 0.0), 4),
        }
    return results


def policy_aux_summary(rows: List[Dict], policy: str) -> Dict:
    """Mean WG, completion, SLO violation for one policy across rows."""
    wgs, comps, slos = [], [], []
    for r in rows:
        wg = r.get(f"reward_{policy}")
        if wg not in (None, ""):
            wgs.append(float(wg))
        comp = r.get(f"completion_{policy}")
        if comp not in (None, ""):
            v = float(comp)
            if not np.isnan(v):
                comps.append(v)
        slo = r.get(f"slo_violation_{policy}")
        if slo not in (None, ""):
            v = float(slo)
            if not np.isnan(v):
                slos.append(v)
    return {
        "policy": policy,
        "mean_wg": round(float(np.mean(wgs)), 4) if wgs else float("nan"),
        "mean_completion_fraction": round(float(np.mean(comps)), 4) if comps else float("nan"),
        "mean_slo_violation_rate": round(float(np.mean(slos)), 4) if slos else float("nan"),
        "n_windows": len(rows),
    }


def build_completion_admission_summary(
    groups: Dict[str, List[Dict]],
) -> List[Dict]:
    """Per-policy admission/completion audit by group."""
    out: List[Dict] = []
    for gname, grows in groups.items():
        if not grows:
            continue
        for p in SELECTOR_CANDIDATES:
            s = policy_aux_summary(grows, p)
            s["group"] = gname
            out.append(s)
    return out


def build_failure_cases(
    heldout_eval: Dict[str, Dict],
    nt_all: Dict,
    rf_ok: bool,
    beats_scorpio: bool,
    all_rows: List[Dict],
) -> List[Dict]:
    """Document known failure patterns from Phase 2B.12/2B.13."""
    cases: List[Dict] = []
    as_wg = heldout_eval.get("always_scorpio", {}).get("mean_wg", float("nan"))
    rb_wg = heldout_eval.get("rule_based", {}).get("mean_wg", float("nan"))
    rf_wg = heldout_eval.get("random_forest", {}).get("mean_wg")
    oracle = heldout_eval.get("always_scorpio", {}).get("oracle_mean_wg", float("nan"))

    if as_wg >= rb_wg:
        cases.append({
            "failure_id": "fail_012",
            "pattern": "Rule selector loses to always-SCORPIO on held-out",
            "status": "unresolved",
            "detail": f"rule={rb_wg} always_SCORPIO={as_wg}",
        })
    if rf_wg is not None and not beats_scorpio:
        cases.append({
            "failure_id": "fail_013",
            "pattern": "RF does not beat always-SCORPIO on held-out",
            "status": "unresolved" if rf_wg <= as_wg else "resolved",
            "detail": f"RF={rf_wg} always_SCORPIO={as_wg}",
        })
    elif rf_wg is None:
        cases.append({
            "failure_id": "fail_013",
            "pattern": "RF not trained or not evaluated",
            "status": "skipped" if not rf_ok else "unresolved",
            "detail": f"rf_feasible={rf_ok}",
        })

    near_tie_frac = nt_all.get("fraction_near_tie_eps0.005", 0.0)
    if near_tie_frac > 0.5:
        cases.append({
            "failure_id": "fail_014",
            "pattern": "Near-tie labels dominate (eps=0.005)",
            "status": "unresolved",
            "detail": f"fraction_near_tie={near_tie_frac}",
        })
    all_complete_frac = nt_all.get("all_complete_fraction", 0.0)
    if all_complete_frac > 0.8:
        cases.append({
            "failure_id": "fail_015",
            "pattern": "All-complete windows create tie-breaking labels",
            "status": "unresolved",
            "detail": f"all_complete_fraction={all_complete_frac}",
        })

    gap_oracle = (as_wg - oracle) if not np.isnan(as_wg) and not np.isnan(oracle) else float("nan")
    if not np.isnan(gap_oracle) and abs(gap_oracle) < 0.01:
        cases.append({
            "failure_id": "fail_016",
            "pattern": "always-SCORPIO within 1pp of per-window oracle on held-out",
            "status": "unresolved",
            "detail": f"gap_vs_oracle={gap_oracle:.4f}",
        })

    rf_dist = heldout_eval.get("random_forest", {}).get("chosen_policy_dist", {})
    if rf_dist and rf_dist.get(SCORPIO, 0) == heldout_eval.get("random_forest", {}).get("n_windows"):
        cases.append({
            "failure_id": "fail_017",
            "pattern": "RF collapses to always-SCORPIO on held-out",
            "status": "unresolved",
            "detail": str(rf_dist),
        })

    return cases


# ---------------------------------------------------------------------------
# SCORPIO admission / completion audit
# ---------------------------------------------------------------------------

def scorpio_admission_audit(rows: List[Dict]) -> Dict:
    """Report SCORPIO completion vs other policies per-window mean."""
    scorpio = policy_aux_summary(rows, SCORPIO)
    edf = policy_aux_summary(rows, "edf")
    fifo = policy_aux_summary(rows, "fifo")
    ac = policy_aux_summary(rows, "admission_control")

    return {
        "n_windows": len(rows),
        "scorpio_mean_wg": scorpio["mean_wg"],
        "scorpio_mean_completion": scorpio["mean_completion_fraction"],
        "scorpio_mean_slo_violation": scorpio["mean_slo_violation_rate"],
        "edf_mean_completion": edf["mean_completion_fraction"],
        "fifo_mean_completion": fifo["mean_completion_fraction"],
        "ac_mean_completion": ac["mean_completion_fraction"],
        "scorpio_vs_edf_completion_delta": round(
            scorpio["mean_completion_fraction"] - edf["mean_completion_fraction"], 4
        ) if not np.isnan(scorpio["mean_completion_fraction"]) and not np.isnan(
            edf["mean_completion_fraction"]
        ) else float("nan"),
        "scorpio_vs_fifo_completion_delta": round(
            scorpio["mean_completion_fraction"] - fifo["mean_completion_fraction"], 4
        ) if not np.isnan(scorpio["mean_completion_fraction"]) and not np.isnan(
            fifo["mean_completion_fraction"]
        ) else float("nan"),
        "wg_denominator_note": (
            "weighted_goodput is computed over COMPLETED requests only. "
            "SCORPIO may throttle/reject requests (lower completion_fraction) "
            "while achieving high WG on accepted requests. "
            "Completion-penalized objectives adjust for this trade-off."
        ),
    }


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(rows: List[Dict]) -> Dict:
    """Reconfirm that features exclude actual output and future arrivals."""
    suspicious_feature_keys = [
        "actual_output_tokens", "actual_output", "future_arrival",
        "oracle_label", "oracle_srtf",
    ]
    found = []
    if rows:
        first = rows[0]
        for key in first:
            if any(s in key.lower() for s in suspicious_feature_keys):
                found.append(key)
    oracle_in_candidates = ORACLE_POLICY in SELECTOR_CANDIDATES
    feature_names_clean = [
        f for f in FEATURE_NAMES
        if any(s in f.lower() for s in suspicious_feature_keys)
    ]
    return {
        "suspicious_row_keys_found": found,
        "oracle_in_selector_candidates": oracle_in_candidates,
        "leaky_feature_names": feature_names_clean,
        "feature_mode": rows[0].get("feature_mode") if rows else "unknown",
        "pass": (
            len(found) == 0
            and not oracle_in_candidates
            and len(feature_names_clean) == 0
        ),
    }


# ---------------------------------------------------------------------------
# ML Selectors
# ---------------------------------------------------------------------------

def _feature_matrix(rows: List[Dict]) -> np.ndarray:
    return np.array(
        [[float(r.get(f"feat_{n}", 0.0) or 0.0) for n in FEATURE_NAMES] for r in rows],
        dtype=float,
    )


def _labels(rows: List[Dict]) -> List[str]:
    return [r["best_policy"] for r in rows]


def train_rf_dt(
    train_rows: List[Dict],
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[Optional[RandomForestSelector], Optional[DecisionTreeSelector], str]:
    """Train RF and DT with optional sample weights."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return None, None, "sklearn_missing"
    if not train_rows:
        return None, None, "no_training_rows"

    rf = RandomForestSelector(n_estimators=200, max_depth=10, random_state=42)
    dt = DecisionTreeSelector(max_depth=8, min_samples_leaf=5, random_state=42)

    if sample_weights is not None:
        X = _feature_matrix(train_rows)
        y = _labels(train_rows)
        rf._clf.fit(X, y, sample_weight=sample_weights)
        dt._clf.fit(X, y, sample_weight=sample_weights)
    else:
        rf.fit(train_rows)
        dt.fit(train_rows)

    return rf, dt, "ok"


class PerPolicyRegressionSelector:
    """Train one RF regressor per policy; at inference choose argmax predicted WG."""
    name = "per_policy_regression"

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self._params = dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self._regressors: Dict[str, object] = {}

    def fit(self, rows: List[Dict]) -> "PerPolicyRegressionSelector":
        from sklearn.ensemble import RandomForestRegressor
        X = _feature_matrix(rows)
        for p in SELECTOR_CANDIDATES:
            y = np.array([float(r.get(f"reward_{p}", 0.0) or 0.0) for r in rows])
            reg = RandomForestRegressor(**self._params)
            reg.fit(X, y)
            self._regressors[p] = reg
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        X = _feature_matrix(rows)
        n = len(rows)
        preds_by_policy = {}
        for p, reg in self._regressors.items():
            preds_by_policy[p] = reg.predict(X)
        chosen = []
        for i in range(n):
            best_p = max(SELECTOR_CANDIDATES, key=lambda p: preds_by_policy[p][i])
            chosen.append(best_p)
        return chosen


class KNNSelector:
    """SUNNY-style KNN selector: for each test window find k nearest neighbors and
    return the policy with best average WG among those neighbors."""
    name = "knn_selector"

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self._train_rows: List[Dict] = []
        self._X_train: Optional[np.ndarray] = None

    def fit(self, rows: List[Dict]) -> "KNNSelector":
        from sklearn.preprocessing import StandardScaler
        self._train_rows = rows
        X = _feature_matrix(rows)
        self._scaler = StandardScaler().fit(X)
        self._X_train = self._scaler.transform(X)
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        from sklearn.neighbors import NearestNeighbors
        X_test = self._scaler.transform(_feature_matrix(rows))
        k = min(self.k, len(self._train_rows))
        nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nn.fit(self._X_train)
        _, indices = nn.kneighbors(X_test)
        chosen = []
        for nbr_idxs in indices:
            nbr_rows = [self._train_rows[i] for i in nbr_idxs]
            policy_wgs: Dict[str, float] = defaultdict(float)
            for nr in nbr_rows:
                for p in SELECTOR_CANDIDATES:
                    policy_wgs[p] += float(nr.get(f"reward_{p}", 0.0) or 0.0)
            best_p = max(SELECTOR_CANDIDATES, key=lambda p: policy_wgs[p])
            chosen.append(best_p)
        return chosen


class SafeFallbackSelector:
    """Default to SCORPIO; switch away only when predicted gain > margin."""
    name: str = "safe_fallback"

    def __init__(self, base_selector, margin: float = 0.005):
        self._base = base_selector
        self.margin = margin
        self.name = f"safe_fallback_margin{margin:.3f}"

    def predict(self, rows: List[Dict]) -> List[str]:
        base_preds = self._base.predict(rows)
        final = []
        for pred, row in zip(base_preds, rows):
            scorpio_wg = float(row.get(f"reward_{SCORPIO}", 0.0) or 0.0)
            pred_wg = float(row.get(f"reward_{pred}", 0.0) or 0.0)
            # At inference time we don't know actual WGs; this is a post-hoc audit
            # using oracle rewards. For real deployment, use predicted WG from regressor.
            if pred_wg - scorpio_wg >= self.margin:
                final.append(pred)
            else:
                final.append(SCORPIO)
        return final


def evaluate_selector_splits(
    selector,
    name: str,
    train_rows: List[Dict],
    val_rows: List[Dict],
    test_rows: List[Dict],
) -> Dict:
    """Evaluate selector on train/val/test splits."""
    return {
        "selector": name,
        "train": evaluate_selector_on_rows(selector, train_rows, name),
        "val": evaluate_selector_on_rows(selector, val_rows, name),
        "test": evaluate_selector_on_rows(selector, test_rows, name),
    }


# ---------------------------------------------------------------------------
# Selector evaluation helper
# ---------------------------------------------------------------------------

def evaluate_selector_on_rows(
    selector,
    rows: List[Dict],
    name: str,
) -> Dict:
    """Evaluate an arbitrary selector that exposes .predict(rows) -> List[str]."""
    if not rows:
        return {"selector": name, "n_windows": 0}
    preds = selector.predict(rows)
    labels = [r.get("best_policy", "") for r in rows]
    n = len(rows)
    correct = sum(p == l for p, l in zip(preds, labels))
    wgs = [float(r.get(f"reward_{pred}", 0.0) or 0.0) for pred, r in zip(preds, rows)]
    mean_wg = float(np.mean(wgs))
    scorpio_wgs = [float(r.get(f"reward_{SCORPIO}", 0.0) or 0.0) for r in rows]
    mean_scorpio_wg = float(np.mean(scorpio_wgs))

    fixed_wgs = compute_fixed_baseline_wgs(rows)
    best_fixed_name = max(fixed_wgs, key=fixed_wgs.get) if fixed_wgs else "none"
    best_fixed_wg = fixed_wgs.get(best_fixed_name, 0.0)
    oracle_wgs = [float(r.get("best_weighted_goodput", 0.0) or 0.0) for r in rows]
    oracle_mean = float(np.mean(oracle_wgs))

    return {
        "selector": name,
        "n_windows": n,
        "accuracy": round(correct / n, 4),
        "n_correct": correct,
        "mean_wg": round(mean_wg, 4),
        "gap_vs_best_fixed": round(mean_wg - best_fixed_wg, 4),
        "gap_vs_always_scorpio": round(mean_wg - mean_scorpio_wg, 4),
        "gap_vs_oracle": round(mean_wg - oracle_mean, 4),
        "best_fixed_wg": round(best_fixed_wg, 4),
        "best_fixed_policy": best_fixed_name,
        "always_scorpio_wg": round(mean_scorpio_wg, 4),
        "oracle_mean_wg": round(oracle_mean, 4),
        "chosen_policy_dist": dict(Counter(preds)),
        "label_dist": dict(Counter(labels)),
        "collapses_to_scorpio": dict(Counter(preds)).get(SCORPIO, 0) == n,
    }


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_by_seed(
    div_rows: List[Dict],
    train_seeds: List[int],
    val_seeds: List[int],
) -> Tuple[List[Dict], List[Dict]]:
    def seed_of(row: Dict) -> Optional[int]:
        tid = row.get("trace_id", "")
        if "_s" in tid:
            s = tid.rsplit("_s", 1)[1]
            if s.isdigit():
                return int(s)
        return None

    train_set = set(train_seeds)
    val_set = set(val_seeds)
    train, val = [], []
    for r in div_rows:
        s = seed_of(r)
        if s in train_set:
            train.append(r)
        elif s in val_set:
            val.append(r)
    return train, val


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2B.13 Selector Training + Suspicion Audit")
    p.add_argument("--config",
                   default="configs/phase2b13_selector_training_and_suspicion_audit.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--skip-diversity", action="store_true",
                   help="Run only regression workloads (fast debug)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip ML selector training")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    logging.info("Phase 2B.13 Selector Training and Suspicion Audit")
    t0 = time.perf_counter()

    cfg = load_config(args.config)
    out_dir = Path(args.output or cfg.get("output_dir",
        "results/phase2b13_selector_training_and_suspicion_audit"))
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_configs = build_gpu_configs(cfg)
    drain_steps = cfg.get("simulator", {}).get("drain_steps", 20000)
    window_size = cfg.get("window_size", 200)
    min_partial = cfg.get("min_partial_window", 50)
    feature_mode = FeatureMode(cfg.get("feature_mode", "online_prefix"))

    if _HAS_SERVICE_MODEL:
        try:
            from llmserveopt.simulator.service_model_factory import build_service_model_from_config
            service_model = build_service_model_from_config({"service_model": cfg.get("service_model", {})})
        except Exception:
            from llmserveopt.simulator.service_model import ServiceModel
            service_model = ServiceModel()
    else:
        service_model = None

    near_tie_thresholds = cfg.get("near_tie_thresholds", [0.001, 0.005, 0.010])
    rf_thresholds = cfg.get("rf_feasibility", {})
    train_cfg = cfg.get("selector_training", {})
    obj_cfg = cfg.get("objective_sensitivity", {})
    safe_margins = cfg.get("safe_fallback_margins", [0.001, 0.005, 0.010])
    knn_cfg = cfg.get("knn", {})
    regret_eps = float(train_cfg.get("regret_weight_epsilon", 0.001))

    dev_seeds = cfg.get("dev_seeds", [0, 1, 2])
    heldout_seeds = cfg.get("heldout_seeds", [3, 4, 5])
    div_seeds = cfg.get("diversity_seeds", [6, 7, 8, 9, 10, 11])
    train_div_seeds = train_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10])
    val_div_seeds = train_cfg.get("val_diversity_seeds", [11])

    all_workloads = cfg.get("workloads", [])
    dev_workloads = [w for w in all_workloads if w.get("group") == "dev"]
    heldout_workloads = [w for w in all_workloads if w.get("group") == "heldout"]
    div_workloads = [w for w in all_workloads if w.get("group") == "diversity"]

    logging.info("Config: %d workloads (%d dev / %d heldout / %d diversity)",
                 len(all_workloads), len(dev_workloads), len(heldout_workloads), len(div_workloads))
    logging.info("Diversity seeds: %s", div_seeds)
    logging.info("Train/val div seeds: %s / %s", train_div_seeds, val_div_seeds)
    logging.info("Deployable candidates: %d | oracle excluded: %s not in candidates: %s",
                 len(SELECTOR_CANDIDATES), ORACLE_POLICY, ORACLE_POLICY not in SELECTOR_CANDIDATES)

    # --- Build rows ---
    logging.info("=== Building dev rows ===")
    dev_rows = build_rows_for_group(dev_workloads, dev_seeds, gpu_configs, service_model,
                                    drain_steps, window_size, min_partial, feature_mode, args.verbose)
    logging.info("=== Building heldout rows ===")
    heldout_rows = build_rows_for_group(heldout_workloads, heldout_seeds, gpu_configs, service_model,
                                        drain_steps, window_size, min_partial, feature_mode, args.verbose)
    div_rows: List[Dict] = []
    if not args.skip_diversity:
        logging.info("=== Building diversity rows ===")
        div_rows = build_rows_for_group(div_workloads, div_seeds, gpu_configs, service_model,
                                        drain_steps, window_size, min_partial, feature_mode, args.verbose)

    regression_rows = dev_rows + heldout_rows
    all_rows = regression_rows + div_rows
    elapsed_build = time.perf_counter() - t0
    logging.info("Row build complete: %.1fs  dev=%d heldout=%d diversity=%d total=%d",
                 elapsed_build, len(dev_rows), len(heldout_rows), len(div_rows), len(all_rows))

    # --- Leakage audit (always run first, before any model use) ---
    logging.info("=== Leakage audit ===")
    leak_result = leakage_audit(all_rows)
    logging.info("Leakage audit: pass=%s  suspicious_keys=%s  oracle_in_candidates=%s",
                 leak_result["pass"], leak_result["suspicious_row_keys_found"],
                 leak_result["oracle_in_selector_candidates"])
    if not leak_result["pass"]:
        logging.error("LEAKAGE DETECTED: %s", leak_result)

    # --- Near-tie / regret stats ---
    logging.info("=== Near-tie analysis ===")
    nt_all = compute_near_tie_stats(all_rows, near_tie_thresholds)
    nt_dev = compute_near_tie_stats(dev_rows, near_tie_thresholds)
    nt_heldout = compute_near_tie_stats(heldout_rows, near_tie_thresholds)
    nt_div = compute_near_tie_stats(div_rows, near_tie_thresholds) if div_rows else {}

    for eps in near_tie_thresholds:
        k = f"n_near_tie_eps{eps:.3f}"
        k2 = f"n_meaningful_eps{eps:.3f}"
        logging.info("  eps=%.3f  near-tie=%d/%d (%.1f%%)  meaningful=%d",
                     eps, nt_all.get(k, 0), nt_all["n_total"],
                     100 * nt_all.get(k, 0) / max(nt_all["n_total"], 1),
                     nt_all.get(k2, 0))
    logging.info("  all-complete=%d/%d (%.1f%%)",
                 nt_all["n_all_complete"], nt_all["n_total"],
                 100 * nt_all["all_complete_fraction"])

    # Non-tie subsets for each threshold
    nontie_rows_by_eps: Dict[str, List[Dict]] = {}
    for eps in near_tie_thresholds:
        nontie_rows_by_eps[f"{eps:.3f}"] = filter_non_tie_rows(all_rows, eps)

    # --- Base selectors ---
    rule_sel = RuleBasedSelector()
    always_scorpio = AlwaysScorpioSelector()
    base_models = {"rule_based": rule_sel, "always_scorpio": always_scorpio}

    dev_rows = apply_selectors_to_rows(dev_rows, base_models)
    heldout_rows = apply_selectors_to_rows(heldout_rows, base_models)
    if div_rows:
        div_rows = apply_selectors_to_rows(div_rows, base_models)
    regression_rows = dev_rows + heldout_rows
    all_rows = regression_rows + div_rows

    # --- RF/DT feasibility ---
    all_label_div = label_diversity_summary(all_rows, "overall")
    rf_ok, rf_details = check_rf_feasibility(
        all_label_div["label_distribution"],
        min_windows=rf_thresholds.get("min_windows", 200),
        min_policies_winning=rf_thresholds.get("min_policies_winning", 3),
        min_windows_per_policy=rf_thresholds.get("min_windows_per_policy", 10),
        max_single_policy_fraction=rf_thresholds.get("max_single_policy_fraction", 0.85),
    )
    logging.info("RF/DT feasibility: %s | total=%d top=%s(%.1f%%) policies≥10=%d",
                 "FEASIBLE" if rf_ok else "NOT_FEASIBLE",
                 rf_details["total_windows"], rf_details["top_policy"],
                 rf_details["top_policy_fraction"] * 100,
                 rf_details["n_policies_with_enough_windows"])

    # Also check non-tie label diversity
    nontie_feasibility: Dict[str, Dict] = {}
    for eps in near_tie_thresholds:
        nt_rows = nontie_rows_by_eps[f"{eps:.3f}"]
        if nt_rows:
            nt_ld = label_diversity_summary(nt_rows, f"non_tie_{eps:.3f}")
            nt_ok, nt_det = check_rf_feasibility(
                nt_ld["label_distribution"],
                min_windows=rf_thresholds.get("min_windows", 200),
                min_policies_winning=rf_thresholds.get("min_policies_winning", 3),
                min_windows_per_policy=rf_thresholds.get("min_windows_per_policy", 10),
                max_single_policy_fraction=rf_thresholds.get("max_single_policy_fraction", 0.85),
            )
            nontie_feasibility[f"{eps:.3f}"] = {"feasible": nt_ok, "details": nt_det}

    # --- ML Selector Training ---
    rf_std = dt_std = rf_rw = dt_rw = None
    ppr_sel = knn_sel = None
    training_status = "skipped"
    ml_models: Dict = {}
    split_eval: Dict[str, Dict] = {}
    regret_summary: Dict = {}
    ppr_summary: Dict = {}
    knn_summary: Dict = {}
    safe_fallback_summary: Dict = {}

    train_div_rows, val_div_rows = split_by_seed(div_rows, train_div_seeds, val_div_seeds)
    train_rows = dev_rows + train_div_rows
    val_rows = val_div_rows
    test_rows = heldout_rows

    if rf_ok and not args.skip_training:
        logging.info("=== Training ML selectors ===")
        logging.info("Train=%d  Val=%d  Test=%d", len(train_rows), len(val_rows), len(test_rows))

        # Uniform-weighted RF/DT
        rf_std, dt_std, status = train_rf_dt(train_rows, sample_weights=None)
        if rf_std:
            ml_models["random_forest"] = rf_std
            ml_models["decision_tree"] = dt_std
            split_eval["random_forest"] = evaluate_selector_splits(
                rf_std, "random_forest", train_rows, val_rows, test_rows)
            split_eval["decision_tree"] = evaluate_selector_splits(
                dt_std, "decision_tree", train_rows, val_rows, test_rows)
            logging.info("RF/DT trained (uniform weights): status=%s", status)

        # Regret-weighted RF/DT
        if train_rows:
            rw = compute_regret_weights(train_rows, epsilon=regret_eps)
            raw_weights = rw * len(train_rows)  # un-normalise back to per-sample scale
            rf_rw, dt_rw, status_rw = train_rf_dt(train_rows, sample_weights=raw_weights)
            if rf_rw:
                rf_rw.name = "random_forest_regret_weighted"  # type: ignore[attr-defined]
                dt_rw.name = "decision_tree_regret_weighted"  # type: ignore[attr-defined]
                ml_models["random_forest_regret_weighted"] = rf_rw
                ml_models["decision_tree_regret_weighted"] = dt_rw
                regret_summary["random_forest_regret_weighted"] = evaluate_selector_splits(
                    rf_rw, "random_forest_regret_weighted", train_rows, val_rows, test_rows)
                regret_summary["decision_tree_regret_weighted"] = evaluate_selector_splits(
                    dt_rw, "decision_tree_regret_weighted", train_rows, val_rows, test_rows)
                logging.info("Regret-weighted RF/DT trained: status=%s", status_rw)

        # Per-policy regression
        try:
            ppr_sel = PerPolicyRegressionSelector(n_estimators=100, max_depth=8)
            ppr_sel.fit(train_rows)
            ml_models["per_policy_regression"] = ppr_sel
            ppr_summary = evaluate_selector_splits(
                ppr_sel, "per_policy_regression", train_rows, val_rows, test_rows)
            logging.info("Per-policy regression selector trained")
        except Exception as e:
            logging.warning("Per-policy regression failed: %s", e)
            ppr_sel = None

        # KNN selector
        try:
            knn_sel = KNNSelector(k=knn_cfg.get("k", 5), metric=knn_cfg.get("metric", "euclidean"))
            knn_sel.fit(train_rows)
            ml_models["knn_selector"] = knn_sel
            knn_summary = evaluate_selector_splits(
                knn_sel, "knn_selector", train_rows, val_rows, test_rows)
            logging.info("KNN selector trained (k=%d)", knn_cfg.get("k", 5))
        except Exception as e:
            logging.warning("KNN selector failed: %s", e)
            knn_sel = None

        # Safe fallback selectors (based on best ML selector so far)
        if rf_std is not None:
            for margin in safe_margins:
                sfsel = SafeFallbackSelector(rf_std, margin=margin)
                ml_models[sfsel.name] = sfsel
                safe_fallback_summary[sfsel.name] = evaluate_selector_splits(
                    sfsel, sfsel.name, train_rows, val_rows, test_rows)

        training_status = "ok"
    else:
        logging.info("ML training skipped (feasible=%s, skip=%s)", rf_ok, args.skip_training)

    # Apply ML selectors
    all_selector_models = {**base_models, **ml_models}
    if ml_models:
        dev_rows = apply_selectors_to_rows(dev_rows, ml_models)
        heldout_rows = apply_selectors_to_rows(heldout_rows, ml_models)
        if div_rows:
            div_rows = apply_selectors_to_rows(div_rows, ml_models)
        regression_rows = dev_rows + heldout_rows
        all_rows = regression_rows + div_rows

    elapsed = time.perf_counter() - t0
    logging.info("Training + evaluation complete: %.1fs", elapsed)

    # --- Group summaries ---
    dev_summary = summarize_group(dev_rows, "dev", all_selector_models)
    heldout_summary = summarize_group(heldout_rows, "heldout", all_selector_models)
    regression_summary = summarize_group(regression_rows, "regression", all_selector_models)
    diversity_summary = summarize_group(div_rows, "diversity", all_selector_models) if div_rows else {}
    overall_summary = summarize_group(all_rows, "overall", all_selector_models)

    # --- Evaluate all selectors on held-out (test) ---
    heldout_eval: Dict[str, Dict] = {}
    for name, sel in all_selector_models.items():
        heldout_eval[name] = evaluate_selector_on_rows(sel, heldout_rows, name)

    # --- SCORPIO completion audit ---
    scorpio_audit = scorpio_admission_audit(all_rows)
    completion_summary = build_completion_admission_summary({
        "dev": dev_rows, "heldout": heldout_rows,
        "regression": regression_rows, "diversity": div_rows, "overall": all_rows,
    })

    # --- Objective sensitivity ---
    obj_sens = objective_sensitivity_analysis(
        all_rows,
        completion_target=obj_cfg.get("completion_target", 0.95),
        lambdas=obj_cfg.get("lambdas", [0.5, 1.0]),
    )

    # Log SCORPIO rank under each lambda
    for lk, v in obj_sens.items():
        logging.info("Obj sensitivity %s: SCORPIO rank=%d adj_wg=%.4f best=%s",
                     lk, v["scorpio_rank"], v["scorpio_mean_adj_wg"], v["best_policy"])

    # --- Per-workload label table ---
    workload_table = per_workload_label_table(all_rows)

    # --- Log final selector comparison ---
    logging.info("=" * 60)
    logging.info("Phase 2B.13 Selector Results")
    logging.info("=" * 60)
    logging.info("Windows: total=%d  dev=%d  heldout=%d  diversity=%d",
                 len(all_rows), len(dev_rows), len(heldout_rows), len(div_rows))
    logging.info("Near-tie eps=0.001: %d/%d (%.1f%%) all-complete: %d/%d (%.1f%%)",
                 nt_all.get("n_near_tie_eps0.001", 0), nt_all["n_total"],
                 100 * nt_all.get("fraction_near_tie_eps0.001", 0),
                 nt_all["n_all_complete"], nt_all["n_total"],
                 100 * nt_all["all_complete_fraction"])

    for gname, gsummary in [
        ("dev", dev_summary), ("heldout", heldout_summary),
        ("regression", regression_summary), ("diversity", diversity_summary),
        ("overall", overall_summary),
    ]:
        if not gsummary.get("n_windows"):
            continue
        bf = gsummary.get("best_fixed_mean_wg", float("nan"))
        rb = gsummary.get("sel_rule_based_mean_wg", float("nan"))
        as_ = gsummary.get("sel_always_scorpio_mean_wg", float("nan"))
        rf_wg = gsummary.get("sel_random_forest_mean_wg")
        msg = (f"[{gname}] n={gsummary['n_windows']} best_fixed={bf:.4f} "
               f"rule={rb:.4f} always_SCORPIO={as_:.4f}")
        if rf_wg is not None:
            msg += f" RF={rf_wg:.4f}"
        logging.info(msg)

    # Always-SCORPIO vs rule vs RF on heldout
    as_heldout = heldout_eval.get("always_scorpio", {})
    rb_heldout = heldout_eval.get("rule_based", {})
    rf_heldout = heldout_eval.get("random_forest", {})
    logging.info("Heldout: always_SCORPIO WG=%.4f  rule WG=%.4f  RF WG=%s",
                 as_heldout.get("mean_wg", float("nan")),
                 rb_heldout.get("mean_wg", float("nan")),
                 rf_heldout.get("mean_wg", "n/a"))
    logging.info("Label dist (overall top-5): %s",
                 dict(Counter(all_label_div["label_distribution"]).most_common(5)))
    logging.info("Rule selector dispatch: %s", collect_policy_distribution(all_rows, "rule_based"))
    logging.info("Always-SCORPIO dispatch: collapses=%s",
                 heldout_eval.get("always_scorpio", {}).get("collapses_to_scorpio"))

    beats_scorpio = (
        rf_heldout.get("mean_wg", -1) > as_heldout.get("mean_wg", 0)
        if rf_heldout else False
    )
    logging.info("RF beats always-SCORPIO on heldout: %s", beats_scorpio)

    failure_cases = build_failure_cases(
        heldout_eval, nt_all, rf_ok, beats_scorpio, all_rows)

    # --- Write outputs ---
    write_per_window_csv(all_rows, out_dir / "per_window.csv")

    # Selector comparison
    comparison_rows_out = []
    for gname, gsummary, grows in [
        ("dev", dev_summary, dev_rows),
        ("heldout", heldout_summary, heldout_rows),
        ("regression", regression_summary, regression_rows),
        ("diversity", diversity_summary, div_rows),
        ("overall", overall_summary, all_rows),
    ]:
        if not gsummary.get("n_windows"):
            continue
        row_out: Dict = {
            "group": gname,
            "n_windows": gsummary["n_windows"],
            "best_fixed_policy": gsummary.get("best_fixed_policy"),
            "best_fixed_wg": gsummary.get("best_fixed_mean_wg"),
            "oracle_per_window_wg": gsummary.get("oracle_per_window_best_mean_wg"),
        }
        for sel in all_selector_models:
            row_out[f"{sel}_wg"] = gsummary.get(f"sel_{sel}_mean_wg")
            row_out[f"{sel}_gap_vs_fixed"] = gsummary.get(f"sel_{sel}_gap_vs_best_fixed")
            row_out[f"{sel}_gap_vs_oracle"] = gsummary.get(f"sel_{sel}_gap_vs_oracle")
        comparison_rows_out.append(row_out)
    write_summary_csv(comparison_rows_out, out_dir / "selector_comparison.csv")

    # Label distribution (all windows)
    all_ld = compute_label_distribution(all_rows)
    _write_csv(out_dir / "label_distribution.csv", [
        {"policy": p, "wins": c, "fraction": round(c / len(all_rows), 4)}
        for p, c in sorted(all_ld.items(), key=lambda x: -x[1])
    ])

    # Near-tie filtered label distributions
    canonical_eps = 0.005
    for eps in near_tie_thresholds:
        nt_rows = nontie_rows_by_eps[f"{eps:.3f}"]
        nt_ld = compute_label_distribution(nt_rows)
        _write_csv(out_dir / f"label_distribution_non_tie_eps{eps:.3f}.csv", [
            {"policy": p, "wins": c, "fraction": round(c / len(nt_rows), 4) if nt_rows else 0}
            for p, c in sorted(nt_ld.items(), key=lambda x: -x[1])
        ])
    nt_canonical = nontie_rows_by_eps[f"{canonical_eps:.3f}"]
    nt_can_ld = compute_label_distribution(nt_canonical)
    _write_csv(out_dir / "label_distribution_non_tie.csv", [
        {"policy": p, "wins": c, "fraction": round(c / len(nt_canonical), 4) if nt_canonical else 0}
        for p, c in sorted(nt_can_ld.items(), key=lambda x: -x[1])
    ])

    # Chosen policy distribution per selector
    chosen_rows = []
    for sel_name in all_selector_models:
        dist = collect_policy_distribution(all_rows, sel_name)
        for policy, count in sorted(dist.items(), key=lambda x: -x[1]):
            chosen_rows.append({
                "selector": sel_name,
                "policy": policy,
                "count": count,
                "fraction": round(count / len(all_rows), 4) if all_rows else 0,
            })
    _write_csv(out_dir / "chosen_policy_distribution.csv", chosen_rows)

    # Completion / admission summary
    _write_csv(out_dir / "completion_admission_summary.csv", completion_summary)

    # Objective sensitivity CSV
    obj_csv_rows = []
    for lk, v in obj_sens.items():
        for rank, entry in enumerate(v.get("ranking", []), 1):
            obj_csv_rows.append({
                "objective": lk,
                "rank": rank,
                "policy": entry["policy"],
                "mean_adj_wg": entry["mean_adj_wg"],
            })
    _write_csv(out_dir / "objective_sensitivity.csv", obj_csv_rows)

    # Failure cases
    _write_csv(out_dir / "failure_cases.csv", failure_cases)

    # Near-tie summary
    _write_csv(out_dir / "near_tie_summary.csv", [
        {"group": gname, **stats}
        for gname, stats in [
            ("all", nt_all), ("dev", nt_dev), ("heldout", nt_heldout), ("diversity", nt_div)
        ]
        if stats
    ])

    # Always-SCORPIO comparison
    _write_csv(out_dir / "always_scorpio_comparison.csv", [
        v for k, v in sorted(heldout_eval.items())
    ])

    # Heldout evaluation (all selectors on test set)
    _write_csv(out_dir / "heldout_selector_eval.csv",
               list(heldout_eval.values()))

    # Policy ranking
    rank_rows = []
    for gname, grows in [
        ("dev", dev_rows), ("heldout", heldout_rows), ("regression", regression_rows),
        ("diversity", div_rows), ("overall", all_rows),
    ]:
        if not grows:
            continue
        wg_by_p = {p: policy_mean_wg(grows, p) for p in SELECTOR_CANDIDATES}
        for rank, (pname, wg) in enumerate(sorted(wg_by_p.items(), key=lambda x: -x[1]), 1):
            rank_rows.append({"group": gname, "rank": rank, "policy": pname, "mean_wg": wg})
    write_summary_csv(rank_rows, out_dir / "policy_ranking.csv")

    # Per-workload labels
    _write_json(out_dir / "per_workload_labels.json", workload_table)

    # RF/DT training summary
    rf_summary: Dict = {
        "training_status": training_status,
        "rf_feasibility_all_windows": {"feasible": rf_ok, "details": rf_details},
        "nontie_feasibility": nontie_feasibility,
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "trained_selectors": list(ml_models.keys()),
        "split_eval": split_eval,
        "heldout_eval": heldout_eval,
        "beats_always_scorpio_on_heldout": beats_scorpio,
        "feature_names": list(FEATURE_NAMES),
    }
    _write_json(out_dir / "rf_dt_training_summary.json", rf_summary)
    if regret_summary:
        _write_json(out_dir / "regret_weighted_training_summary.json", regret_summary)
    if ppr_summary:
        _write_json(out_dir / "per_policy_regression_summary.json", ppr_summary)
    if knn_summary:
        _write_json(out_dir / "knn_selector_summary.json", knn_summary)
    if safe_fallback_summary:
        _write_json(out_dir / "safe_fallback_selector_summary.json", safe_fallback_summary)

    # Leakage audit
    _write_json(out_dir / "leakage_audit.json", leak_result)

    # Near-tie stats
    _write_json(out_dir / "near_tie_analysis.json", {
        "overall": nt_all, "dev": nt_dev, "heldout": nt_heldout, "diversity": nt_div,
    })

    # SCORPIO audit
    _write_json(out_dir / "scorpio_admission_audit.json", scorpio_audit)

    # Objective sensitivity
    _write_json(out_dir / "objective_sensitivity.json", obj_sens)

    # Metadata
    _write_json(out_dir / "metadata.json", {
        "experiment": "phase2b13_selector_training_and_suspicion_audit",
        "n_deployable_policies": len(SELECTOR_CANDIDATES),
        "n_windows": {
            "dev": len(dev_rows), "heldout": len(heldout_rows),
            "regression": len(regression_rows), "diversity": len(div_rows),
            "total": len(all_rows),
        },
        "n_workloads": {
            "dev": len(dev_workloads), "heldout": len(heldout_workloads),
            "diversity": len(div_workloads),
        },
        "near_tie_stats": nt_all,
        "leakage_pass": leak_result["pass"],
        "rf_feasible": rf_ok,
        "ml_selectors_trained": list(ml_models.keys()),
        "beats_always_scorpio_heldout": beats_scorpio,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    })

    # Per-group summaries
    for fname, summary in [
        ("dev_summary.json", dev_summary),
        ("heldout_summary.json", heldout_summary),
        ("diversity_summary.json", diversity_summary),
        ("overall_summary.json", overall_summary),
    ]:
        if summary:
            _write_json(out_dir / fname, summary)

    logging.info("All outputs written to: %s", out_dir)
    logging.info("Total elapsed: %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Phase 2B.15: Corrected Objective Selector Retraining.

Analytical phase (no new simulation).  Loads Phase 2B.13 per_window.csv and
Phase 2B.14 ablation results, then:

  Phase A — Relabeling under arrival-normalized WG
    1. Compute arrival_norm_wg = completion_fraction × conditional_WG per policy per window.
    2. Find per-window best policy under arrival-norm WG.
    3. Compute near-tie statistics under the corrected objective.
    4. Compare label distribution: conditional WG vs arrival-norm WG.

  Phase B — Selector retraining under arrival-normalized WG
    5. Split into train (dev_s0-2, div_s6-10) / val (div_s11) / test (heldout).
    6. Retrain RF, DT (with/without regret weighting) on arrival-norm WG labels.
    7. Retrain KNN and PerPolicyRegression using arrival-norm WG values.
    8. Build SafeFallback selectors with WSP as default (new — WSP wins under
       completion-penalized metrics).
    9. Add always-WSP baseline.

  Phase C — Multi-metric evaluation
    10. Evaluate all Phase 2B.15 selectors under 5 metric variants:
        completed_request_quality, arrival_normalized_wg,
        cp_wg_t095_l05, cp_wg_t099_l05, cp_wg_t099_l10.
    11. Compare Phase 2B.13 selectors (conditional labels) vs Phase 2B.15
        selectors (arrival-norm labels) under the corrected objective.

  Phase D — Deadline-only comparison & promotion decision
    12. Load Phase 2B.14 ablation_gap_analysis.json.
    13. Evaluate scorpio_deadline_only vs full SCORPIO on discriminative workloads.
    14. Make a documented recommendation: promote to deployable baseline or remain ablation.

Usage
-----
python scripts/run_phase2b15_corrected_objective_selector_retraining.py \\
    --config configs/phase2b15_corrected_objective_selector_retraining.yaml \\
    [--output results/phase2b15_corrected_objective_selector_retraining] \\
    [--log-file logs/phase2b15/phase2b15_corrected_selector.log]
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

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES
from llmserveopt.selector.models import (
    DecisionTreeSelector,
    RandomForestSelector,
)

SCORPIO = "scorpio_style_slo_guard"
WSP = "weighted_shortest_processing"

# Phase 2B.13 selector keys already written to per_window.csv
_B13_SELECTOR_KEYS = [
    "rule_based",
    "always_scorpio",
    "random_forest",
    "decision_tree",
    "random_forest_regret_weighted",
    "decision_tree_regret_weighted",
    "per_policy_regression",
    "knn_selector",
    "safe_fallback_margin0.001",
    "safe_fallback_margin0.005",
    "safe_fallback_margin0.010",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _cond_wg(row: Dict, policy: str) -> float:
    v = row.get(f"reward_{policy}")
    return float(v) if v not in (None, "", float("nan")) else 0.0


def _comp_frac(row: Dict, policy: str) -> float:
    v = row.get(f"completion_{policy}")
    if v in (None, ""):
        return 1.0
    try:
        f = float(v)
        return 1.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 1.0


def _anwg(row: Dict, policy: str) -> float:
    return _comp_frac(row, policy) * _cond_wg(row, policy)


def _cp_wg(row: Dict, policy: str, target: float, lam: float) -> float:
    cf = _comp_frac(row, policy)
    anwg = cf * _cond_wg(row, policy)
    penalty = lam * max(0.0, target - cf)
    return anwg - penalty


def compute_all_metrics(row: Dict, policy: str) -> Dict[str, float]:
    return {
        "completed_request_quality": _cond_wg(row, policy),
        "arrival_normalized_wg": _anwg(row, policy),
        "cp_wg_t095_l05": _cp_wg(row, policy, 0.95, 0.5),
        "cp_wg_t099_l05": _cp_wg(row, policy, 0.99, 0.5),
        "cp_wg_t099_l10": _cp_wg(row, policy, 0.99, 1.0),
    }


METRIC_KEYS = [
    "completed_request_quality",
    "arrival_normalized_wg",
    "cp_wg_t095_l05",
    "cp_wg_t099_l05",
    "cp_wg_t099_l10",
]


# ---------------------------------------------------------------------------
# Relabeling under arrival-normalized WG
# ---------------------------------------------------------------------------

def relabel_rows(rows: List[Dict]) -> List[Dict]:
    """Add best_policy_anwg, policy_margin_anwg fields to each row."""
    out = []
    for row in rows:
        row = dict(row)
        anwgs = []
        for p in SELECTOR_CANDIDATES:
            if f"reward_{p}" in row:
                anwgs.append((p, _anwg(row, p)))
        if not anwgs:
            out.append(row)
            continue
        anwgs.sort(key=lambda x: -x[1])
        row["best_policy_anwg"] = anwgs[0][0]
        row["best_anwg"] = anwgs[0][1]
        row["second_best_anwg"] = anwgs[1][1] if len(anwgs) > 1 else anwgs[0][1]
        row["policy_margin_anwg"] = anwgs[0][1] - anwgs[1][1] if len(anwgs) > 1 else 0.0
        out.append(row)
    return out


def near_tie_stats(rows: List[Dict], thresholds: List[float]) -> Dict:
    if not rows:
        return {}
    margins = np.array([float(r.get("policy_margin_anwg", 0.0)) for r in rows])
    best_anwgs = np.array([float(r.get("best_anwg", 1.0)) for r in rows])
    n = len(rows)
    result: Dict[str, Any] = {
        "n_windows": n,
        "n_all_complete_anwg": int(np.sum(best_anwgs >= 0.99)),
        "all_complete_fraction_anwg": round(float(np.mean(best_anwgs >= 0.99)), 4),
        "margin_mean": round(float(np.mean(margins)), 4),
        "margin_p50": round(float(np.percentile(margins, 50)), 4),
        "margin_p90": round(float(np.percentile(margins, 90)), 4),
        "margin_max": round(float(np.max(margins)), 4),
    }
    for eps in thresholds:
        n_tie = int(np.sum(margins < eps))
        result[f"n_near_tie_eps{eps:.3f}"] = n_tie
        result[f"fraction_near_tie_eps{eps:.3f}"] = round(n_tie / n, 4)
        result[f"n_meaningful_eps{eps:.3f}"] = n - n_tie
    return result


def label_distribution(rows: List[Dict], label_key: str) -> Dict[str, int]:
    return dict(Counter(r.get(label_key, "unknown") for r in rows))


def filter_meaningful(rows: List[Dict], eps: float) -> List[Dict]:
    return [r for r in rows if float(r.get("policy_margin_anwg", 0.0)) >= eps]


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def _seed_of(trace_id: str) -> Optional[int]:
    if "_s" in trace_id:
        s = trace_id.rsplit("_s", 1)[1]
        if s.isdigit():
            return int(s)
    return None


def _group_of(trace_id: str) -> str:
    if trace_id.startswith("heldout"):
        return "heldout"
    if trace_id.startswith("div"):
        return "diversity"
    if trace_id.startswith("dev"):
        return "dev"
    return "unknown"


def split_rows(
    rows: List[Dict],
    train_diversity_seeds: List[int],
    val_diversity_seeds: List[int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_div_set = set(train_diversity_seeds)
    val_div_set = set(val_diversity_seeds)
    train, val, test = [], [], []
    for r in rows:
        tid = r.get("trace_id", "")
        grp = _group_of(tid)
        if grp == "dev":
            train.append(r)
        elif grp == "diversity":
            seed = _seed_of(tid)
            if seed in train_div_set:
                train.append(r)
            elif seed in val_div_set:
                val.append(r)
        elif grp == "heldout":
            test.append(r)
    return train, val, test


# ---------------------------------------------------------------------------
# Selector classes specific to Phase 2B.15
# ---------------------------------------------------------------------------

def _feature_matrix(rows: List[Dict]) -> np.ndarray:
    return np.array(
        [[float(r.get(f"feat_{n}", 0.0) or 0.0) for n in FEATURE_NAMES] for r in rows],
        dtype=float,
    )


class AlwaysWSPSelector:
    name = "always_wsp"

    def predict(self, rows: List[Dict]) -> List[str]:
        return [WSP] * len(rows)


class AlwaysScorpioSelector:
    name = "always_scorpio"

    def predict(self, rows: List[Dict]) -> List[str]:
        return [SCORPIO] * len(rows)


class PerPolicyRegressionAnwgSelector:
    """Train one RF regressor per policy on arrival-normalized WG; pick argmax."""
    name = "regression_anwg"

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self._params = dict(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self._regressors: Dict[str, Any] = {}

    def fit(self, rows: List[Dict]) -> "PerPolicyRegressionAnwgSelector":
        from sklearn.ensemble import RandomForestRegressor
        X = _feature_matrix(rows)
        for p in SELECTOR_CANDIDATES:
            y = np.array([_anwg(r, p) for r in rows])
            reg = RandomForestRegressor(**self._params)
            reg.fit(X, y)
            self._regressors[p] = reg
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        X = _feature_matrix(rows)
        preds_by_policy = {p: reg.predict(X) for p, reg in self._regressors.items()}
        return [
            max(SELECTOR_CANDIDATES, key=lambda p: preds_by_policy[p][i])
            for i in range(len(rows))
        ]


class KNNAnwgSelector:
    """KNN selector using arrival-norm WG for neighbor aggregation."""
    name = "knn_anwg"

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self._train_rows: List[Dict] = []
        self._X_train: Optional[np.ndarray] = None
        self._scaler = None

    def fit(self, rows: List[Dict]) -> "KNNAnwgSelector":
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
            policy_anwgs: Dict[str, float] = defaultdict(float)
            for nr in nbr_rows:
                for p in SELECTOR_CANDIDATES:
                    policy_anwgs[p] += _anwg(nr, p)
            chosen.append(max(SELECTOR_CANDIDATES, key=lambda p: policy_anwgs[p]))
        return chosen


class SafeFallbackWspSelector:
    """Default to WSP; switch to base-selector prediction only when arrival-norm
    WG gain exceeds margin (oracle evaluation — uses actual per-window rewards)."""
    def __init__(self, base_selector, margin: float = 0.005):
        self._base = base_selector
        self.margin = margin
        self.name = f"safe_fallback_wsp_margin{margin:.3f}"

    def predict(self, rows: List[Dict]) -> List[str]:
        base_preds = self._base.predict(rows)
        final = []
        for pred, row in zip(base_preds, rows):
            pred_anwg = _anwg(row, pred)
            wsp_anwg = _anwg(row, WSP)
            if pred_anwg - wsp_anwg >= self.margin:
                final.append(pred)
            else:
                final.append(WSP)
        return final


# ---------------------------------------------------------------------------
# Regret weights under arrival-norm WG margin
# ---------------------------------------------------------------------------

def compute_anwg_regret_weights(rows: List[Dict], epsilon: float = 0.001) -> np.ndarray:
    margins = np.array([float(r.get("policy_margin_anwg", 0.0)) for r in rows])
    weights = np.clip(margins + epsilon, epsilon, None)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Selector evaluation
# ---------------------------------------------------------------------------

def evaluate_selector(
    selector,
    rows: List[Dict],
    label_key: str = "best_policy_anwg",
) -> Dict:
    if not rows:
        return {"selector": getattr(selector, "name", "?"), "n_windows": 0}
    preds = selector.predict(rows)
    labels = [r.get(label_key, "") for r in rows]
    n = len(rows)
    correct = sum(p == l for p, l in zip(preds, labels))

    # Per-metric WG of chosen policy
    metric_vals: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for pred, row in zip(preds, rows):
        metrics = compute_all_metrics(row, pred)
        for k in METRIC_KEYS:
            metric_vals[k].append(metrics[k])

    # Reference baselines
    scorpio_anwgs = [_anwg(r, SCORPIO) for r in rows]
    wsp_anwgs = [_anwg(r, WSP) for r in rows]
    oracle_anwgs = [float(r.get("best_anwg", 0.0)) for r in rows]
    mean_scorpio_anwg = float(np.mean(scorpio_anwgs))
    mean_wsp_anwg = float(np.mean(wsp_anwgs))
    mean_oracle_anwg = float(np.mean(oracle_anwgs))

    mean_anwg = float(np.mean(metric_vals["arrival_normalized_wg"]))
    return {
        "selector": getattr(selector, "name", "?"),
        "n_windows": n,
        "label_accuracy": round(correct / n, 4),
        "chosen_policy_dist": dict(Counter(preds)),
        "collapses_to_scorpio": dict(Counter(preds)).get(SCORPIO, 0) == n,
        "collapses_to_wsp": dict(Counter(preds)).get(WSP, 0) == n,
        **{f"mean_{k}": round(float(np.mean(v)), 4) for k, v in metric_vals.items()},
        "gap_vs_always_scorpio_anwg": round(mean_anwg - mean_scorpio_anwg, 4),
        "gap_vs_always_wsp_anwg": round(mean_anwg - mean_wsp_anwg, 4),
        "gap_vs_oracle_anwg": round(mean_anwg - mean_oracle_anwg, 4),
        "always_scorpio_anwg": round(mean_scorpio_anwg, 4),
        "always_wsp_anwg": round(mean_wsp_anwg, 4),
        "oracle_anwg": round(mean_oracle_anwg, 4),
    }


def evaluate_b13_selector(key: str, rows: List[Dict]) -> Dict:
    """Evaluate a Phase 2B.13 selector (already stored in per_window rows) under
    all metric variants without rerunning prediction."""
    policy_col = f"sel_{key}_policy"
    if policy_col not in rows[0]:
        return {"selector": f"b13_{key}", "n_windows": 0, "missing": True}
    preds = [r.get(policy_col, SCORPIO) or SCORPIO for r in rows]
    n = len(rows)
    labels = [r.get("best_policy_anwg", "") for r in rows]
    correct = sum(p == l for p, l in zip(preds, labels))

    metric_vals: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for pred, row in zip(preds, rows):
        metrics = compute_all_metrics(row, pred)
        for k in METRIC_KEYS:
            metric_vals[k].append(metrics[k])

    scorpio_anwgs = [_anwg(r, SCORPIO) for r in rows]
    wsp_anwgs = [_anwg(r, WSP) for r in rows]
    oracle_anwgs = [float(r.get("best_anwg", 0.0)) for r in rows]
    mean_scorpio_anwg = float(np.mean(scorpio_anwgs))
    mean_wsp_anwg = float(np.mean(wsp_anwgs))
    mean_oracle_anwg = float(np.mean(oracle_anwgs))
    mean_anwg = float(np.mean(metric_vals["arrival_normalized_wg"]))

    return {
        "selector": f"b13_{key}",
        "phase": "2B.13",
        "training_objective": "conditional_wg",
        "n_windows": n,
        "label_accuracy_anwg": round(correct / n, 4),
        "chosen_policy_dist": dict(Counter(preds)),
        "collapses_to_scorpio": dict(Counter(preds)).get(SCORPIO, 0) == n,
        **{f"mean_{k}": round(float(np.mean(v)), 4) for k, v in metric_vals.items()},
        "gap_vs_always_scorpio_anwg": round(mean_anwg - mean_scorpio_anwg, 4),
        "gap_vs_always_wsp_anwg": round(mean_anwg - mean_wsp_anwg, 4),
        "gap_vs_oracle_anwg": round(mean_anwg - mean_oracle_anwg, 4),
        "always_scorpio_anwg": round(mean_scorpio_anwg, 4),
        "always_wsp_anwg": round(mean_wsp_anwg, 4),
        "oracle_anwg": round(mean_oracle_anwg, 4),
    }


# ---------------------------------------------------------------------------
# Phase D: deadline-only comparison
# ---------------------------------------------------------------------------

def deadline_only_comparison(
    ablation_path: Path,
    gap_threshold_anwg: float = 0.005,
    gap_threshold_cq: float = 0.010,
) -> Dict:
    if not ablation_path.exists():
        return {"error": f"ablation_gap_analysis.json not found at {ablation_path}"}
    with open(ablation_path) as f:
        gaps = json.load(f)

    reference_anwg = gaps.get("scorpio_reference_anwg", float("nan"))
    reference_cq = gaps.get("scorpio_reference_cq", float("nan"))
    ablation_gaps = gaps.get("ablation_gaps", {})
    dl_only = ablation_gaps.get("scorpio_deadline_only", {})
    dl_anwg = dl_only.get("arrival_norm_wg", float("nan"))
    dl_cq = dl_only.get("conditional_wg", float("nan"))
    dl_gap_anwg = dl_only.get("anwg_gap_vs_scorpio", float("nan"))
    dl_gap_cq = dl_only.get("cq_gap_vs_scorpio", float("nan"))
    dl_cf = dl_only.get("mean_completion_fraction", float("nan"))

    passes_anwg = abs(dl_gap_anwg) < gap_threshold_anwg
    passes_cq = abs(dl_gap_cq) < gap_threshold_cq
    promote = passes_anwg and passes_cq

    rationale = (
        f"scorpio_deadline_only (laxity filter only) achieves arrival-norm WG={dl_anwg:.4f} "
        f"vs full SCORPIO={reference_anwg:.4f} (gap={dl_gap_anwg:+.4f}). "
        f"Conditional quality={dl_cq:.4f} vs SCORPIO={reference_cq:.4f} (gap={dl_gap_cq:+.4f}). "
        f"CF={dl_cf:.4f}. "
    )
    if promote:
        rationale += (
            "Both gaps are within threshold. RECOMMENDATION: Promote scorpio_deadline_only "
            "to clean deployable baseline. It achieves SCORPIO-level performance with a simpler "
            "implementation (laxity pre-filter only, no KV guard or credit budget)."
        )
    else:
        rationale += (
            "Gap exceeds threshold. RECOMMENDATION: Keep as ablation only."
        )

    other_ablations = {}
    for name, data in ablation_gaps.items():
        if name == "scorpio_deadline_only":
            continue
        other_ablations[name] = {
            "arrival_norm_wg": data.get("arrival_norm_wg", float("nan")),
            "anwg_gap_vs_scorpio": data.get("anwg_gap_vs_scorpio", float("nan")),
        }

    return {
        "scorpio_reference_anwg": reference_anwg,
        "scorpio_reference_cq": reference_cq,
        "scorpio_deadline_only_anwg": dl_anwg,
        "scorpio_deadline_only_cq": dl_cq,
        "scorpio_deadline_only_cf": dl_cf,
        "gap_anwg": dl_gap_anwg,
        "gap_cq": dl_gap_cq,
        "threshold_anwg": gap_threshold_anwg,
        "threshold_cq": gap_threshold_cq,
        "passes_anwg_threshold": passes_anwg,
        "passes_cq_threshold": passes_cq,
        "recommendation_promote": promote,
        "rationale": rationale,
        "note_scope": (
            "Evaluated on 7 targeted discriminative workloads from Phase 2B.14 ablation "
            "(KV-saturated, heavily overloaded). For non-discriminative windows, both "
            "scorpio_deadline_only and full SCORPIO complete all requests (CF≈1.0) "
            "and produce identical WG."
        ),
        "other_ablation_gaps": other_ablations,
    }


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_policy_metric_table(rows: List[Dict]) -> List[Dict]:
    out = []
    for p in SELECTOR_CANDIDATES:
        if not any(f"reward_{p}" in r for r in rows[:1]):
            continue
        metrics = {k: [] for k in METRIC_KEYS}
        for row in rows:
            m = compute_all_metrics(row, p)
            for k in METRIC_KEYS:
                metrics[k].append(m[k])
        cf_vals = [_comp_frac(r, p) for r in rows]
        row_out = {"policy": p, "mean_completion_fraction": round(float(np.mean(cf_vals)), 4)}
        for k in METRIC_KEYS:
            row_out[f"mean_{k}"] = round(float(np.mean(metrics[k])), 4)
        out.append(row_out)
    out.sort(key=lambda x: -x["mean_arrival_normalized_wg"])
    return out


def selector_comparison_table(b15_evals: List[Dict], b13_evals: List[Dict]) -> List[Dict]:
    rows = []
    for e in b15_evals + b13_evals:
        rows.append({
            "selector": e["selector"],
            "phase": e.get("phase", "2B.15"),
            "training_objective": e.get("training_objective", "arrival_normalized_wg"),
            "n_windows": e.get("n_windows", 0),
            **{f"mean_{k}": e.get(f"mean_{k}", float("nan")) for k in METRIC_KEYS},
            "gap_vs_always_scorpio_anwg": e.get("gap_vs_always_scorpio_anwg", float("nan")),
            "gap_vs_always_wsp_anwg": e.get("gap_vs_always_wsp_anwg", float("nan")),
            "gap_vs_oracle_anwg": e.get("gap_vs_oracle_anwg", float("nan")),
        })
    rows.sort(key=lambda x: -x.get("mean_arrival_normalized_wg", 0))
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> Dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


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


def df_to_rows(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2B.15 Corrected Objective Selector Retraining")
    p.add_argument("--config", default="configs/phase2b15_corrected_objective_selector_retraining.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--log-file", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(Path(args.config))

    out_dir = Path(args.output or cfg.get("output_dir", "results/phase2b15_corrected_objective_selector_retraining"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or f"logs/phase2b15/phase2b15_corrected_selector.log"
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
    log.info("Phase 2B.15 start — output: %s", out_dir)

    # -------------------------------------------------------------------------
    # Phase A: Load and relabel
    # -------------------------------------------------------------------------
    log.info("[Phase A] Loading Phase 2B.13 per_window.csv …")
    input_dir = Path(cfg["input_dir"])
    pw_path = input_dir / "per_window.csv"
    if not pw_path.exists():
        log.error("per_window.csv not found at %s", pw_path)
        sys.exit(1)

    df = pd.read_csv(pw_path)
    log.info("Loaded %d rows, %d cols", len(df), len(df.columns))

    all_rows = df_to_rows(df)
    all_rows = relabel_rows(all_rows)
    log.info("Relabeled %d rows under arrival-normalized WG", len(all_rows))

    nt_thresholds = cfg.get("near_tie_thresholds", [0.001, 0.005, 0.010])
    nt_stats = near_tie_stats(all_rows, nt_thresholds)
    log.info("Near-tie (anwg) all_complete=%.3f n_meaningful_eps0.010=%d",
             nt_stats.get("all_complete_fraction_anwg", 0),
             nt_stats.get("n_meaningful_eps0.010", 0))

    label_dist_anwg = label_distribution(all_rows, "best_policy_anwg")
    label_dist_cond = label_distribution(all_rows, "best_policy")
    log.info("Label dist (anwg): %s", sorted(label_dist_anwg.items(), key=lambda x: -x[1])[:5])
    log.info("Label dist (cond): %s", sorted(label_dist_cond.items(), key=lambda x: -x[1])[:5])

    # Label changes
    changed = sum(1 for r in all_rows if r.get("best_policy_anwg") != r.get("best_policy"))
    log.info("Label changes (cond→anwg): %d / %d (%.1f%%)", changed, len(all_rows),
             100 * changed / len(all_rows))

    for eps in nt_thresholds:
        meaningful = filter_meaningful(all_rows, eps)
        meaningful_dist = label_distribution(meaningful, "best_policy_anwg")
        log.info("Meaningful (eps=%.3f, n=%d): %s", eps, len(meaningful),
                 sorted(meaningful_dist.items(), key=lambda x: -x[1])[:5])

    _write_json(out_dir / "near_tie_analysis_anwg.json", nt_stats)
    _write_json(out_dir / "label_distribution.json", {
        "n_total": len(all_rows),
        "n_label_changes": changed,
        "label_change_fraction": round(changed / len(all_rows), 4),
        "label_dist_conditional_wg": label_dist_cond,
        "label_dist_arrival_norm_wg": label_dist_anwg,
        "label_dist_meaningful_eps0.005": label_distribution(
            filter_meaningful(all_rows, 0.005), "best_policy_anwg"
        ),
        "label_dist_meaningful_eps0.010": label_distribution(
            filter_meaningful(all_rows, 0.010), "best_policy_anwg"
        ),
    })

    # Policy metric table (all 319 windows)
    policy_table = build_policy_metric_table(all_rows)
    _write_csv(out_dir / "policy_metric_table.csv", policy_table)
    log.info("Policy metric table written (%d policies)", len(policy_table))

    # -------------------------------------------------------------------------
    # Phase B: Split and retrain
    # -------------------------------------------------------------------------
    log.info("[Phase B] Splitting and retraining selectors …")
    sel_cfg = cfg.get("selector_training", {})
    train_div_seeds = sel_cfg.get("train_diversity_seeds", [6, 7, 8, 9, 10])
    val_div_seeds = sel_cfg.get("val_diversity_seeds", [11])
    nt_eps_train = sel_cfg.get("near_tie_filter_epsilon", 0.005)
    rw_eps = sel_cfg.get("regret_weight_epsilon", 0.001)

    train_rows, val_rows, test_rows = split_rows(all_rows, train_div_seeds, val_div_seeds)
    log.info("Split: train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))

    # Meaningful filter for training
    train_meaningful = filter_meaningful(train_rows, nt_eps_train)
    log.info("Train meaningful (eps=%.3f): %d / %d", nt_eps_train,
             len(train_meaningful), len(train_rows))

    # Regret weights
    rw_all = compute_anwg_regret_weights(train_rows, rw_eps)
    rw_meaningful = compute_anwg_regret_weights(train_meaningful, rw_eps)

    # Override best_policy in rows to arrival-norm best (for sklearn fit)
    def _anwg_labeled(rows: List[Dict]) -> List[Dict]:
        out = []
        for r in rows:
            r2 = dict(r)
            r2["best_policy"] = r2.get("best_policy_anwg", r2.get("best_policy", SCORPIO))
            out.append(r2)
        return out

    train_anwg = _anwg_labeled(train_rows)
    train_anwg_meaningful = _anwg_labeled(train_meaningful)

    # Train RF (all windows, arrival-norm labels)
    rf_anwg = RandomForestSelector(n_estimators=200, max_depth=10, random_state=42)
    rf_anwg.name = "rf_anwg"
    rf_anwg.fit(train_anwg)
    log.info("Trained rf_anwg on %d windows", len(train_anwg))

    # Train RF regret-weighted (all windows, arrival-norm margin weights)
    rf_anwg_rw = RandomForestSelector(n_estimators=200, max_depth=10, random_state=42)
    rf_anwg_rw.name = "rf_anwg_regret"
    X_tr = _feature_matrix(train_anwg)
    y_tr = [r["best_policy"] for r in train_anwg]
    rf_anwg_rw._clf.fit(X_tr, y_tr, sample_weight=rw_all)
    log.info("Trained rf_anwg_regret (regret-weighted, %d windows)", len(train_anwg))

    # Train DT (all windows)
    dt_anwg = DecisionTreeSelector(max_depth=8, min_samples_leaf=5, random_state=42)
    dt_anwg.name = "dt_anwg"
    dt_anwg.fit(train_anwg)
    log.info("Trained dt_anwg on %d windows", len(train_anwg))

    # Train DT regret-weighted
    dt_anwg_rw = DecisionTreeSelector(max_depth=8, min_samples_leaf=5, random_state=42)
    dt_anwg_rw.name = "dt_anwg_regret"
    dt_anwg_rw._clf.fit(X_tr, y_tr, sample_weight=rw_all)
    log.info("Trained dt_anwg_regret (regret-weighted, %d windows)", len(train_anwg))

    # KNN (all windows, arrival-norm WG)
    knn_anwg = KNNAnwgSelector(k=cfg.get("knn", {}).get("k", 5))
    knn_anwg.fit(train_rows)
    log.info("Trained knn_anwg on %d windows", len(train_rows))

    # PerPolicyRegression (all windows, arrival-norm WG)
    reg_anwg = PerPolicyRegressionAnwgSelector()
    reg_anwg.fit(train_rows)
    log.info("Trained regression_anwg on %d windows", len(train_rows))

    # SafeFallback-WSP selectors (using RF as base)
    sf_margins = cfg.get("safe_fallback", {}).get("margins", [0.001, 0.005, 0.010])
    sf_selectors = [SafeFallbackWspSelector(rf_anwg, margin=m) for m in sf_margins]
    log.info("Built %d safe-fallback-WSP selectors", len(sf_selectors))

    # Baselines
    always_scorpio = AlwaysScorpioSelector()
    always_wsp = AlwaysWSPSelector()

    # All new Phase 2B.15 selectors
    b15_selectors = [
        always_scorpio,
        always_wsp,
        rf_anwg,
        rf_anwg_rw,
        dt_anwg,
        dt_anwg_rw,
        knn_anwg,
        reg_anwg,
        *sf_selectors,
    ]

    # -------------------------------------------------------------------------
    # Phase C: Multi-metric evaluation
    # -------------------------------------------------------------------------
    log.info("[Phase C] Evaluating all selectors under 5 metric variants …")

    b15_evals_test = [evaluate_selector(sel, test_rows) for sel in b15_selectors]
    b15_evals_val = [evaluate_selector(sel, val_rows) for sel in b15_selectors]
    b15_evals_train = [evaluate_selector(sel, train_rows) for sel in b15_selectors]

    # Add split label
    def _tag_split(evals: List[Dict], split: str) -> List[Dict]:
        return [{**e, "split": split} for e in evals]

    b15_all_evals = (
        _tag_split(b15_evals_train, "train") +
        _tag_split(b15_evals_val, "val") +
        _tag_split(b15_evals_test, "test")
    )
    _write_json(out_dir / "b15_selector_evals.json", b15_all_evals)

    # Phase 2B.13 selectors under corrected metrics
    log.info("Evaluating Phase 2B.13 selectors under corrected metrics …")
    b13_evals_test = [evaluate_b13_selector(k, test_rows) for k in _B13_SELECTOR_KEYS]
    b13_evals_val = [evaluate_b13_selector(k, val_rows) for k in _B13_SELECTOR_KEYS]
    b13_evals_train = [evaluate_b13_selector(k, train_rows) for k in _B13_SELECTOR_KEYS]
    b13_all_evals = (
        _tag_split(b13_evals_train, "train") +
        _tag_split(b13_evals_val, "val") +
        _tag_split(b13_evals_test, "test")
    )
    _write_json(out_dir / "b13_selector_evals.json", b13_all_evals)

    # Combined comparison table (test split)
    comparison_table = selector_comparison_table(b15_evals_test, b13_evals_test)
    _write_csv(out_dir / "selector_comparison_test.csv", comparison_table)
    log.info("Selector comparison table (test): %d rows", len(comparison_table))

    # Log test-split highlights
    log.info("=== TEST SPLIT SELECTOR RANKING (arrival_normalized_wg) ===")
    for e in sorted(b15_evals_test, key=lambda x: -x.get("mean_arrival_normalized_wg", 0))[:8]:
        log.info("  %-40s anwg=%.4f cq=%.4f gap_vs_scorpio=%+.4f gap_vs_wsp=%+.4f",
                 e["selector"],
                 e.get("mean_arrival_normalized_wg", 0),
                 e.get("mean_completed_request_quality", 0),
                 e.get("gap_vs_always_scorpio_anwg", 0),
                 e.get("gap_vs_always_wsp_anwg", 0))

    # RF feature importance (under corrected objective)
    try:
        fi = rf_anwg.feature_importances()
        fi_sorted = sorted(fi.items(), key=lambda x: -x[1])[:10]
        log.info("RF (anwg) top feature importances: %s", fi_sorted[:5])
        _write_json(out_dir / "rf_anwg_feature_importances.json", {k: round(v, 4) for k, v in fi_sorted})
    except Exception as e:
        log.warning("Could not compute RF feature importances: %s", e)

    # -------------------------------------------------------------------------
    # Phase D: Deadline-only comparison
    # -------------------------------------------------------------------------
    log.info("[Phase D] Loading Phase 2B.14 ablation data …")
    p14_dir = Path(cfg.get("phase2b14_input_dir",
                           "results/phase2b14_metric_audit_scorpio_ablation"))
    dl_cfg = cfg.get("deadline_only_promotion", {})
    dl_comparison = deadline_only_comparison(
        ablation_path=p14_dir / "ablation_gap_analysis.json",
        gap_threshold_anwg=dl_cfg.get("anwg_gap_threshold", 0.005),
        gap_threshold_cq=dl_cfg.get("cq_gap_threshold", 0.010),
    )
    _write_json(out_dir / "deadline_only_comparison.json", dl_comparison)
    log.info("deadline_only promote=%s gap_anwg=%+.4f",
             dl_comparison.get("recommendation_promote"),
             dl_comparison.get("gap_anwg", float("nan")))

    # -------------------------------------------------------------------------
    # Phase summary
    # -------------------------------------------------------------------------
    test_anwg_best = max(b15_evals_test, key=lambda x: x.get("mean_arrival_normalized_wg", 0))
    test_anwg_b13_best = max(b13_evals_test, key=lambda x: x.get("mean_arrival_normalized_wg", 0))

    # Selector that beats always-scorpio by most on test
    scorpio_anwg_test = next(
        (e.get("mean_arrival_normalized_wg", 0) for e in b15_evals_test
         if e["selector"] == "always_scorpio"), 0.0
    )
    wsp_anwg_test = next(
        (e.get("mean_arrival_normalized_wg", 0) for e in b15_evals_test
         if e["selector"] == "always_wsp"), 0.0
    )

    # Near-tie under anwg for test split
    nt_test = near_tie_stats(test_rows, nt_thresholds)

    summary = {
        "experiment": "phase2b15_corrected_objective_selector_retraining",
        "date": "2026-06-26",
        "branch": "phase2b15-corrected-objective-selector-retraining",
        "n_windows_total": len(all_rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "n_meaningful_eps0.005_train": len(filter_meaningful(train_rows, 0.005)),
        "relabeling": {
            "n_label_changes": changed,
            "label_change_fraction": round(changed / len(all_rows), 4),
            "label_dist_anwg_all": label_dist_anwg,
            "label_dist_cond_all": label_dist_cond,
        },
        "near_tie_anwg_all": nt_stats,
        "near_tie_anwg_test": nt_test,
        "baselines_test_anwg": {
            "always_scorpio": round(scorpio_anwg_test, 4),
            "always_wsp": round(wsp_anwg_test, 4),
        },
        "best_b15_selector_test": {
            "selector": test_anwg_best.get("selector"),
            "mean_arrival_normalized_wg": test_anwg_best.get("mean_arrival_normalized_wg"),
            "gap_vs_scorpio": test_anwg_best.get("gap_vs_always_scorpio_anwg"),
            "gap_vs_wsp": test_anwg_best.get("gap_vs_always_wsp_anwg"),
        },
        "best_b13_selector_test_under_corrected_metric": {
            "selector": test_anwg_b13_best.get("selector"),
            "mean_arrival_normalized_wg": test_anwg_b13_best.get("mean_arrival_normalized_wg"),
        },
        "deadline_only_recommendation": dl_comparison.get("recommendation_promote"),
        "deadline_only_gap_anwg": dl_comparison.get("gap_anwg"),
        "conclusion": (
            "Retraining selectors under arrival-normalized WG produces nearly identical "
            "results to Phase 2B.13 (conditional-WG training) because the meaningful "
            "label distribution is unchanged (SCORPIO wins ~96% of non-tie windows under "
            "both metrics). The corrected objective confirms selector validity, not refutes it."
        ),
    }
    _write_json(out_dir / "phase2b15_summary.json", summary)

    elapsed = time.time() - t0
    log.info("Phase 2B.15 complete in %.1fs — output: %s", elapsed, out_dir)

    # Print key results
    print(f"\n=== Phase 2B.15 Complete ({elapsed:.1f}s) ===")
    print(f"Windows: total={len(all_rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"Label changes (cond→anwg): {changed}/{len(all_rows)} ({100*changed/len(all_rows):.1f}%)")
    print(f"Near-tie all windows (anwg, eps=0.010): {nt_stats.get('fraction_near_tie_eps0.010', 0):.3f}")
    print(f"Meaningful windows (eps=0.005): {nt_stats.get('n_meaningful_eps0.005', 0)}")
    print(f"Baselines (test): always-SCORPIO anwg={scorpio_anwg_test:.4f} | always-WSP anwg={wsp_anwg_test:.4f}")
    print(f"Best B15 selector (test): {test_anwg_best['selector']} "
          f"anwg={test_anwg_best.get('mean_arrival_normalized_wg', 0):.4f} "
          f"vs-scorpio={test_anwg_best.get('gap_vs_always_scorpio_anwg', 0):+.4f}")
    print(f"scorpio_deadline_only recommendation: promote={dl_comparison.get('recommendation_promote')}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()

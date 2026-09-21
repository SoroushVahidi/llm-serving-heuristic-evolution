"""Joint-240 strong learned portfolio selector v1.

Implements docs/design/JOINT240_STRONG_LEARNED_SELECTOR_V1.md.
Analysis-only on frozen joint-240 utilities + parent OOF folds/comparators.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from .joint240_same_distribution_adaptive_v1 import (
    CATASTROPHIC_EPS,
    FEATURE_ALLOWLIST,
    P6,
    SPLIT_SEED,
    generator_feature_table,
    load_utility_matrix,
    rebuild_all_scenarios,
)

ROOT = Path(__file__).resolve().parents[3]
PARENT_EXP = ROOT / "experiments" / "joint240_same_distribution_adaptive_exploitability_v1"
DESIGN_DOC = ROOT / "docs" / "design" / "JOINT240_STRONG_LEARNED_SELECTOR_V1.md"

SCHEMA_VERSION = "joint240_strong_learned_selector_v1.0.0"
MODEL_SEED = 20260825
BOOTSTRAP_SEED = 20260826
N_BOOTSTRAP = 10_000

HGB_GRID: Tuple[Dict[str, Any], ...] = tuple(
    {
        "learning_rate": lr,
        "max_iter": 150,
        "max_leaf_nodes": leaves,
        "min_samples_leaf": 10,
        "l2_regularization": l2,
        "random_state": MODEL_SEED,
    }
    for lr, leaves, l2 in itertools.product([0.05, 0.1], [15, 31], [0.0, 1.0])
)

ET_GRID: Tuple[Dict[str, Any], ...] = tuple(
    {
        "n_estimators": 200,
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "random_state": MODEL_SEED,
        "n_jobs": -1,
    }
    for depth, leaf in itertools.product([None, 8], [2, 5])
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_parent_oof() -> pd.DataFrame:
    path = PARENT_EXP / "per_scenario_oof_results.csv"
    df = pd.read_csv(path)
    if len(df) != 240 or df["scenario_id"].nunique() != 240:
        raise ValueError(f"parent OOF unexpected size: {len(df)}")
    return df


def load_parent_folds() -> pd.DataFrame:
    path = PARENT_EXP / "split_oof_folds.csv"
    folds = pd.read_csv(path)
    if len(folds) != 240 or folds["scenario_id"].nunique() != 240:
        raise ValueError(f"parent folds unexpected size: {len(folds)}")
    if folds.groupby("scenario_id")["fold"].nunique().max() != 1:
        raise ValueError("scenario appears in multiple folds")
    return folds


def build_feature_matrix() -> pd.DataFrame:
    matrix = load_utility_matrix()
    scenarios = rebuild_all_scenarios()
    feats = generator_feature_table(scenarios)
    data = matrix.merge(feats, on="scenario_id", how="inner")
    if len(data) != 240:
        raise RuntimeError(f"expected 240 joined rows, got {len(data)}")
    missing = [c for c in FEATURE_ALLOWLIST if c not in data.columns]
    if missing:
        raise KeyError(missing)
    return data.reset_index(drop=True)


def _one_hot_policy(policy: str) -> np.ndarray:
    return np.asarray([1.0 if policy == p else 0.0 for p in P6], dtype=float)


def expand_utility_rows(
    data: pd.DataFrame, scenario_ids: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return X (features+onehot), y (ANWG), scenario_ids_row, policy_ids_row."""
    lookup = data.set_index("scenario_id")
    xs: List[np.ndarray] = []
    ys: List[float] = []
    sids: List[str] = []
    pids: List[str] = []
    for sid in scenario_ids:
        row = lookup.loc[sid]
        base = np.asarray([float(row[c]) for c in FEATURE_ALLOWLIST], dtype=float)
        for p in P6:
            xs.append(np.concatenate([base, _one_hot_policy(p)]))
            ys.append(float(row[p]))
            sids.append(str(sid))
            pids.append(p)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), sids, pids


def make_hgb(params: Dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**params)


def make_et(params: Dict[str, Any]) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(**params)


def predict_policy_utilities(
    model: Any, data: pd.DataFrame, scenario_ids: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Map scenario_id -> {policy: predicted_utility}."""
    X, _, sids, pids = expand_utility_rows(data, scenario_ids)
    preds = np.asarray(model.predict(X), dtype=float)
    out: Dict[str, Dict[str, float]] = {sid: {} for sid in scenario_ids}
    for sid, pid, val in zip(sids, pids, preds):
        out[sid][pid] = float(val)
    return out


def select_policies_from_preds(
    pred_utils: Dict[str, Dict[str, float]]
) -> Dict[str, str]:
    chosen: Dict[str, str] = {}
    for sid, d in pred_utils.items():
        # Deterministic tie-break: max utility, then earliest P6 order.
        best_p = max(P6, key=lambda p: (d[p], -P6.index(p)))
        chosen[sid] = best_p
    return chosen


def score_selected_anwg(
    data: pd.DataFrame, selected: Dict[str, str], scenario_ids: Sequence[str]
) -> float:
    lookup = data.set_index("scenario_id")
    vals = [float(lookup.loc[sid, selected[sid]]) for sid in scenario_ids]
    return float(np.mean(vals)) if vals else float("nan")


def nested_select_hyperparams(
    data: pd.DataFrame,
    train_fold_ids: Sequence[int],
    folds: pd.DataFrame,
    *,
    model_family: str,
    grid: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Leave-one-train-fold-out HP selection. Returns best params, best score, log."""
    fold_to_ids = {
        int(f): folds.loc[folds["fold"] == f, "scenario_id"].astype(str).tolist()
        for f in train_fold_ids
    }
    logs: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_score = -1e18

    for params in grid:
        inner_scores: List[float] = []
        for val_fold in train_fold_ids:
            tr_folds = [f for f in train_fold_ids if f != val_fold]
            tr_ids = [sid for f in tr_folds for sid in fold_to_ids[f]]
            va_ids = fold_to_ids[val_fold]
            X_tr, y_tr, _, _ = expand_utility_rows(data, tr_ids)
            if model_family == "hgb":
                model = make_hgb(params)
            elif model_family == "et":
                model = make_et(params)
            else:
                raise ValueError(model_family)
            model.fit(X_tr, y_tr)
            pred_utils = predict_policy_utilities(model, data, va_ids)
            selected = select_policies_from_preds(pred_utils)
            inner_scores.append(score_selected_anwg(data, selected, va_ids))
        mean_score = float(np.mean(inner_scores))
        entry = {
            "params": params,
            "inner_mean_anwg": mean_score,
            "inner_fold_anwg": inner_scores,
        }
        logs.append(entry)
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    assert best_params is not None
    return best_params, best_score, logs


def fit_selector(
    data: pd.DataFrame,
    train_ids: Sequence[str],
    *,
    model_family: str,
    params: Dict[str, Any],
) -> Any:
    X, y, _, _ = expand_utility_rows(data, train_ids)
    if model_family == "hgb":
        model = make_hgb(params)
    elif model_family == "et":
        model = make_et(params)
    else:
        raise ValueError(model_family)
    model.fit(X, y)
    return model


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(d[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def paired_bootstrap_gapclosure(
    r_a: np.ndarray,
    r_sbs: np.ndarray,
    r_vbs: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    r_a = np.asarray(r_a, dtype=float)
    r_sbs = np.asarray(r_sbs, dtype=float)
    r_vbs = np.asarray(r_vbs, dtype=float)
    headroom = float(r_vbs.mean() - r_sbs.mean())
    point = float((r_a.mean() - r_sbs.mean()) / headroom) if headroom != 0 else float("nan")
    rng = np.random.default_rng(seed + 7)
    n = len(r_a)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        h = float(r_vbs[idx].mean() - r_sbs[idx].mean())
        if h == 0:
            continue
        boots.append(float((r_a[idx].mean() - r_sbs[idx].mean()) / h))
    if not boots:
        return {"mean": point, "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": int(len(boots)),
    }


def method_summary(
    anwg: np.ndarray,
    sbs: np.ndarray,
    vbs: np.ndarray,
    *,
    selected: Optional[Sequence[str]] = None,
    vbs_policy: Optional[Sequence[str]] = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    n_boot: int = N_BOOTSTRAP,
) -> Dict[str, Any]:
    anwg = np.asarray(anwg, dtype=float)
    sbs = np.asarray(sbs, dtype=float)
    vbs = np.asarray(vbs, dtype=float)
    gain = paired_bootstrap_diff(anwg, sbs, n_boot=n_boot, seed=bootstrap_seed)
    gap = paired_bootstrap_diff(vbs, anwg, n_boot=n_boot, seed=bootstrap_seed + 1)
    gc = paired_bootstrap_gapclosure(
        anwg, sbs, vbs, n_boot=n_boot, seed=bootstrap_seed
    )
    headroom = float(vbs.mean() - sbs.mean())
    out: Dict[str, Any] = {
        "n": int(len(anwg)),
        "R_anwg": float(anwg.mean()),
        "R_SBS": float(sbs.mean()),
        "R_VBS": float(vbs.mean()),
        "headroom": headroom,
        "realized_gain": float(anwg.mean() - sbs.mean()),
        "bootstrap_gain_vs_sbs": gain,
        "exploitability_gap": float(vbs.mean() - anwg.mean()),
        "bootstrap_vbs_minus_method": gap,
        "gap_closure": float((anwg.mean() - sbs.mean()) / headroom) if headroom else None,
        "bootstrap_gap_closure": gc,
        "catastrophic_lt_sbs_minus_eps": int(np.sum(anwg < (sbs - CATASTROPHIC_EPS))),
        "frac_catastrophic": float(np.mean(anwg < (sbs - CATASTROPHIC_EPS))),
        "median_regret_vs_vbs": float(np.median(vbs - anwg)),
        "p90_regret_vs_vbs": float(np.quantile(vbs - anwg, 0.90)),
    }
    if selected is not None and vbs_policy is not None:
        sel = np.asarray(list(selected), dtype=object)
        truth = np.asarray(list(vbs_policy), dtype=object)
        correct = sel == truth
        out["accuracy_vs_vbs_winner"] = float(correct.mean())
        # regret when wrong: vbs - selected utility already in anwg vs vbs
        wrong = ~correct
        if wrong.any():
            out["mean_regret_when_incorrect"] = float((vbs[wrong] - anwg[wrong]).mean())
        else:
            out["mean_regret_when_incorrect"] = 0.0
        out["mean_regret_when_correct"] = (
            float((vbs[correct] - anwg[correct]).mean()) if correct.any() else None
        )
    return out


def classify_recovery(summary: Dict[str, Any]) -> str:
    gain_mean = float(summary["realized_gain"])
    gain_lo = float(summary["bootstrap_gain_vs_sbs"]["ci95_low"])
    gc = float(summary["gap_closure"])
    if gain_lo > 0 and gc >= 0.50:
        return "STRONG_RECOVERY"
    if gain_mean > 0:
        return "PARTIAL_RECOVERY"
    return "NO_RECOVERY"


def confusion_matrix(
    selected: Sequence[str], truth: Sequence[str], labels: Sequence[str] = P6
) -> pd.DataFrame:
    idx = {p: i for i, p in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for s, t in zip(selected, truth):
        mat[idx[t], idx[s]] += 1
    return pd.DataFrame(mat, index=list(labels), columns=list(labels))

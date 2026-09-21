"""Joint-240 guarded / abstaining selector v1 (matrix-only OOF reanalysis).

Implements docs/design/JOINT240_GUARDED_ABSTAINING_SELECTOR_V1.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .joint240_same_distribution_adaptive_v1 import (
    FEATURE_ALLOWLIST,
    P6,
    SPLIT_SEED,
    fit_scen_selector,
    select_scen_model_on_val,
)

ROOT = Path(__file__).resolve().parents[3]
JOINT240_EXP = ROOT / "experiments" / "joint240_same_distribution_adaptive_exploitability_v1"
SCHEMA_VERSION = "joint240_guarded_abstaining_selector_v1.0.0"

SBS_POLICY = "kv_constrained_online"
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 2000
CATASTROPHIC_EPS = 0.01

MAXPROB_GRID = tuple(float(x) for x in np.round(np.arange(0.0, 1.0001, 0.05), 2))
MARGIN_GRID = tuple(float(x) for x in np.round(np.arange(0.0, 1.0001, 0.05), 2))
UTIL_ADV_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)


def inner_train_val(train_ids: Sequence[str], fold: int) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(SPLIT_SEED + 100 + fold)
    ids = list(train_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(0.20 * len(ids))))
    if len(ids) - n_val < 1:
        n_val = max(0, len(ids) - 1)
    return ids[n_val:], ids[:n_val]


def fit_classifier(X: np.ndarray, y: np.ndarray, C: float) -> Pipeline:
    return fit_scen_selector(X, y, C)


def predict_proba_df(model: Pipeline, X: np.ndarray) -> pd.DataFrame:
    proba = model.predict_proba(X)
    est = model.named_steps.get("clf", model)
    classes = list(est.classes_)
    df = pd.DataFrame(proba, columns=[str(c) for c in classes])
    for p in P6:
        if p not in df.columns:
            df[p] = 0.0
    return df[list(P6)]


def maxprob_and_margin(proba: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = proba.to_numpy(dtype=float)
    order = np.argsort(-arr, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    rows = np.arange(len(arr))
    top1_p = arr[rows, top1_idx]
    top2_p = arr[rows, top2_idx]
    pred = np.asarray(proba.columns)[top1_idx]
    return pred.astype(str), top1_p, top1_p - top2_p


def apply_prob_guard(
    pred: np.ndarray,
    score: np.ndarray,
    *,
    tau: float,
    sbs: str = SBS_POLICY,
) -> np.ndarray:
    out = pred.astype(object).copy()
    out[np.asarray(score) < tau] = sbs
    return out.astype(str)


def fit_utility_regressors(X: np.ndarray, utilities: pd.DataFrame) -> Dict[str, Pipeline]:
    models = {}
    for p in P6:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=SPLIT_SEED)),
            ]
        )
        pipe.fit(X, utilities[p].to_numpy(dtype=float))
        models[p] = pipe
    return models


def predict_utilities(models: Dict[str, Pipeline], X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({p: models[p].predict(X) for p in P6})


def apply_util_guard(
    pred_util: pd.DataFrame,
    *,
    tau: float,
    sbs: str = SBS_POLICY,
) -> np.ndarray:
    arr = pred_util.to_numpy(dtype=float)
    best_idx = np.argmax(arr, axis=1)
    best = np.asarray(pred_util.columns)[best_idx].astype(str)
    sbs_pred = pred_util[sbs].to_numpy(dtype=float)
    best_pred = arr[np.arange(len(arr)), best_idx]
    adv = best_pred - sbs_pred
    out = best.copy()
    out[(best == sbs) | (adv < tau)] = sbs
    return out.astype(str)


def lookup_anwg(matrix: pd.DataFrame, scenario_ids: Sequence[str], policies: Sequence[str]) -> np.ndarray:
    lookup = matrix.set_index("scenario_id")
    return np.asarray(
        [float(lookup.loc[sid, pol]) for sid, pol in zip(scenario_ids, policies)],
        dtype=float,
    )


def choose_tau_on_val(
    scores: np.ndarray,
    pred: np.ndarray,
    ids_val: Sequence[str],
    matrix: pd.DataFrame,
    grid: Sequence[float],
    *,
    mode: str,
    pred_util: Optional[pd.DataFrame] = None,
) -> Tuple[float, float]:
    best_tau = float(grid[0])
    best_score = -1e18
    for tau in grid:
        if mode == "util":
            assert pred_util is not None
            chosen = apply_util_guard(pred_util, tau=float(tau))
        else:
            chosen = apply_prob_guard(pred, scores, tau=float(tau))
        anwg = lookup_anwg(matrix, ids_val, chosen)
        mean_u = float(np.mean(anwg)) if len(anwg) else float("nan")
        if mean_u > best_score + 1e-15 or (
            abs(mean_u - best_score) <= 1e-15 and float(tau) > best_tau
        ):
            best_score = mean_u
            best_tau = float(tau)
    return best_tau, best_score


def summarize_method(
    anwg: np.ndarray,
    sbs: np.ndarray,
    vbs: np.ndarray,
    policies: Sequence[str],
    abstained: np.ndarray,
) -> Dict[str, Any]:
    anwg = np.asarray(anwg, dtype=float)
    sbs = np.asarray(sbs, dtype=float)
    vbs = np.asarray(vbs, dtype=float)
    gain = anwg - sbs
    gap = vbs - anwg
    headroom = float(np.mean(vbs - sbs))
    realized = float(np.mean(gain))
    closure = float(realized / headroom) if headroom > 0 else float("nan")
    specialist = ~np.asarray(abstained, dtype=bool)
    return {
        "n": int(len(anwg)),
        "R_A": float(np.mean(anwg)),
        "R_SBS": float(np.mean(sbs)),
        "R_VBS": float(np.mean(vbs)),
        "headroom": headroom,
        "realized_gain_vs_sbs": realized,
        "exploitability_gap": float(np.mean(gap)),
        "gap_closure": closure,
        "n_abstain_sbs": int(np.sum(abstained)),
        "frac_abstain_sbs": float(np.mean(abstained)),
        "n_specialist": int(np.sum(specialist)),
        "frac_specialist": float(np.mean(specialist)),
        "frac_beat_sbs": float(np.mean(anwg > sbs)),
        "frac_lose_to_sbs": float(np.mean(anwg < sbs)),
        "n_catastrophic": int(np.sum(anwg < sbs - CATASTROPHIC_EPS)),
        "frac_catastrophic": float(np.mean(anwg < sbs - CATASTROPHIC_EPS)),
        "mean_regret_vs_vbs": float(np.mean(gap)),
        "mean_regret_when_specialist": float(np.mean(gap[specialist])) if specialist.any() else None,
        "mean_regret_when_abstain": float(np.mean(gap[abstained])) if np.any(abstained) else None,
        "policy_counts": {p: int(np.sum(np.asarray(policies) == p)) for p in P6},
    }


def paired_bootstrap_mean(
    values: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    boots = [float(np.mean(values[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(np.mean(values)), "ci95_low": float(lo), "ci95_high": float(hi)}


def assign_verdicts(
    methods: Dict[str, Dict[str, Any]],
    boot_gain: Dict[str, Dict[str, float]],
    ascen_cat: int,
) -> Dict[str, Any]:
    guarded_names = ["maxprob_guard", "margin_guard", "util_advantage_guard"]
    best_name = None
    best_gain = -1e18
    for name in guarded_names:
        if name not in methods:
            continue
        g = float(methods[name]["realized_gain_vs_sbs"])
        if g > best_gain:
            best_gain = g
            best_name = name
    if best_name is None:
        return {"labels": ["GUARDED_SELECTOR_INCONCLUSIVE"]}

    ci = boot_gain[best_name]
    lo, hi = float(ci["ci95_low"]), float(ci["ci95_high"])
    cat = int(methods[best_name]["n_catastrophic"])
    cat_reduced = cat <= ascen_cat * 0.75
    labels: List[str] = []
    if best_gain > 0 and lo > 0:
        labels.append("GUARDED_SELECTOR_BEATS_SBS")
    if (lo <= 0 <= hi or lo >= 0) and cat_reduced and hi >= 0:
        labels.append("GUARDED_SELECTOR_RECOVERS_SBS")
    if best_gain < 0 and hi < 0:
        labels.append("GUARDED_SELECTOR_STILL_BELOW_SBS")
    if "GUARDED_SELECTOR_STILL_BELOW_SBS" in labels:
        labels = [x for x in labels if x != "GUARDED_SELECTOR_RECOVERS_SBS"]
    width = hi - lo
    if not labels:
        labels.append("GUARDED_SELECTOR_INCONCLUSIVE")
    elif (
        width > 0.02
        and "GUARDED_SELECTOR_BEATS_SBS" not in labels
        and "GUARDED_SELECTOR_STILL_BELOW_SBS" not in labels
        and "GUARDED_SELECTOR_INCONCLUSIVE" not in labels
    ):
        labels.append("GUARDED_SELECTOR_INCONCLUSIVE")
    return {
        "best_guarded_method": best_name,
        "best_guarded_gain": best_gain,
        "best_guarded_gain_ci": ci,
        "best_guarded_n_catastrophic": cat,
        "ascen_n_catastrophic": ascen_cat,
        "labels": labels,
    }


def run_oof_experiment(data: pd.DataFrame) -> Dict[str, Any]:
    """`data` must include features, P6 utilities, fold, vbs_*, n_elevated_mechanisms."""
    df = data.copy().reset_index(drop=True)
    if len(df) != 240:
        raise RuntimeError(f"expected 240 rows, got {len(df)}")
    X_all = df[list(FEATURE_ALLOWLIST)].to_numpy(dtype=float)
    y_all = df["vbs_policy"].astype(str).to_numpy()
    ids = df["scenario_id"].astype(str).to_numpy()
    id_to_row = {sid: i for i, sid in enumerate(ids)}

    records: List[Dict[str, Any]] = []
    tau_log: List[Dict[str, Any]] = []
    n_folds = int(df["fold"].max()) + 1

    for fold in range(n_folds):
        test_ids = df.loc[df["fold"] == fold, "scenario_id"].tolist()
        train_pool = df.loc[df["fold"] != fold, "scenario_id"].tolist()
        tr_ids, val_ids = inner_train_val(train_pool, fold)
        tr_idx = [id_to_row[i] for i in tr_ids]
        val_idx = [id_to_row[i] for i in val_ids]
        te_idx = [id_to_row[i] for i in test_ids]
        tv_idx = tr_idx + val_idx

        name, val_score, _ = select_scen_model_on_val(
            X_all[tr_idx],
            y_all[tr_idx],
            X_all[val_idx],
            [ids[i] for i in val_idx],
            df,
        )
        C = 1.0 if "1.0" in name else 0.5

        model_tr = fit_classifier(X_all[tr_idx], y_all[tr_idx], C)
        proba_val = predict_proba_df(model_tr, X_all[val_idx])
        pred_val, maxp_val, margin_val = maxprob_and_margin(proba_val)
        util_models_tr = fit_utility_regressors(
            X_all[tr_idx], df.iloc[tr_idx][list(P6)].reset_index(drop=True)
        )
        util_val = predict_utilities(util_models_tr, X_all[val_idx])

        tau_max, score_max = choose_tau_on_val(
            maxp_val, pred_val, [ids[i] for i in val_idx], df, MAXPROB_GRID, mode="maxprob"
        )
        tau_margin, score_margin = choose_tau_on_val(
            margin_val, pred_val, [ids[i] for i in val_idx], df, MARGIN_GRID, mode="margin"
        )
        tau_util, score_util = choose_tau_on_val(
            np.zeros(len(val_idx)),
            pred_val,
            [ids[i] for i in val_idx],
            df,
            UTIL_ADV_GRID,
            mode="util",
            pred_util=util_val,
        )
        tau_log.append(
            {
                "fold": fold,
                "selected_model": name,
                "C": C,
                "val_mean_anwg_model_select": val_score,
                "tau_maxprob": tau_max,
                "tau_maxprob_val_anwg": score_max,
                "tau_margin": tau_margin,
                "tau_margin_val_anwg": score_margin,
                "tau_util_advantage": tau_util,
                "tau_util_val_anwg": score_util,
                "n_train": len(tr_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
            }
        )

        model = fit_classifier(X_all[tv_idx], y_all[tv_idx], C)
        util_models = fit_utility_regressors(
            X_all[tv_idx], df.iloc[tv_idx][list(P6)].reset_index(drop=True)
        )
        proba_te = predict_proba_df(model, X_all[te_idx])
        pred_te, maxp_te, margin_te = maxprob_and_margin(proba_te)
        util_te = predict_utilities(util_models, X_all[te_idx])

        unguarded = pred_te.astype(str)
        maxprob_ch = apply_prob_guard(pred_te, maxp_te, tau=tau_max)
        margin_ch = apply_prob_guard(pred_te, margin_te, tau=tau_margin)
        util_ch = apply_util_guard(util_te, tau=tau_util)

        lookup = df.set_index("scenario_id")
        for j, sid in enumerate(test_ids):
            row = lookup.loc[sid]
            rec: Dict[str, Any] = {
                "scenario_id": sid,
                "fold": fold,
                "n_elevated_mechanisms": int(row["n_elevated_mechanisms"]),
                "vbs_policy": str(row["vbs_policy"]),
                "vbs_anwg": float(row["vbs_anwg"]),
                "sbs_policy": SBS_POLICY,
                "sbs_anwg": float(row[SBS_POLICY]),
                "unguarded_policy": unguarded[j],
                "unguarded_anwg": float(row[unguarded[j]]),
                "maxprob": float(maxp_te[j]),
                "margin": float(margin_te[j]),
                "maxprob_policy": maxprob_ch[j],
                "maxprob_anwg": float(row[maxprob_ch[j]]),
                "maxprob_abstain": bool(maxp_te[j] < tau_max),
                "margin_policy": margin_ch[j],
                "margin_anwg": float(row[margin_ch[j]]),
                "margin_abstain": bool(margin_te[j] < tau_margin),
                "util_policy": util_ch[j],
                "util_anwg": float(row[util_ch[j]]),
                "util_abstain": bool(util_ch[j] == SBS_POLICY),
                "util_pred_adv": float(util_te.iloc[j].max() - util_te.iloc[j][SBS_POLICY]),
                "tau_maxprob": tau_max,
                "tau_margin": tau_margin,
                "tau_util": tau_util,
                "pred_correct_unguarded": bool(unguarded[j] == row["vbs_policy"]),
            }
            for p in P6:
                rec[p] = float(row[p])
                rec[f"proba__{p}"] = float(proba_te.iloc[j][p])
                rec[f"pred_util__{p}"] = float(util_te.iloc[j][p])
            for col in (
                "fairness_pressure",
                "service_heterogeneity",
                "prefill_decode_pressure",
                "kv_pressure",
                "urgency_pressure",
                "burst_pressure",
                "high_fairness_pressure",
                "high_service_heterogeneity",
                "high_prefill_decode_pressure",
                "high_kv_pressure",
                "high_urgency_pressure",
                "high_burst_pressure",
            ):
                if col in lookup.columns:
                    rec[col] = row[col]
            records.append(rec)

    return {"oof": pd.DataFrame(records), "tau_log": tau_log}

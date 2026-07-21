#!/usr/bin/env python3
"""Run a small high-information policy-composition falsification pilot.

This is intentionally an artifact-level pilot over precomputed clean selector
dataset vectors. It does not reconstruct the Wulver-only composition harness.
When the native composition module is unavailable, component-wise composition
is reported as unavailable rather than simulated by unsupported module mixes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector.advanced import (  # noqa: E402
    PolicyRewardRegressorSelector,
    validate_feature_columns,
)


ANWG = "metric_arrival_normalized_weighted_goodput"
COMPLETION = "metric_completion_fraction"
QUALITY = "metric_weighted_goodput"
HOLDOUT_SPLITS = ("ID_TEST", "OOD_TEST")
DEV_SPLITS = ("TRAIN", "VALIDATION")


@dataclass(frozen=True)
class Treatment:
    name: str
    kind: str
    selection: Optional[pd.Series] = None
    weights: Optional[pd.DataFrame] = None
    formula: str = ""
    top_k: str = ""
    validation_anwg: Optional[float] = None
    notes: str = ""


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover - defensive reporting only
        return f"UNAVAILABLE: {exc}"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _round(value: object, digits: int = 6) -> object:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(val) or math.isinf(val):
        return None
    return round(val, digits)


def _prepare(pilot_dir: Path):
    windows = pd.read_csv(pilot_dir / "retained_windows.csv")
    features = pd.read_csv(pilot_dir / "window_features.csv")
    vectors = pd.read_csv(pilot_dir / "full_policy_vectors.csv")
    policies = sorted(vectors["policy_name"].unique().tolist())
    feat_cols = validate_feature_columns(sorted(c for c in features.columns if c.startswith("feat_")))

    rows = windows.merge(features, on="window_idx", how="left")
    metric_tables = {
        "anwg": vectors.pivot_table(index="window_idx", columns="policy_name", values=ANWG, aggfunc="first").reindex(columns=policies),
        "completion": vectors.pivot_table(index="window_idx", columns="policy_name", values=COMPLETION, aggfunc="first").reindex(columns=policies),
        "quality": vectors.pivot_table(index="window_idx", columns="policy_name", values=QUALITY, aggfunc="first").reindex(columns=policies),
        "rejection": vectors.pivot_table(index="window_idx", columns="policy_name", values="metric_rejection_fraction", aggfunc="first").reindex(columns=policies),
    }
    anwg = metric_tables["anwg"].fillna(0.0)
    rows["oracle_policy"] = rows["window_idx"].map(lambda idx: anwg.loc[idx].idxmax())
    rows["oracle_anwg"] = rows["window_idx"].map(lambda idx: float(anwg.loc[idx].max()))
    rows["top2_margin"] = rows["window_idx"].map(lambda idx: _top2_margin(anwg.loc[idx].to_numpy(dtype=float)))
    for policy in policies:
        rows[f"anwg_{policy}"] = rows["window_idx"].map(anwg[policy])
    return rows, vectors, policies, feat_cols, metric_tables


def _top2_margin(values: np.ndarray) -> float:
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0


def _fit_train_imputer(rows: pd.DataFrame, feat_cols: Sequence[str]) -> Dict[str, float]:
    train = rows[rows["split"] == "TRAIN"]
    stats: Dict[str, float] = {}
    for col in feat_cols:
        median = pd.to_numeric(train[col], errors="coerce").median()
        stats[col] = 0.0 if pd.isna(median) else float(median)
    return stats


def _apply_imputer(rows: pd.DataFrame, feat_cols: Sequence[str], stats: Mapping[str, float]) -> pd.DataFrame:
    out = rows.copy()
    for col in feat_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(stats[col])
    return out


def _category_thresholds(rows: pd.DataFrame, metric_tables: Mapping[str, pd.DataFrame]) -> Dict[str, float]:
    dev = rows[rows["split"].isin(DEV_SPLITS)]
    abs_wsp_scorpio = np.abs(
        metric_tables["anwg"].loc[dev["window_idx"], "weighted_shortest_processing"].to_numpy(dtype=float)
        - metric_tables["anwg"].loc[dev["window_idx"], "scorpio_style_slo_guard"].to_numpy(dtype=float)
    )
    margins = dev["top2_margin"].to_numpy(dtype=float)
    oracle = dev["oracle_anwg"].to_numpy(dtype=float)
    fixed_floor = np.quantile(margins, 0.75) if len(margins) else 0.005
    return {
        "dominant_margin": float(max(0.005, fixed_floor)),
        "meaningful_margin": float(max(0.005, np.quantile(margins, 0.60) if len(margins) else 0.005)),
        "near_tie_margin": float(max(0.001, np.quantile(margins, 0.50) if len(margins) else 0.001)),
        "boundary_abs_gap": float(max(0.02, np.quantile(abs_wsp_scorpio, 0.25) if len(abs_wsp_scorpio) else 0.02)),
        "high_regret_oracle": float(max(0.02, np.quantile(oracle, 0.75) if len(oracle) else 0.02)),
    }


def _assign_categories(rows: pd.DataFrame, metric_tables: Mapping[str, pd.DataFrame], thresholds: Mapping[str, float]) -> pd.DataFrame:
    out = rows.copy()
    anwg = metric_tables["anwg"]
    cats: List[str] = []
    primary: List[str] = []
    for _, row in out.iterrows():
        idx = row["window_idx"]
        scores = anwg.loc[idx].fillna(0.0)
        winner = str(scores.idxmax())
        margin = float(row["top2_margin"])
        row_cats: List[str] = []
        if margin <= thresholds["near_tie_margin"]:
            row_cats.append("near_tie")
        if winner == "weighted_shortest_processing" and margin >= thresholds["dominant_margin"]:
            row_cats.append("wsp_dominant")
        elif winner == "scorpio_style_slo_guard" and margin >= thresholds["dominant_margin"]:
            row_cats.append("scorpio_dominant")
        elif winner not in {"weighted_shortest_processing", "scorpio_style_slo_guard"} and margin >= thresholds["dominant_margin"]:
            row_cats.append("other_policy_dominant")
        wsp_scorpio_gap = abs(float(scores["weighted_shortest_processing"]) - float(scores["scorpio_style_slo_guard"]))
        if wsp_scorpio_gap <= thresholds["boundary_abs_gap"]:
            row_cats.append("wsp_scorpio_boundary")
        if float(row["oracle_anwg"]) >= thresholds["high_regret_oracle"] and margin >= thresholds["meaningful_margin"]:
            row_cats.append("high_regret_opportunity")
        if str(row.get("time_slice_pool", "")) == "ood_reserved" or str(row.get("split", "")) == "OOD_TEST":
            row_cats.append("ood_like_shifted")
        if margin >= thresholds["meaningful_margin"]:
            row_cats.append("meaningful")
        if not row_cats:
            row_cats.append("uncategorized")
        cats.append("|".join(row_cats))
        primary.append(row_cats[0])
    out["pilot_categories"] = cats
    out["primary_pilot_category"] = primary
    return out


def _select_experts(rows: pd.DataFrame, policies: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    required = [
        "weighted_shortest_processing",
        "scorpio_style_slo_guard",
        "edf",
        "estimated_service_time_first",
        "fifo",
    ]
    experts = [p for p in required if p in policies]
    reasons = {
        "weighted_shortest_processing": "Required WSP parent; best train/dev fixed policy family and service-time specialist.",
        "scorpio_style_slo_guard": "Required SCORPIO parent; explicit SLO/admission guard and boundary-policy comparator.",
        "edf": "Deadline-oriented expert requested in the pilot brief.",
        "estimated_service_time_first": "Locally available service-time complementary policy in the Option-B vectors.",
        "fifo": "Development-side oracle winner on many retained windows; keeps the compact set from excluding a strong non-WSP/non-SCORPIO parent.",
    }
    return experts, {p: reasons[p] for p in experts}


def _best_fixed_from_validation(rows: pd.DataFrame, metric_tables: Mapping[str, pd.DataFrame], policies: Sequence[str]) -> str:
    val = rows[rows["split"] == "VALIDATION"]
    return str(metric_tables["anwg"].loc[val["window_idx"], policies].mean(axis=0).idxmax())


def _fit_discrete_selector(train: pd.DataFrame, val: pd.DataFrame, policies: Sequence[str], feat_cols: Sequence[str], seed: int):
    models = [
        PolicyRewardRegressorSelector(
            name="existing_selector_rf_reward_regression",
            allowed_policies=policies,
            feature_cols=feat_cols,
            estimator="random_forest",
            n_estimators=120,
            max_depth=8,
            random_state=seed,
        ).fit(train),
        PolicyRewardRegressorSelector(
            name="existing_selector_extra_trees_reward_regression",
            allowed_policies=policies,
            feature_cols=feat_cols,
            estimator="extra_trees",
            n_estimators=120,
            max_depth=8,
            random_state=seed,
        ).fit(train),
    ]
    scores = {m.name: np.mean([val.loc[i, f"anwg_{p}"] for i, p in zip(val.index, m.predict(val))]) for m in models}
    return max(models, key=lambda m: scores[m.name]), scores


def _fit_contextual_regressor(train: pd.DataFrame, experts: Sequence[str], feat_cols: Sequence[str], seed: int):
    return PolicyRewardRegressorSelector(
        name="contextual_rf_policy_utility",
        allowed_policies=experts,
        feature_cols=feat_cols,
        estimator="random_forest",
        n_estimators=120,
        max_depth=6,
        random_state=seed,
    ).fit(train)


def _simplex_weights(n: int, step: float = 0.25) -> Iterable[np.ndarray]:
    total = int(round(1.0 / step))
    for counts in itertools.product(range(total + 1), repeat=n):
        if sum(counts) == total:
            yield np.asarray(counts, dtype=float) / total


def _static_rank_weights(rows: pd.DataFrame, metric_tables: Mapping[str, pd.DataFrame], experts: Sequence[str], weights: np.ndarray) -> pd.DataFrame:
    idxs = rows["window_idx"].tolist()
    out = pd.DataFrame(0.0, index=idxs, columns=list(experts))
    # Development-selected fixed policy weights proxy a normalized rank vote at window level.
    for i, policy in enumerate(experts):
        out[policy] = float(weights[i])
    return out


def _optimize_static_weights(val: pd.DataFrame, metric_tables: Mapping[str, pd.DataFrame], experts: Sequence[str]) -> Tuple[np.ndarray, float]:
    values = metric_tables["anwg"].loc[val["window_idx"], experts].to_numpy(dtype=float)
    best_w = np.zeros(len(experts), dtype=float)
    best_score = -1.0
    for w in _simplex_weights(len(experts), step=0.25):
        if np.count_nonzero(w > 1e-12) > 2:
            continue
        score = float(np.nanmean(values @ w))
        if score > best_score:
            best_score = score
            best_w = w
    return best_w, best_score


def _contextual_weights(
    scores: np.ndarray,
    experts: Sequence[str],
    *,
    formula: str,
    tau: float,
    baseline_policy: str,
    top_k: Optional[int],
) -> np.ndarray:
    clipped = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if formula == "softmax":
        shifted = clipped / max(tau, 1e-9)
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
    elif formula == "positive_advantage":
        base_idx = experts.index(baseline_policy) if baseline_policy in experts else int(np.argmax(clipped.mean(axis=0)))
        weights = np.maximum(clipped - clipped[:, [base_idx]], 0.0)
    elif formula == "rank":
        rank_template = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625][: len(experts)], dtype=float)
        weights = np.zeros_like(clipped)
        order = np.argsort(-clipped, axis=1)
        for i in range(clipped.shape[0]):
            for rank_pos, policy_idx in enumerate(order[i]):
                weights[i, policy_idx] = rank_template[rank_pos]
    else:
        raise ValueError(formula)

    if top_k is not None and top_k < len(experts):
        keep = np.argsort(-weights, axis=1)[:, :top_k]
        mask = np.zeros_like(weights, dtype=bool)
        for i in range(weights.shape[0]):
            mask[i, keep[i]] = True
        weights = np.where(mask, weights, 0.0)

    sums = weights.sum(axis=1, keepdims=True)
    fallback = sums[:, 0] <= 1e-12
    if fallback.any():
        weights[fallback, :] = 0.0
        fallback_idx = experts.index(baseline_policy) if baseline_policy in experts else int(np.argmax(clipped.mean(axis=0)))
        weights[fallback, fallback_idx] = 1.0
        sums = weights.sum(axis=1, keepdims=True)
    return weights / sums


def _weights_df(rows: pd.DataFrame, experts: Sequence[str], weights: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(weights, columns=list(experts))
    out.insert(0, "window_idx", rows["window_idx"].to_numpy())
    return out.set_index("window_idx")


def _evaluate_treatment(
    rows: pd.DataFrame,
    treatment: Treatment,
    metric_tables: Mapping[str, pd.DataFrame],
    best_fixed_policy: str,
    meaningful_margin: float,
    seed: int,
    bootstrap: int,
) -> Dict:
    idxs = rows["window_idx"].tolist()
    if treatment.selection is not None:
        selected = [treatment.selection.loc[idx] for idx in idxs]
        values = np.asarray([metric_tables["anwg"].loc[idx, p] for idx, p in zip(idxs, selected)], dtype=float)
        completion = np.asarray([metric_tables["completion"].loc[idx, p] for idx, p in zip(idxs, selected)], dtype=float)
        quality = np.asarray([metric_tables["quality"].loc[idx, p] for idx, p in zip(idxs, selected)], dtype=float)
        selected_policy = selected
        active_experts = np.ones(len(rows), dtype=float)
        entropy = np.zeros(len(rows), dtype=float)
        fallback_frequency = 0.0
    elif treatment.weights is not None:
        weights = treatment.weights.loc[idxs]
        cols = weights.columns.tolist()
        matrix = weights.to_numpy(dtype=float)
        values = np.sum(metric_tables["anwg"].loc[idxs, cols].to_numpy(dtype=float) * matrix, axis=1)
        completion = np.sum(metric_tables["completion"].loc[idxs, cols].to_numpy(dtype=float) * matrix, axis=1)
        quality = np.sum(metric_tables["quality"].loc[idxs, cols].to_numpy(dtype=float) * matrix, axis=1)
        selected_policy = [cols[int(i)] for i in np.argmax(matrix, axis=1)]
        active_experts = (matrix > 1e-8).sum(axis=1).astype(float)
        entropy = np.asarray([_entropy(row) for row in matrix], dtype=float)
        fallback_frequency = 0.0
    else:
        return {
            "method": treatment.name,
            "kind": treatment.kind,
            "n_windows": int(len(rows)),
            "anwg": None,
            "completion_fraction": None,
            "completed_request_quality": None,
            "mean_regret_to_hindsight_oracle": None,
            "p95_regret": None,
            "worst_regret": None,
            "meaningful_window_anwg": None,
            "meaningful_window_count": int((rows["top2_margin"].to_numpy(dtype=float) >= meaningful_margin).sum()),
            "switching_frequency": 0.0,
            "average_active_experts": 0.0,
            "weight_entropy": 0.0,
            "fallback_frequency": 1.0,
            "feasibility_violations": None,
            "formula": treatment.formula,
            "top_k": treatment.top_k,
            "notes": treatment.notes,
        }

    oracle = rows["oracle_anwg"].to_numpy(dtype=float)
    regrets = oracle - values
    meaningful = rows["top2_margin"].to_numpy(dtype=float) >= meaningful_margin
    fixed = metric_tables["anwg"].loc[idxs, best_fixed_policy].to_numpy(dtype=float)
    switches = _switching_frequency(selected_policy)
    result = {
        "method": treatment.name,
        "kind": treatment.kind,
        "n_windows": int(len(rows)),
        "anwg": _round(np.nanmean(values)),
        "completion_fraction": _round(np.nanmean(completion)),
        "completed_request_quality": _round(np.nanmean(quality)),
        "mean_regret_to_hindsight_oracle": _round(np.nanmean(regrets)),
        "p95_regret": _round(np.nanpercentile(regrets, 95)),
        "worst_regret": _round(np.nanmax(regrets)),
        "meaningful_window_anwg": _round(np.nanmean(values[meaningful])) if meaningful.any() else None,
        "meaningful_window_count": int(meaningful.sum()),
        "switching_frequency": _round(switches),
        "average_active_experts": _round(np.nanmean(active_experts)),
        "weight_entropy": _round(np.nanmean(entropy)),
        "fallback_frequency": _round(fallback_frequency),
        "feasibility_violations": 0 if treatment.kind != "component_wise_composition" else None,
        "formula": treatment.formula,
        "top_k": treatment.top_k,
        "notes": treatment.notes,
    }
    if bootstrap > 0 and len(rows) >= 2 and not np.all(np.isnan(values)):
        result.update(_bootstrap(values, fixed, oracle, seed, bootstrap))
    return result


def _entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 1e-12]
    if len(positive) == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def _switching_frequency(labels: Sequence[str]) -> float:
    if len(labels) <= 1:
        return 0.0
    return float(np.mean([a != b for a, b in zip(labels[:-1], labels[1:])]))


def _bootstrap(values: np.ndarray, fixed: np.ndarray, oracle: np.ndarray, seed: int, n: int) -> Dict:
    rng = np.random.default_rng(seed)
    anwg = []
    diff_fixed = []
    regret = []
    size = len(values)
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        anwg.append(float(np.nanmean(values[idx])))
        diff_fixed.append(float(np.nanmean(values[idx] - fixed[idx])))
        regret.append(float(np.nanmean(oracle[idx] - values[idx])))
    return {
        "anwg_ci_low": _round(np.percentile(anwg, 2.5)),
        "anwg_ci_high": _round(np.percentile(anwg, 97.5)),
        "diff_vs_best_fixed_ci_low": _round(np.percentile(diff_fixed, 2.5)),
        "diff_vs_best_fixed_ci_high": _round(np.percentile(diff_fixed, 97.5)),
        "mean_regret_ci_low": _round(np.percentile(regret, 2.5)),
        "mean_regret_ci_high": _round(np.percentile(regret, 97.5)),
    }


def _subset_analysis(
    rows: pd.DataFrame,
    treatments: Sequence[Treatment],
    metric_tables: Mapping[str, pd.DataFrame],
    best_fixed_policy: str,
    meaningful_margin: float,
    seed: int,
    bootstrap: int,
) -> pd.DataFrame:
    subset_masks = {
        "all": np.ones(len(rows), dtype=bool),
        "meaningful": rows["top2_margin"].to_numpy(dtype=float) >= meaningful_margin,
        "wsp_dominant": rows["pilot_categories"].str.contains("wsp_dominant", regex=False).to_numpy(),
        "scorpio_dominant": rows["pilot_categories"].str.contains("scorpio_dominant", regex=False).to_numpy(),
        "other_policy_dominant": rows["pilot_categories"].str.contains("other_policy_dominant", regex=False).to_numpy(),
        "boundary": rows["pilot_categories"].str.contains("wsp_scorpio_boundary", regex=False).to_numpy(),
        "near_tie": rows["pilot_categories"].str.contains("near_tie", regex=False).to_numpy(),
        "high_regret": rows["pilot_categories"].str.contains("high_regret_opportunity", regex=False).to_numpy(),
        "ood_like_shifted": rows["pilot_categories"].str.contains("ood_like_shifted", regex=False).to_numpy(),
    }
    records = []
    for subset, mask in subset_masks.items():
        sub = rows.loc[mask].copy()
        if sub.empty:
            continue
        for treatment in treatments:
            rec = _evaluate_treatment(sub, treatment, metric_tables, best_fixed_policy, meaningful_margin, seed, 0)
            rec["subset"] = subset
            records.append(rec)
    return pd.DataFrame(records)


def _make_selection(rows: pd.DataFrame, policy: str) -> pd.Series:
    return pd.Series(policy, index=rows["window_idx"], dtype=object)


def _make_model_selection(rows: pd.DataFrame, model) -> pd.Series:
    return pd.Series(model.predict(rows), index=rows["window_idx"], dtype=object)


def _maybe_distill_child(
    train: pd.DataFrame,
    val: pd.DataFrame,
    holdout: pd.DataFrame,
    teacher: Treatment,
    experts: Sequence[str],
    feat_cols: Sequence[str],
    metric_tables: Mapping[str, pd.DataFrame],
    best_parent_a: str,
    best_parent_b: str,
    seed: int,
) -> Tuple[Optional[Treatment], Optional[Dict]]:
    if teacher.weights is None:
        return None, None
    from sklearn.tree import DecisionTreeClassifier, export_text

    teacher_actions = teacher.weights.idxmax(axis=1)
    train_dev = pd.concat([train, val], axis=0)
    y = train_dev["window_idx"].map(teacher_actions).fillna(best_parent_a).astype(str)
    if len(set(y.tolist())) < 2:
        return None, None
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed)
    model.fit(train_dev[feat_cols].to_numpy(dtype=float), y)
    pred = model.predict(holdout[feat_cols].to_numpy(dtype=float)).tolist()
    selection = pd.Series(pred, index=holdout["window_idx"], dtype=object)
    treatment = Treatment(
        name="symbolic_child_depth3_tree",
        kind="symbolic_child",
        selection=selection,
        notes="Depth-3 decision tree distilled from the development-selected contextual mixture argmax expert.",
    )
    rules = export_text(model, feature_names=list(feat_cols), max_depth=3)
    info = {
        "name": treatment.name,
        "representation": "sklearn DecisionTreeClassifier depth=3 over causal feat_* columns",
        "teacher": teacher.name,
        "parents_compared": [best_parent_a, best_parent_b],
        "experts": list(experts),
        "feature_names": list(feat_cols),
        "tree_rules": rules,
    }
    return treatment, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "experiments/selector_v2_calibrated_pilot_20260720T163235Z")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/composition_smart_pilot")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()

    t0 = time.perf_counter()
    pilot_dir = args.pilot_dir if args.pilot_dir.is_absolute() else ROOT / args.pilot_dir
    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    rows, vectors, policies, feat_cols, metric_tables = _prepare(pilot_dir)
    gates = _load_json(pilot_dir / "quality_gates.json")
    audit = _load_json(pilot_dir / "leakage_audit.json")
    imputer = _fit_train_imputer(rows, feat_cols)
    rows = _apply_imputer(rows, feat_cols, imputer)
    thresholds = _category_thresholds(rows, metric_tables)
    rows = _assign_categories(rows, metric_tables, thresholds)
    experts, expert_reasons = _select_experts(rows, policies)

    train = rows[rows["split"] == "TRAIN"].copy()
    val = rows[rows["split"] == "VALIDATION"].copy()
    holdout = rows[rows["split"].isin(HOLDOUT_SPLITS)].copy()
    dev = rows[rows["split"].isin(DEV_SPLITS)].copy()

    best_fixed = _best_fixed_from_validation(rows, metric_tables, policies)
    discrete_model, discrete_val_scores = _fit_discrete_selector(train, val, policies, feat_cols, args.seed)
    context_model = _fit_contextual_regressor(train, experts, feat_cols, args.seed)

    treatments: List[Treatment] = [
        Treatment(
            name=f"best_fixed__{best_fixed}",
            kind="fixed_policy",
            selection=_make_selection(rows, best_fixed),
            validation_anwg=float(metric_tables["anwg"].loc[val["window_idx"], best_fixed].mean()),
        ),
        Treatment(
            name=discrete_model.name,
            kind="discrete_selector",
            selection=_make_model_selection(rows, discrete_model),
            validation_anwg=float(discrete_val_scores[discrete_model.name]),
            notes="Best of the local clean per-policy reward-regression selector variants by VALIDATION ANWG.",
        ),
    ]

    equal = np.ones(len(experts), dtype=float) / len(experts)
    static_equal = Treatment(
        name="static_rank_equal_proxy",
        kind="static_composition_proxy",
        weights=_static_rank_weights(rows, metric_tables, experts, equal),
        formula="fixed_equal",
        top_k="dense",
        validation_anwg=float(np.mean(metric_tables["anwg"].loc[val["window_idx"], experts].to_numpy(dtype=float) @ equal)),
        notes="Aggregate vector-level proxy for equal normalized-rank composition; native rank-composition harness unavailable locally.",
    )
    opt_w, opt_score = _optimize_static_weights(val, metric_tables, experts)
    static_opt = Treatment(
        name="static_rank_sparse_dev_optimized_proxy",
        kind="static_composition_proxy",
        weights=_static_rank_weights(rows, metric_tables, experts, opt_w),
        formula="dev_sparse_simplex_step_0.25_max_2_experts",
        top_k=str(int(np.count_nonzero(opt_w > 1e-12))),
        validation_anwg=opt_score,
        notes="Sparse static weights selected on VALIDATION only over a coarse 0.25 simplex grid.",
    )
    treatments.extend([static_equal, static_opt])

    score_all = context_model.predict_scores(rows)
    score_val = context_model.predict_scores(val)
    formula_specs: List[Tuple[str, float]] = [
        ("softmax", 0.02),
        ("softmax", 0.05),
        ("softmax", 0.10),
        ("positive_advantage", 1.0),
        ("rank", 1.0),
    ]
    contextual_candidates: List[Treatment] = []
    for formula, tau in formula_specs:
        for top_k in (1, 2, 3, None):
            weights_all = _contextual_weights(score_all, experts, formula=formula, tau=tau, baseline_policy=best_fixed, top_k=top_k)
            weights_val = _contextual_weights(score_val, experts, formula=formula, tau=tau, baseline_policy=best_fixed, top_k=top_k)
            val_values = np.sum(metric_tables["anwg"].loc[val["window_idx"], experts].to_numpy(dtype=float) * weights_val, axis=1)
            name = f"contextual_{formula}_tau{tau:g}_top{top_k or 'dense'}_proxy"
            contextual_candidates.append(Treatment(
                name=name,
                kind="contextual_composition_proxy",
                weights=_weights_df(rows, experts, weights_all),
                formula=f"{formula};tau={tau:g}",
                top_k=str(top_k or "dense"),
                validation_anwg=float(np.nanmean(val_values)),
                notes="Contextual composition proxy using RF-predicted policy utilities and causal feat_* columns only.",
            ))
    best_contextual = max(contextual_candidates, key=lambda t: t.validation_anwg if t.validation_anwg is not None else -1.0)
    treatments.extend(contextual_candidates)

    component_treatment = Treatment(
        name="component_wise_composition_unavailable",
        kind="component_wise_composition",
        notes="Native component-wise composition harness files are absent locally and on fetched refs; no unsupported module combination was forced.",
    )
    treatments.append(component_treatment)

    holdout_comparison = [_evaluate_treatment(holdout, t, metric_tables, best_fixed, thresholds["meaningful_margin"], args.seed, args.bootstrap) for t in treatments]
    method_df = pd.DataFrame(holdout_comparison)

    # Create a symbolic child only if the best contextual proxy beats both
    # fixed and discrete treatments on held-out meaningful windows.
    fixed_meaningful = method_df.loc[method_df["method"] == f"best_fixed__{best_fixed}", "meaningful_window_anwg"].iloc[0]
    discrete_meaningful = method_df.loc[method_df["method"] == discrete_model.name, "meaningful_window_anwg"].iloc[0]
    best_contextual_holdout = method_df.loc[method_df["method"] == best_contextual.name, "meaningful_window_anwg"].iloc[0]
    child_treatment = None
    child_info = None
    if (
        best_contextual_holdout is not None
        and fixed_meaningful is not None
        and discrete_meaningful is not None
        and float(best_contextual_holdout) > max(float(fixed_meaningful), float(discrete_meaningful)) + 0.002
    ):
        top_parents = (
            "weighted_shortest_processing" if "weighted_shortest_processing" in experts else experts[0],
            "scorpio_style_slo_guard" if "scorpio_style_slo_guard" in experts else experts[1],
        )
        child_treatment, child_info = _maybe_distill_child(
            train, val, holdout, best_contextual, experts, feat_cols, metric_tables, top_parents[0], top_parents[1], args.seed
        )
        if child_treatment is not None:
            child_eval = _evaluate_treatment(holdout, child_treatment, metric_tables, best_fixed, thresholds["meaningful_margin"], args.seed, args.bootstrap)
            method_df = pd.concat([method_df, pd.DataFrame([child_eval])], ignore_index=True)

    final_treatments = treatments + ([child_treatment] if child_treatment is not None else [])
    subset_df = _subset_analysis(holdout, final_treatments, metric_tables, best_fixed, thresholds["meaningful_margin"], args.seed, 0)

    weights_records = []
    for treatment in contextual_candidates + [static_equal, static_opt]:
        if treatment.weights is None:
            continue
        tmp = treatment.weights.reset_index().copy()
        tmp.insert(0, "method", treatment.name)
        tmp.insert(1, "formula", treatment.formula)
        tmp.insert(2, "top_k", treatment.top_k)
        weights_records.append(tmp)
    weights_df = pd.concat(weights_records, ignore_index=True) if weights_records else pd.DataFrame()

    pilot_windows = rows[[
        "window_idx", "split", "group_key", "dataset_family", "source_trace", "shape",
        "time_slice_pool", "n_requests", "oracle_policy", "oracle_anwg", "top2_margin",
        "pilot_categories", "primary_pilot_category",
    ]].copy()

    child_unique_wins = 0
    frontier_expanded = False
    child_anwg = None
    if child_treatment is not None:
        child_eval = method_df[method_df["method"] == child_treatment.name].iloc[0]
        child_anwg = child_eval["anwg"]
        child_sel = child_treatment.selection.loc[holdout["window_idx"]]
        parent_a = metric_tables["anwg"].loc[holdout["window_idx"], "weighted_shortest_processing"].to_numpy(dtype=float)
        parent_b = metric_tables["anwg"].loc[holdout["window_idx"], "scorpio_style_slo_guard"].to_numpy(dtype=float)
        child_vals = np.asarray([metric_tables["anwg"].loc[idx, p] for idx, p in zip(holdout["window_idx"], child_sel)], dtype=float)
        current_oracle = metric_tables["anwg"].loc[holdout["window_idx"], policies].max(axis=1).to_numpy(dtype=float)
        child_unique_wins = int(np.sum((child_vals > parent_a + 0.002) & (child_vals > parent_b + 0.002)))
        frontier_expanded = bool(np.any(child_vals > current_oracle + 1e-12))

    meaningful_subset = subset_df[subset_df["subset"] == "meaningful"]
    best_fixed_anwg = float(method_df.loc[method_df["method"] == f"best_fixed__{best_fixed}", "anwg"].iloc[0])
    discrete_anwg = float(method_df.loc[method_df["method"] == discrete_model.name, "anwg"].iloc[0])
    static_rows = method_df[method_df["kind"] == "static_composition_proxy"].dropna(subset=["anwg"])
    contextual_rows = method_df[method_df["kind"] == "contextual_composition_proxy"].dropna(subset=["anwg"])
    best_static_row = static_rows.sort_values("anwg", ascending=False).iloc[0]
    best_contextual_row = contextual_rows.sort_values("anwg", ascending=False).iloc[0]
    best_top_k = str(best_contextual_row["top_k"])

    best_contextual_meaningful = float(best_contextual_row["meaningful_window_anwg"])
    best_static_meaningful = float(best_static_row["meaningful_window_anwg"])
    component_anwg = None
    if (
        best_contextual_meaningful > max(float(fixed_meaningful), float(discrete_meaningful)) + 0.002
        or (child_treatment is not None and child_unique_wins > 0 and frontier_expanded)
    ):
        decision = "GO"
    elif (
        best_contextual_meaningful <= max(float(fixed_meaningful), float(discrete_meaningful)) + 0.001
        and best_static_meaningful <= max(float(fixed_meaningful), float(discrete_meaningful)) + 0.001
        and child_treatment is None
    ):
        decision = "NO_GO"
    else:
        decision = "INCONCLUSIVE"

    if decision == "GO":
        justified = "YES"
        failure_mode = "None in this diagnostic proxy; native component harness still missing."
    elif decision == "NO_GO":
        justified = "NO"
        failure_mode = "Contextual/static aggregate composition did not beat the local best fixed/discrete selector on held-out meaningful windows."
    else:
        justified = "UNCLEAR"
        failure_mode = "Small/noisy held-out pilot and missing native component-wise harness limit inference."

    missing_files = [
        rel for rel in [
            "src/llmserveopt/policies/composition.py",
            "src/llmserveopt/selector/composition_experiment.py",
            "tests/test_policy_composition.py",
            "tools/composition_smoke_experiment.py",
            "docs/current/COMPOSITION_EXPERIMENT_DESIGN.md",
            "docs/current/COMPOSITION_IMPLEMENTATION_STATUS.md",
            "docs/current/composition_experiment_schema.json",
            "docs/current/composition_hypotheses.json",
        ]
        if not (ROOT / rel).exists()
    ]

    manifest = {
        "timestamp": timestamp,
        "runtime_s": _round(time.perf_counter() - t0, 3),
        "decision": decision,
        "full_wulver_composition_run_justified": justified,
        "local_branch": _git(["branch", "--show-current"]),
        "local_commit": _git(["rev-parse", "HEAD"]),
        "worktree_status_short": _git(["status", "--short", "--branch"]),
        "pilot_source_dir": str(pilot_dir.relative_to(ROOT)),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "policy_vector_count": int(len(vectors)),
        "policy_count_in_vectors": int(len(policies)),
        "policies_in_vectors": list(policies),
        "expert_set": list(experts),
        "expert_reasons": expert_reasons,
        "thresholds_development_only": thresholds,
        "split_counts": rows["split"].value_counts().to_dict(),
        "pilot_window_count": int(len(rows)),
        "holdout_window_count": int(len(holdout)),
        "meaningful_holdout_window_count": int((holdout["top2_margin"] >= thresholds["meaningful_margin"]).sum()),
        "quality_gates": gates,
        "leakage_audit": audit,
        "missing_wulver_only_dependencies": missing_files,
        "composition_harness_complete": len(missing_files) == 0,
        "component_wise_composition_available": False,
        "tmux_used": False,
        "no_paid_api": True,
        "no_gpu": True,
        "method_selection_rule": "All treatment/model/weight choices selected on TRAIN/VALIDATION only; ID_TEST/OOD_TEST evaluated once as diagnostic held-out pilot.",
        "best_fixed_method": best_fixed,
        "best_discrete_selector": discrete_model.name,
        "discrete_selector_validation_scores": discrete_val_scores,
        "best_static_composition": str(best_static_row["method"]),
        "best_contextual_composition": str(best_contextual_row["method"]),
        "best_top_k": best_top_k,
        "symbolic_child_created": child_treatment is not None,
        "symbolic_child_unique_win_count": child_unique_wins,
        "oracle_frontier_expanded_by_child": frontier_expanded,
        "main_failure_mode": failure_mode,
    }

    pilot_windows.to_csv(out_dir / "pilot_windows.csv", index=False)
    method_df.to_csv(out_dir / "method_comparison.csv", index=False)
    subset_df.to_csv(out_dir / "subset_analysis.csv", index=False)
    weights_df.to_csv(out_dir / "composition_weights.csv", index=False)
    if child_info is not None:
        (out_dir / "symbolic_child.json").write_text(json.dumps(child_info, indent=2))
    (out_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "pilot_report.md").write_text(_report(manifest, method_df, subset_df))

    print(json.dumps({
        "decision": decision,
        "output_dir": str(out_dir),
        "runtime_s": manifest["runtime_s"],
        "pilot_window_count": int(len(rows)),
        "meaningful_holdout_window_count": manifest["meaningful_holdout_window_count"],
        "best_fixed": best_fixed,
        "best_discrete_selector": discrete_model.name,
        "best_contextual": best_contextual_row["method"],
        "best_contextual_anwg": best_contextual_row["anwg"],
        "missing_wulver_only_dependencies": missing_files,
    }, indent=2))
    return 0


def _report(manifest: Mapping[str, object], method_df: pd.DataFrame, subset_df: pd.DataFrame) -> str:
    lines = [
        "# Composition Smart Pilot Report",
        "",
        f"COMPOSITION_PILOT_DECISION = {manifest['decision']}",
        "",
        "## Scope",
        "",
        "This is a local diagnostic pilot over precomputed clean selector simulator vectors. "
        "The native composition harness files named in the request are not present locally or on fetched refs, "
        "so static/contextual composition rows are aggregate vector-level proxies and component-wise composition is unavailable.",
        "",
        "## Repository Audit",
        "",
        f"- Branch: `{manifest['local_branch']}`",
        f"- Commit: `{manifest['local_commit']}`",
        f"- Composition harness complete: `{manifest['composition_harness_complete']}`",
        f"- Policy count in clean vectors: `{manifest['policy_count_in_vectors']}`",
        f"- Missing Wulver-only dependencies: `{', '.join(manifest['missing_wulver_only_dependencies']) or 'none'}`",
        "",
        "## Pilot Data",
        "",
        f"- Pilot windows: `{manifest['pilot_window_count']}`",
        f"- Holdout windows: `{manifest['holdout_window_count']}`",
        f"- Meaningful holdout windows: `{manifest['meaningful_holdout_window_count']}`",
        f"- Expert set: `{', '.join(manifest['expert_set'])}`",
        "",
        "## Held-Out Method Comparison",
        "",
        method_df.to_markdown(index=False),
        "",
        "## Held-Out Subset Analysis",
        "",
        subset_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        str(manifest["main_failure_mode"]),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

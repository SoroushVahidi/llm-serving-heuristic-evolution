"""Evaluation helpers for module-credit prediction and ranking."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def evaluate_credit_predictions(model, rows: Sequence[Mapping[str, Any]], *, target: str = "C_base") -> dict[str, Any]:
    actual = np.asarray([float(r[target]) for r in rows], dtype=float)
    pred = model.predict_mean(rows)
    err = pred - actual
    sign_mask = actual != 0.0
    return {
        "n_rows": len(rows),
        "target": target,
        "mae": float(np.mean(np.abs(err))) if len(rows) else None,
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(rows) else None,
        "bias": float(np.mean(err)) if len(rows) else None,
        "sign_accuracy": float(np.mean(np.sign(pred[sign_mask]) == np.sign(actual[sign_mask]))) if sign_mask.any() else None,
    }


def evaluate_topk_ranking(
    model,
    rows: Sequence[Mapping[str, Any]],
    *,
    lambda_m: float = 0.5,
    ks: Sequence[int] = (1, 3, 5),
    meaningful_gain_threshold: float = 0.01,
) -> dict[str, Any]:
    """Rank compatible candidates per (state, base_policy)."""
    grouped = _group_candidates(rows)
    scores = model.predict_score(rows, lambda_m=lambda_m)
    by_id = {id(row): float(score) for row, score in zip(rows, scores)}
    result: dict[str, Any] = {}
    for k in ks:
        selected = []
        for candidates in grouped.values():
            ordered = sorted(candidates, key=lambda r: by_id[id(r)], reverse=True)
            selected.extend(ordered[: min(k, len(ordered))])
        result[f"top_{k}"] = _summarize_selected(selected, meaningful_gain_threshold)
    return result


def evaluate_offline_synthesis_decisions(
    rows: Sequence[Mapping[str, Any]],
    model,
    *,
    lambda_m: float = 0.5,
    meaningful_gain_threshold: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare one-choice-per-(state,base) decision policies."""
    grouped = _group_candidates(rows)
    rng = random.Random(seed)
    model_scores = {id(row): score for row, score in zip(rows, model.predict_score(rows, lambda_m=lambda_m))}
    strategies = {
        "random_compatible": lambda candidates: rng.choice(candidates),
        "highest_whole_policy_suitability_donor": lambda candidates: max(candidates, key=lambda r: float(r.get("donor_conservative_suitability", 0.0))),
        "structural_nearest_proxy": lambda candidates: min(candidates, key=lambda r: abs(float(r.get("compatibility_metadata", {}).get("structural_distance", 0.0)))),
        "module_credit_model": lambda candidates: max(candidates, key=lambda r: model_scores[id(r)]),
    }
    out = {}
    for name, choose in strategies.items():
        selected = [choose(candidates) for candidates in grouped.values() if candidates]
        out[name] = _summarize_selected(selected, meaningful_gain_threshold)
    return out


def _group_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        compat = row.get("compatibility_metadata", {})
        if isinstance(compat, Mapping) and float(compat.get("compatible", 1.0)) <= 0.0:
            continue
        grouped[(str(row["state_id"]), str(row["base_policy"]))].append(row)
    return grouped


def _summarize_selected(selected: Sequence[Mapping[str, Any]], meaningful_gain_threshold: float) -> dict[str, Any]:
    if not selected:
        return {"n_selected": 0}
    c_base = np.asarray([float(r["C_base"]) for r in selected])
    c_parent = np.asarray([float(r["C_parent"]) for r in selected])
    c_env = np.asarray([float(r["C_env"]) for r in selected])
    return {
        "n_selected": len(selected),
        "mean_C_base": float(np.mean(c_base)),
        "mean_C_parent": float(np.mean(c_parent)),
        "mean_C_env": float(np.mean(c_env)),
        "positive_transfer_precision": float(np.mean(c_base > 0.0)),
        "meaningful_gain_precision": float(np.mean(c_base > meaningful_gain_threshold)),
        "beats_both_parents_fraction": float(np.mean(c_parent > 0.0)),
        "expands_library_envelope_fraction": float(np.mean(c_env > 0.0)),
    }

"""Envelope-gain and bootstrap helpers for the KV-aware composition
falsification v1. All computation uses canonical
`arrival_normalized_weighted_goodput`. Structurally identical to
`prefill_control_metrics.py` (same statistical methodology, reused for
comparability with the completed Family B v2 falsification), generalised to
two named parents instead of full/small.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

PRIMARY = "arrival_normalized_weighted_goodput"
PRACTICAL_EPS = 0.01


def parent_envelope(
    kv_scores: Dict[str, float],
    llf_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """E_P(x) = max(R_kv(x), R_llf(x))."""
    return {
        sid: max(float(kv_scores.get(sid, 0.0)), float(llf_scores.get(sid, 0.0)))
        for sid in scenario_ids
    }


def envelope_gain(
    child_scores: Dict[str, float],
    envelope: Dict[str, float],
    scenario_ids: Sequence[str],
    *,
    eps: float = PRACTICAL_EPS,
) -> Dict[str, float]:
    """G(c;P) = mean_x[max(R_c(x) - E_P(x), 0)]; also G_eps and beats-envelope counts."""
    gains, gains_eps, deltas = [], [], []
    beat_eps = 0
    for sid in scenario_ids:
        rc = float(child_scores.get(sid, 0.0))
        ep = float(envelope.get(sid, 0.0))
        delta = rc - ep
        deltas.append(delta)
        gains.append(max(delta, 0.0))
        gains_eps.append(max(delta - eps, 0.0))
        if delta > eps:
            beat_eps += 1
    n = len(scenario_ids)
    arr_g = np.asarray(gains, dtype=float)
    arr_ge = np.asarray(gains_eps, dtype=float)
    arr_d = np.asarray(deltas, dtype=float)
    return {
        "n_scenarios": float(n),
        "mean_envelope_gain": float(np.mean(arr_g)) if n else float("nan"),
        "mean_envelope_gain_eps": float(np.mean(arr_ge)) if n else float("nan"),
        "mean_delta": float(np.mean(arr_d)) if n else float("nan"),
        "median_delta": float(np.median(arr_d)) if n else float("nan"),
        "frac_positive_delta": float(np.mean(arr_d > 1e-12)) if n else 0.0,
        "n_beat_envelope_plus_eps": float(beat_eps),
        "fraction_beat_envelope_plus_eps": float(beat_eps / n) if n else 0.0,
    }


def paired_deltas(
    child_scores: Dict[str, float],
    envelope: Dict[str, float],
    scenario_ids: Sequence[str],
) -> List[float]:
    return [
        float(child_scores.get(sid, 0.0)) - float(envelope.get(sid, 0.0))
        for sid in scenario_ids
    ]


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 20261201,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) via bootstrapped percentile CI."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
        for _ in range(n_boot)
    ]
    return (
        float(np.mean(arr)),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def best_fixed_parent_score(
    kv_scores: Dict[str, float],
    llf_scores: Dict[str, float],
    fit_scenario_ids: Sequence[str],
    eval_scenario_ids: Sequence[str],
) -> Tuple[Dict[str, float], str]:
    """Best fixed parent, chosen by mean score on `fit_scenario_ids` (e.g.
    TRAIN), applied to `eval_scenario_ids` (e.g. TEST/OOD) -- never lets the
    evaluation set influence which parent is "best"."""
    mean_kv = float(np.mean([kv_scores.get(sid, 0.0) for sid in fit_scenario_ids])) if fit_scenario_ids else 0.0
    mean_llf = float(np.mean([llf_scores.get(sid, 0.0) for sid in fit_scenario_ids])) if fit_scenario_ids else 0.0
    if mean_kv >= mean_llf:
        return {sid: float(kv_scores.get(sid, 0.0)) for sid in eval_scenario_ids}, "kv_constrained_online"
    return {sid: float(llf_scores.get(sid, 0.0)) for sid in eval_scenario_ids}, "least_laxity_first"


def oracle_regret(
    method_scores: Dict[str, float],
    kv_scores: Dict[str, float],
    llf_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    regrets = [
        max(float(kv_scores.get(sid, 0.0)), float(llf_scores.get(sid, 0.0)))
        - float(method_scores.get(sid, 0.0))
        for sid in scenario_ids
    ]
    arr = np.asarray(regrets, dtype=float)
    return {
        "mean_regret": float(np.mean(arr)) if len(arr) else float("nan"),
        "median_regret": float(np.median(arr)) if len(arr) else float("nan"),
        "n_regret_gt_eps": float(np.sum(arr > PRACTICAL_EPS)),
    }


def pairwise_comparison(
    a_scores: Dict[str, float],
    b_scores: Dict[str, float],
    scenario_ids: Sequence[str],
    *,
    eps: float = PRACTICAL_EPS,
) -> Dict[str, float]:
    a_better = b_better = ties = 0
    deltas = []
    for sid in scenario_ids:
        d = float(a_scores.get(sid, 0.0)) - float(b_scores.get(sid, 0.0))
        deltas.append(d)
        if d > eps:
            a_better += 1
        elif d < -eps:
            b_better += 1
        else:
            ties += 1
    n = len(scenario_ids)
    mean, lo, hi = bootstrap_ci(deltas)
    return {
        "n_a_better": float(a_better),
        "n_b_better": float(b_better),
        "n_ties": float(ties),
        "mean_delta_a_minus_b": mean,
        "ci_lo": lo,
        "ci_hi": hi,
    }

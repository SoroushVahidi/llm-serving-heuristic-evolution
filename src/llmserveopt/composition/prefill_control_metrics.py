"""Envelope-gain and bootstrap helpers for PrefillControl composition.

All computation uses canonical `arrival_normalized_weighted_goodput`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PRIMARY = "arrival_normalized_weighted_goodput"
PRACTICAL_EPS = 0.01


# ===================================================================
# Parent envelope
# ===================================================================

def parent_envelope(
    full_scores: Dict[str, float],
    small_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """Compute E_P(x) = max(R_full(x), R_small(x)) for each scenario."""
    env = {}
    for sid in scenario_ids:
        ea = float(full_scores.get(sid, 0.0))
        eb = float(small_scores.get(sid, 0.0))
        env[sid] = max(ea, eb)
    return env


def envelope_gain(
    child_scores: Dict[str, float],
    parent_envelope: Dict[str, float],
    scenario_ids: Sequence[str],
    *,
    eps: float = PRACTICAL_EPS,
) -> Dict[str, float]:
    """G(c;P) = mean_x[max(R_c(x) - E_P(x), 0)] with E_P = max(R_full, R_small).

    Also computes delta_child = R_c - E_P, beats-both counts, and
    practical envelope expansion at epsilon.
    """
    gains = []
    deltas = []
    beat_both_gt0 = 0
    beat_both_gt_eps = 0
    lose_both = 0
    for sid in scenario_ids:
        rc = float(child_scores.get(sid, 0.0))
        ep = float(parent_envelope.get(sid, 0.0))
        delta = rc - ep
        gain = max(delta, 0.0)
        gains.append(gain)
        deltas.append(delta)
        if rc > ep + eps:
            beat_both_gt_eps += 1
            beat_both_gt0 += 1
        # Check against each parent individually
        # lose_both: only matters when child < both parents
    n = len(scenario_ids)
    arr_gains = np.asarray(gains, dtype=float)
    arr_deltas = np.asarray(deltas, dtype=float)
    return {
        "n_scenarios": float(n),
        "mean_envelope_gain": float(np.mean(arr_gains)) if n else float("nan"),
        "median_envelope_gain": float(np.median(arr_gains)) if n else float("nan"),
        "mean_delta": float(np.mean(arr_deltas)) if n else float("nan"),
        "median_delta": float(np.median(arr_deltas)) if n else float("nan"),
        "frac_positive_delta": float(np.mean(arr_deltas > 1e-12)),
        "frac_positive_gain": float(np.mean(arr_gains > 1e-12)),
        "n_beat_envelope_plus_eps": float(beat_both_gt_eps),
        "fraction_beat_envelope_plus_eps": float(beat_both_gt_eps / n) if n else 0.0,
        "mean_delta_clipped_0p01": float(np.mean([max(d - eps, 0.0) for d in deltas])) if n else 0.0,
    }


def paired_deltas(
    child_scores: Dict[str, float],
    parent_envelope: Dict[str, float],
    scenario_ids: Sequence[str],
) -> List[float]:
    """Return per-scenario delta_child(x) = R_c(x) - E_P(x)."""
    return [
        float(child_scores.get(sid, 0.0)) - float(parent_envelope.get(sid, 0.0))
        for sid in scenario_ids
    ]


# ===================================================================
# Bootstrap CI
# ===================================================================

def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 20261201,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for the mean via bootstrapped percentile CI."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    mean = float(np.mean(arr))
    return mean, lo, hi


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 20261201,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Alias for bootstrap_ci — matches ESTF/WFS naming convention."""
    return bootstrap_ci(values, n_boot=n_boot, seed=seed, alpha=alpha)


# ===================================================================
# Oracle / best fixed parent
# ===================================================================

def oracle_scores(
    full_scores: Dict[str, float],
    small_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """Oracle = max(R_full, R_small) per scenario."""
    return parent_envelope(full_scores, small_scores, scenario_ids)


def best_fixed_parent_score(
    full_scores: Dict[str, float],
    small_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """Best fixed parent = the globally better parent across all candidate scenarios.

    Returns per-scenario score using that single best parent.
    """
    mean_full = np.mean([float(full_scores.get(sid, 0.0)) for sid in scenario_ids])
    mean_small = np.mean([float(small_scores.get(sid, 0.0)) for sid in scenario_ids])

    if mean_full >= mean_small:
        return {sid: float(full_scores.get(sid, 0.0)) for sid in scenario_ids}
    return {sid: float(small_scores.get(sid, 0.0)) for sid in scenario_ids}


def oracle_regret(
    method_scores: Dict[str, float],
    full_scores: Dict[str, float],
    small_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """Mean regret vs oracle = mean_x[oracle(x) - R_method(x)]."""
    regrets = []
    for sid in scenario_ids:
        oracle = max(
            float(full_scores.get(sid, 0.0)),
            float(small_scores.get(sid, 0.0)),
        )
        regrets.append(oracle - float(method_scores.get(sid, 0.0)))
    arr = np.asarray(regrets, dtype=float)
    return {
        "mean_regret": float(np.mean(arr)),
        "median_regret": float(np.median(arr)),
        "p95_regret": float(np.quantile(arr, 0.95)),
        "n_regret_gt_0.01": float(np.mean(arr > 0.01)),
    }


# ===================================================================
# Pairwise comparison helpers
# ===================================================================

def pairwise_comparison(
    a_scores: Dict[str, float],
    b_scores: Dict[str, float],
    scenario_ids: Sequence[str],
    *,
    eps: float = PRACTICAL_EPS,
) -> Dict[str, float]:
    """Pairwise comparison of two methods: a wins / b wins / ties."""
    a_better = 0
    b_better = 0
    ties = 0
    deltas = []
    for sid in scenario_ids:
        sa = float(a_scores.get(sid, 0.0))
        sb = float(b_scores.get(sid, 0.0))
        d = sa - sb
        deltas.append(d)
        if d > eps:
            a_better += 1
        elif d < -eps:
            b_better += 1
        else:
            ties += 1
    n = len(scenario_ids)
    arr = np.asarray(deltas, dtype=float)
    ci = bootstrap_ci(list(deltas))
    return {
        "n_a_better": float(a_better),
        "n_b_better": float(b_better),
        "n_ties": float(ties),
        "mean_delta_a_minus_b": ci[0],
        "ci_lo": ci[1],
        "ci_hi": ci[2],
        "wins": float(a_better / n) if n else 0.0,
        "losses": float(b_better / n) if n else 0.0,
    }


# ===================================================================
# Best fixed intermediate child
# ===================================================================

def best_fixed_intermediate_score(
    intermediate_scores: Dict[str, float],
    full_scores: Dict[str, float],
    small_scores: Dict[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """Best fixed intermediate = the intermediate chunk that has the
    highest mean score across all candidate scenarios.

    Returns per-scenario scores using that single best intermediate.
    """
    chunk_options = [
        c for c in intermediate_scores
        if c.startswith("chunk_") and c not in ("chunk_64", "chunk_65536")
    ]
    if not chunk_options:
        return {}
    # Compute mean score per intermediate chunk
    means = {}
    for chunk in chunk_options:
        vals = [float(intermediate_scores.get(sid, 0.0)) for sid in scenario_ids]
        means[chunk] = np.mean(vals)
    best_chunk = max(means, key=means.get)
    return {sid: float(intermediate_scores.get(f"{best_chunk}.{sid}", 0.0)) for sid in scenario_ids}

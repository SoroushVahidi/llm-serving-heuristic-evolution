"""Envelope-gain and bootstrap helpers for ESTF/WFS composition."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def envelope_gain(
    child: Mapping[str, float],
    parent_a: Mapping[str, float],
    parent_b: Mapping[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    """G(c;P) = mean_x[max(R_c, E_P) - E_P] with E_P = max(parent_a, parent_b)."""
    gains = []
    beat_both = 0
    lose_both = 0
    unique_child = 0
    for sid in scenario_ids:
        rc = float(child[sid])
        ea = float(parent_a[sid])
        eb = float(parent_b[sid])
        env = max(ea, eb)
        gains.append(max(rc, env) - env)
        if rc > ea + 1e-12 and rc > eb + 1e-12:
            beat_both += 1
            unique_child += 1
        if rc < ea - 1e-12 and rc < eb - 1e-12:
            lose_both += 1
    arr = np.asarray(gains, dtype=float)
    n = len(arr)
    return {
        "n": float(n),
        "mean_envelope_gain": float(np.mean(arr)) if n else float("nan"),
        "median_envelope_gain": float(np.median(arr)) if n else float("nan"),
        "frac_positive_gain": float(np.mean(arr > 1e-12)) if n else 0.0,
        "frac_gain_gt_0.01": float(np.mean(arr > 0.01)) if n else 0.0,
        "n_beat_both": float(beat_both),
        "n_lose_both": float(lose_both),
        "unique_winner_fraction": float(unique_child / n) if n else 0.0,
    }


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 20260816,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for the mean via paired bootstrap."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(arr)), lo, hi


def regret_to_oracle(
    method: Mapping[str, float],
    parent_a: Mapping[str, float],
    parent_b: Mapping[str, float],
    scenario_ids: Sequence[str],
) -> Dict[str, float]:
    regrets = []
    for sid in scenario_ids:
        oracle = max(float(parent_a[sid]), float(parent_b[sid]))
        regrets.append(oracle - float(method[sid]))
    arr = np.asarray(regrets, dtype=float)
    return {
        "mean_regret": float(np.mean(arr)),
        "median_regret": float(np.median(arr)),
        "p95_regret": float(np.quantile(arr, 0.95)),
        "frac_regret_gt_0.01": float(np.mean(arr > 0.01)),
    }

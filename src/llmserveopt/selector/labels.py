"""
Window label construction: assign each window to the best deployable policy
under weighted_goodput, with deterministic tie-breaking.

Label rule
----------
    label = argmax_{p in SELECTOR_CANDIDATES} weighted_goodput(p, window)

Tie-breaking order (ascending preference)
------------------------------------------
1. lower slo_violation_rate
2. lower p95_ttft
3. lower p95_latency
4. higher throughput
5. alphabetical policy name (deterministic)

Oracle exclusion
----------------
oracle_srtf is never a label candidate.  It may appear in the oracle_reward
column if provided, but is never in best_policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.metrics import RunMetrics
from ..policies.registry import ORACLE_POLICY_NAMES
from .candidates import SELECTOR_CANDIDATES


@dataclass
class TieBreaker:
    """All metrics used in tie-breaking, lower-is-better unless noted."""
    slo_violation_rate: float = float("nan")
    p95_ttft: float = float("nan")
    p95_latency: float = float("nan")
    throughput: float = float("nan")        # higher is better → negate for sort
    policy_name: str = ""

    def sort_key(self):
        thr = -self.throughput if not np.isnan(self.throughput) else float("inf")
        p95t = self.p95_ttft if not np.isnan(self.p95_ttft) else float("inf")
        p95l = self.p95_latency if not np.isnan(self.p95_latency) else float("inf")
        slor = self.slo_violation_rate if not np.isnan(self.slo_violation_rate) else float("inf")
        return (slor, p95t, p95l, thr, self.policy_name)


@dataclass
class WindowLabel:
    """Label record for one (window, seed) combination."""
    best_policy: str
    best_weighted_goodput: float
    second_best_policy: str
    second_best_weighted_goodput: float
    policy_margin: float                # best - second_best weighted_goodput
    regret_to_best_fixed: float         # filled in at dataset level
    reward_vector: Dict[str, float] = field(default_factory=dict)   # per-policy weighted_goodput
    oracle_weighted_goodput: float = float("nan")                    # oracle_srtf if available
    # Tie-break columns
    slo_violation_rate_best: float = float("nan")
    p95_ttft_best: float = float("nan")
    p95_latency_best: float = float("nan")


def label_window(
    metrics_by_policy: Dict[str, RunMetrics],
) -> WindowLabel:
    """Compute the label for one window given per-policy RunMetrics.

    Parameters
    ----------
    metrics_by_policy : dict mapping policy_name -> RunMetrics.
        May include oracle entries; they are excluded from label selection.

    Returns
    -------
    WindowLabel with best_policy, reward vectors, etc.

    Raises
    ------
    ValueError if no deployable candidate has metrics.
    """
    oracle_wg = float("nan")
    for oname in ORACLE_POLICY_NAMES:
        if oname in metrics_by_policy:
            oracle_wg = metrics_by_policy[oname].weighted_goodput

    candidates = {
        name: m for name, m in metrics_by_policy.items()
        if name in set(SELECTOR_CANDIDATES)
    }
    if not candidates:
        raise ValueError(
            f"No deployable candidate metrics found. Got policies: {list(metrics_by_policy.keys())}. "
            f"Expected candidates: {SELECTOR_CANDIDATES[:4]}..."
        )

    # Build (policy_name, weighted_goodput, TieBreaker) triples
    entries = []
    for name, m in candidates.items():
        wg = m.weighted_goodput if not np.isnan(m.weighted_goodput) else -float("inf")
        tb = TieBreaker(
            slo_violation_rate=m.slo_violation_rate,
            p95_ttft=m.p95_ttft,
            p95_latency=m.p95_latency,
            throughput=m.request_throughput,
            policy_name=name,
        )
        entries.append((wg, tb, name, m))

    # Sort: best weighted_goodput first, then tie-break ascending
    entries.sort(key=lambda e: (-e[0], e[1].sort_key()))

    best_wg, best_tb, best_name, best_m = entries[0]
    reward_vec = {name: m.weighted_goodput for name, _, name2, m in [(None, None, n, m) for n, m in candidates.items()]}
    # Rebuild cleanly
    reward_vec = {name: m.weighted_goodput for name, m in candidates.items()}

    if len(entries) >= 2:
        second_wg, _, second_name, _ = entries[1]
    else:
        second_wg = float("nan")
        second_name = ""

    margin = best_wg - second_wg if not np.isnan(second_wg) else float("nan")

    return WindowLabel(
        best_policy=best_name,
        best_weighted_goodput=float(best_wg),
        second_best_policy=second_name,
        second_best_weighted_goodput=float(second_wg),
        policy_margin=float(margin) if not np.isnan(margin) else float("nan"),
        regret_to_best_fixed=float("nan"),  # filled in at dataset level
        reward_vector=reward_vec,
        oracle_weighted_goodput=oracle_wg,
        slo_violation_rate_best=float(best_m.slo_violation_rate) if not np.isnan(best_m.slo_violation_rate) else float("nan"),
        p95_ttft_best=float(best_m.p95_ttft) if not np.isnan(best_m.p95_ttft) else float("nan"),
        p95_latency_best=float(best_m.p95_latency) if not np.isnan(best_m.p95_latency) else float("nan"),
    )


def label_windows(
    window_metrics: List[Dict[str, RunMetrics]],
) -> List[WindowLabel]:
    """Label a list of windows and fill in regret_to_best_fixed.

    regret_to_best_fixed = best_per_window_goodput - best_fixed_policy_mean_goodput
    where best_fixed_policy is the single policy that maximises mean goodput across
    all windows (an oracle that picks one policy for the whole trace).
    """
    labels = [label_window(m) for m in window_metrics]

    # Compute per-policy mean goodput across all windows
    all_names = set()
    for lbl in labels:
        all_names.update(lbl.reward_vector.keys())

    policy_mean: Dict[str, float] = {}
    for name in all_names:
        vals = [lbl.reward_vector.get(name, float("nan")) for lbl in labels]
        valid = [v for v in vals if not np.isnan(v)]
        policy_mean[name] = float(np.mean(valid)) if valid else float("nan")

    best_fixed_wg = max(
        (v for v in policy_mean.values() if not np.isnan(v)),
        default=float("nan"),
    )

    for lbl in labels:
        if not np.isnan(best_fixed_wg):
            lbl.regret_to_best_fixed = lbl.best_weighted_goodput - best_fixed_wg

    return labels

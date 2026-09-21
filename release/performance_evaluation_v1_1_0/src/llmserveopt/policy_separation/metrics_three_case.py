"""Pairwise/scenario-level separation metrics for the three-case policy
separation diagnostic. Reuses selector.dataset_v2.discriminativeness's
margin constants and compute_discriminativeness so this experiment's notion
of "tie"/"discriminative" agrees with Selector Dataset v2's, per
docs/design/POLICY_SEPARATION_DATASET_V1.md section 10 -- not re-derived.

Raw variance is never used alone as a success criterion; it is one field of
several (see `scenario_dispersion`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..selector.dataset_v2.discriminativeness import (
    PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN,
    PRIMARY_SELECTOR_OBJECTIVE,
    STANDARD_OBJECTIVES,
    compute_discriminativeness,
)
from ..selector.dataset_v2.schema import PolicyOutcomeVector

EPS = 1e-9

PRIMARY_OBJECTIVE = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)


@dataclass(frozen=True)
class PolicyResultRow:
    scenario_id: str
    policy_name: str
    arrival_normalized_weighted_goodput: Optional[float]
    weighted_goodput: Optional[float]
    completion_fraction: Optional[float]
    slo_violation_rate: Optional[float]
    mean_latency: Optional[float]
    mean_ttft: Optional[float]
    mean_tpot: Optional[float]
    num_completed: int
    num_dropped: int
    num_total: int


def _finite(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def pairwise_rows(scenario_id: str, rows: List[PolicyResultRow]) -> List[Dict]:
    """One row per unordered policy pair (alphabetically ordered i<j) with
    signed advantage, absolute separation, and a latency log-ratio."""
    out = []
    by_name = {r.policy_name: r for r in rows}
    names = sorted(by_name)
    for idx_i in range(len(names)):
        for idx_j in range(idx_i + 1, len(names)):
            i, j = names[idx_i], names[idx_j]
            ri, rj = by_name[i], by_name[j]
            anwg_i = _finite(ri.arrival_normalized_weighted_goodput)
            anwg_j = _finite(rj.arrival_normalized_weighted_goodput)
            signed_advantage = (anwg_i - anwg_j) if (anwg_i is not None and anwg_j is not None) else None
            abs_separation = abs(signed_advantage) if signed_advantage is not None else None

            lat_i = _finite(ri.mean_latency)
            lat_j = _finite(rj.mean_latency)
            log_ratio = None
            if lat_i is not None and lat_j is not None:
                log_ratio = abs(math.log((lat_i + EPS) / (lat_j + EPS)))

            out.append({
                "scenario_id": scenario_id,
                "policy_i": i,
                "policy_j": j,
                "anwg_i": anwg_i,
                "anwg_j": anwg_j,
                "signed_advantage_i_minus_j": signed_advantage,
                "abs_separation": abs_separation,
                "latency_log_ratio": log_ratio,
                "practically_equivalent": (
                    abs_separation is not None and abs_separation <= PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN
                ),
            })
    return out


def scenario_summary(scenario_id: str, rows: List[PolicyResultRow]) -> Dict:
    """Scenario-level diagnostics: rank, top-two margin, unique winner /
    tie set, dispersion, and the reused discriminativeness classification."""
    valid = [(r.policy_name, _finite(r.arrival_normalized_weighted_goodput)) for r in rows]
    valid = [(name, v) for name, v in valid if v is not None]
    if not valid:
        return {
            "scenario_id": scenario_id, "n_valid_policies": 0,
            "winner_policy": None, "unique_winner": None, "tie_set": None,
            "top_two_margin": None, "ranking": None,
            "dispersion_std": None, "dispersion_mad": None,
            "classification": "INSUFFICIENT_DATA",
        }

    ranked = sorted(valid, key=lambda kv: -kv[1])
    best_name, best_val = ranked[0]
    tie_set = [name for name, v in ranked if (best_val - v) <= PRACTICAL_EQUIVALENCE_ABSOLUTE_MARGIN]
    unique_winner = len(tie_set) == 1
    top_two_margin = (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else None

    values = [v for _, v in valid]
    mean_v = sum(values) / len(values)
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median_v = (sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2)
    mad = sum(abs(v - median_v) for v in values) / len(values)

    outcomes = [
        PolicyOutcomeVector(
            policy_name=r.policy_name,
            fidelity_class="historical",
            weighted_goodput=_finite(r.weighted_goodput),
            arrival_normalized_weighted_goodput=_finite(r.arrival_normalized_weighted_goodput),
            completion_fraction=_finite(r.completion_fraction),
            slo_attainment=(1.0 - _finite(r.slo_violation_rate)) if _finite(r.slo_violation_rate) is not None else None,
            p95_latency=None,
            request_throughput=None,
            slo_success_throughput=None,
        )
        for r in rows
    ]
    disc = compute_discriminativeness(outcomes, PRIMARY_OBJECTIVE)
    classification = disc.classification if disc is not None else "INSUFFICIENT_DATA"

    return {
        "scenario_id": scenario_id,
        "n_valid_policies": len(valid),
        "winner_policy": best_name,
        "unique_winner": unique_winner,
        "tie_set": ";".join(tie_set),
        "top_two_margin": top_two_margin,
        "ranking": ";".join(f"{name}:{v:.6f}" for name, v in ranked),
        "dispersion_std": std,
        "dispersion_mad": mad,
        "classification": classification,
    }

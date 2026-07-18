"""
Multi-objective regret and discriminativeness analysis for Selector
Dataset v2. See docs/selector_dataset_v2.md §5-6.

Never reduces a window to a single winner label at this layer -- every
function here operates over (and returns) the FULL per-policy outcome
vector's relevant slice for one objective at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .schema import DiscriminativenessResult, PolicyOutcomeVector, RegretRecord

# Near-tie threshold: continuity with Phase 2B.16's own established
# "margin < 0.005" near-tie criterion (see docs/research_status.md's
# Phase 2B.16 summary) -- reused deliberately, not reinvented.
NEAR_TIE_RELATIVE_MARGIN = 0.005
# Strongly-discriminative threshold: this project's own new, disclosed
# choice for Dataset v2 -- no prior phase defined this boundary.
STRONGLY_DISCRIMINATIVE_RELATIVE_MARGIN = 0.03


@dataclass(frozen=True)
class Objective:
    name: str
    higher_is_better: bool
    extractor: Callable[[PolicyOutcomeVector], Optional[float]]


STANDARD_OBJECTIVES: List[Objective] = [
    Objective("weighted_goodput", True, lambda o: o.weighted_goodput),
    Objective("arrival_normalized_weighted_goodput", True, lambda o: o.arrival_normalized_weighted_goodput),
    Objective("p95_latency", False, lambda o: o.p95_latency),
    Objective("slo_attainment", True, lambda o: o.slo_attainment),
    Objective("request_throughput", True, lambda o: o.request_throughput),
]


def _valid_values(outcomes: List[PolicyOutcomeVector], objective: Objective) -> Dict[str, float]:
    out = {}
    for o in outcomes:
        v = objective.extractor(o)
        if v is not None:
            out[o.policy_name] = v
    return out


def compute_discriminativeness(
    outcomes: List[PolicyOutcomeVector], objective: Objective, epsilon: float = 1e-9,
) -> Optional[DiscriminativenessResult]:
    """None if fewer than 2 policies have a valid value for this
    objective (nothing to discriminate between)."""
    values = _valid_values(outcomes, objective)
    if len(values) < 2:
        return None

    sign = 1.0 if objective.higher_is_better else -1.0
    ranked = sorted(values.items(), key=lambda kv: -sign * kv[1])
    best_name, best_val = ranked[0]
    second_name, second_val = ranked[1]
    all_vals = list(values.values())
    max_min_spread = max(all_vals) - min(all_vals)
    abs_margin = sign * (best_val - second_val)
    denom = abs(best_val) if abs(best_val) > epsilon else epsilon
    rel_margin = abs_margin / denom

    tie_epsilon = max(epsilon, abs(best_val) * 0.001)
    tie_set = sorted(name for name, val in values.items() if abs(val - best_val) <= tie_epsilon)

    if max_min_spread <= epsilon:
        classification = "ALL_COMPLETE_OR_EFFECTIVELY_TIED"
    elif rel_margin < NEAR_TIE_RELATIVE_MARGIN:
        classification = "NEAR_TIE"
    elif rel_margin < STRONGLY_DISCRIMINATIVE_RELATIVE_MARGIN:
        classification = "MODERATELY_DISCRIMINATIVE"
    else:
        classification = "STRONGLY_DISCRIMINATIVE"

    return DiscriminativenessResult(
        objective_name=objective.name, best_policy=best_name, best_value=float(best_val),
        second_best_policy=second_name, second_best_value=float(second_val),
        absolute_winner_margin=float(abs_margin), relative_winner_margin=float(rel_margin),
        max_min_spread=float(max_min_spread), tie_set=tie_set, classification=classification,
    )


def compute_regrets(
    outcomes: List[PolicyOutcomeVector], objective: Objective,
    best_fixed_policy_value: Optional[float] = None,
) -> List[RegretRecord]:
    """regret(s, p) = score(best compatible policy for scenario s) -
    score(policy p), sign-adjusted so regret is always >= 0 regardless of
    whether the objective is higher- or lower-is-better. `regret_to_best_fixed`
    additionally requires the dataset-wide best-on-average policy's value
    for this objective (see `compute_best_fixed_policy_values`) -- NaN if
    not supplied."""
    values = _valid_values(outcomes, objective)
    if not values:
        return []
    sign = 1.0 if objective.higher_is_better else -1.0
    best_val = max(values.values(), key=lambda v: sign * v)

    records = []
    for pname, val in values.items():
        regret = sign * (best_val - val)
        regret_to_fixed = float("nan")
        if best_fixed_policy_value is not None:
            regret_to_fixed = sign * (best_fixed_policy_value - val)
        records.append(RegretRecord(
            objective_name=objective.name, policy_name=pname,
            regret=float(regret), regret_to_best_fixed=float(regret_to_fixed),
        ))
    return records


def compute_best_fixed_policy_values(
    all_window_outcomes: List[List[PolicyOutcomeVector]], objective: Objective,
) -> Dict[str, float]:
    """Dataset-level statistic (needs every window): for each candidate
    policy, its MEAN value for `objective` across every window it has a
    valid value in. Mirrors selector/labels.py's `label_windows` own
    `regret_to_best_fixed` computation, generalized to any objective.
    Returns {policy_name: mean_value} -- the caller picks
    max/min(..., key=sign) themselves per objective direction."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for window_outcomes in all_window_outcomes:
        for pname, val in _valid_values(window_outcomes, objective).items():
            sums[pname] = sums.get(pname, 0.0) + val
            counts[pname] = counts.get(pname, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def best_fixed_policy_and_value(
    all_window_outcomes: List[List[PolicyOutcomeVector]], objective: Objective,
) -> Optional[tuple]:
    means = compute_best_fixed_policy_values(all_window_outcomes, objective)
    if not means:
        return None
    sign = 1.0 if objective.higher_is_better else -1.0
    best_name = max(means, key=lambda n: sign * means[n])
    return best_name, means[best_name]

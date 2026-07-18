"""Adaptive scenario-search utilities for Selector Dataset v2."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from .schema import WindowRecordV2
from .scenario_families import ScenarioFamilySpec
from .scenario_redesign import DISCRIMINATIVE_POOL, REPRESENTATIVE_POOL, with_pool


WG_OBJECTIVE = "weighted_goodput"
DISCRIMINATIVE_CLASSES = {
    "MODERATELY_DISCRIMINATIVE",
    "STRONGLY_DISCRIMINATIVE",
}


@dataclass(frozen=True)
class TrialSummary:
    family_id: str
    seed: int
    bottleneck_class: str | None
    source_trace: str
    request_plan_ancestor_id: str | None
    num_windows: int
    class_counts: Dict[str, int]
    winner_counts: Dict[str, int]
    max_spread: float
    mean_spread: float
    mean_best_score: float
    mean_best_fixed_score: float
    oracle_headroom: float
    retained_pool: str | None = None
    retention_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_trial(
    spec: ScenarioFamilySpec,
    seed: int,
    records: Sequence[WindowRecordV2],
) -> TrialSummary:
    class_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    spreads: list[float] = []
    best_scores: list[float] = []
    policy_values: dict[str, list[float]] = {}

    for record in records:
        disc = next((d for d in record.discriminativeness if d.objective_name == WG_OBJECTIVE), None)
        if disc is None:
            continue
        class_counts[disc.classification] += 1
        winner_counts[disc.best_policy] += 1
        spreads.append(disc.max_min_spread)
        best_scores.append(disc.best_value)
        for outcome in record.outcomes:
            if outcome.weighted_goodput is not None:
                policy_values.setdefault(outcome.policy_name, []).append(outcome.weighted_goodput)

    means = {
        policy: sum(vals) / len(vals)
        for policy, vals in policy_values.items()
        if vals
    }
    best_fixed = max(means.values()) if means else 0.0
    oracle = sum(best_scores) / len(best_scores) if best_scores else 0.0
    return TrialSummary(
        family_id=spec.family_id,
        seed=seed,
        bottleneck_class=spec.bottleneck_class,
        source_trace=spec.source_trace,
        request_plan_ancestor_id=spec.request_plan_ancestor_id,
        num_windows=len(records),
        class_counts=dict(class_counts),
        winner_counts=dict(winner_counts),
        max_spread=max(spreads) if spreads else 0.0,
        mean_spread=sum(spreads) / len(spreads) if spreads else 0.0,
        mean_best_score=oracle,
        mean_best_fixed_score=best_fixed,
        oracle_headroom=oracle - best_fixed,
    )


def retained_pool_for_trial(
    summary: TrialSummary,
    current_winner_counts: Counter[str],
    representative_windows: int,
    discriminative_windows: int,
    *,
    max_representative_fraction: float = 0.30,
) -> tuple[str | None, str | None]:
    disc_windows = sum(summary.class_counts.get(cls, 0) for cls in DISCRIMINATIVE_CLASSES)
    all_complete = summary.class_counts.get("ALL_COMPLETE_OR_EFFECTIVELY_TIED", 0)

    if disc_windows > 0 and summary.max_spread >= 0.01:
        winners = [p for p, c in summary.winner_counts.items() if c > 0]
        underrepresented = any(current_winner_counts.get(p, 0) == 0 for p in winners)
        if underrepresented:
            return DISCRIMINATIVE_POOL, "discriminative_underrepresented_winner"
        if summary.oracle_headroom >= 0.005:
            return DISCRIMINATIVE_POOL, "discriminative_oracle_headroom"
        if disc_windows >= max(2, summary.num_windows // 4):
            return DISCRIMINATIVE_POOL, "discriminative_density"

    total_after = representative_windows + discriminative_windows + summary.num_windows
    representative_share_after = (representative_windows + summary.num_windows) / max(total_after, 1)
    if all_complete < summary.num_windows and representative_share_after <= max_representative_fraction:
        return REPRESENTATIVE_POOL, "nontrivial_representative"
    if summary.source_trace != "synthetic" and representative_share_after <= max_representative_fraction:
        return REPRESENTATIVE_POOL, "real_trace_representative_cap"
    return None, "redundant_or_equivalent"


def attach_retention(
    summary: TrialSummary,
    pool: str | None,
    reason: str | None,
) -> TrialSummary:
    return TrialSummary(
        family_id=summary.family_id,
        seed=summary.seed,
        bottleneck_class=summary.bottleneck_class,
        source_trace=summary.source_trace,
        request_plan_ancestor_id=summary.request_plan_ancestor_id,
        num_windows=summary.num_windows,
        class_counts=summary.class_counts,
        winner_counts=summary.winner_counts,
        max_spread=summary.max_spread,
        mean_spread=summary.mean_spread,
        mean_best_score=summary.mean_best_score,
        mean_best_fixed_score=summary.mean_best_fixed_score,
        oracle_headroom=summary.oracle_headroom,
        retained_pool=pool,
        retention_reason=reason,
    )


def spec_with_retained_pool(spec: ScenarioFamilySpec, pool: str) -> ScenarioFamilySpec:
    return with_pool(spec, pool)

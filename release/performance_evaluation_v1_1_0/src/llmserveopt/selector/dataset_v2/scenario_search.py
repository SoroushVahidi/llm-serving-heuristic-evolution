"""Adaptive scenario-search utilities for Selector Dataset v2."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

from .schema import WindowRecordV2
from .discriminativeness import PRIMARY_SELECTOR_OBJECTIVE
from .scenario_families import ScenarioFamilySpec
from .scenario_redesign import DISCRIMINATIVE_POOL, REPRESENTATIVE_POOL, with_pool


PRIMARY_OBJECTIVE = PRIMARY_SELECTOR_OBJECTIVE
WG_OBJECTIVE = PRIMARY_OBJECTIVE
DISCRIMINATIVE_CLASSES = {
    "MODERATELY_DISCRIMINATIVE",
    "STRONGLY_DISCRIMINATIVE",
}
STRONG_CLASS = "STRONGLY_DISCRIMINATIVE"


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
    strong_winner_counts: Dict[str, int] = field(default_factory=dict)
    retained_pool: str | None = None
    retention_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_trial(
    spec: ScenarioFamilySpec,
    seed: int,
    records: Sequence[WindowRecordV2],
    objective_name: str = PRIMARY_OBJECTIVE,
) -> TrialSummary:
    class_counts: Counter[str] = Counter()
    winner_counts: Counter[str] = Counter()
    strong_winner_counts: Counter[str] = Counter()
    spreads: list[float] = []
    best_scores: list[float] = []
    policy_values: dict[str, list[float]] = {}

    for record in records:
        disc = next((d for d in record.discriminativeness if d.objective_name == objective_name), None)
        if disc is None:
            continue
        class_counts[disc.classification] += 1
        winner_counts[disc.best_policy] += 1
        if disc.classification == STRONG_CLASS:
            strong_winner_counts[disc.best_policy] += 1
        spreads.append(disc.max_min_spread)
        best_scores.append(disc.best_value)
        for outcome in record.outcomes:
            value = getattr(outcome, objective_name, None)
            if value is not None:
                policy_values.setdefault(outcome.policy_name, []).append(value)

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
        strong_winner_counts=dict(strong_winner_counts),
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


def diversity_aware_retained_pool_for_trial(
    summary: TrialSummary,
    current_winner_counts: Counter[str],
    strong_winner_counts: Counter[str],
    representative_windows: int,
    discriminative_windows: int,
    *,
    target_policies: set[str] | None = None,
    max_representative_fraction: float = 0.30,
    max_single_strong_winner_share: float = 0.85,
    dominant_policy: str = "scorpio_style_slo_guard",
) -> tuple[str | None, str | None]:
    """Retain informative trials while rewarding discovered specialization.

    This function does not rebalance labels. It only changes which already
    observed simulation outcomes are retained: strong/moderate wins by
    policies that are rare in the retained pool get preference, and additional
    windows from a dominant policy are capped once they stop adding diversity.
    """
    disc_windows = sum(summary.class_counts.get(cls, 0) for cls in DISCRIMINATIVE_CLASSES)
    strong_windows = summary.class_counts.get(STRONG_CLASS, 0)
    all_complete = summary.class_counts.get("ALL_COMPLETE_OR_EFFECTIVELY_TIED", 0)
    if target_policies is None:
        target_policies = set()

    winners = {p for p, count in summary.winner_counts.items() if count > 0}
    strong_winners = {p for p, count in summary.strong_winner_counts.items() if count > 0}
    non_dominant_winners = winners - {dominant_policy}
    strong_non_dominant_winners = strong_winners - {dominant_policy}
    target_winners = strong_winners & target_policies

    if strong_windows > 0 and summary.max_spread >= 0.02:
        if target_winners:
            return DISCRIMINATIVE_POOL, "strong_target_policy_winner"
        if strong_non_dominant_winners and any(strong_winner_counts.get(p, 0) < 10 for p in strong_non_dominant_winners):
            return DISCRIMINATIVE_POOL, "strong_underrepresented_winner"
        if dominant_policy not in strong_winners:
            return DISCRIMINATIVE_POOL, "strong_non_dominant_winner"

        retained_strong = sum(strong_winner_counts.values())
        dominant_share = (
            strong_winner_counts.get(dominant_policy, 0) / retained_strong
            if retained_strong else 0.0
        )
        if dominant_share < max_single_strong_winner_share and summary.oracle_headroom >= 0.01:
            return DISCRIMINATIVE_POOL, "strong_dominant_below_cap"
        return None, "skipped_dominant_strong_winner_cap"

    if disc_windows > 0 and summary.max_spread >= 0.005:
        if target_winners:
            return DISCRIMINATIVE_POOL, "moderate_target_policy_winner"
        if non_dominant_winners and any(current_winner_counts.get(p, 0) < 10 for p in non_dominant_winners):
            return DISCRIMINATIVE_POOL, "moderate_underrepresented_winner"
        if summary.oracle_headroom >= 0.01 and dominant_policy not in winners:
            return DISCRIMINATIVE_POOL, "moderate_non_dominant_headroom"

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
        strong_winner_counts=summary.strong_winner_counts,
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


def strongly_discriminative_winner_counts(
    records: Sequence[WindowRecordV2],
    objective_name: str = PRIMARY_OBJECTIVE,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        disc = next((d for d in record.discriminativeness if d.objective_name == objective_name), None)
        if disc is not None and disc.classification == STRONG_CLASS:
            counts[disc.best_policy] += 1
    return counts

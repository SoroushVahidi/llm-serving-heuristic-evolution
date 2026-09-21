"""Parent selection and composition-gating utilities for structural synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ParentScoreConfig:
    alpha_expected_advantage: float = 1.0
    beta_complementarity: float = 0.5
    gamma_marginal_frontier_value: float = 1.0
    delta_incompatibility: float = 1.0
    eta_uncertainty: float = 0.25


@dataclass(frozen=True)
class ParentEvidence:
    expected_advantage: Mapping[tuple[str, str], float]
    complementarity: Mapping[tuple[str, str], float]
    marginal_frontier_value: Mapping[str, float]
    incompatibility: Mapping[tuple[str, str], float]
    uncertainty: Mapping[tuple[str, str], float]


@dataclass(frozen=True)
class ParentPairScore:
    parent_a: str
    parent_b: str
    score: float
    expected_advantage: float
    complementarity: float
    marginal_frontier_value: float
    incompatibility: float
    uncertainty: float


@dataclass(frozen=True)
class CompositionGateConfig:
    min_top1_top2_margin: float = 0.005
    min_pairwise_advantage: float = 0.002
    min_complementarity: float = 0.10
    max_incompatibility: float = 0.50
    min_uncertainty_for_composition: float = 0.0


@dataclass(frozen=True)
class CompositionGateDecision:
    action: str
    reason: str
    top1_top2_margin: float
    parent_pair_score: float


def _pair_value(values: Mapping[tuple[str, str], float], a: str, b: str, default: float = 0.0) -> float:
    return float(values.get((a, b), values.get((b, a), default)))


def score_parent_pair(a: str, b: str, evidence: ParentEvidence, config: ParentScoreConfig = ParentScoreConfig()) -> ParentPairScore:
    expected_advantage = _pair_value(evidence.expected_advantage, a, b)
    complementarity = _pair_value(evidence.complementarity, a, b)
    marginal = max(float(evidence.marginal_frontier_value.get(a, 0.0)), float(evidence.marginal_frontier_value.get(b, 0.0)))
    incompatibility = _pair_value(evidence.incompatibility, a, b)
    uncertainty = _pair_value(evidence.uncertainty, a, b)
    score = (
        config.alpha_expected_advantage * expected_advantage
        + config.beta_complementarity * complementarity
        + config.gamma_marginal_frontier_value * marginal
        - config.delta_incompatibility * incompatibility
        + config.eta_uncertainty * uncertainty
    )
    return ParentPairScore(a, b, score, expected_advantage, complementarity, marginal, incompatibility, uncertainty)


def select_parent_pairs(
    candidates: Sequence[str],
    evidence: ParentEvidence,
    *,
    top_n: int = 5,
    config: ParentScoreConfig = ParentScoreConfig(),
) -> list[ParentPairScore]:
    scores = []
    ordered = sorted(candidates)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            scores.append(score_parent_pair(a, b, evidence, config))
    scores.sort(key=lambda item: (-item.score, item.parent_a, item.parent_b))
    return scores[:top_n]


def composition_gate(
    *,
    top1_top2_margin: float,
    pair_score: ParentPairScore,
    config: CompositionGateConfig = CompositionGateConfig(),
) -> CompositionGateDecision:
    if pair_score.incompatibility > config.max_incompatibility:
        return CompositionGateDecision("SELECT_SINGLE", "parent modules are incompatible", top1_top2_margin, pair_score.score)
    if top1_top2_margin >= config.min_top1_top2_margin and pair_score.uncertainty < config.min_uncertainty_for_composition:
        return CompositionGateDecision("SELECT_SINGLE", "top-1 margin is confident", top1_top2_margin, pair_score.score)
    if pair_score.expected_advantage < config.min_pairwise_advantage and pair_score.complementarity < config.min_complementarity:
        return CompositionGateDecision("SELECT_SINGLE", "weak advantage and complementarity", top1_top2_margin, pair_score.score)
    return CompositionGateDecision("ATTEMPT_STRUCTURAL_COMPOSITION", "pair has enough advantage, complementarity, or uncertainty", top1_top2_margin, pair_score.score)

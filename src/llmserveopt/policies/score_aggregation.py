"""Weighted score aggregation over comparable per-request policy scores.

Companion to the rank-aggregation operators in ``composition.py``. Rank
aggregation is safe for any policy that can produce a deterministic
ordering; score aggregation additionally requires that each participating
expert emit a genuine single comparable scalar (see
``capabilities.SCORE_CAPABLE_EXPERTS``), because unlike ranks, raw scores
must be normalized onto a shared scale before they can be combined.

Convention: every score function in this module returns higher = more
preferred to schedule first, matching the normalized-rank convention used
in ``composition.py`` (1.0 = most preferred).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Sequence, cast

from ..core.action import Action
from ..core.types import ObservableState
from .base import BasePolicy
from .capabilities import CapabilityError, require_score_capable
from .composition import (
    CompositionDecisionLog,
    CompositionError,
    RankExpertSpec,
    _normalize_weights,
    _weight_entropy,
)
from .policy_library_v2_helpers import deterministic_place
from .scoring import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    predicted_service_proxy,
    urgency_score,
    weighted_shortest_processing_score,
)
from .weighted_shortest_processing import WeightedShortestProcessingPolicy

_EPS = 1e-12


class NormalizationMode(str, Enum):
    NONE = "none"
    MIN_MAX = "min_max"
    ZSCORE = "zscore"
    ROBUST_MAD = "robust_mad"


def score_with_named_expert(
    name: str,
    state: ObservableState,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> Dict[int, float]:
    """Return {request_id: score} for a score-capable named expert.

    Higher score = more preferred. Raises CapabilityError for any name not
    in SCORE_CAPABLE_EXPERTS rather than silently returning an empty or
    fabricated result -- callers that want missing-as-empty behavior (as
    rank_with_named_expert provides for rank-only experts) should check
    capabilities.capabilities_for(name) first.
    """
    require_score_capable([name])
    requests = list(state.waiting_queue)
    if not requests:
        return {}
    now = state.time

    if name == "fifo":
        return {r.request_id: -r.arrival_time for r in requests}
    if name == "edf":
        return {r.request_id: -r.slo_deadline for r in requests}
    if name == "shortest_output_first":
        return {r.request_id: -float(r.predicted_output_tokens) for r in requests}
    if name == "shortest_prompt_first":
        return {r.request_id: -float(r.prompt_tokens) for r in requests}
    if name == "weighted_shortest_processing":
        return {r.request_id: -weighted_shortest_processing_score(r, alpha, beta) for r in requests}
    if name == "estimated_service_time_first":
        return {r.request_id: -predicted_service_proxy(r, alpha, beta) for r in requests}
    if name == "least_laxity_first":
        # Mirrors rank_with_named_expert's step_size-scaled laxity primary key.
        step_size = 0.001
        return {
            r.request_id: -(r.slo_deadline - now - predicted_service_proxy(r, alpha, beta) * step_size)
            for r in requests
        }
    if name == "slo_slack_score":
        return {r.request_id: urgency_score(r, now, alpha, beta) + r.priority for r in requests}
    raise CapabilityError(f"score_with_named_expert has no formula registered for {name!r}")


def normalize_scores(scores: Mapping[int, float], mode: NormalizationMode | str) -> Dict[int, float]:
    """Normalize a {request_id: raw_score} mapping under the given mode.

    Handles degenerate cases explicitly rather than raising:
    * empty input -> {}
    * constant vector under MIN_MAX/ZSCORE -> every value maps to 0.5 (MIN_MAX)
      or 0.0 (ZSCORE), since there is no meaningful spread to normalize against.
    * zero MAD under ROBUST_MAD -> falls back to 0.0 for every value (median
      absolute deviation cannot distinguish values around the median).

    Raises CompositionError if any score is NaN or +/-inf: silently
    propagating a non-finite value into an aggregate score would produce an
    unpredictable final ranking, which is unsafe.
    """
    mode = NormalizationMode(mode)
    if not scores:
        return {}
    for request_id, value in scores.items():
        if not math.isfinite(value):
            raise CompositionError(f"Non-finite score {value!r} for request_id={request_id}")

    if mode is NormalizationMode.NONE:
        return dict(scores)

    values = list(scores.values())
    if mode is NormalizationMode.MIN_MAX:
        lo, hi = min(values), max(values)
        if hi - lo <= _EPS:
            return {rid: 0.5 for rid in scores}
        return {rid: (v - lo) / (hi - lo) for rid, v in scores.items()}

    if mode is NormalizationMode.ZSCORE:
        mean = statistics.fmean(values)
        if len(values) < 2:
            return {rid: 0.0 for rid in scores}
        std = statistics.pstdev(values)
        if std <= _EPS:
            return {rid: 0.0 for rid in scores}
        return {rid: (v - mean) / std for rid, v in scores.items()}

    if mode is NormalizationMode.ROBUST_MAD:
        median = statistics.median(values)
        abs_devs = [abs(v - median) for v in values]
        mad = statistics.median(abs_devs)
        if mad <= _EPS:
            return {rid: 0.0 for rid in scores}
        # 1.4826 is the standard consistency constant making MAD a
        # normal-consistent estimator of scale, matching common robust-z usage.
        scale = 1.4826 * mad
        return {rid: (v - median) / scale for rid, v in scores.items()}

    raise CompositionError(f"Unhandled normalization mode {mode!r}")


@dataclass
class ScoreExpertSpec:
    name: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScoreAggregationResult:
    aggregate: Dict[int, float]
    normalized_by_expert: Dict[str, Dict[int, float]]
    weights: Dict[str, float]
    ranked_request_ids: list[int]


def weighted_score_aggregate(
    expert_scores: Mapping[str, Mapping[int, float]],
    weights: Mapping[str, float],
    *,
    normalization: NormalizationMode | str = NormalizationMode.MIN_MAX,
) -> ScoreAggregationResult:
    """Normalize each expert's scores independently, then combine by weight.

    A request absent from a given expert's score mapping contributes
    nothing from that expert (same missing-value convention as the rank
    aggregators). Requests absent from every active expert do not appear in
    the result.
    """
    normalized_by_expert: Dict[str, Dict[int, float]] = {
        name: normalize_scores(expert_scores[name], normalization) for name in weights
    }
    aggregate: Dict[int, float] = {}
    for name, weight in weights.items():
        for request_id, value in normalized_by_expert[name].items():
            aggregate[request_id] = aggregate.get(request_id, 0.0) + weight * value
    ranked_ids = [rid for rid, _ in sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))]
    return ScoreAggregationResult(
        aggregate=aggregate,
        normalized_by_expert=normalized_by_expert,
        weights=dict(weights),
        ranked_request_ids=ranked_ids,
    )


def build_score_weights(
    specs: Sequence[ScoreExpertSpec],
    *,
    top_k: int | None = None,
) -> Dict[str, float]:
    """Validate and normalize score-expert weights to sum to one.

    Reuses composition._normalize_weights (nonnegative-finite validation,
    deterministic top_k-by-weight selection, deterministic sum-to-one
    normalization) so score and rank ensembles share identical weight
    semantics.
    """
    require_score_capable([s.name for s in specs if s.enabled and s.weight > 0.0])
    # _normalize_weights only reads .name/.weight/.enabled at runtime, so
    # ScoreExpertSpec (identical shape to composition.RankExpertSpec) works
    # unchanged; the cast is for static typing only.
    return _normalize_weights(cast(Sequence[RankExpertSpec], specs), top_k=top_k)


class StaticScoreEnsemblePolicy(BasePolicy):
    """Weighted normalized-score ensemble over score-capable experts.

    Sibling of composition.StaticRankEnsemblePolicy: same sparse top-k
    weight handling, deterministic fallback, feasibility projection via
    deterministic_place, and CompositionDecisionLog instrumentation, but
    aggregating normalized comparable scores instead of normalized ranks.
    """

    name = "composition_static_score_ensemble"

    def __init__(
        self,
        experts: Sequence[ScoreExpertSpec],
        *,
        normalization: NormalizationMode | str = NormalizationMode.MIN_MAX,
        fallback_policy: BasePolicy | None = None,
        top_k: int | None = None,
        max_admits: int | None = None,
    ) -> None:
        require_score_capable([e.name for e in experts])
        self.experts = list(experts)
        self.normalization = NormalizationMode(normalization)
        self.fallback_policy = fallback_policy or WeightedShortestProcessingPolicy()
        self.top_k = top_k
        self.max_admits = max_admits
        self.decision_logs: list[CompositionDecisionLog] = []

    def reset(self) -> None:
        self.decision_logs.clear()
        self.fallback_policy.reset()

    def select_action(self, state: ObservableState) -> Action:
        weights = build_score_weights(self.experts, top_k=self.top_k)
        by_id = {req.request_id: req for req in state.waiting_queue}

        if not weights or not state.waiting_queue:
            action = self.fallback_policy.select_action(state)
            self._log(state, weights, {}, True, "no active experts or empty queue", action)
            return action

        expert_scores = {name: score_with_named_expert(name, state) for name in weights}
        result = weighted_score_aggregate(expert_scores, weights, normalization=self.normalization)

        ranked = [
            by_id[rid]
            for rid in sorted(
                result.ranked_request_ids,
                key=lambda rid: (-result.aggregate[rid], by_id[rid].arrival_time, rid),
            )
        ]
        contributions = {
            rid: {name: result.normalized_by_expert[name].get(rid, 0.0) * weights[name] for name in weights}
            for rid in result.aggregate
        }

        action = deterministic_place(state, ranked, max_admits=self.max_admits)
        fallback_used = False
        reason = None
        if action.is_empty() and state.waiting_queue:
            fallback_used = True
            reason = "score composition produced no feasible admission"
            action = self.fallback_policy.select_action(state)
        self._log(state, weights, contributions, fallback_used, reason, action)
        return action

    def _log(
        self,
        state: ObservableState,
        weights: Dict[str, float],
        contributions: Dict[int, Dict[str, float]],
        fallback_used: bool,
        reason: str | None,
        action: Action,
    ) -> None:
        dominant = max(weights.items(), key=lambda item: (item[1], item[0]))[0] if weights else None
        self.decision_logs.append(
            CompositionDecisionLog(
                policy_name=self.name,
                step=state.step,
                selected_request_ids=sorted(action.all_admitted_ids()),
                expert_weights=weights,
                expert_contributions=contributions,
                fallback_used=fallback_used,
                weight_entropy=_weight_entropy(weights),
                dominant_expert=dominant,
                switching_count=0,
                invalid_reason=reason,
            )
        )

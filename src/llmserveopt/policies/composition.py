"""Experimental policy-composition harness.

This module deliberately does not register composed policies in the baseline
registry.  It provides typed building blocks for composition experiments while
preserving the behavior of every existing production policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, Mapping, Sequence

from ..core.action import Action
from ..core.types import ObservableGPUState, ObservableRequest, ObservableState
from .base import BasePolicy
from .policy_library_v2_helpers import deterministic_place, gpu_pressure, system_pressure
from .registry import make_policy, make_policy_library_v2
from .scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy
from .scoring import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    predicted_service_proxy,
    urgency_score,
    weighted_shortest_processing_score,
)
from .weighted_shortest_processing import WeightedShortestProcessingPolicy

_EPS = 1e-12


class CompositionError(ValueError):
    """Raised when a requested policy composition is invalid."""


class ModuleKind(str, Enum):
    ADMISSION = "admission"
    PRIORITY = "priority"
    BATCHING_PREFILL = "batching_prefill"
    PLACEMENT = "placement"
    PREEMPTION = "preemption"
    KV_CACHE = "kv_cache"
    FAIRNESS_AGING = "fairness_aging"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CompositionModuleSpec:
    name: str
    kind: ModuleKind
    input_type: str
    output_type: str
    supported: bool
    strictly_causal: bool
    notes: str = ""


@dataclass(frozen=True)
class RankExpertSpec:
    name: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class RankExpertOutput:
    expert_name: str
    normalized_ranks: Dict[int, float]
    ranked_request_ids: list[int]
    missing_request_ids: list[int] = field(default_factory=list)


@dataclass
class CompositionDecisionLog:
    policy_name: str
    step: int
    selected_request_ids: list[int]
    expert_weights: Dict[str, float]
    expert_contributions: Dict[int, Dict[str, float]]
    fallback_used: bool
    weight_entropy: float
    dominant_expert: str | None
    switching_count: int
    invalid_reason: str | None = None


@dataclass(frozen=True)
class ContextualWeightRule:
    expert_name: str
    intercept: float = 0.0
    coefficients: Mapping[str, float] = field(default_factory=dict)


def default_module_specs() -> list[CompositionModuleSpec]:
    """Return the typed module surface currently supported by the simulator."""
    return [
        CompositionModuleSpec("AdmissionRule", ModuleKind.ADMISSION, "ObservableState, ObservableRequest", "admit|skip|defer", True, True),
        CompositionModuleSpec("PriorityRule", ModuleKind.PRIORITY, "ObservableState, Sequence[ObservableRequest]", "ranked requests or normalized ranks", True, True),
        CompositionModuleSpec("BatchingPrefillRule", ModuleKind.BATCHING_PREFILL, "ObservableState, ranked requests", "candidate subset or budget", True, True, "Only admission/budget approximations are supported; no literal chunk-size action."),
        CompositionModuleSpec("PlacementRule", ModuleKind.PLACEMENT, "ObservableState, ObservableRequest", "gpu_id or infeasible", True, True),
        CompositionModuleSpec("PreemptionRule", ModuleKind.PREEMPTION, "ObservableState", "Action.preempt", False, True, "Action supports preempt, but current deployable library does not expose reusable preemption modules."),
        CompositionModuleSpec("KVCacheRule", ModuleKind.KV_CACHE, "ObservableState, ObservableGPUState, ObservableRequest", "constraint or placement penalty", True, True, "KV capacity is supported; prefix/cache reuse is not."),
        CompositionModuleSpec("FairnessAgingRule", ModuleKind.FAIRNESS_AGING, "ObservableState, ObservableRequest", "priority bonus or rank key", True, True, "Durable tenant-history fairness needs a small extension."),
        CompositionModuleSpec("CacheReuseRule", ModuleKind.UNSUPPORTED, "prefix/cache state", "cache-affinity action", False, False),
    ]


def _request_by_id(state: ObservableState) -> Dict[int, ObservableRequest]:
    return {req.request_id: req for req in state.waiting_queue}


def _finite_nonnegative_weight(weight: float, *, expert_name: str) -> float:
    if not math.isfinite(weight):
        raise CompositionError(f"Expert {expert_name!r} has non-finite weight {weight!r}")
    if weight < 0.0:
        raise CompositionError(f"Expert {expert_name!r} has negative weight {weight!r}")
    return float(weight)


def _normalize_weights(specs: Sequence[RankExpertSpec], *, top_k: int | None = None) -> Dict[str, float]:
    active: list[tuple[str, float]] = []
    for spec in specs:
        if not spec.enabled:
            continue
        weight = _finite_nonnegative_weight(spec.weight, expert_name=spec.name)
        if weight > 0.0:
            active.append((spec.name, weight))
    active.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        if top_k <= 0:
            raise CompositionError("top_k must be positive when provided")
        active = active[:top_k]
    total = sum(weight for _name, weight in active)
    if total <= 0.0:
        return {}
    return {name: weight / total for name, weight in active}


def _weight_entropy(weights: Mapping[str, float]) -> float:
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for weight in weights.values():
        p = max(0.0, weight) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def _normalized_ranks_from_sorted(
    expert_name: str,
    sorted_requests: Sequence[ObservableRequest],
    all_requests: Sequence[ObservableRequest],
) -> RankExpertOutput:
    ranked_ids = [req.request_id for req in sorted_requests]
    n = len(ranked_ids)
    if n == 0:
        ranks: Dict[int, float] = {}
    elif n == 1:
        ranks = {ranked_ids[0]: 1.0}
    else:
        ranks = {
            request_id: 1.0 - (rank / (n - 1))
            for rank, request_id in enumerate(ranked_ids)
        }
    ranked = set(ranked_ids)
    missing = sorted(req.request_id for req in all_requests if req.request_id not in ranked)
    return RankExpertOutput(expert_name, ranks, ranked_ids, missing)


def _decode_pressure(state: ObservableState) -> float:
    if not state.gpu_states:
        return 0.0
    return max(g.decoding_count / max(g.max_active_sequences, 1) for g in state.gpu_states)


def _prefill_pressure(state: ObservableState) -> float:
    if not state.gpu_states:
        return 0.0
    return max(g.prefilling_count / max(g.max_active_sequences, 1) for g in state.gpu_states)


def _kv_pressure(state: ObservableState) -> float:
    if not state.gpu_states:
        return 0.0
    return max(g.current_kv_tokens / max(g.max_kv_tokens, 1) for g in state.gpu_states)


def _queue_pressure(state: ObservableState) -> float:
    cap = sum(g.max_active_sequences for g in state.gpu_states)
    return len(state.waiting_queue) / max(cap, 1)


def causal_context_features(state: ObservableState) -> Dict[str, float]:
    """Small causal feature vector for contextual composition weights."""
    waiting = state.waiting_queue
    if waiting:
        mean_prompt = sum(r.prompt_tokens for r in waiting) / len(waiting)
        mean_output = sum(r.predicted_output_tokens for r in waiting) / len(waiting)
        urgent_frac = sum(1 for r in waiting if r.slo_deadline <= state.time) / len(waiting)
    else:
        mean_prompt = 0.0
        mean_output = 0.0
        urgent_frac = 0.0
    return {
        "system_pressure": system_pressure(state),
        "queue_pressure": _queue_pressure(state),
        "kv_pressure": _kv_pressure(state),
        "decode_pressure": _decode_pressure(state),
        "prefill_pressure": _prefill_pressure(state),
        "mean_prompt_tokens": mean_prompt,
        "mean_predicted_output_tokens": mean_output,
        "urgent_deadline_fraction": urgent_frac,
    }


def rank_with_named_expert(
    name: str,
    state: ObservableState,
    *,
    step_size: float = 0.001,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> RankExpertOutput:
    """Return normalized ranks for a supported causal priority expert.

    Higher normalized rank means stronger preference.  Unsupported experts are
    represented as missing outputs for every request rather than as raw scores.
    """
    requests = list(state.waiting_queue)
    if not requests:
        return RankExpertOutput(name, {}, [], [])

    if name == "fifo":
        ranked = sorted(requests, key=lambda r: (r.arrival_time, r.request_id))
    elif name == "edf":
        ranked = sorted(requests, key=lambda r: (r.slo_deadline, r.arrival_time, r.request_id))
    elif name == "shortest_output_first":
        ranked = sorted(requests, key=lambda r: (r.predicted_output_tokens, r.arrival_time, r.request_id))
    elif name == "shortest_prompt_first":
        ranked = sorted(requests, key=lambda r: (r.prompt_tokens, r.arrival_time, r.request_id))
    elif name == "weighted_shortest_processing":
        ranked = sorted(requests, key=lambda r: (weighted_shortest_processing_score(r, alpha, beta), r.arrival_time, r.request_id))
    elif name == "estimated_service_time_first":
        ranked = sorted(requests, key=lambda r: (predicted_service_proxy(r, alpha, beta), r.slo_deadline, -r.priority, r.arrival_time, r.request_id))
    elif name == "least_laxity_first":
        ranked = sorted(
            requests,
            key=lambda r: (
                r.slo_deadline - state.time - predicted_service_proxy(r, alpha, beta) * step_size,
                r.slo_deadline,
                -r.priority,
                r.arrival_time,
                r.request_id,
            ),
        )
    elif name == "slo_slack_score":
        ranked = sorted(
            requests,
            key=lambda r: (-urgency_score(r, state.time, alpha, beta), -r.priority, r.arrival_time, r.request_id),
        )
    elif name == "aging_priority":
        ranked = sorted(
            requests,
            key=lambda r: (
                -((r.priority + max(0.0, state.time - r.arrival_time)) / max(predicted_service_proxy(r, alpha, beta), _EPS)),
                r.arrival_time,
                r.request_id,
            ),
        )
    elif name == "kv_constrained_online":
        ranked = sorted(
            requests,
            key=lambda r: (
                -(1.0 / max(r.slo_deadline - state.time, _EPS)) / max(r.prompt_tokens + r.predicted_output_tokens, 1),
                r.arrival_time,
                r.request_id,
            ),
        )
    elif name == "adaptive_chunked_prefill":
        pressure = system_pressure(state)
        ranked = sorted(
            requests,
            key=lambda r: (
                r.prompt_tokens * (1.0 + pressure),
                r.predicted_output_tokens,
                r.arrival_time,
                r.request_id,
            ),
        )
    elif name == "scorpio_style_slo_guard":
        scorpio = ScorpioStyleSloGuardPolicy(step_size=step_size, alpha=alpha, beta=beta)
        kv_p, dec_p, queue_p = scorpio._system_pressures(state)
        candidates: list[ObservableRequest] = []
        laxities: list[float] = []
        for req in requests:
            lax = scorpio._laxity(req, state.time)
            ttft_slack = scorpio._ttft_proxy_slack(req, state.time)
            if lax < -scorpio.laxity_threshold or ttft_slack < -scorpio.ttft_slack_threshold:
                continue
            candidates.append(req)
            laxities.append(lax)
        if not candidates:
            return _normalized_ranks_from_sorted(name, [], requests)
        guard_active = scorpio._guard_active(kv_p, dec_p, queue_p, sum(laxities) / len(laxities))
        if guard_active and kv_p >= scorpio.kv_utilization_threshold:
            candidates = [
                req for req in candidates
                if req.predicted_output_tokens <= scorpio.long_decode_token_threshold
                or scorpio._laxity(req, state.time) < 0.5
            ]
        ranked = sorted(candidates, key=lambda r: scorpio._sort_key(r, state.time, guard_active, dec_p))
    else:
        return RankExpertOutput(name, {}, [], sorted(req.request_id for req in requests))

    return _normalized_ranks_from_sorted(name, ranked, requests)


class StaticRankEnsemblePolicy(BasePolicy):
    """Weighted normalized-rank ensemble over compatible causal priority experts."""

    name = "composition_static_rank_ensemble"

    def __init__(
        self,
        experts: Sequence[RankExpertSpec],
        *,
        fallback_policy: BasePolicy | None = None,
        top_k: int | None = None,
        min_expert_support: int = 1,
        max_admits: int | None = None,
    ) -> None:
        self.experts = list(experts)
        self.fallback_policy = fallback_policy or WeightedShortestProcessingPolicy()
        self.top_k = top_k
        self.min_expert_support = min_expert_support
        self.max_admits = max_admits
        self.decision_logs: list[CompositionDecisionLog] = []
        self._last_dominant_expert: str | None = None
        self.switching_count = 0

    def reset(self) -> None:
        self.decision_logs.clear()
        self._last_dominant_expert = None
        self.switching_count = 0
        self.fallback_policy.reset()

    def _current_weights(self, state: ObservableState) -> Dict[str, float]:
        return _normalize_weights(self.experts, top_k=self.top_k)

    def _ranked_requests_and_log(self, state: ObservableState) -> tuple[list[ObservableRequest], CompositionDecisionLog]:
        weights = self._current_weights(state)
        if not weights:
            return [], self._make_log(state, {}, {}, True, "all expert weights are zero")

        by_id = _request_by_id(state)
        aggregate: Dict[int, float] = {req.request_id: 0.0 for req in state.waiting_queue}
        support: Dict[int, int] = {req.request_id: 0 for req in state.waiting_queue}
        contributions: Dict[int, Dict[str, float]] = {req.request_id: {} for req in state.waiting_queue}

        for expert_name, weight in weights.items():
            expert_output = rank_with_named_expert(expert_name, state)
            for request_id, normalized_rank in expert_output.normalized_ranks.items():
                value = weight * normalized_rank
                aggregate[request_id] = aggregate.get(request_id, 0.0) + value
                support[request_id] = support.get(request_id, 0) + 1
                contributions.setdefault(request_id, {})[expert_name] = value

        ranked_ids = [
            request_id
            for request_id, value in sorted(
                aggregate.items(),
                key=lambda item: (-item[1], -support[item[0]], by_id[item[0]].arrival_time, item[0]),
            )
            if support.get(request_id, 0) >= self.min_expert_support
        ]
        ranked = [by_id[request_id] for request_id in ranked_ids]
        dominant = max(weights.items(), key=lambda item: (item[1], item[0]))[0] if weights else None
        if dominant != self._last_dominant_expert and self._last_dominant_expert is not None:
            self.switching_count += 1
        self._last_dominant_expert = dominant
        log = CompositionDecisionLog(
            policy_name=self.name,
            step=state.step,
            selected_request_ids=[],
            expert_weights=weights,
            expert_contributions=contributions,
            fallback_used=False,
            weight_entropy=_weight_entropy(weights),
            dominant_expert=dominant,
            switching_count=self.switching_count,
        )
        if not ranked:
            log.fallback_used = True
            log.invalid_reason = "no request had enough expert support"
        return ranked, log

    def _make_log(
        self,
        state: ObservableState,
        weights: Dict[str, float],
        contributions: Dict[int, Dict[str, float]],
        fallback_used: bool,
        reason: str | None,
    ) -> CompositionDecisionLog:
        return CompositionDecisionLog(
            policy_name=self.name,
            step=state.step,
            selected_request_ids=[],
            expert_weights=weights,
            expert_contributions=contributions,
            fallback_used=fallback_used,
            weight_entropy=_weight_entropy(weights),
            dominant_expert=None,
            switching_count=self.switching_count,
            invalid_reason=reason,
        )

    def select_action(self, state: ObservableState) -> Action:
        ranked, log = self._ranked_requests_and_log(state)
        if not ranked:
            action = self.fallback_policy.select_action(state)
            log.selected_request_ids = sorted(action.all_admitted_ids())
            self.decision_logs.append(log)
            return action
        action = deterministic_place(state, ranked, max_admits=self.max_admits)
        if action.is_empty() and state.waiting_queue:
            log.fallback_used = True
            log.invalid_reason = "composition produced no feasible admission"
            action = self.fallback_policy.select_action(state)
        log.selected_request_ids = sorted(action.all_admitted_ids())
        self.decision_logs.append(log)
        return action


class ContextualRankEnsemblePolicy(StaticRankEnsemblePolicy):
    """Rank ensemble whose expert weights are deterministic functions of causal state."""

    name = "composition_contextual_rank_ensemble"

    def __init__(
        self,
        experts: Sequence[RankExpertSpec],
        *,
        contextual_rules: Sequence[ContextualWeightRule] | None = None,
        min_commitment_steps: int = 0,
        fallback_policy: BasePolicy | None = None,
        top_k: int | None = None,
        min_expert_support: int = 1,
        max_admits: int | None = None,
    ) -> None:
        super().__init__(
            experts,
            fallback_policy=fallback_policy,
            top_k=top_k,
            min_expert_support=min_expert_support,
            max_admits=max_admits,
        )
        self.contextual_rules = {rule.expert_name: rule for rule in contextual_rules or []}
        self.min_commitment_steps = min_commitment_steps
        self._committed_until_step = -1
        self._committed_weights: Dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
        self._committed_until_step = -1
        self._committed_weights = {}

    def _current_weights(self, state: ObservableState) -> Dict[str, float]:
        if self.min_commitment_steps > 0 and state.step <= self._committed_until_step:
            return dict(self._committed_weights)
        base = _normalize_weights(self.experts, top_k=self.top_k)
        if not base:
            return {}
        features = causal_context_features(state)
        logits: Dict[str, float] = {}
        for expert_name, base_weight in base.items():
            rule = self.contextual_rules.get(expert_name)
            logit = math.log(max(base_weight, _EPS))
            if rule is not None:
                logit += rule.intercept
                for feature, coefficient in rule.coefficients.items():
                    logit += coefficient * features.get(feature, 0.0)
            logits[expert_name] = logit
        max_logit = max(logits.values())
        exp_values = {name: math.exp(value - max_logit) for name, value in logits.items()}
        denom = sum(exp_values.values())
        weights = {name: value / denom for name, value in exp_values.items()}
        if self.min_commitment_steps > 0:
            self._committed_weights = dict(weights)
            self._committed_until_step = state.step + self.min_commitment_steps - 1
        return weights


class ScorpioAdmissionComponent:
    """Reusable SCORPIO-style causal admission filter and budget."""

    def __init__(self, scorpio: ScorpioStyleSloGuardPolicy | None = None) -> None:
        self.scorpio = scorpio or ScorpioStyleSloGuardPolicy()

    def reset(self) -> None:
        self.scorpio.reset()

    def filter(self, state: ObservableState) -> tuple[list[ObservableRequest], int | None]:
        self.scorpio._admission_budget = min(
            self.scorpio.admission_budget_max,
            self.scorpio._admission_budget + self.scorpio.admission_budget_refill,
        )
        kv_p, dec_p, queue_p = self.scorpio._system_pressures(state)
        candidates: list[ObservableRequest] = []
        laxities: list[float] = []
        for req in state.waiting_queue:
            lax = self.scorpio._laxity(req, state.time)
            ttft_slack = self.scorpio._ttft_proxy_slack(req, state.time)
            if lax < -self.scorpio.laxity_threshold or ttft_slack < -self.scorpio.ttft_slack_threshold:
                continue
            candidates.append(req)
            laxities.append(lax)
        if not candidates:
            return [], 0
        mean_laxity = sum(laxities) / len(laxities)
        guard_active = self.scorpio._guard_active(kv_p, dec_p, queue_p, mean_laxity)
        if guard_active and kv_p >= self.scorpio.kv_utilization_threshold:
            candidates = [
                req for req in candidates
                if req.predicted_output_tokens <= self.scorpio.long_decode_token_threshold
                or self.scorpio._laxity(req, state.time) < 0.5
            ]
        max_admits = max(1, int(self.scorpio._admission_budget)) if guard_active else None
        return candidates, max_admits

    def consume(self, admitted_count: int) -> None:
        self.scorpio._admission_budget = max(
            0.0,
            self.scorpio._admission_budget - admitted_count * self.scorpio.admission_cost,
        )


class KVReservePlacementComponent:
    def __init__(self, target_kv_utilization: float = 0.90) -> None:
        if target_kv_utilization <= 0.0 or target_kv_utilization > 1.0:
            raise CompositionError("target_kv_utilization must be in (0, 1]")
        self.target_kv_utilization = target_kv_utilization

    def gpu_key(self, gpu: ObservableGPUState, req: ObservableRequest) -> tuple:
        post_kv = (gpu.current_kv_tokens + req.prompt_tokens) / max(gpu.max_kv_tokens, 1)
        over_target = max(0.0, post_kv - self.target_kv_utilization)
        return (over_target, post_kv, gpu_pressure(gpu), gpu.gpu_id)

    def admit_filter(self, req: ObservableRequest, gpu: ObservableGPUState, _admitted: list[ObservableRequest]) -> bool:
        post_kv = (gpu.current_kv_tokens + req.prompt_tokens) / max(gpu.max_kv_tokens, 1)
        return post_kv <= self.target_kv_utilization or req.slo_deadline <= 0.0


class AdaptivePrefillGuardComponent:
    def __init__(self, long_prompt_threshold: int = 1024, max_long_prefill_admits: int = 1, pressure_threshold: float = 0.65) -> None:
        self.long_prompt_threshold = long_prompt_threshold
        self.max_long_prefill_admits = max_long_prefill_admits
        self.pressure_threshold = pressure_threshold

    def filter(self, state: ObservableState, candidates: Iterable[ObservableRequest]) -> list[ObservableRequest]:
        pressure = max(system_pressure(state), _prefill_pressure(state))
        if pressure < self.pressure_threshold:
            return list(candidates)
        ranked: list[ObservableRequest] = []
        long_count = 0
        for req in candidates:
            if req.prompt_tokens >= self.long_prompt_threshold:
                if long_count >= self.max_long_prefill_admits:
                    continue
                long_count += 1
            ranked.append(req)
        return ranked


class ComponentWiseCompositionPolicy(BasePolicy):
    """Prototype: SCORPIO-style admission plus WSP ranking plus safe guards."""

    name = "composition_component_wise_scorpio_wsp"

    def __init__(
        self,
        *,
        admission: ScorpioAdmissionComponent | None = None,
        kv: KVReservePlacementComponent | None = None,
        prefill: AdaptivePrefillGuardComponent | None = None,
        fallback_policy: BasePolicy | None = None,
        aging_tiebreak: bool = True,
    ) -> None:
        self.admission = admission or ScorpioAdmissionComponent()
        self.kv = kv or KVReservePlacementComponent()
        self.prefill = prefill or AdaptivePrefillGuardComponent()
        self.fallback_policy = fallback_policy or WeightedShortestProcessingPolicy()
        self.aging_tiebreak = aging_tiebreak
        self.decision_logs: list[CompositionDecisionLog] = []

    def reset(self) -> None:
        self.admission.reset()
        self.fallback_policy.reset()
        self.decision_logs.clear()

    def _priority_key(self, state: ObservableState, req: ObservableRequest) -> tuple:
        age = max(0.0, state.time - req.arrival_time) if self.aging_tiebreak else 0.0
        return (
            weighted_shortest_processing_score(req),
            -age,
            req.slo_deadline,
            r_priority_tiebreak(req),
            req.arrival_time,
            req.request_id,
        )

    def select_action(self, state: ObservableState) -> Action:
        candidates, max_admits = self.admission.filter(state)
        candidates = self.prefill.filter(state, candidates)
        ranked = sorted(candidates, key=lambda req: self._priority_key(state, req))
        action = deterministic_place(
            state,
            ranked,
            gpu_key=self.kv.gpu_key,
            admit_filter=self.kv.admit_filter,
            max_admits=max_admits,
        )
        admitted_count = sum(len(v) for v in action.admit.values())
        self.admission.consume(admitted_count)
        fallback_used = False
        reason = None
        if action.is_empty() and state.waiting_queue:
            fallback_used = True
            reason = "component-wise composition admitted nothing; deterministic fallback used"
            action = self.fallback_policy.select_action(state)
        self.decision_logs.append(
            CompositionDecisionLog(
                policy_name=self.name,
                step=state.step,
                selected_request_ids=sorted(action.all_admitted_ids()),
                expert_weights={"scorpio_admission": 1.0, "wsp_priority": 1.0, "kv_reserve": 1.0, "adaptive_prefill": 1.0},
                expert_contributions={req.request_id: {"wsp_rank_key": float(idx)} for idx, req in enumerate(ranked)},
                fallback_used=fallback_used,
                weight_entropy=0.0,
                dominant_expert="component_wise",
                switching_count=0,
                invalid_reason=reason,
            )
        )
        return action


class ConditionalRegimeCompositionPolicy(BasePolicy):
    """Simple causal regime switch between WSP and SCORPIO/WSP components."""

    name = "composition_conditional_overload_slo"

    def __init__(
        self,
        *,
        queue_pressure_threshold: float = 1.0,
        kv_pressure_threshold: float = 0.70,
        urgent_fraction_threshold: float = 0.10,
        low_pressure_policy: BasePolicy | None = None,
        high_pressure_policy: ComponentWiseCompositionPolicy | None = None,
        min_commitment_steps: int = 0,
    ) -> None:
        self.queue_pressure_threshold = queue_pressure_threshold
        self.kv_pressure_threshold = kv_pressure_threshold
        self.urgent_fraction_threshold = urgent_fraction_threshold
        self.low_pressure_policy = low_pressure_policy or WeightedShortestProcessingPolicy()
        self.high_pressure_policy = high_pressure_policy or ComponentWiseCompositionPolicy()
        self.min_commitment_steps = min_commitment_steps
        self.decision_logs: list[CompositionDecisionLog] = []
        self._committed_regime: str | None = None
        self._committed_until_step = -1
        self._last_regime: str | None = None
        self.switching_count = 0

    def reset(self) -> None:
        self.low_pressure_policy.reset()
        self.high_pressure_policy.reset()
        self.decision_logs.clear()
        self._committed_regime = None
        self._committed_until_step = -1
        self._last_regime = None
        self.switching_count = 0

    def _choose_regime(self, state: ObservableState) -> str:
        if self.min_commitment_steps > 0 and state.step <= self._committed_until_step and self._committed_regime:
            return self._committed_regime
        features = causal_context_features(state)
        high_pressure = (
            features["queue_pressure"] >= self.queue_pressure_threshold
            or features["kv_pressure"] >= self.kv_pressure_threshold
            or features["urgent_deadline_fraction"] >= self.urgent_fraction_threshold
        )
        regime = "high_pressure_component_wise" if high_pressure else "low_pressure_wsp"
        if self.min_commitment_steps > 0:
            self._committed_regime = regime
            self._committed_until_step = state.step + self.min_commitment_steps - 1
        return regime

    def select_action(self, state: ObservableState) -> Action:
        regime = self._choose_regime(state)
        if self._last_regime is not None and regime != self._last_regime:
            self.switching_count += 1
        self._last_regime = regime
        if regime == "high_pressure_component_wise":
            action = self.high_pressure_policy.select_action(state)
            fallback_used = bool(self.high_pressure_policy.decision_logs and self.high_pressure_policy.decision_logs[-1].fallback_used)
        else:
            action = self.low_pressure_policy.select_action(state)
            fallback_used = False
        self.decision_logs.append(
            CompositionDecisionLog(
                policy_name=self.name,
                step=state.step,
                selected_request_ids=sorted(action.all_admitted_ids()),
                expert_weights={regime: 1.0},
                expert_contributions={},
                fallback_used=fallback_used,
                weight_entropy=0.0,
                dominant_expert=regime,
                switching_count=self.switching_count,
            )
        )
        return action


def r_priority_tiebreak(req: ObservableRequest) -> float:
    return -req.priority


def make_static_rank_ensemble(
    expert_names: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> StaticRankEnsemblePolicy:
    specs = [
        RankExpertSpec(name=name, weight=(weights or {}).get(name, 1.0))
        for name in expert_names
    ]
    return StaticRankEnsemblePolicy(specs, top_k=top_k)


def make_contextual_wsp_scorpio_ensemble() -> ContextualRankEnsemblePolicy:
    """Small ready-to-train placeholder with causal load-aware hand weights."""
    experts = [
        RankExpertSpec("weighted_shortest_processing", 1.0),
        RankExpertSpec("scorpio_style_slo_guard", 1.0),
        RankExpertSpec("kv_constrained_online", 0.5),
        RankExpertSpec("aging_priority", 0.25),
    ]
    rules = [
        ContextualWeightRule(
            "scorpio_style_slo_guard",
            coefficients={"queue_pressure": 0.8, "kv_pressure": 0.6, "urgent_deadline_fraction": 1.0},
        ),
        ContextualWeightRule(
            "weighted_shortest_processing",
            coefficients={"queue_pressure": -0.2, "mean_predicted_output_tokens": -0.0005},
        ),
        ContextualWeightRule("kv_constrained_online", coefficients={"kv_pressure": 1.2}),
        ContextualWeightRule("aging_priority", coefficients={"queue_pressure": 0.3}),
    ]
    return ContextualRankEnsemblePolicy(experts, contextual_rules=rules, top_k=3, min_commitment_steps=2)


def instantiate_policy_for_treatment(treatment: str) -> BasePolicy:
    """Factory for the smoke harness and future experiment configs."""
    if treatment == "best_fixed_placeholder":
        return make_policy("weighted_shortest_processing")
    if treatment == "static_rank_ensemble":
        return make_static_rank_ensemble([
            "weighted_shortest_processing",
            "scorpio_style_slo_guard",
            "estimated_service_time_first",
            "kv_constrained_online",
            "aging_priority",
        ], top_k=3)
    if treatment == "contextual_rank_ensemble":
        return make_contextual_wsp_scorpio_ensemble()
    if treatment == "component_wise_scorpio_wsp":
        return ComponentWiseCompositionPolicy()
    if treatment == "conditional_overload_slo":
        return ConditionalRegimeCompositionPolicy(min_commitment_steps=2)
    if treatment in {"weighted_shortest_processing", "scorpio_style_slo_guard"}:
        return make_policy_library_v2(treatment)
    raise CompositionError(f"Unknown composition treatment {treatment!r}")

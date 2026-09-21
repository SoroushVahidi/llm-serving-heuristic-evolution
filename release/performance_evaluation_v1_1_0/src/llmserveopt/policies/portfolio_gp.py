"""Typed portfolio-GP representation for the v1 screening gate.

This module is deliberately narrower than the existing heuristic DSL.  Its
first responsibility is exact reproduction of the six frozen parent policies
used by ``portfolio_guided_typed_gp_screen_v1``.  Where the scalar DSL cannot
represent a parent mechanism faithfully (WFS class deficit, KV placement, or
prefill chunk execution), this module keeps that mechanism as a typed native
module instead of pretending it is a single ranking expression.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.action import Action
from ..core.types import ObservableGPUState, ObservableRequest, ObservableState
from .base import BasePolicy
from .estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from .kv_constrained_online import KVConstrainedOnlinePolicy
from .least_laxity_first import LeastLaxityFirstPolicy
from .policy_library_v2_helpers import (
    deterministic_place,
    est_steps,
    gpu_pressure,
    laxity_seconds,
    queue_class_counts,
)
from .prefill_control_variants import (
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    GreedyArrivalPrefillControlPolicy,
    _arrival_rank,
)
from .weighted_fair_share import WeightedFairSharePolicy

PORTFOLIO_GP_SCHEMA_VERSION = "PortfolioGuidedTypedGPGenomeV1"
PARENT_POLICY_IDS: tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
)

MODULE_TYPES = frozenset({
    "Policy",
    "RankingRule",
    "PlacementRule",
    "PrefillRule",
    "KVGuard",
})

MAX_AST_DEPTH = 4
MAX_NODES = 31
MAX_FREE_NUMERIC_CONSTANTS = 4


class PortfolioGPError(ValueError):
    """Raised for invalid portfolio-GP genomes or operators."""


@dataclass(frozen=True)
class TypedModule:
    module_type: str
    module_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    children: tuple["TypedModule", ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "module_id": self.module_id,
            "parameters": copy.deepcopy(self.parameters),
            "children": [child.to_dict() for child in self.children],
            "description": self.description,
        }


@dataclass(frozen=True)
class PortfolioGPGenomeV1:
    name: str
    root: TypedModule
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PORTFOLIO_GP_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "root": self.root.to_dict(),
            "metadata": copy.deepcopy(self.metadata),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def genome_string(self) -> str:
        return self.canonical_json()

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self)

    def build_policy(self) -> BasePolicy:
        self.validate()
        return PortfolioGPPolicy(self)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_genome_string(genome_string: str) -> PortfolioGPGenomeV1:
    return parse_genome(json.loads(genome_string))


def parse_genome(payload: Mapping[str, Any]) -> PortfolioGPGenomeV1:
    if payload.get("schema_version") != PORTFOLIO_GP_SCHEMA_VERSION:
        raise PortfolioGPError(f"Unsupported genome schema {payload.get('schema_version')!r}")

    def module_from(raw: Mapping[str, Any]) -> TypedModule:
        return TypedModule(
            module_type=str(raw["module_type"]),
            module_id=str(raw["module_id"]),
            parameters=copy.deepcopy(raw.get("parameters", {})),
            children=tuple(module_from(child) for child in raw.get("children", [])),
            description=str(raw.get("description", "")),
        )

    genome = PortfolioGPGenomeV1(
        name=str(payload["name"]),
        root=module_from(payload["root"]),
        metadata=copy.deepcopy(payload.get("metadata", {})),
    )
    genome.validate()
    return genome


def _count_nodes(module: TypedModule) -> int:
    return 1 + sum(_count_nodes(child) for child in module.children)


def _depth(module: TypedModule) -> int:
    if not module.children:
        return 1
    return 1 + max(_depth(child) for child in module.children)


def _numeric_constants(module: TypedModule) -> list[float]:
    values: list[float] = []
    free_keys = set(module.parameters.get("_free_numeric_parameters", []))
    for key, value in module.parameters.items():
        if key not in free_keys:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    for child in module.children:
        values.extend(_numeric_constants(child))
    return values


def validate_genome(genome: PortfolioGPGenomeV1) -> None:
    if genome.schema_version != PORTFOLIO_GP_SCHEMA_VERSION:
        raise PortfolioGPError(f"Unsupported schema {genome.schema_version!r}")
    if genome.root.module_type != "Policy":
        raise PortfolioGPError("Root module must have type Policy")
    if _depth(genome.root) > MAX_AST_DEPTH:
        raise PortfolioGPError("Genome exceeds max AST depth")
    if _count_nodes(genome.root) > MAX_NODES:
        raise PortfolioGPError("Genome exceeds max node count")
    constants = _numeric_constants(genome.root)
    if len(constants) > MAX_FREE_NUMERIC_CONSTANTS:
        raise PortfolioGPError("Genome exceeds max numeric constant count")
    for value in constants:
        if not math.isfinite(value):
            raise PortfolioGPError(f"Non-finite numeric parameter {value!r}")
    for _, module in _walk_modules(genome.root):
        if module.module_type not in MODULE_TYPES:
            raise PortfolioGPError(f"Unknown module type {module.module_type!r}")
    _module_by_type(genome.root, "RankingRule", required=True)
    _module_by_type(genome.root, "PlacementRule", required=True)


def _walk_modules(module: TypedModule, path: tuple[int, ...] = ()) -> Iterable[tuple[tuple[int, ...], TypedModule]]:
    yield path, module
    for idx, child in enumerate(module.children):
        yield from _walk_modules(child, path + (idx,))


def _module_by_type(root: TypedModule, module_type: str, *, required: bool = False) -> TypedModule | None:
    matches = [module for _, module in _walk_modules(root) if module.module_type == module_type]
    if not matches:
        if required:
            raise PortfolioGPError(f"Genome missing required module type {module_type}")
        return None
    if len(matches) > 1 and module_type in {"RankingRule", "PlacementRule", "PrefillRule", "KVGuard"}:
        raise PortfolioGPError(f"Genome has duplicate singleton module type {module_type}")
    return matches[0]


def _module(module_type: str, module_id: str, *, description: str = "", **parameters: Any) -> TypedModule:
    return TypedModule(module_type=module_type, module_id=module_id, parameters=dict(parameters), description=description)


def source_policy_implementation(policy_id: str) -> str:
    return {
        "estimated_service_time_first": "src/llmserveopt/policies/estimated_service_time_first.py",
        "least_laxity_first": "src/llmserveopt/policies/least_laxity_first.py",
        "weighted_fair_share": "src/llmserveopt/policies/weighted_fair_share.py",
        "kv_constrained_online": "src/llmserveopt/policies/kv_constrained_online.py",
        "full_prefill": "src/llmserveopt/policies/prefill_control_variants.py",
        "chunked_prefill_small": "src/llmserveopt/policies/prefill_control_variants.py",
    }[policy_id]


def _policy_module(
    policy_id: str,
    *,
    ranking: TypedModule,
    placement: TypedModule,
    prefill: TypedModule | None = None,
    kv_guard: TypedModule | None = None,
) -> TypedModule:
    children = [ranking, placement]
    if prefill is not None:
        children.append(prefill)
    if kv_guard is not None:
        children.append(kv_guard)
    return TypedModule(
        module_type="Policy",
        module_id="policy.module_composition",
        parameters={
            "canonical_parent_id": policy_id,
            "exactness_status": "EXACT_PARENT_REPRESENTATION",
        },
        children=tuple(children),
        description=f"Typed exact parent representation for {policy_id}",
    )


def parent_genome(policy_id: str) -> PortfolioGPGenomeV1:
    if policy_id == "estimated_service_time_first":
        root = _policy_module(
            policy_id,
            ranking=_module(
                "RankingRule", "ranking.estf_service_time",
                alpha=0.5, beta=1.0, _free_numeric_parameters=["alpha", "beta"],
            ),
            placement=_module("PlacementRule", "placement.round_robin_scan"),
        )
    elif policy_id == "least_laxity_first":
        root = _policy_module(
            policy_id,
            ranking=_module(
                "RankingRule", "ranking.llf_laxity",
                alpha=0.5, beta=1.0, _free_numeric_parameters=["alpha", "beta"],
            ),
            placement=_module("PlacementRule", "placement.round_robin_scan"),
        )
    elif policy_id == "weighted_fair_share":
        root = _policy_module(
            policy_id,
            ranking=_module(
                "RankingRule", "ranking.wfs_deficit_priority_service",
                alpha=0.5, beta=1.0, _free_numeric_parameters=["alpha", "beta"],
            ),
            placement=_module("PlacementRule", "placement.default_gpu_pressure"),
        )
    elif policy_id == "kv_constrained_online":
        root = _policy_module(
            policy_id,
            ranking=_module(
                "RankingRule", "ranking.kv_urgent_kv_cost",
                step_size=0.001, alpha=0.5, beta=1.0, urgent_laxity_seconds=0.25,
                _free_numeric_parameters=["urgent_laxity_seconds"],
            ),
            placement=_module("PlacementRule", "placement.kv_low_post_util"),
            kv_guard=_module(
                "KVGuard", "kv_guard.target_util_or_urgent_laxity",
                step_size=0.001, alpha=0.5, beta=1.0,
                target_kv_utilization=0.82, urgent_laxity_seconds=0.25,
                _free_numeric_parameters=["target_kv_utilization", "urgent_laxity_seconds"],
            ),
        )
    elif policy_id == "full_prefill":
        root = _policy_module(
            policy_id,
            ranking=_module("RankingRule", "ranking.arrival_order"),
            placement=_module("PlacementRule", "placement.default_gpu_pressure"),
            prefill=_module(
                "PrefillRule", "prefill.full",
                max_prefill_chunk_tokens=UNLIMITED_PREFILL_CHUNK,
                decode_first=False,
                _free_numeric_parameters=["max_prefill_chunk_tokens"],
            ),
        )
    elif policy_id == "chunked_prefill_small":
        root = _policy_module(
            policy_id,
            ranking=_module("RankingRule", "ranking.arrival_order"),
            placement=_module("PlacementRule", "placement.default_gpu_pressure"),
            prefill=_module(
                "PrefillRule", "prefill.chunked_small",
                max_prefill_chunk_tokens=DEFAULT_CHUNK_SMALL,
                decode_first=False,
                _free_numeric_parameters=["max_prefill_chunk_tokens"],
            ),
        )
    else:
        raise PortfolioGPError(f"Unknown frozen parent policy {policy_id!r}")
    genome = PortfolioGPGenomeV1(
        name=f"portfolio_gp_parent::{policy_id}",
        root=root,
        metadata={
            "source_policy": policy_id,
            "source_policy_implementation": source_policy_implementation(policy_id),
            "screen": "portfolio_guided_typed_gp_screen_v1",
        },
    )
    genome.validate()
    return genome


PARENT_GENOMES_V1: dict[str, PortfolioGPGenomeV1] = {
    policy_id: parent_genome(policy_id) for policy_id in PARENT_POLICY_IDS
}


class PortfolioGPPolicy(BasePolicy):
    """Executable interpreter for the typed modules used in the screen gate."""

    def __init__(self, genome: PortfolioGPGenomeV1) -> None:
        self.genome = genome
        self.name = genome.name

    def reset(self) -> None:
        pass

    def select_action(self, state: ObservableState) -> Action:
        ranking = _module_by_type(self.genome.root, "RankingRule", required=True)
        placement = _module_by_type(self.genome.root, "PlacementRule", required=True)
        assert ranking is not None and placement is not None
        ranked = self._rank_requests(ranking, state)
        if placement.module_id == "placement.round_robin_scan":
            action = self._round_robin_place(state, ranked)
        elif placement.module_id == "placement.default_gpu_pressure":
            action = deterministic_place(state, ranked)
        elif placement.module_id == "placement.kv_low_post_util":
            action = self._kv_place(state, ranked)
        else:
            raise PortfolioGPError(f"Unsupported placement module {placement.module_id!r}")
        prefill = _module_by_type(self.genome.root, "PrefillRule")
        if prefill is not None:
            action.prefill_chunk_override = {
                g.gpu_id: int(prefill.parameters["max_prefill_chunk_tokens"])
                for g in state.gpu_states
            }
        return action

    def _rank_requests(self, ranking: TypedModule, state: ObservableState) -> list[ObservableRequest]:
        if ranking.module_id == "ranking.arrival_order":
            return sorted(state.waiting_queue, key=_arrival_rank)
        if ranking.module_id == "ranking.estf_service_time":
            alpha = float(ranking.parameters.get("alpha", 0.5))
            beta = float(ranking.parameters.get("beta", 1.0))
            return sorted(
                state.waiting_queue,
                key=lambda r: (est_steps(r, alpha, beta), r.slo_deadline, -r.priority, r.request_id),
            )
        if ranking.module_id == "ranking.llf_laxity":
            alpha = float(ranking.parameters.get("alpha", 0.5))
            beta = float(ranking.parameters.get("beta", 1.0))
            now = state.time
            return sorted(
                state.waiting_queue,
                key=lambda r: (
                    r.slo_deadline - now - est_steps(r, alpha, beta),
                    r.slo_deadline,
                    -r.priority,
                    r.request_id,
                ),
            )
        if ranking.module_id == "ranking.wfs_deficit_priority_service":
            alpha = float(ranking.parameters.get("alpha", 0.5))
            beta = float(ranking.parameters.get("beta", 1.0))
            active = queue_class_counts(r for g in state.gpu_states for r in g.active_requests_info)
            demand = queue_class_counts(state.waiting_queue)
            admitted_counts: Counter[str] = Counter()

            def score(req: ObservableRequest) -> float:
                cls = req.class_id or "unknown"
                served_share = active[cls] + admitted_counts[cls]
                deficit = demand[cls] / max(1, served_share + 1)
                return deficit * req.priority / max(est_steps(req, alpha, beta), 1e-9)

            return sorted(state.waiting_queue, key=lambda r: (-score(r), r.arrival_time, r.request_id))
        if ranking.module_id == "ranking.kv_urgent_kv_cost":
            step_size = float(ranking.parameters.get("step_size", 0.001))
            alpha = float(ranking.parameters.get("alpha", 0.5))
            beta = float(ranking.parameters.get("beta", 1.0))
            urgent_laxity_seconds = float(ranking.parameters.get("urgent_laxity_seconds", 0.25))

            def key(req: ObservableRequest) -> tuple[bool, float, float, int]:
                laxity = laxity_seconds(req, state.time, step_size, alpha, beta)
                kv_cost = req.prompt_tokens + 0.25 * req.predicted_output_tokens
                return (laxity > urgent_laxity_seconds, kv_cost / max(req.priority, 1e-9), laxity, req.request_id)

            return sorted(state.waiting_queue, key=key)
        raise PortfolioGPError(f"Unsupported ranking module {ranking.module_id!r}")

    @staticmethod
    def _round_robin_place(state: ObservableState, ranked: Sequence[ObservableRequest]) -> Action:
        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}
        if not state.gpu_states:
            return Action(admit=admit)
        gpu_idx = 0
        n_gpus = len(state.gpu_states)
        for req in ranked:
            for offset in range(n_gpus):
                gpu = state.gpu_states[(gpu_idx + offset) % n_gpus]
                if BasePolicy._feasible_on_gpu(gpu, req):
                    admit[gpu.gpu_id].append(req.request_id)
                    gpu.active_request_ids.append(req.request_id)
                    gpu.current_kv_tokens += req.prompt_tokens
                    gpu_idx = (gpu_idx + offset + 1) % n_gpus
                    break
        return Action(admit=admit)

    def _kv_place(self, state: ObservableState, ranked: Sequence[ObservableRequest]) -> Action:
        guard = _module_by_type(self.genome.root, "KVGuard", required=True)
        assert guard is not None
        step_size = float(guard.parameters.get("step_size", 0.001))
        alpha = float(guard.parameters.get("alpha", 0.5))
        beta = float(guard.parameters.get("beta", 1.0))
        target = float(guard.parameters.get("target_kv_utilization", 0.82))
        urgent_laxity_seconds = float(guard.parameters.get("urgent_laxity_seconds", 0.25))

        def admit_filter(req: ObservableRequest, gpu: ObservableGPUState, _admitted: list[ObservableRequest]) -> bool:
            post_util = (gpu.current_kv_tokens + req.prompt_tokens) / max(gpu.max_kv_tokens, 1)
            urgent = laxity_seconds(req, state.time, step_size, alpha, beta) <= urgent_laxity_seconds
            return post_util <= target or urgent

        return deterministic_place(
            state,
            list(ranked),
            gpu_key=lambda g, r: ((g.current_kv_tokens + r.prompt_tokens) / max(g.max_kv_tokens, 1), g.gpu_id),
            admit_filter=admit_filter,
        )


class PrefillExecutionControlPolicy(BasePolicy):
    """Exact action-level wrapper for fixed prefill chunk controls."""

    def __init__(self, parent_id: str, chunk_size: int) -> None:
        self.parent_id = parent_id
        self.chunk_size = int(chunk_size)
        self.base = GreedyArrivalPrefillControlPolicy()
        self.name = parent_id

    def reset(self) -> None:
        self.base.reset()

    def select_action(self, state: ObservableState) -> Action:
        action = self.base.select_action(state)
        action.prefill_chunk_override = {g.gpu_id: self.chunk_size for g in state.gpu_states}
        return action


def make_original_parent_policy(policy_id: str) -> BasePolicy:
    if policy_id == "estimated_service_time_first":
        return EstimatedServiceTimeFirstPolicy()
    if policy_id == "least_laxity_first":
        return LeastLaxityFirstPolicy()
    if policy_id == "weighted_fair_share":
        return WeightedFairSharePolicy()
    if policy_id == "kv_constrained_online":
        return KVConstrainedOnlinePolicy()
    if policy_id == "full_prefill":
        return PrefillExecutionControlPolicy(policy_id, UNLIMITED_PREFILL_CHUNK)
    if policy_id == "chunked_prefill_small":
        return PrefillExecutionControlPolicy(policy_id, DEFAULT_CHUNK_SMALL)
    raise PortfolioGPError(f"Unknown parent policy {policy_id!r}")


def action_signature(action: Action) -> dict[str, Any]:
    return {
        "admit": {str(k): list(v) for k, v in sorted(action.admit.items())},
        "preempt": {str(k): list(v) for k, v in sorted(action.preempt.items())},
        "swap": {str(k): list(v) for k, v in sorted(action.swap.items())},
        "migrate": {str(k): [list(pair) for pair in v] for k, v in sorted(action.migrate.items())},
        "hold_decode": {str(k): list(v) for k, v in sorted(action.hold_decode.items())},
        "prefill_chunk_override": {str(k): int(v) for k, v in sorted(action.prefill_chunk_override.items())},
    }


def state_signature(state: ObservableState) -> dict[str, Any]:
    return {
        "time": state.time,
        "step": state.step,
        "waiting_ids": [r.request_id for r in state.waiting_queue],
        "gpu": [
            {
                "gpu_id": g.gpu_id,
                "active_request_ids": list(g.active_request_ids),
                "current_kv_tokens": g.current_kv_tokens,
                "prefilling_count": g.prefilling_count,
                "decoding_count": g.decoding_count,
            }
            for g in state.gpu_states
        ],
    }


def policy_behavior_fingerprint(policy: BasePolicy, probe_set: Sequence[ObservableState]) -> str:
    records = []
    for idx, probe in enumerate(probe_set):
        state = copy.deepcopy(probe)
        if hasattr(policy, "reset"):
            policy.reset()
        action = policy.select_action(state)
        records.append({
            "probe_index": idx,
            "state": state_signature(probe),
            "action": action_signature(action),
        })
    return hashlib.sha256(canonical_json({"records": records}).encode("utf-8")).hexdigest()


def decision_overlap(policy_a: BasePolicy, policy_b: BasePolicy, probe_set: Sequence[ObservableState]) -> float:
    if not probe_set:
        raise ValueError("probe_set must not be empty")
    matches = 0
    for probe in probe_set:
        state_a = copy.deepcopy(probe)
        state_b = copy.deepcopy(probe)
        if hasattr(policy_a, "reset"):
            policy_a.reset()
        if hasattr(policy_b, "reset"):
            policy_b.reset()
        if action_signature(policy_a.select_action(state_a)) == action_signature(policy_b.select_action(state_b)):
            matches += 1
    return matches / len(probe_set)


@dataclass(frozen=True)
class ParentReproductionResult:
    policy_id: str
    scenarios_checked: int
    decision_points: int
    exact_action_agreement: bool
    metric_equality_checked: bool = False
    metric_equality: bool | None = None
    first_mismatch: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.exact_action_agreement and (self.metric_equality is not False):
            return "PARENT_REPRODUCTION_PASS"
        return "PARENT_REPRODUCTION_FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "scenarios_checked": self.scenarios_checked,
            "decision_points": self.decision_points,
            "exact_action_agreement": self.exact_action_agreement,
            "metric_equality_checked": self.metric_equality_checked,
            "metric_equality": self.metric_equality,
            "first_mismatch": copy.deepcopy(self.first_mismatch),
            "status": self.status,
        }


def compare_parent_on_probe_states(
    policy_id: str,
    probe_set: Sequence[ObservableState],
) -> ParentReproductionResult:
    original = make_original_parent_policy(policy_id)
    decoded = PARENT_GENOMES_V1[policy_id].build_policy()
    for idx, probe in enumerate(probe_set):
        state_a = copy.deepcopy(probe)
        state_b = copy.deepcopy(probe)
        if hasattr(original, "reset"):
            original.reset()
        if hasattr(decoded, "reset"):
            decoded.reset()
        action_a = original.select_action(state_a)
        action_b = decoded.select_action(state_b)
        sig_a = action_signature(action_a)
        sig_b = action_signature(action_b)
        if sig_a != sig_b:
            return ParentReproductionResult(
                policy_id=policy_id,
                scenarios_checked=len(probe_set),
                decision_points=idx + 1,
                exact_action_agreement=False,
                first_mismatch={
                    "probe_index": idx,
                    "original_action": sig_a,
                    "decoded_action": sig_b,
                    "state": state_signature(probe),
                },
            )
    return ParentReproductionResult(
        policy_id=policy_id,
        scenarios_checked=len(probe_set),
        decision_points=len(probe_set),
        exact_action_agreement=True,
    )


def _req(
    request_id: int,
    *,
    arrival: float,
    prompt: int,
    output: int,
    deadline: float,
    priority: float,
    class_id: str,
) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=priority,
        class_id=class_id,
    )


def _gpu(
    gpu_id: int,
    *,
    active: Sequence[ObservableRequest] = (),
    current_kv_tokens: int = 0,
    max_kv_tokens: int = 4096,
    max_active_sequences: int = 4,
    max_batch_tokens: int = 16,
    prefilling_count: int = 0,
    decoding_count: int = 0,
) -> ObservableGPUState:
    return ObservableGPUState(
        gpu_id=gpu_id,
        max_active_sequences=max_active_sequences,
        max_batch_tokens=max_batch_tokens,
        max_kv_tokens=max_kv_tokens,
        active_request_ids=[r.request_id for r in active],
        active_requests_info=list(active),
        current_kv_tokens=current_kv_tokens,
        tokens_decoded_per_request={r.request_id: 1 for r in active},
        prefilling_count=prefilling_count,
        decoding_count=decoding_count,
    )


def make_parent_reproduction_probe_states() -> list[ObservableState]:
    active_long = _req(90, arrival=0.0, prompt=256, output=128, deadline=2.0, priority=2.0, class_id="long")
    active_short = _req(91, arrival=0.0, prompt=64, output=32, deadline=1.5, priority=1.0, class_id="short")
    return [
        ObservableState(
            time=0.10,
            step=10,
            waiting_queue=[
                _req(1, arrival=0.00, prompt=512, output=256, deadline=5.0, priority=1.0, class_id="long"),
                _req(2, arrival=0.01, prompt=64, output=32, deadline=5.5, priority=1.0, class_id="short"),
                _req(3, arrival=0.02, prompt=128, output=80, deadline=1.0, priority=3.0, class_id="urgent"),
            ],
            gpu_states=[_gpu(0), _gpu(1)],
            completed_count=0,
        ),
        ObservableState(
            time=0.40,
            step=40,
            waiting_queue=[
                _req(4, arrival=0.00, prompt=256, output=64, deadline=0.60, priority=1.0, class_id="short"),
                _req(5, arrival=0.01, prompt=64, output=16, deadline=2.0, priority=5.0, class_id="short"),
                _req(6, arrival=0.02, prompt=128, output=512, deadline=1.2, priority=2.0, class_id="long"),
            ],
            gpu_states=[_gpu(0, active=[active_long], current_kv_tokens=512), _gpu(1, active=[active_short], current_kv_tokens=128)],
            completed_count=2,
        ),
        ObservableState(
            time=0.80,
            step=80,
            waiting_queue=[
                _req(7, arrival=0.00, prompt=700, output=100, deadline=4.0, priority=1.0, class_id="bulk"),
                _req(8, arrival=0.01, prompt=100, output=20, deadline=0.90, priority=3.0, class_id="urgent"),
                _req(9, arrival=0.02, prompt=900, output=40, deadline=3.0, priority=2.0, class_id="bulk"),
            ],
            gpu_states=[
                _gpu(0, current_kv_tokens=3300, max_kv_tokens=4096, max_active_sequences=3),
                _gpu(1, current_kv_tokens=1000, max_kv_tokens=4096, max_active_sequences=3),
            ],
            completed_count=4,
        ),
        ObservableState(
            time=0.05,
            step=5,
            waiting_queue=[
                _req(10, arrival=0.00, prompt=4096, output=80, deadline=1.0, priority=1.0, class_id="hog"),
                _req(11, arrival=0.03, prompt=96, output=70, deadline=2.0, priority=1.0, class_id="late"),
            ],
            gpu_states=[_gpu(0, max_kv_tokens=10000, max_active_sequences=64, max_batch_tokens=1000)],
            completed_count=0,
        ),
    ]


def _get_module(root: TypedModule, path: tuple[int, ...]) -> TypedModule:
    module = root
    for idx in path:
        module = module.children[idx]
    return module


def _replace_module(root: TypedModule, path: tuple[int, ...], replacement: TypedModule) -> TypedModule:
    if not path:
        return replacement
    idx = path[0]
    children = list(root.children)
    children[idx] = _replace_module(children[idx], path[1:], replacement)
    params = dict(root.parameters)
    if root.module_type == "Policy":
        params["canonical_parent_id"] = None
        params["exactness_status"] = "COMPOSED_CANDIDATE"
    return replace(root, parameters=params, children=tuple(children))


def typed_subtree_crossover(
    parent_a: PortfolioGPGenomeV1,
    parent_b: PortfolioGPGenomeV1,
    module_type: str,
    *,
    seed: int,
    child_name: str | None = None,
) -> PortfolioGPGenomeV1:
    if module_type not in {"Policy", "RankingRule", "PrefillRule", "KVGuard"}:
        raise PortfolioGPError(f"Unsupported crossover module type {module_type!r}")
    paths_a = [path for path, module in _walk_modules(parent_a.root) if module.module_type == module_type]
    paths_b = [path for path, module in _walk_modules(parent_b.root) if module.module_type == module_type]
    if not paths_a or not paths_b:
        raise PortfolioGPError(f"No compatible {module_type} crossover point in both parents")
    rng = np.random.default_rng(seed)
    path_a = paths_a[int(rng.integers(0, len(paths_a)))]
    path_b = paths_b[int(rng.integers(0, len(paths_b)))]
    donor = _get_module(parent_b.root, path_b)
    child_root = _replace_module(parent_a.root, path_a, donor)
    child = PortfolioGPGenomeV1(
        name=child_name or f"{parent_a.name}__x_{module_type}__{parent_b.name}",
        root=child_root,
        metadata={
            "operator": "typed_subtree_crossover",
            "parent_a": parent_a.stable_hash(),
            "parent_b": parent_b.stable_hash(),
            "module_type": module_type,
            "path_a": list(path_a),
            "path_b": list(path_b),
            "seed": int(seed),
        },
    )
    child.validate()
    return child


def mutate_genome(
    genome: PortfolioGPGenomeV1,
    *,
    seed: int,
    scale: float = 0.10,
    child_name: str | None = None,
) -> PortfolioGPGenomeV1:
    rng = np.random.default_rng(seed)
    payload = genome.to_dict()
    mutated = False

    def mutate_module(raw: dict[str, Any]) -> None:
        nonlocal mutated
        if mutated:
            return
        params = raw.get("parameters", {})
        free_keys = set(params.get("_free_numeric_parameters", []))
        numeric_keys = [
            key for key, value in params.items()
            if key in free_keys and not isinstance(value, bool) and isinstance(value, (int, float))
        ]
        if numeric_keys:
            key = numeric_keys[int(rng.integers(0, len(numeric_keys)))]
            base = float(params[key])
            delta = rng.uniform(-scale, scale) * max(abs(base), 1.0)
            params[key] = max(0.0, base + delta) if key not in {"alpha", "beta"} else max(1e-9, base + delta)
            mutated = True
            return
        for child in raw.get("children", []):
            mutate_module(child)

    mutate_module(payload["root"])
    if not mutated:
        raise PortfolioGPError("No bounded numeric mutation point found")
    payload["name"] = child_name or f"{genome.name}__mut"
    payload["metadata"] = {
        "operator": "bounded_parameter_mutation",
        "parent": genome.stable_hash(),
        "seed": int(seed),
        "scale": float(scale),
    }
    payload["root"]["parameters"]["canonical_parent_id"] = None
    payload["root"]["parameters"]["exactness_status"] = "COMPOSED_CANDIDATE"
    child = parse_genome(payload)
    child.validate()
    return child


def envelope_values(parent_rewards: Mapping[str, Sequence[float]]) -> list[float]:
    if not parent_rewards:
        raise ValueError("parent_rewards must not be empty")
    lengths = {len(values) for values in parent_rewards.values()}
    if len(lengths) != 1:
        raise ValueError("all parent reward vectors must have the same length")
    return [max(float(parent_rewards[p][i]) for p in parent_rewards) for i in range(next(iter(lengths)))]


def marginal_gains(candidate_rewards: Sequence[float], envelope: Sequence[float]) -> list[float]:
    if len(candidate_rewards) != len(envelope):
        raise ValueError("candidate_rewards and envelope length mismatch")
    return [max(float(c), float(e)) - float(e) for c, e in zip(candidate_rewards, envelope)]


def summarize_marginal_gain(
    candidate_rewards: Sequence[float],
    parent_rewards: Mapping[str, Sequence[float]],
    families: Sequence[str],
    *,
    epsilon: float = 0.005,
) -> dict[str, Any]:
    env = envelope_values(parent_rewards)
    gains = marginal_gains(candidate_rewards, env)
    unique = [float(c) > float(e) + epsilon for c, e in zip(candidate_rewards, env)]
    positive_by_family: Counter[str] = Counter()
    for family, gain in zip(families, gains):
        if gain > 0.0:
            positive_by_family[str(family)] += 1
    total_positive = sum(gains)
    max_family_share = 0.0
    if total_positive > 0.0:
        by_family_gain: Counter[str] = Counter()
        for family, gain in zip(families, gains):
            by_family_gain[str(family)] += max(0.0, gain)
        max_family_share = max(by_family_gain.values(), default=0.0) / total_positive
    return {
        "mean_MG": float(np.mean(gains)) if gains else 0.0,
        "total_MG": float(np.sum(gains)),
        "unique_wins_eps": int(sum(unique)),
        "positive_regions": int(len(positive_by_family)),
        "max_family_positive_MG_share": float(max_family_share),
        "gains": gains,
    }


@dataclass
class TreatmentBudgetAccountant:
    treatment_id: str
    target_evaluated_candidates: int
    proposed_candidates: int = 0
    valid_candidates: int = 0
    unique_candidates: int = 0
    evaluated_candidates: int = 0
    duplicate_candidates: int = 0
    rejected_candidates: int = 0

    def record_proposed(self) -> None:
        self.proposed_candidates += 1

    def record_rejected(self) -> None:
        self.rejected_candidates += 1

    def record_duplicate(self) -> None:
        self.duplicate_candidates += 1

    def record_valid_unique(self, *, evaluated: bool) -> None:
        self.valid_candidates += 1
        self.unique_candidates += 1
        if evaluated:
            self.evaluated_candidates += 1

    @property
    def complete(self) -> bool:
        return self.evaluated_candidates >= self.target_evaluated_candidates

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment_id": self.treatment_id,
            "target_evaluated_candidates": self.target_evaluated_candidates,
            "proposed_candidates": self.proposed_candidates,
            "valid_candidates": self.valid_candidates,
            "unique_candidates": self.unique_candidates,
            "evaluated_candidates": self.evaluated_candidates,
            "duplicate_candidates": self.duplicate_candidates,
            "rejected_candidates": self.rejected_candidates,
            "complete": self.complete,
        }


def equal_budget_summary(accountants: Sequence[TreatmentBudgetAccountant]) -> dict[str, Any]:
    evaluated = {a.treatment_id: a.evaluated_candidates for a in accountants}
    return {
        "evaluated_by_treatment": evaluated,
        "equal_evaluated_candidates": len(set(evaluated.values())) == 1,
        "treatments": [a.to_dict() for a in accountants],
    }

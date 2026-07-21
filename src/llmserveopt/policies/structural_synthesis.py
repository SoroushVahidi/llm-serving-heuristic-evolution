"""Structural synthesis operators for scheduler genomes."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .genome import (
    GenomeModule,
    GenomeValidationError,
    RegimeCondition,
    SchedulerGenomeV1,
    compile_genome,
    const,
    module,
    op,
    parse_genome,
    var,
    weighted_sum,
)


def _service_proxy_expr() -> dict[str, Any]:
    return weighted_sum((var("req.prompt_tokens"), -0.5), (var("req.predicted_output_tokens"), -1.0))


def _wsp_priority_expr() -> dict[str, Any]:
    return op("div_safe", _service_proxy_expr(), op("max", var("req.priority_weight"), const(1e-3)))


def _positive_slack_expr() -> dict[str, Any]:
    return var("req.deadline_slack")


def _kv_capacity_expr(threshold: float = 0.90) -> dict[str, Any]:
    return op("sub", const(threshold), sys_var("sys.kv_utilization"))


def sys_var(name: str) -> dict[str, Any]:
    return var(name)


def map_policy_to_genome(policy_name: str) -> SchedulerGenomeV1:
    """Best-effort parent genome mapping for representative policies."""
    if policy_name == "weighted_shortest_processing":
        return SchedulerGenomeV1(
            name="genome_weighted_shortest_processing",
            admission_rule=None,
            priority_rule=module("priority_rule", _wsp_priority_expr(), description="WSP predicted service divided by priority"),
            tie_breaker="arrival_order",
            metadata={"source_policy": policy_name, "mapping_status": "EXACT"},
        )
    if policy_name == "edf":
        return SchedulerGenomeV1(
            name="genome_edf",
            admission_rule=None,
            priority_rule=module("priority_rule", op("neg", var("req.deadline_slack")), description="earliest deadline via minimum slack"),
            tie_breaker="earliest_deadline",
            metadata={"source_policy": policy_name, "mapping_status": "EXACT"},
        )
    if policy_name == "aging_priority":
        return SchedulerGenomeV1(
            name="genome_aging_priority",
            admission_rule=None,
            priority_rule=module(
                "priority_rule",
                op("div_safe", op("add", var("req.priority_weight"), var("req.waiting_time")), op("max", op("neg", _service_proxy_expr()), const(1.0))),
                description="priority plus waiting-time age bonus over estimated service",
            ),
            fairness_rule=module("fairness_rule", var("req.waiting_time"), description="age bonus"),
            tie_breaker="arrival_order",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "exact implementation has tuned aging coefficients"},
        )
    if policy_name == "scorpio_style_slo_guard":
        return SchedulerGenomeV1(
            name="genome_scorpio_style_slo_guard",
            admission_rule=module("admission_rule", _positive_slack_expr(), status="APPROXIMATE", description="positive slack admission proxy"),
            priority_rule=module(
                "priority_rule",
                weighted_sum((var("req.deadline_urgency"), 1.0), (var("req.priority_weight"), 1.0), (var("req.waiting_time"), 0.05), (var("req.predicted_output_tokens"), -0.01)),
                status="APPROXIMATE",
                description="urgency/priority/age/decode penalty proxy",
            ),
            kv_guard=module("kv_guard", _kv_capacity_expr(0.85), status="APPROXIMATE", description="aggregate KV pressure guard"),
            tie_breaker="earliest_deadline",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "stateful budget and per-GPU decode/KV filters are not exactly representable"},
        )
    if policy_name == "kv_constrained_online":
        return SchedulerGenomeV1(
            name="genome_kv_constrained_online",
            admission_rule=module("admission_rule", _kv_capacity_expr(0.90), status="APPROXIMATE", description="aggregate KV utilization threshold"),
            priority_rule=module("priority_rule", op("div_safe", var("req.deadline_urgency"), op("max", var("req.estimated_kv_cost"), const(1.0))), status="APPROXIMATE"),
            kv_guard=module("kv_guard", _kv_capacity_expr(0.90), status="APPROXIMATE"),
            tie_breaker="earliest_deadline",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "per-GPU post-placement KV reserve is not exactly encoded"},
        )
    if policy_name == "adaptive_chunked_prefill":
        return SchedulerGenomeV1(
            name="genome_adaptive_chunked_prefill",
            admission_rule=module("admission_rule", op("sub", const(2.0), sys_var("sys.slo_pressure")), status="APPROXIMATE"),
            priority_rule=module("priority_rule", weighted_sum((var("req.prompt_tokens"), -1.0), (var("req.deadline_urgency"), 0.5)), status="APPROXIMATE"),
            prefill_rule=module("prefill_rule", op("sub", const(0.85), sys_var("sys.token_budget_utilization")), status="APPROXIMATE", description="prefill budget proxy"),
            tie_breaker="shortest_prompt",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "true chunk-size actions are unsupported"},
        )
    return SchedulerGenomeV1(
        name=f"genome_{policy_name}",
        admission_rule=None,
        priority_rule=module("priority_rule", var("req.priority_weight"), status="UNSUPPORTED", description="placeholder priority"),
        metadata={"source_policy": policy_name, "mapping_status": "UNSUPPORTED", "limitation": "no canonical parent mapping implemented"},
    )


def module_swap(base: SchedulerGenomeV1, donor: SchedulerGenomeV1, module_type: str, *, child_name: str | None = None) -> SchedulerGenomeV1:
    if module_type not in {"admission_rule", "priority_rule", "prefill_rule", "kv_guard", "fairness_rule"}:
        raise GenomeValidationError(f"Cannot swap unsupported module type {module_type!r}")
    donor_module = getattr(donor, module_type)
    if donor_module is None:
        raise GenomeValidationError(f"Donor {donor.name} does not have module {module_type}")
    child = replace(base, name=child_name or f"{base.name}__swap_{module_type}_from__{donor.name}", **{module_type: donor_module})
    child.validate()
    return child


def conditional_composition(
    condition: dict[str, Any],
    then_parent: SchedulerGenomeV1,
    else_parent: SchedulerGenomeV1,
    *,
    child_name: str,
) -> SchedulerGenomeV1:
    child = SchedulerGenomeV1(
        name=child_name,
        admission_rule=else_parent.admission_rule,
        priority_rule=else_parent.priority_rule,
        prefill_rule=else_parent.prefill_rule,
        kv_guard=else_parent.kv_guard,
        fairness_rule=else_parent.fairness_rule,
        regime_conditions=(RegimeCondition("then_parent", condition, then_parent.priority_rule, then_parent.admission_rule),),
        tie_breaker=else_parent.tie_breaker,
        metadata={"operator": "conditional_composition", "then_parent": then_parent.name, "else_parent": else_parent.name},
    )
    child.validate()
    return child


def typed_subtree_crossover(
    parent_a: SchedulerGenomeV1,
    parent_b: SchedulerGenomeV1,
    module_type: str,
    *,
    child_name: str,
) -> SchedulerGenomeV1:
    module_a = getattr(parent_a, module_type, None)
    module_b = getattr(parent_b, module_type, None)
    if module_a is None or module_b is None:
        raise GenomeValidationError(f"Both parents must expose {module_type}")
    if module_a.module_type != module_b.module_type:
        raise GenomeValidationError("Cannot cross over incompatible module types")
    return module_swap(parent_a, parent_b, module_type, child_name=child_name)


def mutate_constants(genome: SchedulerGenomeV1, *, scale: float = 0.10, seed: int = 0, child_name: str | None = None) -> SchedulerGenomeV1:
    rng = np.random.default_rng(seed)
    payload = genome.to_dict()

    def mutate(node: Any) -> None:
        if isinstance(node, dict):
            if "const" in node and isinstance(node["const"], (int, float)):
                base = float(node["const"])
                delta = rng.uniform(-scale, scale) * max(abs(base), 1.0)
                node["const"] = max(-1000.0, min(1000.0, base + delta))
            for value in node.values():
                mutate(value)
        elif isinstance(node, list):
            for item in node:
                mutate(item)

    mutate(payload)
    payload["name"] = child_name or f"{genome.name}__const_mutation"
    child = parse_genome(payload)
    child.validate()
    return child


def mutate_feature_or_operator(genome: SchedulerGenomeV1, *, child_name: str | None = None) -> SchedulerGenomeV1:
    payload = genome.to_dict()
    changed = False

    def mutate(node: Any) -> None:
        nonlocal changed
        if changed:
            return
        if isinstance(node, dict):
            if node.get("var") == "req.predicted_output_tokens":
                node["var"] = "req.estimated_decode_cost"
                changed = True
                return
            if node.get("op") == "add":
                node["op"] = "max"
                changed = True
                return
            for value in node.values():
                mutate(value)
        elif isinstance(node, list):
            for item in node:
                mutate(item)

    mutate(payload)
    if not changed:
        raise GenomeValidationError("No whitelisted feature/operator mutation point found")
    payload["name"] = child_name or f"{genome.name}__feature_operator_mutation"
    child = parse_genome(payload)
    child.validate()
    return child


def frontier_value(
    policy_rewards: Mapping[str, Sequence[float]],
    child_rewards: Sequence[float],
    *,
    meaningful_margin: float = 0.002,
    complexity_penalty: float = 0.0,
) -> dict[str, float]:
    if not policy_rewards:
        raise ValueError("policy_rewards must not be empty")
    base = np.vstack([np.asarray(values, dtype=float) for values in policy_rewards.values()])
    child = np.asarray(child_rewards, dtype=float)
    envelope = base.max(axis=0)
    new_envelope = np.maximum(envelope, child)
    gains = new_envelope - envelope
    return {
        "marginal_frontier_value": float(np.mean(gains) - complexity_penalty),
        "unique_win_count": int(np.sum(child > envelope)),
        "meaningful_unique_win_count": int(np.sum(child > envelope + meaningful_margin)),
        "mean_gain_on_wins": float(np.mean(gains[gains > 0.0])) if np.any(gains > 0.0) else 0.0,
        "complexity_penalty": float(complexity_penalty),
    }


def render_llm_synthesis_prompt(
    *,
    target_workload_niche: str,
    parent_genomes: Sequence[SchedulerGenomeV1],
    parent_strengths: Mapping[str, str],
    pairwise_advantage_evidence: Mapping[str, Any],
    frontier_gap: str,
    allowed_primitives: Sequence[str],
    forbidden_features: Sequence[str],
) -> str:
    parent_payload = [json.loads(parent.canonical_json()) for parent in parent_genomes]
    request = {
        "task": "propose_scheduler_genome_child",
        "target_workload_niche": target_workload_niche,
        "parent_genomes": parent_payload,
        "parent_strengths": dict(parent_strengths),
        "pairwise_advantage_evidence": dict(pairwise_advantage_evidence),
        "frontier_gap": frontier_gap,
        "allowed_primitives": list(allowed_primitives),
        "forbidden_features": list(forbidden_features),
        "output_contract": "Return one SchedulerGenomeV1 JSON object only. Do not use forbidden or future-looking features.",
    }
    return json.dumps(request, indent=2, sort_keys=True)


def verify_child(genome: SchedulerGenomeV1) -> bool:
    compile_genome(genome)
    return True

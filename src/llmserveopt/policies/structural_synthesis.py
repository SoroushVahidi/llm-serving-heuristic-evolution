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


# ---------------------------------------------------------------------------
# Shared expression builders for the expanded policy audit (see
# docs/current/POLICY_GENOME_COVERAGE_AUDIT.md). These follow
# policies/scoring.py's exact formulas:
#   service_proxy = alpha * prompt_tokens + beta * predicted_output_tokens
#   slack (unitless)  = deadline_slack - service_proxy
#   slack (seconds)   = deadline_slack - service_proxy * step_size
# `req.deadline_slack` (DSL built-in) is exactly `slo_deadline - now`, with no
# service-time subtraction -- these helpers add that subtraction explicitly
# so the DSL expression matches each policy's real formula, not just its
# built-in urgency proxy.
# ---------------------------------------------------------------------------

def _service_score_expr(alpha: float, beta: float, step_size: float | None = None) -> dict[str, Any]:
    """-(alpha*prompt + beta*output), optionally scaled by step_size.

    This is the DSL-score-convention (higher score = ranked first) form of
    "prefer smaller estimated service time": ascending real service time
    maps to descending DSL score.
    """
    scale = step_size if step_size is not None else 1.0
    return weighted_sum(
        (var("req.prompt_tokens"), -alpha * scale),
        (var("req.predicted_output_tokens"), -beta * scale),
    )


def _slack_expr(alpha: float, beta: float, step_size: float | None = None) -> dict[str, Any]:
    """deadline_slack - service_proxy[*step_size] -- the real "laxity"/"slack"
    used throughout policies/scoring.py and policy_library_v2_helpers.py,
    as opposed to the DSL's raw req.deadline_slack built-in."""
    scale = step_size if step_size is not None else 1.0
    return weighted_sum(
        (var("req.deadline_slack"), 1.0),
        (var("req.prompt_tokens"), -alpha * scale),
        (var("req.predicted_output_tokens"), -beta * scale),
    )


def _urgency_expr(slack_expr: dict[str, Any], eps: float = 1e-9) -> dict[str, Any]:
    """1 / max(slack, eps) -- matches policies/scoring.py::urgency_score exactly."""
    return op("div_safe", const(1.0), op("max", slack_expr, const(eps)))


def _if_then_else(cond: dict[str, Any], then: dict[str, Any], else_: dict[str, Any]) -> dict[str, Any]:
    return {"op": "if_then_else", "cond": cond, "then": then, "else": else_}


def _bucket_penalty_expr(var_name: str, edges: list[float]) -> dict[str, Any]:
    """Nested if_then_else approximating an ascending step function of
    `var_name` over `edges` -- 0 for the first bucket, 1 for the next, etc.
    Used only where the real policy buckets a continuous field (e.g.
    multi_bin_batching's predicted-output bins); this is an explicit,
    disclosed approximation of discrete binning, not a claim of exact
    reproduction of bin-affinity placement."""
    expr = const(float(len(edges)))
    for i in range(len(edges) - 1, -1, -1):
        edge = edges[i]
        expr = _if_then_else(op("sub", const(edge), var(var_name)), const(float(i)), expr)
    return expr


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
    if policy_name == "fifo":
        return SchedulerGenomeV1(
            name="genome_fifo",
            admission_rule=None,
            priority_rule=module("priority_rule", const(0.0), description="no ranking signal; pure arrival order via tie-breaker"),
            tie_breaker="arrival_order",
            metadata={"source_policy": policy_name, "mapping_status": "EXACT"},
        )
    if policy_name == "shortest_output_first":
        return SchedulerGenomeV1(
            name="genome_shortest_output_first",
            admission_rule=None,
            priority_rule=module("priority_rule", op("neg", var("req.predicted_output_tokens")), description="ascending predicted_output_tokens"),
            tie_breaker="shortest_output",
            metadata={"source_policy": policy_name, "mapping_status": "EXACT"},
        )
    if policy_name == "shortest_prompt_first":
        return SchedulerGenomeV1(
            name="genome_shortest_prompt_first",
            admission_rule=None,
            priority_rule=module("priority_rule", op("neg", var("req.prompt_tokens")), description="ascending prompt_tokens"),
            tie_breaker="shortest_prompt",
            metadata={"source_policy": policy_name, "mapping_status": "EXACT"},
        )
    if policy_name == "first_fit":
        return SchedulerGenomeV1(
            name="genome_first_fit",
            admission_rule=None,
            priority_rule=module("priority_rule", const(0.0), description="arrival/id order; placement matches the DSL's default scan-first-feasible-GPU order"),
            tie_breaker="arrival_order",
            metadata={
                "source_policy": policy_name, "mapping_status": "EXACT",
                "limitation": "faithful only if ObservableState.gpu_states is already gpu_id-ordered, matching first_fit.py's explicit sort",
            },
        )
    if policy_name == "orca_style":
        return SchedulerGenomeV1(
            name="genome_orca_style",
            admission_rule=None,
            priority_rule=module("priority_rule", var("req.priority_weight"), description="priority-class descending, FCFS within class"),
            tie_breaker="highest_priority",
            metadata={
                "source_policy": policy_name, "mapping_status": "EXACT",
                "limitation": "tie-break uses request_id, not arrival_time, as the final key (equivalent given request IDs are arrival-ordered)",
            },
        )
    if policy_name == "slo_slack_score":
        slack = _slack_expr(0.5, 1.0)
        return SchedulerGenomeV1(
            name="genome_slo_slack_score",
            admission_rule=None,
            priority_rule=module(
                "priority_rule",
                op("add", _urgency_expr(slack), var("req.priority_weight")),
                description="urgency(deadline slack, service proxy) + priority, matches scoring.urgency_score exactly",
                alpha=0.5, beta=1.0, priority_weight=1.0,
            ),
            tie_breaker="arrival_order",
            metadata={"source_policy": policy_name, "mapping_status": "EXACT"},
        )
    if policy_name == "least_laxity_first":
        return SchedulerGenomeV1(
            name="genome_least_laxity_first",
            admission_rule=None,
            priority_rule=module(
                "priority_rule", op("neg", _slack_expr(0.5, 1.0)),
                description="ascending laxity = descending DSL score; laxity = deadline_slack - service_proxy exactly",
                alpha=0.5, beta=1.0,
            ),
            tie_breaker="earliest_deadline",
            metadata={
                "source_policy": policy_name, "mapping_status": "EXACT",
                "limitation": "primary ranking (laxity) is exact; the 3rd/4th tie-break keys (priority, request_id) are not reproducible -- the DSL supports one tie-breaker only",
            },
        )
    if policy_name == "estimated_service_time_first":
        return SchedulerGenomeV1(
            name="genome_estimated_service_time_first",
            admission_rule=None,
            priority_rule=module(
                "priority_rule", _service_score_expr(0.5, 1.0),
                description="ascending alpha*prompt+beta*output = descending DSL score, matches predicted_service_proxy exactly",
                alpha=0.5, beta=1.0,
            ),
            tie_breaker="earliest_deadline",
            metadata={
                "source_policy": policy_name, "mapping_status": "EXACT",
                "limitation": "primary ranking (estimated service time) is exact; the 3rd/4th tie-break keys are not reproducible",
            },
        )
    if policy_name == "admission_control":
        slack_seconds = _slack_expr(0.5, 1.0, step_size=0.001)
        return SchedulerGenomeV1(
            name="genome_admission_control",
            admission_rule=module(
                "admission_rule", op("add", slack_seconds, const(0.0)), status="APPROXIMATE",
                description="laxity(seconds) >= -laxity_threshold; genome encodes laxity_threshold=0.0 (meaningful filtering) rather than the deployed default (inf, a no-op)",
                laxity_threshold_encoded=0.0, laxity_threshold_registry_default="inf", alpha=0.5, beta=1.0, step_size=0.001,
            ),
            priority_rule=module(
                "priority_rule", op("neg", slack_seconds),
                description="ascending laxity(seconds) = descending DSL score",
                alpha=0.5, beta=1.0, step_size=0.001,
            ),
            tie_breaker="highest_priority",
            metadata={
                "source_policy": policy_name, "mapping_status": "APPROXIMATE",
                "limitation": "the registry-default instance (laxity_threshold=inf) makes admission a no-op; genome represents the mechanism using the documented laxity_threshold=0.0 case instead. Tie-break keys 3-5 are not reproducible.",
            },
        )
    if policy_name == "aging_priority":
        laxity = _slack_expr(0.5, 1.0, step_size=0.001)
        aged_priority_over_service = op(
            "div_safe",
            weighted_sum((var("req.priority_weight"), 1.0), (var("req.waiting_time"), 0.15)),
            op("max", weighted_sum((var("req.prompt_tokens"), 0.5), (var("req.predicted_output_tokens"), 1.0)), const(1e-9)),
        )
        return SchedulerGenomeV1(
            name="genome_aging_priority",
            admission_rule=None,
            priority_rule=module(
                "priority_rule",
                op("add", aged_priority_over_service, op("mul", const(0.2), _urgency_expr(laxity))),
                description="(priority + 0.15*waiting_time)/service + 0.2/max(laxity,eps), matching the real aging_rate=0.15 and urgency weight=0.2",
                aging_rate=0.15, urgency_weight=0.2, alpha=0.5, beta=1.0,
            ),
            fairness_rule=module("fairness_rule", var("req.waiting_time"), description="age bonus term", aging_rate=0.15),
            tie_breaker="arrival_order",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "coefficients now match the real defaults exactly; the only gap is unit precision (deadline_slack-based laxity, matching scoring.py's convention)"},
        )
    if policy_name == "scorpio_style_slo_guard":
        default_score = weighted_sum(
            (var("req.deadline_urgency"), 1.0), (var("req.priority_weight"), 1.0), (var("req.waiting_time"), 0.05),
        )
        guard_score = op("add", default_score, op("mul", const(-0.35), var("req.predicted_output_tokens")))
        return SchedulerGenomeV1(
            name="genome_scorpio_style_slo_guard",
            admission_rule=module("admission_rule", _positive_slack_expr(), status="APPROXIMATE", description="positive slack admission proxy"),
            priority_rule=module(
                "priority_rule", default_score, status="APPROXIMATE",
                description="urgency + priority_weight*priority + age_bonus*wait (real coefficients: priority_weight=1.0, age_bonus=0.05)",
                priority_weight=1.0, age_bonus=0.05,
            ),
            kv_guard=module("kv_guard", _kv_capacity_expr(0.65), status="APPROXIMATE", description="aggregate KV pressure guard", kv_utilization_threshold=0.65),
            regime_conditions=(
                RegimeCondition(
                    name="kv_pressure_guard",
                    condition=op("sub", sys_var("sys.kv_utilization"), const(0.65)),
                    priority_rule=module(
                        "priority_rule", guard_score, status="APPROXIMATE",
                        description="decode-length penalty (decode_penalty_weight*beta=0.35) applied only while sys.kv_utilization exceeds the real threshold 0.65",
                        decode_penalty_weight=0.35,
                    ),
                    admission_rule=module("admission_rule", _kv_capacity_expr(0.65), status="APPROXIMATE", description="tighter admission under the guard regime"),
                ),
            ),
            tie_breaker="earliest_deadline",
            metadata={
                "source_policy": policy_name, "mapping_status": "APPROXIMATE",
                "limitation": "the stateful admission-budget refill/consume mechanism and per-GPU decode-pressure multiplier are not representable in a stateless DSL genome; regime_conditions now captures the guard_active branching using the real kv_utilization_threshold instead of a single static formula",
            },
        )
    if policy_name == "kv_constrained_online":
        return SchedulerGenomeV1(
            name="genome_kv_constrained_online",
            admission_rule=module("admission_rule", _kv_capacity_expr(0.82), status="APPROXIMATE", description="aggregate KV utilization threshold (real target_kv_utilization=0.82)", target_kv_utilization=0.82),
            priority_rule=module(
                "priority_rule", op("div_safe", var("req.deadline_urgency"), op("max", var("req.estimated_kv_cost"), const(1.0))),
                status="APPROXIMATE", description="urgency per KV cost",
            ),
            kv_guard=module("kv_guard", _kv_capacity_expr(0.82), status="APPROXIMATE", target_kv_utilization=0.82),
            tie_breaker="earliest_deadline",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "per-GPU post-placement KV reserve and the urgent-laxity-seconds admission override are not exactly encoded"},
        )
    if policy_name == "adaptive_chunked_prefill":
        return SchedulerGenomeV1(
            name="genome_adaptive_chunked_prefill",
            admission_rule=module("admission_rule", op("sub", const(0.55), sys_var("sys.slo_pressure")), status="APPROXIMATE", description="real pressure_threshold=0.55", pressure_threshold=0.55),
            priority_rule=module("priority_rule", weighted_sum((var("req.prompt_tokens"), -1.0), (var("req.deadline_urgency"), 0.5)), status="APPROXIMATE"),
            prefill_rule=module(
                "prefill_rule", op("sub", const(2.048), op("div_safe", var("req.prompt_tokens"), const(1000.0))),
                status="APPROXIMATE", description="real long_prompt_threshold=2048 tokens, scaled by /1000 to stay within the DSL's [-1000,1000] constant range",
                long_prompt_threshold=2048,
            ),
            tie_breaker="shortest_prompt",
            metadata={"source_policy": policy_name, "mapping_status": "APPROXIMATE", "limitation": "the concurrent-long-prefill admission cap (a running count over currently-active requests) is not representable; true chunk-size actions are unsupported"},
        )
    if policy_name == "sola_style_state_aware":
        load_proxy = weighted_sum(
            (sys_var("sys.kv_utilization"), 0.5),
            (op("sub", const(1.0), sys_var("sys.free_sequence_ratio")), 0.5),
        )
        return SchedulerGenomeV1(
            name="genome_sola_style_state_aware",
            admission_rule=None,
            priority_rule=module(
                "priority_rule",
                weighted_sum(
                    (var("req.priority_weight"), 1.8),
                    (var("req.deadline_urgency"), 1.2),
                    (var("req.waiting_time"), 0.04),
                ),
                status="APPROXIMATE",
                description="real coefficients 1.8*priority + 1.2*urgency + 0.04*wait; the load-and-KV-scaled service/KV penalty terms are approximated via kv_guard rather than folded into the score",
                priority_coef=1.8, urgency_coef=1.2, wait_coef=0.04,
            ),
            kv_guard=module("kv_guard", op("sub", const(1.0), load_proxy), status="APPROXIMATE", description="load_proxy = 0.5*kv_utilization + 0.5*(1-free_sequence_ratio), approximating system_pressure"),
            tie_breaker="arrival_order",
            metadata={
                "source_policy": policy_name, "mapping_status": "APPROXIMATE",
                "limitation": "system_pressure's queue-length term and the load/KV-scaled service penalty are not exactly reproduced; GPU-placement pressure ordering is not representable",
            },
        )
    if policy_name == "flow_control_stability":
        return SchedulerGenomeV1(
            name="genome_flow_control_stability",
            admission_rule=module("admission_rule", op("sub", const(0.62), sys_var("sys.slo_pressure")), status="APPROXIMATE", description="overload proxy (real overload_threshold=0.62 on system_pressure, approximated via sys.slo_pressure)", overload_threshold=0.62),
            priority_rule=module(
                "priority_rule", op("neg", _slack_expr(0.5, 1.0, step_size=0.001)),
                status="APPROXIMATE", description="laxity-first ranking (secondary WSPT-over-priority tiebreak not representable in one score)",
            ),
            tie_breaker="earliest_deadline",
            metadata={
                "source_policy": policy_name, "mapping_status": "APPROXIMATE",
                "limitation": "the stateful refilling admission budget (this policy's defining 'flow control' mechanism) and arrival-slope overload trigger are not representable in a stateless DSL genome; only the underlying laxity ranking is captured",
            },
        )
    if policy_name == "weighted_fair_share":
        return SchedulerGenomeV1(
            name="genome_weighted_fair_share",
            admission_rule=None,
            priority_rule=module(
                "priority_rule", op("div_safe", var("req.priority_weight"), op("max", weighted_sum((var("req.prompt_tokens"), 0.5), (var("req.predicted_output_tokens"), 1.0)), const(1e-9))),
                status="APPROXIMATE", description="residual priority/service ranking only",
            ),
            fairness_rule=module("fairness_rule", const(0.0), status="UNSUPPORTED", description="class-demand-vs-active-service deficit requires per-class_id aggregate counts, which are not in the DSL's causal variable whitelist"),
            tie_breaker="arrival_order",
            metadata={
                "source_policy": policy_name, "mapping_status": "APPROXIMATE",
                "limitation": "the class-fairness deficit term -- this policy's defining fairness mechanism -- cannot be represented; the genome captures only the residual priority/service ranking that remains without it",
            },
        )
    if policy_name == "multi_bin_batching":
        return SchedulerGenomeV1(
            name="genome_multi_bin_batching",
            admission_rule=None,
            priority_rule=module(
                "priority_rule", op("neg", _bucket_penalty_expr("req.predicted_output_tokens", [32.0, 64.0, 128.0, 256.0])),
                status="APPROXIMATE", description="approximates the real [32,64,128,256]-edge bin grouping via nested thresholds; ranking only",
                bin_edges=[32.0, 64.0, 128.0, 256.0],
            ),
            tie_breaker="shortest_output",
            metadata={
                "source_policy": policy_name, "mapping_status": "APPROXIMATE",
                "limitation": "bin-affinity GPU placement (this policy's defining length-mismatch-reduction mechanism) is not representable; only an approximate bucketed ranking is captured",
            },
        )
    return SchedulerGenomeV1(
        name=f"genome_{policy_name}",
        admission_rule=None,
        priority_rule=module("priority_rule", var("req.priority_weight"), status="UNSUPPORTED", description="placeholder priority"),
        metadata={
            "source_policy": policy_name, "mapping_status": "UNSUPPORTED",
            "limitation": UNMAPPABLE_REASONS.get(policy_name, "no canonical parent mapping implemented"),
        },
    )


UNMAPPABLE_REASONS: dict[str, str] = {
    "greedy_token_fill": "defining behavior is KV-remaining-capacity GPU placement; the DSL genome has no placement-strategy module, only admission/ranking",
    "least_loaded": "defining behavior is active-sequence-count GPU placement; not representable without a placement-strategy module",
    "best_fit": "defining behavior is tightest-remaining-KV GPU placement; not representable without a placement-strategy module",
    "random_feasible": "the DSL is fully deterministic (no RNG primitive in ALLOWED_OPS/ALLOWED_VARS); a stochastic permutation policy cannot be expressed",
    "vllm_style_token_budget": "the per-step, per-GPU block-rounded KV admission loop with running local totals is not verified representable via the DSL's admission_condition in this pass; not attempted to avoid an unverified, possibly-misleading mapping",
    "sarathi_style": "the per-GPU decode-priority-halved prefill-chunk-budget admission loop with a starvation safety valve is not verified representable via the DSL's admission_condition in this pass; not attempted to avoid an unverified, possibly-misleading mapping",
    "splitfuse_style": "the per-GPU fixed-token-budget-fill admission loop with an oversized-prefill safety valve is not verified representable via the DSL's admission_condition in this pass; not attempted to avoid an unverified, possibly-misleading mapping",
    "slai_style_phase_aware": "the defining phase-interference penalty depends on aggregate prefilling_count/decoding_count phase shares, which are not in the DSL's causal variable whitelist (ALLOWED_VARS); representing this policy without its phase term would misrepresent it",
}


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

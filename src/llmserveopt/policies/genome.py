"""Typed scheduler genome representation for structural synthesis.

The genome is a conservative wrapper around the existing verified heuristic
DSL.  It separates conceptual scheduling modules while compiling only the
subset that the current simulator can execute safely: admission conditions,
request priority scores, and simple regime-dependent priority rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from ..heuristics.compiler import CompilationError, compile_heuristic
from ..heuristics.dsl_schema import ALLOWED_OPS, ALLOWED_TIE_BREAKERS, ALLOWED_VARS
from ..heuristics.policy import HeuristicPolicy, build_heuristic_policy
from ..heuristics.verifier import verify_heuristic

GENOME_SCHEMA_VERSION = "SchedulerGenomeV1"
SUPPORTED_MODULE_TYPES = frozenset({
    "admission_rule",
    "priority_rule",
    "prefill_rule",
    "kv_guard",
    "fairness_rule",
    "regime_conditions",
})


class GenomeValidationError(ValueError):
    """Raised when a scheduler genome is syntactically or semantically invalid."""


@dataclass(frozen=True)
class GenomeModule:
    module_type: str
    expression: dict[str, Any] | None = None
    status: str = "EXACT"
    description: str = ""
    unsupported_reason: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "expression": copy.deepcopy(self.expression),
            "status": self.status,
            "description": self.description,
            "unsupported_reason": self.unsupported_reason,
            "parameters": copy.deepcopy(self.parameters),
        }


@dataclass(frozen=True)
class RegimeCondition:
    name: str
    condition: dict[str, Any]
    priority_rule: GenomeModule
    admission_rule: GenomeModule | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition": copy.deepcopy(self.condition),
            "priority_rule": self.priority_rule.to_dict(),
            "admission_rule": self.admission_rule.to_dict() if self.admission_rule else None,
        }


@dataclass(frozen=True)
class SchedulerGenomeV1:
    name: str
    admission_rule: GenomeModule | None
    priority_rule: GenomeModule
    prefill_rule: GenomeModule | None = None
    kv_guard: GenomeModule | None = None
    fairness_rule: GenomeModule | None = None
    regime_conditions: tuple[RegimeCondition, ...] = ()
    tie_breaker: str = "arrival_order"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = GENOME_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "tie_breaker": self.tie_breaker,
            "admission_rule": self.admission_rule.to_dict() if self.admission_rule else None,
            "priority_rule": self.priority_rule.to_dict(),
            "prefill_rule": self.prefill_rule.to_dict() if self.prefill_rule else None,
            "kv_guard": self.kv_guard.to_dict() if self.kv_guard else None,
            "fairness_rule": self.fairness_rule.to_dict() if self.fairness_rule else None,
            "regime_conditions": [r.to_dict() for r in self.regime_conditions],
            "metadata": copy.deepcopy(self.metadata),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def validate(self) -> None:
        validate_genome(self)

    def to_heuristic_dict(self) -> dict[str, Any]:
        return genome_to_heuristic_dict(self)

    def build_policy(self) -> HeuristicPolicy:
        self.validate()
        return build_heuristic_policy(self.to_heuristic_dict())


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_genome(payload: Mapping[str, Any]) -> SchedulerGenomeV1:
    if payload.get("schema_version") != GENOME_SCHEMA_VERSION:
        raise GenomeValidationError(f"Unsupported genome schema {payload.get('schema_version')!r}")

    def module_from(raw: Mapping[str, Any] | None) -> GenomeModule | None:
        if raw is None:
            return None
        return GenomeModule(
            module_type=str(raw["module_type"]),
            expression=copy.deepcopy(raw.get("expression")),
            status=str(raw.get("status", "EXACT")),
            description=str(raw.get("description", "")),
            unsupported_reason=raw.get("unsupported_reason"),
            parameters=copy.deepcopy(raw.get("parameters", {})),
        )

    regimes = []
    for raw in payload.get("regime_conditions", []):
        priority = module_from(raw.get("priority_rule"))
        if priority is None:
            raise GenomeValidationError("Regime missing priority_rule")
        regimes.append(RegimeCondition(
            name=str(raw["name"]),
            condition=copy.deepcopy(raw["condition"]),
            priority_rule=priority,
            admission_rule=module_from(raw.get("admission_rule")),
        ))

    priority_rule = module_from(payload.get("priority_rule"))
    if priority_rule is None:
        raise GenomeValidationError("Genome missing priority_rule")
    genome = SchedulerGenomeV1(
        name=str(payload["name"]),
        admission_rule=module_from(payload.get("admission_rule")),
        priority_rule=priority_rule,
        prefill_rule=module_from(payload.get("prefill_rule")),
        kv_guard=module_from(payload.get("kv_guard")),
        fairness_rule=module_from(payload.get("fairness_rule")),
        regime_conditions=tuple(regimes),
        tie_breaker=str(payload.get("tie_breaker", "arrival_order")),
        metadata=copy.deepcopy(payload.get("metadata", {})),
    )
    genome.validate()
    return genome


def _walk_expressions(expr: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(expr, dict):
        return
    yield expr
    if "op" not in expr:
        return
    op = expr.get("op")
    if op == "weighted_sum":
        for item in expr.get("terms", []):
            if isinstance(item, (list, tuple)) and item:
                yield from _walk_expressions(item[0])
    elif op == "if_then_else":
        for key in ("cond", "then", "else"):
            yield from _walk_expressions(expr.get(key))
    else:
        for sub in expr.get("args", []):
            yield from _walk_expressions(sub)


def _validate_expression(expr: dict[str, Any], location: str) -> None:
    for node in _walk_expressions(expr):
        if "var" in node and node["var"] not in ALLOWED_VARS:
            raise GenomeValidationError(f"{location}: forbidden or unknown causal feature {node['var']!r}")
        if "op" in node and node["op"] not in ALLOWED_OPS:
            raise GenomeValidationError(f"{location}: unsupported operator {node['op']!r}")
        if "const" in node:
            value = node["const"]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise GenomeValidationError(f"{location}: non-finite constant {value!r}")


def _validate_module(module: GenomeModule | None, expected_type: str, *, executable_required: bool = False) -> None:
    if module is None:
        return
    if module.module_type != expected_type:
        raise GenomeValidationError(f"Expected module type {expected_type!r}, got {module.module_type!r}")
    if module.module_type not in SUPPORTED_MODULE_TYPES:
        raise GenomeValidationError(f"Unknown module type {module.module_type!r}")
    if module.status not in {"EXACT", "APPROXIMATE", "UNSUPPORTED"}:
        raise GenomeValidationError(f"Invalid mapping status {module.status!r}")
    if executable_required and module.expression is None:
        raise GenomeValidationError(f"Executable module {module.module_type!r} needs an expression")
    if module.expression is not None:
        _validate_expression(module.expression, module.module_type)


def validate_genome(genome: SchedulerGenomeV1) -> None:
    if genome.schema_version != GENOME_SCHEMA_VERSION:
        raise GenomeValidationError(f"Unsupported genome schema {genome.schema_version!r}")
    if genome.tie_breaker not in ALLOWED_TIE_BREAKERS:
        raise GenomeValidationError(f"Unsupported tie_breaker {genome.tie_breaker!r}")
    _validate_module(genome.admission_rule, "admission_rule")
    _validate_module(genome.priority_rule, "priority_rule", executable_required=True)
    _validate_module(genome.prefill_rule, "prefill_rule")
    _validate_module(genome.kv_guard, "kv_guard")
    _validate_module(genome.fairness_rule, "fairness_rule")
    for regime in genome.regime_conditions:
        _validate_expression(regime.condition, f"regime_conditions.{regime.name}.condition")
        _validate_module(regime.priority_rule, "priority_rule", executable_required=True)
        _validate_module(regime.admission_rule, "admission_rule")
    try:
        heuristic = genome_to_heuristic_dict(genome)
        result = verify_heuristic(heuristic)
    except Exception as exc:
        raise GenomeValidationError(f"Genome cannot be converted to verified heuristic: {exc}") from exc
    if not result.valid:
        raise GenomeValidationError(f"Genome heuristic verification failed: {result.errors}")


def _combine_conditions(expressions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not expressions:
        return None
    expr = expressions[0]
    for other in expressions[1:]:
        expr = {"op": "min", "args": [expr, other]}
    return expr


def genome_to_heuristic_dict(genome: SchedulerGenomeV1) -> dict[str, Any]:
    default_rule: dict[str, Any] = {"request_score": copy.deepcopy(genome.priority_rule.expression)}
    conditions = [
        m.expression for m in (genome.admission_rule, genome.prefill_rule, genome.kv_guard)
        if m is not None and m.expression is not None and m.status != "UNSUPPORTED"
    ]
    admission = _combine_conditions(conditions)
    if admission is not None:
        default_rule["admission_condition"] = admission

    regimes = []
    for regime in genome.regime_conditions:
        rule = {
            "condition": copy.deepcopy(regime.condition),
            "request_score": copy.deepcopy(regime.priority_rule.expression),
        }
        regime_conditions = []
        if regime.admission_rule and regime.admission_rule.expression is not None:
            regime_conditions.append(regime.admission_rule.expression)
        if regime_conditions:
            rule["admission_condition"] = _combine_conditions(copy.deepcopy(regime_conditions))
        regimes.append(rule)

    return {
        "name": genome.name,
        "description": genome.metadata.get("description", "generated scheduler genome"),
        "tie_breaker": genome.tie_breaker,
        "default": default_rule,
        "regimes": regimes,
    }


def compile_genome(genome: SchedulerGenomeV1):
    genome.validate()
    try:
        return compile_heuristic(genome.to_heuristic_dict())
    except CompilationError as exc:
        raise GenomeValidationError(str(exc)) from exc


def const(value: float) -> dict[str, Any]:
    return {"const": float(value)}


def var(name: str) -> dict[str, Any]:
    return {"var": name}


def op(name: str, *args: dict[str, Any]) -> dict[str, Any]:
    return {"op": name, "args": list(args)}


def weighted_sum(*terms: tuple[dict[str, Any], float]) -> dict[str, Any]:
    return {"op": "weighted_sum", "terms": [[expr, float(weight)] for expr, weight in terms]}


def module(module_type: str, expression: dict[str, Any] | None, *, status: str = "EXACT", description: str = "", unsupported_reason: str | None = None, **parameters: Any) -> GenomeModule:
    return GenomeModule(module_type, expression, status, description, unsupported_reason, parameters)

"""
Verifier for the scheduling heuristic DSL.

verify_heuristic(heuristic_dict) → VerificationResult

Error codes
-----------
SCHEMA_MISSING_FIELD        — required top-level field absent
SCHEMA_INVALID_TYPE         — field has wrong type
FORBIDDEN_VARIABLE          — var references an explicitly forbidden name
FORBIDDEN_SUBSTRING         — var name contains a forbidden substring
UNKNOWN_VARIABLE            — var references a name not in ALLOWED_VARS
FORBIDDEN_OP                — expression uses a forbidden operation name
UNKNOWN_OP                  — expression uses an op not in ALLOWED_OPS
EXPRESSION_DEPTH_EXCEEDED   — expression tree deeper than max_expression_depth
EXPRESSION_NODE_EXCEEDED    — total expression nodes exceed max_nodes
TERMS_EXCEEDED              — weighted_sum has more terms than max_terms
CONSTANT_OUT_OF_RANGE       — const value outside [min_constant, max_constant]
FORBIDDEN_TIE_BREAKER       — tie_breaker not in ALLOWED_TIE_BREAKERS
EXPRESSION_NOT_FINITE       — expression produces non-finite value on dummy input
TOO_MANY_REGIMES            — regime count exceeds max_regimes
REGIME_MISSING_CONDITION    — a regime block lacks a condition expression
REGIME_MISSING_SCORE        — a regime block lacks request_score
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .dsl_schema import (
    ALLOWED_OPS,
    ALLOWED_TIE_BREAKERS,
    ALLOWED_VARS,
    DEFAULT_LIMITS,
    FORBIDDEN_OPS,
    FORBIDDEN_SUBSTRINGS,
    FORBIDDEN_VARS,
    REQUIRED_FIELDS,
    REQUIRED_RULE_FIELDS,
)
from .expressions import ExpressionError, evaluate_expression


@dataclass
class VerificationResult:
    valid: bool
    errors: List[Tuple[str, str]] = field(default_factory=list)  # [(code, message), ...]
    warnings: List[str] = field(default_factory=list)

    def add_error(self, code: str, message: str) -> None:
        self.errors.append((code, message))
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _check_var_name(name: str, result: VerificationResult) -> None:
    if name in FORBIDDEN_VARS:
        result.add_error("FORBIDDEN_VARIABLE", f"Variable '{name}' is explicitly forbidden")
        return
    for sub in FORBIDDEN_SUBSTRINGS:
        if sub in name:
            result.add_error("FORBIDDEN_SUBSTRING", f"Variable '{name}' contains forbidden substring '{sub}'")
            return
    if name not in ALLOWED_VARS:
        result.add_error("UNKNOWN_VARIABLE", f"Variable '{name}' is not in ALLOWED_VARS")


def _count_nodes(expr: Any, *, _depth: int = 0) -> Tuple[int, int]:
    """Return (total_nodes, max_depth) for expression tree."""
    if not isinstance(expr, dict):
        return (0, _depth)

    nodes = 1
    max_d = _depth

    if "const" in expr or "var" in expr:
        return (1, _depth)

    if "op" in expr:
        op = expr.get("op", "")
        if op == "weighted_sum":
            for term in expr.get("terms", []):
                if isinstance(term, (list, tuple)) and len(term) == 2:
                    n, d = _count_nodes(term[0], _depth=_depth + 1)
                    nodes += n
                    max_d = max(max_d, d)
        elif op == "if_then_else":
            for sub_key in ("cond", "then", "else"):
                if sub_key in expr:
                    n, d = _count_nodes(expr[sub_key], _depth=_depth + 1)
                    nodes += n
                    max_d = max(max_d, d)
        else:
            for arg in expr.get("args", []):
                n, d = _count_nodes(arg, _depth=_depth + 1)
                nodes += n
                max_d = max(max_d, d)

    return (nodes, max_d)


def _verify_expression(
    expr: Any,
    result: VerificationResult,
    limits: Dict[str, Any],
    location: str,
    *,
    _depth: int = 0,
) -> None:
    max_depth = limits.get("max_expression_depth", DEFAULT_LIMITS["max_expression_depth"])
    max_nodes = limits.get("max_nodes", DEFAULT_LIMITS["max_nodes"])
    max_terms = limits.get("max_terms", DEFAULT_LIMITS["max_terms"])
    min_const = limits.get("min_constant", DEFAULT_LIMITS["min_constant"])
    max_const = limits.get("max_constant", DEFAULT_LIMITS["max_constant"])

    if not isinstance(expr, dict):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}: expression must be a dict, got {type(expr).__name__}")
        return

    if _depth == 0:
        # Run global node count and depth check once at root
        total_nodes, actual_depth = _count_nodes(expr)
        if actual_depth > max_depth:
            result.add_error(
                "EXPRESSION_DEPTH_EXCEEDED",
                f"{location}: depth {actual_depth} > max_expression_depth {max_depth}",
            )
        if total_nodes > max_nodes:
            result.add_error(
                "EXPRESSION_NODE_EXCEEDED",
                f"{location}: {total_nodes} nodes > max_nodes {max_nodes}",
            )

    if "const" in expr:
        v = expr["const"]
        if not isinstance(v, (int, float)):
            result.add_error("SCHEMA_INVALID_TYPE", f"{location}: const must be numeric")
        elif not math.isfinite(v):
            result.add_error("EXPRESSION_NOT_FINITE", f"{location}: const is non-finite: {v}")
        elif v < min_const or v > max_const:
            result.add_error(
                "CONSTANT_OUT_OF_RANGE",
                f"{location}: const {v} outside [{min_const}, {max_const}]",
            )
        return

    if "var" in expr:
        _check_var_name(expr["var"], result)
        return

    if "op" not in expr:
        result.add_error(
            "SCHEMA_MISSING_FIELD",
            f"{location}: expression has neither 'const', 'var', nor 'op'",
        )
        return

    op = expr["op"]
    if not isinstance(op, str):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}: op must be a string")
        return

    if op in FORBIDDEN_OPS:
        result.add_error("FORBIDDEN_OP", f"{location}: operation '{op}' is forbidden")
        return

    if op not in ALLOWED_OPS:
        result.add_error("UNKNOWN_OP", f"{location}: operation '{op}' is not in ALLOWED_OPS")
        return

    def _sub(sub_expr: Any, suffix: str) -> None:
        _verify_expression(sub_expr, result, limits, f"{location}/{suffix}", _depth=_depth + 1)

    if op == "weighted_sum":
        terms = expr.get("terms", [])
        if not isinstance(terms, list):
            result.add_error("SCHEMA_INVALID_TYPE", f"{location}/weighted_sum: terms must be a list")
            return
        if len(terms) > max_terms:
            result.add_error(
                "TERMS_EXCEEDED",
                f"{location}/weighted_sum: {len(terms)} terms > max_terms {max_terms}",
            )
        for i, item in enumerate(terms):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                result.add_error(
                    "SCHEMA_INVALID_TYPE",
                    f"{location}/weighted_sum/term[{i}]: must be [expr, weight]",
                )
                continue
            _sub(item[0], f"term[{i}]/expr")
            w = item[1]
            if not isinstance(w, (int, float)):
                result.add_error(
                    "SCHEMA_INVALID_TYPE",
                    f"{location}/weighted_sum/term[{i}]/weight: must be numeric",
                )
            elif not math.isfinite(w):
                result.add_error(
                    "EXPRESSION_NOT_FINITE",
                    f"{location}/weighted_sum/term[{i}]/weight: non-finite {w}",
                )
            elif w < min_const or w > max_const:
                result.add_error(
                    "CONSTANT_OUT_OF_RANGE",
                    f"{location}/weighted_sum/term[{i}]/weight: {w} outside [{min_const}, {max_const}]",
                )
        return

    if op == "if_then_else":
        for sub_key in ("cond", "then", "else"):
            if sub_key not in expr:
                result.add_error(
                    "SCHEMA_MISSING_FIELD",
                    f"{location}/if_then_else: missing '{sub_key}'",
                )
            else:
                _sub(expr[sub_key], sub_key)
        return

    args = expr.get("args", [])
    if not isinstance(args, list):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}/{op}: args must be a list")
        return
    for i, arg in enumerate(args):
        _sub(arg, f"args[{i}]")


def _build_dummy_context() -> Dict[str, float]:
    """Build a dummy variable context with all allowed vars set to 0.5."""
    from .dsl_schema import ALLOWED_VARS
    return {v: 0.5 for v in ALLOWED_VARS}


def _verify_expression_finite(
    expr: Any,
    result: VerificationResult,
    location: str,
) -> None:
    """Try to evaluate expr with dummy inputs; flag if ExpressionError is raised."""
    if not isinstance(expr, dict):
        return
    ctx = _build_dummy_context()
    try:
        val = evaluate_expression(expr, ctx)
        if not math.isfinite(val):
            result.add_error(
                "EXPRESSION_NOT_FINITE",
                f"{location}: expression returned non-finite value {val} on dummy input",
            )
    except ExpressionError as e:
        result.add_error(
            "EXPRESSION_NOT_FINITE",
            f"{location}: expression raised ExpressionError on dummy input: {e}",
        )


def _verify_rule_block(
    block: Any,
    result: VerificationResult,
    limits: Dict[str, Any],
    location: str,
) -> None:
    if not isinstance(block, dict):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}: rule block must be a dict")
        return
    for req_field in REQUIRED_RULE_FIELDS:
        if req_field not in block:
            result.add_error(
                "SCHEMA_MISSING_FIELD",
                f"{location}: rule block missing required field '{req_field}'",
            )
    if "request_score" in block:
        expr = block["request_score"]
        _verify_expression(expr, result, limits, f"{location}/request_score")
        _verify_expression_finite(expr, result, f"{location}/request_score")
    if "batch_score" in block:
        expr = block["batch_score"]
        _verify_expression(expr, result, limits, f"{location}/batch_score")
        _verify_expression_finite(expr, result, f"{location}/batch_score")
    if "admission_condition" in block:
        expr = block["admission_condition"]
        _verify_expression(expr, result, limits, f"{location}/admission_condition")
        _verify_expression_finite(expr, result, f"{location}/admission_condition")


def verify_heuristic(heuristic: Any, *, extra_limits: Optional[Dict[str, Any]] = None) -> VerificationResult:
    """Verify that `heuristic` is a valid, safe scheduling heuristic DSL document.

    Parameters
    ----------
    heuristic : dict — parsed JSON heuristic document.
    extra_limits : optional overrides to DEFAULT_LIMITS.

    Returns
    -------
    VerificationResult — .valid is True iff all checks pass.
    """
    result = VerificationResult(valid=True)
    limits = {**DEFAULT_LIMITS, **(extra_limits or {})}

    if not isinstance(heuristic, dict):
        result.add_error("SCHEMA_INVALID_TYPE", "Heuristic must be a JSON object (dict)")
        return result

    # Required top-level fields
    for req in REQUIRED_FIELDS:
        if req not in heuristic:
            result.add_error("SCHEMA_MISSING_FIELD", f"Missing required field: '{req}'")

    # name
    if "name" in heuristic:
        if not isinstance(heuristic["name"], str) or not heuristic["name"].strip():
            result.add_error("SCHEMA_INVALID_TYPE", "'name' must be a non-empty string")
        else:
            name = heuristic["name"]
            for sub in FORBIDDEN_SUBSTRINGS:
                if sub in name:
                    result.add_warning(f"'name' contains suspicious substring '{sub}'")

    # tie_breaker
    if "tie_breaker" in heuristic:
        tb = heuristic["tie_breaker"]
        if not isinstance(tb, str):
            result.add_error("SCHEMA_INVALID_TYPE", "'tie_breaker' must be a string")
        elif tb not in ALLOWED_TIE_BREAKERS:
            result.add_error(
                "FORBIDDEN_TIE_BREAKER",
                f"tie_breaker '{tb}' is not in ALLOWED_TIE_BREAKERS: {sorted(ALLOWED_TIE_BREAKERS)}",
            )

    # default rule block
    if "default" in heuristic:
        _verify_rule_block(heuristic["default"], result, limits, "default")

    # optional regimes
    if "regimes" in heuristic:
        regimes = heuristic["regimes"]
        if not isinstance(regimes, list):
            result.add_error("SCHEMA_INVALID_TYPE", "'regimes' must be a list")
        else:
            max_regimes = limits.get("max_regimes", DEFAULT_LIMITS["max_regimes"])
            if len(regimes) > max_regimes:
                result.add_error(
                    "TOO_MANY_REGIMES",
                    f"regimes count {len(regimes)} > max_regimes {max_regimes}",
                )
            for i, regime in enumerate(regimes):
                loc = f"regimes[{i}]"
                if not isinstance(regime, dict):
                    result.add_error("SCHEMA_INVALID_TYPE", f"{loc}: must be a dict")
                    continue
                if "condition" not in regime:
                    result.add_error("REGIME_MISSING_CONDITION", f"{loc}: missing 'condition' expression")
                else:
                    _verify_expression(regime["condition"], result, limits, f"{loc}/condition")
                    _verify_expression_finite(regime["condition"], result, f"{loc}/condition")
                if "request_score" not in regime:
                    result.add_error("REGIME_MISSING_SCORE", f"{loc}: missing 'request_score'")
                else:
                    _verify_rule_block(regime, result, limits, loc)

    # optional description / metadata fields (no validation needed, just type check)
    if "description" in heuristic and not isinstance(heuristic["description"], str):
        result.add_warning("'description' should be a string")

    return result

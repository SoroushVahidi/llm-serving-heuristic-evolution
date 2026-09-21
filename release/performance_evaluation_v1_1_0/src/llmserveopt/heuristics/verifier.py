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

CC3 (compositional DSL) error codes
------------------------------------
RESERVED_VAR_NAME              — a literal var uses a compiler-reserved prefix
PRIMITIVE_UNKNOWN              — {"primitive"/"primitive_gate"} name not in the CC2 registry
PRIMITIVE_WRONG_SHAPE          — name resolves to a primitive of the wrong shape for this node kind
STATEFUL_PRIMITIVE_MISPLACED   — a stateful primitive was referenced inline instead of via "admission_budget"
PRIMITIVE_PARAM_INVALID        — primitive parameter unsupported, non-numeric, NaN, or out of bounds
PRIMITIVE_BUDGET_EXCEEDED      — distinct primitive references exceed max_active_primitives
MIXTURE_EMPTY                  — weighted_sum/topk_mixture has zero terms
MIXTURE_FAMILY_INCOMPATIBLE    — mixture's primitive-rooted terms share no compatible family
TOPK_INVALID_K                 — topk_mixture.k is not an int in [1, len(terms)]
PARAM_UNDECLARED                — {"param": name} does not match any declared "parameters" entry
PARAM_SCHEMA_INVALID            — a declared parameter is missing a required field or has min>max or default outside bounds
PARAM_DUPLICATE_NAME            — two declared parameters share the same name
FALLBACK_INVALID                — "fallback.policy" is not in ALLOWED_FALLBACK_POLICIES
ON_NO_ADMITS_MISSING            — admission_condition is used somewhere but "on_no_admits" is not declared
ON_NO_ADMITS_INVALID            — "on_no_admits" is not one of ALLOWED_ON_NO_ADMITS_MODES
PLACEMENT_EMPTY                 — "placement.keys" is empty
PLACEMENT_TOO_MANY_KEYS         — "placement.keys" exceeds max_placement_keys
PLACEMENT_KEY_UNKNOWN           — a placement key name is not a known PLACEMENT primitive
ADMISSION_BUDGET_INVALID        — "admission_budget" does not reference admission_credit_budget
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import primitive_bridge as bridge
from .dsl_schema import (
    ALLOWED_FALLBACK_POLICIES,
    ALLOWED_ON_NO_ADMITS_MODES,
    ALLOWED_OPS,
    ALLOWED_PARAMETER_TYPES,
    ALLOWED_TIE_BREAKERS,
    ALLOWED_VARS,
    DEFAULT_LIMITS,
    FORBIDDEN_OPS,
    FORBIDDEN_SUBSTRINGS,
    FORBIDDEN_VARS,
    PARAM_VAR_PREFIX,
    PRIMITIVE_VAR_PREFIX,
    REQUIRED_FIELDS,
    REQUIRED_PARAMETER_FIELDS,
    REQUIRED_RULE_FIELDS,
)
from .expressions import ExpressionError, evaluate_expression
from .primitive_bridge import PrimitiveError


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
    if name.startswith(PRIMITIVE_VAR_PREFIX) or name.startswith(PARAM_VAR_PREFIX):
        result.add_error(
            "RESERVED_VAR_NAME",
            f"Variable '{name}' uses a compiler-reserved prefix; use "
            "{\"primitive\": ...}/{\"primitive_gate\": ...}/{\"param\": ...} instead of a literal var",
        )
        return
    if name in FORBIDDEN_VARS:
        result.add_error("FORBIDDEN_VARIABLE", f"Variable '{name}' is explicitly forbidden")
        return
    for sub in FORBIDDEN_SUBSTRINGS:
        if sub in name:
            result.add_error("FORBIDDEN_SUBSTRING", f"Variable '{name}' contains forbidden substring '{sub}'")
            return
    if name not in ALLOWED_VARS:
        result.add_error("UNKNOWN_VARIABLE", f"Variable '{name}' is not in ALLOWED_VARS")


def _check_primitive_ref(
    kind: str,
    name: Any,
    params: Any,
    result: VerificationResult,
    location: str,
) -> None:
    """Validate a {"primitive": name, "params": {...}} or {"primitive_gate": ...} leaf."""
    if not isinstance(name, str):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}: {kind} name must be a string")
        return
    if not isinstance(params, dict):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}: {kind} params must be a dict")
        return
    if name in bridge.STATEFUL_PRIMITIVE_NAMES:
        result.add_error(
            "STATEFUL_PRIMITIVE_MISPLACED",
            f"{location}: stateful primitive '{name}' cannot be referenced inline; "
            "declare it via the top-level 'admission_budget' block",
        )
        return
    if not bridge.is_known_primitive(name):
        result.add_error("PRIMITIVE_UNKNOWN", f"{location}: unknown primitive '{name}'")
        return
    actual_kind = bridge.primitive_kind(name)
    expected_kinds = {"primitive": ("value", "system_value"), "primitive_gate": ("gate",)}[kind]
    if actual_kind not in expected_kinds:
        result.add_error(
            "PRIMITIVE_WRONG_SHAPE",
            f"{location}: primitive '{name}' has shape '{actual_kind}', expected one of {expected_kinds} for '{kind}' node",
        )
        return
    try:
        bridge.validate_primitive_params(name, params)
    except PrimitiveError as exc:
        result.add_error("PRIMITIVE_PARAM_INVALID", f"{location}: {exc}")


def _check_param_ref(name: Any, declared_params: frozenset, result: VerificationResult, location: str) -> None:
    if not isinstance(name, str):
        result.add_error("SCHEMA_INVALID_TYPE", f"{location}: param name must be a string")
        return
    if name not in declared_params:
        result.add_error(
            "PARAM_UNDECLARED",
            f"{location}: param '{name}' is not declared in the top-level 'parameters' list",
        )


def _count_nodes(expr: Any, *, _depth: int = 0) -> Tuple[int, int]:
    """Return (total_nodes, max_depth) for expression tree."""
    if not isinstance(expr, dict):
        return (0, _depth)

    nodes = 1
    max_d = _depth

    if "const" in expr or "var" in expr or "primitive" in expr or "primitive_gate" in expr or "param" in expr:
        return (1, _depth)

    if "op" in expr:
        op = expr.get("op", "")
        if op in ("weighted_sum", "topk_mixture"):
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


def _mixture_root_primitive_name(node: Any) -> Optional[str]:
    """If `node` (a mixture term's expression) is itself a bare {"primitive": name}
    leaf, return name; else None (arbitrary sub-expressions are not family-checked)."""
    if isinstance(node, dict) and "primitive" in node and isinstance(node["primitive"], str):
        return node["primitive"]
    return None


def _verify_expression(
    expr: Any,
    result: VerificationResult,
    limits: Dict[str, Any],
    location: str,
    *,
    _depth: int = 0,
    declared_params: frozenset = frozenset(),
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

    if "primitive" in expr:
        _check_primitive_ref("primitive", expr["primitive"], expr.get("params", {}), result, location)
        return

    if "primitive_gate" in expr:
        _check_primitive_ref("primitive_gate", expr["primitive_gate"], expr.get("params", {}), result, location)
        return

    if "param" in expr:
        _check_param_ref(expr["param"], declared_params, result, location)
        return

    if "op" not in expr:
        result.add_error(
            "SCHEMA_MISSING_FIELD",
            f"{location}: expression has neither 'const', 'var', 'primitive', 'primitive_gate', 'param', nor 'op'",
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
        _verify_expression(
            sub_expr, result, limits, f"{location}/{suffix}", _depth=_depth + 1, declared_params=declared_params
        )

    def _verify_mixture_terms(terms: Any, op_name: str) -> None:
        if not isinstance(terms, list):
            result.add_error("SCHEMA_INVALID_TYPE", f"{location}/{op_name}: terms must be a list")
            return
        if not terms:
            result.add_error("MIXTURE_EMPTY", f"{location}/{op_name}: mixture has zero terms")
            return
        if len(terms) > max_terms:
            result.add_error(
                "TERMS_EXCEEDED",
                f"{location}/{op_name}: {len(terms)} terms > max_terms {max_terms}",
            )
        root_primitives: List[str] = []
        for i, item in enumerate(terms):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                result.add_error(
                    "SCHEMA_INVALID_TYPE",
                    f"{location}/{op_name}/term[{i}]: must be [expr, weight]",
                )
                continue
            _sub(item[0], f"term[{i}]/expr")
            root_name = _mixture_root_primitive_name(item[0])
            if root_name is not None and bridge.is_known_primitive(root_name):
                root_primitives.append(root_name)
            w = item[1]
            if not isinstance(w, (int, float)):
                result.add_error(
                    "SCHEMA_INVALID_TYPE",
                    f"{location}/{op_name}/term[{i}]/weight: must be numeric",
                )
            elif not math.isfinite(w):
                result.add_error(
                    "EXPRESSION_NOT_FINITE",
                    f"{location}/{op_name}/term[{i}]/weight: non-finite {w}",
                )
            elif w < min_const or w > max_const:
                result.add_error(
                    "CONSTANT_OUT_OF_RANGE",
                    f"{location}/{op_name}/term[{i}]/weight: {w} outside [{min_const}, {max_const}]",
                )
        if len(root_primitives) >= 2 and not bridge.families_compatible(root_primitives):
            result.add_error(
                "MIXTURE_FAMILY_INCOMPATIBLE",
                f"{location}/{op_name}: primitives {root_primitives} share no compatible family",
            )

    if op == "weighted_sum":
        _verify_mixture_terms(expr.get("terms", []), "weighted_sum")
        return

    if op == "topk_mixture":
        terms = expr.get("terms", [])
        _verify_mixture_terms(terms, "topk_mixture")
        k = expr.get("k")
        n_terms = len(terms) if isinstance(terms, list) else 0
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or (n_terms and k > n_terms):
            result.add_error(
                "TOPK_INVALID_K",
                f"{location}/topk_mixture: k={k!r} invalid for {n_terms} term(s)",
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

    if op in ("bool_and", "bool_or"):
        args = expr.get("args", [])
        if not isinstance(args, list) or len(args) < 2:
            result.add_error("SCHEMA_INVALID_TYPE", f"{location}/{op}: requires at least 2 args")
            return
        for i, arg in enumerate(args):
            _sub(arg, f"args[{i}]")
        return

    if op == "bool_not":
        args = expr.get("args", [])
        if not isinstance(args, list) or len(args) != 1:
            result.add_error("SCHEMA_INVALID_TYPE", f"{location}/bool_not: requires exactly 1 arg")
            return
        _sub(args[0], "args[0]")
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


def _collect_var_names(node: Any) -> set:
    names: set = set()

    def _walk(n: Any) -> None:
        if isinstance(n, dict):
            if "var" in n and isinstance(n["var"], str):
                names.add(n["var"])
            for v in n.values():
                _walk(v)
        elif isinstance(n, (list, tuple)):
            for item in n:
                _walk(item)

    _walk(node)
    return names


def _verify_expression_finite(
    expr: Any,
    result: VerificationResult,
    location: str,
) -> None:
    """Lower {"primitive"/"primitive_gate"/"param"} leaves (dummy-valued) and
    try to evaluate expr with dummy inputs; flag if ExpressionError is raised.

    Primitive/param-shape errors are already reported by _verify_expression's
    own per-node checks, so a lowering failure here is silently skipped
    rather than raising a second, less specific EXPRESSION_NOT_FINITE error.
    """
    if not isinstance(expr, dict):
        return
    try:
        lowered = bridge.lower_expression(expr)
    except PrimitiveError:
        return
    ctx = _build_dummy_context()
    for name in _collect_var_names(lowered) - set(ctx):
        ctx[name] = 0.5
    try:
        val = evaluate_expression(lowered, ctx)
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
    *,
    declared_params: frozenset = frozenset(),
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
        _verify_expression(expr, result, limits, f"{location}/request_score", declared_params=declared_params)
        _verify_expression_finite(expr, result, f"{location}/request_score")
    if "batch_score" in block:
        expr = block["batch_score"]
        _verify_expression(expr, result, limits, f"{location}/batch_score", declared_params=declared_params)
        _verify_expression_finite(expr, result, f"{location}/batch_score")
    if "admission_condition" in block:
        expr = block["admission_condition"]
        _verify_expression(expr, result, limits, f"{location}/admission_condition", declared_params=declared_params)
        _verify_expression_finite(expr, result, f"{location}/admission_condition")


def _verify_parameters_block(
    heuristic: Dict[str, Any],
    result: VerificationResult,
    limits: Dict[str, Any],
) -> frozenset:
    """Validate the optional top-level "parameters" declaration list.

    Returns the frozenset of successfully declared parameter names (used to
    validate {"param": name} references elsewhere in the document)."""
    if "parameters" not in heuristic:
        return frozenset()
    params = heuristic["parameters"]
    if not isinstance(params, list):
        result.add_error("SCHEMA_INVALID_TYPE", "'parameters' must be a list")
        return frozenset()
    max_parameters = limits.get("max_parameters", DEFAULT_LIMITS["max_parameters"])
    if len(params) > max_parameters:
        result.add_error(
            "PARAM_SCHEMA_INVALID",
            f"'parameters' has {len(params)} entries > max_parameters {max_parameters}",
        )
    declared: List[str] = []
    for i, p in enumerate(params):
        loc = f"parameters[{i}]"
        if not isinstance(p, dict):
            result.add_error("SCHEMA_INVALID_TYPE", f"{loc}: must be a dict")
            continue
        missing = [f for f in REQUIRED_PARAMETER_FIELDS if f not in p]
        if missing:
            result.add_error("PARAM_SCHEMA_INVALID", f"{loc}: missing field(s) {missing}")
            continue
        name = p["name"]
        if not isinstance(name, str) or not name.strip():
            result.add_error("PARAM_SCHEMA_INVALID", f"{loc}: 'name' must be a non-empty string")
            continue
        if name in declared:
            result.add_error("PARAM_DUPLICATE_NAME", f"{loc}: duplicate parameter name '{name}'")
            continue
        if p["type"] not in ALLOWED_PARAMETER_TYPES:
            result.add_error("PARAM_SCHEMA_INVALID", f"{loc}: type '{p['type']}' not in {sorted(ALLOWED_PARAMETER_TYPES)}")
            continue
        lo, hi, default = p["min"], p["max"], p["default"]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (lo, hi, default)):
            result.add_error("PARAM_SCHEMA_INVALID", f"{loc}: min/max/default must be finite numbers")
            continue
        if lo > hi:
            result.add_error("PARAM_SCHEMA_INVALID", f"{loc}: min {lo} > max {hi}")
            continue
        if default < lo or default > hi:
            result.add_error("PARAM_SCHEMA_INVALID", f"{loc}: default {default} outside [{lo}, {hi}]")
            continue
        declared.append(name)
    return frozenset(declared)


def _verify_fallback_block(heuristic: Dict[str, Any], result: VerificationResult) -> None:
    if "fallback" not in heuristic:
        return
    fb = heuristic["fallback"]
    if not isinstance(fb, dict) or "policy" not in fb:
        result.add_error("FALLBACK_INVALID", "'fallback' must be a dict with a 'policy' field")
        return
    if fb["policy"] not in ALLOWED_FALLBACK_POLICIES:
        result.add_error(
            "FALLBACK_INVALID",
            f"'fallback.policy' {fb['policy']!r} not in {sorted(ALLOWED_FALLBACK_POLICIES)}",
        )


def _admission_condition_uses_primitive_gate(heuristic: Dict[str, Any]) -> bool:
    """True iff any admission_condition block references a {"primitive_gate": ...}
    (the new CC3 "admission gate" construct). Plain pre-CC3 admission_condition
    expressions (arbitrary var/op composition, no primitive_gate) are legacy
    and do not require "on_no_admits" -- this keeps every existing genome- and
    hand-authored heuristic backward compatible."""
    blocks = []
    if isinstance(heuristic.get("default"), dict) and "admission_condition" in heuristic["default"]:
        blocks.append(heuristic["default"]["admission_condition"])
    for regime in heuristic.get("regimes", []) or []:
        if isinstance(regime, dict) and "admission_condition" in regime:
            blocks.append(regime["admission_condition"])
    for expr in blocks:
        refs = bridge.collect_primitive_refs(expr)
        if any(kind == "primitive_gate" for kind, _, _ in refs):
            return True
    return False


def _verify_on_no_admits(heuristic: Dict[str, Any], result: VerificationResult) -> None:
    uses_gate = _admission_condition_uses_primitive_gate(heuristic)
    if "on_no_admits" not in heuristic:
        if uses_gate:
            result.add_error(
                "ON_NO_ADMITS_MISSING",
                "'admission_condition' uses a primitive_gate but top-level 'on_no_admits' is not declared",
            )
        return
    if heuristic["on_no_admits"] not in ALLOWED_ON_NO_ADMITS_MODES:
        result.add_error(
            "ON_NO_ADMITS_INVALID",
            f"'on_no_admits' {heuristic['on_no_admits']!r} not in {sorted(ALLOWED_ON_NO_ADMITS_MODES)}",
        )


def _verify_placement_block(heuristic: Dict[str, Any], result: VerificationResult, limits: Dict[str, Any]) -> None:
    if "placement" not in heuristic:
        return
    placement = heuristic["placement"]
    if not isinstance(placement, dict) or "keys" not in placement:
        result.add_error("SCHEMA_INVALID_TYPE", "'placement' must be a dict with a 'keys' list")
        return
    keys = placement["keys"]
    if not isinstance(keys, list):
        result.add_error("SCHEMA_INVALID_TYPE", "'placement.keys' must be a list")
        return
    if not keys:
        result.add_error("PLACEMENT_EMPTY", "'placement.keys' must have at least one entry")
        return
    max_keys = limits.get("max_placement_keys", DEFAULT_LIMITS["max_placement_keys"])
    if len(keys) > max_keys:
        result.add_error(
            "PLACEMENT_TOO_MANY_KEYS",
            f"'placement.keys' has {len(keys)} entries > max_placement_keys {max_keys}",
        )
    for i, k in enumerate(keys):
        loc = f"placement.keys[{i}]"
        if not isinstance(k, dict) or "name" not in k:
            result.add_error("SCHEMA_INVALID_TYPE", f"{loc}: must be a dict with a 'name' field")
            continue
        name = k["name"]
        params = k.get("params", {}) or {}
        if name not in bridge.PLACEMENT_PRIMITIVE_NAMES:
            result.add_error(
                "PLACEMENT_KEY_UNKNOWN",
                f"{loc}: '{name}' is not a known PLACEMENT primitive "
                f"({sorted(bridge.PLACEMENT_PRIMITIVE_NAMES)})",
            )
            continue
        if not isinstance(params, dict):
            result.add_error("SCHEMA_INVALID_TYPE", f"{loc}.params: must be a dict")
            continue
        try:
            bridge.validate_primitive_params(name, params)
        except PrimitiveError as exc:
            result.add_error("PRIMITIVE_PARAM_INVALID", f"{loc}: {exc}")


def _verify_admission_budget_block(heuristic: Dict[str, Any], result: VerificationResult) -> None:
    if "admission_budget" not in heuristic:
        return
    budget = heuristic["admission_budget"]
    if not isinstance(budget, dict) or "primitive" not in budget:
        result.add_error("ADMISSION_BUDGET_INVALID", "'admission_budget' must be a dict with a 'primitive' field")
        return
    name = budget["primitive"]
    if name != "admission_credit_budget":
        result.add_error(
            "ADMISSION_BUDGET_INVALID",
            f"'admission_budget.primitive' must be 'admission_credit_budget', got {name!r}",
        )
        return
    params = budget.get("params", {}) or {}
    if not isinstance(params, dict):
        result.add_error("SCHEMA_INVALID_TYPE", "'admission_budget.params' must be a dict")
        return
    try:
        bridge.validate_primitive_params(name, params)
    except PrimitiveError as exc:
        result.add_error("PRIMITIVE_PARAM_INVALID", f"admission_budget: {exc}")


def _verify_primitive_budget(heuristic: Dict[str, Any], result: VerificationResult, limits: Dict[str, Any]) -> None:
    refs: List[Tuple[str, str, Dict[str, float]]] = []
    for block in bridge.iter_expression_blocks(heuristic):
        refs.extend(bridge.collect_primitive_refs(block))
    distinct = {(kind, name, tuple(sorted(params.items()))) for kind, name, params in refs}
    max_active = limits.get("max_active_primitives", DEFAULT_LIMITS["max_active_primitives"])
    if len(distinct) > max_active:
        result.add_error(
            "PRIMITIVE_BUDGET_EXCEEDED",
            f"{len(distinct)} distinct primitive reference(s) > max_active_primitives {max_active}",
        )


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

    # CC3: declared external parameters (parsed first so {"param": ...} refs
    # in default/regimes can be validated against the declared name set)
    declared_params = _verify_parameters_block(heuristic, result, limits)

    # default rule block
    if "default" in heuristic:
        _verify_rule_block(heuristic["default"], result, limits, "default", declared_params=declared_params)

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
                    _verify_expression(regime["condition"], result, limits, f"{loc}/condition", declared_params=declared_params)
                    _verify_expression_finite(regime["condition"], result, f"{loc}/condition")
                if "request_score" not in regime:
                    result.add_error("REGIME_MISSING_SCORE", f"{loc}: missing 'request_score'")
                else:
                    _verify_rule_block(regime, result, limits, loc, declared_params=declared_params)

    # CC3: fallback / on_no_admits / placement / admission_budget / primitive budget
    _verify_fallback_block(heuristic, result)
    _verify_on_no_admits(heuristic, result)
    _verify_placement_block(heuristic, result, limits)
    _verify_admission_budget_block(heuristic, result)
    _verify_primitive_budget(heuristic, result, limits)

    # optional description / metadata fields (no validation needed, just type check)
    if "description" in heuristic and not isinstance(heuristic["description"], str):
        result.add_warning("'description' should be a string")

    return result

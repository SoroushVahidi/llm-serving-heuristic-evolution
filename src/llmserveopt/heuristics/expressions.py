"""
Safe expression evaluator for the scheduling heuristic DSL.

Expression nodes are plain Python dicts (parsed from JSON).  No eval, no exec,
no imports.  All intermediate values must be finite; non-finite results raise
ExpressionError so the verifier can detect them during static analysis.

Expression node shapes
----------------------
{"const": <number>}
{"var": "<namespace>.<name>"}
{"op": "add",          "args": [e1, e2]}
{"op": "sub",          "args": [e1, e2]}
{"op": "mul",          "args": [e1, e2]}
{"op": "div_safe",     "args": [e1, e2]}          # e1 / max(|e2|, eps)
{"op": "min",          "args": [e1, e2]}
{"op": "max",          "args": [e1, e2]}
{"op": "neg",          "args": [e1]}
{"op": "abs",          "args": [e1]}
{"op": "clip",         "args": [e1, lo, hi]}       # len==3
{"op": "sqrt_safe",    "args": [e1]}               # sqrt(max(0, e1))
{"op": "log1p_safe",   "args": [e1]}               # log1p(max(0, e1))
{"op": "weighted_sum", "terms": [[e1, w1], ...]}   # sum(evaluate(e) * w)
{"op": "if_then_else", "cond": e1, "then": e2, "else": e3}  # e1>0 → e2 else e3
"""
from __future__ import annotations

import math
from typing import Any, Dict

# Type alias — an expression is any JSON-serializable dict
Expression = Dict[str, Any]

_EPS = 1e-9


class ExpressionError(Exception):
    """Raised when an expression cannot be evaluated safely."""


def evaluate_expression(
    expr: Expression,
    ctx: Dict[str, float],
    *,
    _depth: int = 0,
    max_depth: int = 64,
) -> float:
    """Recursively evaluate `expr` against variable context `ctx`.

    Parameters
    ----------
    expr : dict — expression node.
    ctx  : dict — variable name → float value.
    _depth : internal recursion counter.
    max_depth : hard recursion limit (default 64, independent of DSL limit).

    Returns
    -------
    float — the evaluated value.

    Raises
    ------
    ExpressionError on invalid structure, unknown variables, or non-finite result.
    """
    if _depth > max_depth:
        raise ExpressionError(f"Expression depth exceeds {max_depth}")

    if not isinstance(expr, dict):
        raise ExpressionError(f"Expression must be a dict, got {type(expr).__name__}")

    def _eval(e: Expression) -> float:
        return evaluate_expression(e, ctx, _depth=_depth + 1, max_depth=max_depth)

    def _check_finite(v: float, label: str = "result") -> float:
        if not math.isfinite(v):
            raise ExpressionError(f"Non-finite {label}: {v}")
        return v

    # --- const node ---
    if "const" in expr:
        v = expr["const"]
        if not isinstance(v, (int, float)):
            raise ExpressionError(f"const must be numeric, got {type(v).__name__}")
        return _check_finite(float(v), "const")

    # --- var node ---
    if "var" in expr:
        name = expr["var"]
        if not isinstance(name, str):
            raise ExpressionError("var name must be a string")
        if name not in ctx:
            raise ExpressionError(f"Unknown variable in context: '{name}'")
        return _check_finite(float(ctx[name]), f"var({name})")

    # --- op node ---
    if "op" not in expr:
        raise ExpressionError(f"Expression node has neither 'const', 'var', nor 'op': {list(expr.keys())}")

    op = expr["op"]
    if not isinstance(op, str):
        raise ExpressionError("op must be a string")

    if op == "weighted_sum":
        terms = expr.get("terms", [])
        if not isinstance(terms, list):
            raise ExpressionError("weighted_sum.terms must be a list")
        total = 0.0
        for item in terms:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ExpressionError("Each weighted_sum term must be [expr, weight]")
            e_val = _eval(item[0])
            w = item[1]
            if not isinstance(w, (int, float)):
                raise ExpressionError(f"weighted_sum weight must be numeric, got {type(w).__name__}")
            w = float(w)
            if not math.isfinite(w):
                raise ExpressionError(f"weighted_sum weight is non-finite: {w}")
            total += e_val * w
        return _check_finite(total, "weighted_sum")

    if op == "if_then_else":
        cond = _eval(expr["cond"])
        return _eval(expr["then"]) if cond > 0.0 else _eval(expr["else"])

    args = expr.get("args", [])
    if not isinstance(args, list):
        raise ExpressionError("op.args must be a list")

    if op == "add":
        if len(args) != 2:
            raise ExpressionError("add requires 2 args")
        return _check_finite(_eval(args[0]) + _eval(args[1]), "add")

    if op == "sub":
        if len(args) != 2:
            raise ExpressionError("sub requires 2 args")
        return _check_finite(_eval(args[0]) - _eval(args[1]), "sub")

    if op == "mul":
        if len(args) != 2:
            raise ExpressionError("mul requires 2 args")
        return _check_finite(_eval(args[0]) * _eval(args[1]), "mul")

    if op == "div_safe":
        if len(args) != 2:
            raise ExpressionError("div_safe requires 2 args")
        num = _eval(args[0])
        den = _eval(args[1])
        return _check_finite(num / (den if abs(den) >= _EPS else math.copysign(_EPS, den) or _EPS), "div_safe")

    if op == "min":
        if len(args) != 2:
            raise ExpressionError("min requires 2 args")
        return _check_finite(min(_eval(args[0]), _eval(args[1])), "min")

    if op == "max":
        if len(args) != 2:
            raise ExpressionError("max requires 2 args")
        return _check_finite(max(_eval(args[0]), _eval(args[1])), "max")

    if op == "neg":
        if len(args) != 1:
            raise ExpressionError("neg requires 1 arg")
        return _check_finite(-_eval(args[0]), "neg")

    if op == "abs":
        if len(args) != 1:
            raise ExpressionError("abs requires 1 arg")
        return _check_finite(abs(_eval(args[0])), "abs")

    if op == "clip":
        if len(args) != 3:
            raise ExpressionError("clip requires 3 args: [value, lo, hi]")
        v = _eval(args[0])
        lo = _eval(args[1])
        hi = _eval(args[2])
        return _check_finite(max(lo, min(hi, v)), "clip")

    if op == "sqrt_safe":
        if len(args) != 1:
            raise ExpressionError("sqrt_safe requires 1 arg")
        return _check_finite(math.sqrt(max(0.0, _eval(args[0]))), "sqrt_safe")

    if op == "log1p_safe":
        if len(args) != 1:
            raise ExpressionError("log1p_safe requires 1 arg")
        return _check_finite(math.log1p(max(0.0, _eval(args[0]))), "log1p_safe")

    raise ExpressionError(f"Unknown operation: '{op}'")

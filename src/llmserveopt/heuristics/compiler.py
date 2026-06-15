"""
Compiler for verified scheduling heuristic DSL documents.

compile_heuristic(heuristic_dict) -> CompiledHeuristic

A CompiledHeuristic is a lightweight callable wrapper around a verified
heuristic JSON object.  It exposes two main operations:

  score_request(req, sys_vars, batch_vars) -> float
      Evaluate the active regime's request_score expression.

  select_regime(sys_vars, batch_vars) -> str
      Return the name of the matched regime (or "default").

All variable binding and expression evaluation happen through the safe
evaluate_expression() function — no exec, no eval, no imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .dsl_schema import ALLOWED_TIE_BREAKERS, DEFAULT_LIMITS
from .expressions import ExpressionError, evaluate_expression
from .verifier import VerificationResult, verify_heuristic


class CompilationError(Exception):
    """Raised when a heuristic fails verification or cannot be compiled."""


@dataclass
class CompiledHeuristic:
    """Compiled, ready-to-use scheduling heuristic.

    Attributes
    ----------
    name : str — heuristic identifier.
    tie_breaker : str — one of ALLOWED_TIE_BREAKERS.
    description : str — optional human-readable description.
    raw : dict — original heuristic JSON.
    regimes : list — ordered list of (condition_expr, rule_block) tuples.
    default_rule : dict — fallback rule block.
    limits : dict — effective limits used during compilation.
    """

    name: str
    tie_breaker: str
    description: str
    raw: Dict[str, Any]
    regimes: List[Tuple[Any, Dict[str, Any]]]   # [(condition_expr, rule_block), ...]
    default_rule: Dict[str, Any]
    limits: Dict[str, Any]

    def _make_context(
        self,
        req_vars: Optional[Dict[str, float]] = None,
        sys_vars: Optional[Dict[str, float]] = None,
        batch_vars: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        ctx: Dict[str, float] = {}
        if req_vars:
            ctx.update(req_vars)
        if sys_vars:
            ctx.update(sys_vars)
        if batch_vars:
            ctx.update(batch_vars)
        return ctx

    def active_rule(
        self,
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
    ) -> Tuple[str, Dict[str, Any]]:
        """Return (regime_name, rule_block) for the first matching regime.

        Falls back to ("default", default_rule) if no regime matches.
        """
        ctx = self._make_context(sys_vars=sys_vars, batch_vars=batch_vars)
        for i, (cond_expr, rule_block) in enumerate(self.regimes):
            try:
                val = evaluate_expression(cond_expr, ctx)
                if val > 0.0:
                    return (f"regime_{i}", rule_block)
            except ExpressionError:
                # Non-fatal: fall through to next regime
                continue
        return ("default", self.default_rule)

    def score_request(
        self,
        req_vars: Dict[str, float],
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
    ) -> float:
        """Evaluate the active request_score expression for one request.

        Returns 0.0 on any evaluation error so the system degrades gracefully.
        """
        _, rule = self.active_rule(sys_vars, batch_vars)
        score_expr = rule.get("request_score")
        if score_expr is None:
            return 0.0
        ctx = self._make_context(req_vars=req_vars, sys_vars=sys_vars, batch_vars=batch_vars)
        try:
            return float(evaluate_expression(score_expr, ctx))
        except (ExpressionError, Exception):
            return 0.0

    def score_batch(
        self,
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
    ) -> float:
        """Evaluate the active batch_score expression (if defined).

        Returns 0.0 when the active rule has no batch_score.
        """
        _, rule = self.active_rule(sys_vars, batch_vars)
        score_expr = rule.get("batch_score")
        if score_expr is None:
            return 0.0
        ctx = self._make_context(sys_vars=sys_vars, batch_vars=batch_vars)
        try:
            return float(evaluate_expression(score_expr, ctx))
        except (ExpressionError, Exception):
            return 0.0

    def check_admission(
        self,
        req_vars: Dict[str, float],
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
    ) -> bool:
        """Evaluate the active admission_condition (if any).

        Returns True when no condition is defined (admit by default).
        Returns True when condition evaluates > 0, else False.
        """
        _, rule = self.active_rule(sys_vars, batch_vars)
        cond_expr = rule.get("admission_condition")
        if cond_expr is None:
            return True
        ctx = self._make_context(req_vars=req_vars, sys_vars=sys_vars, batch_vars=batch_vars)
        try:
            return evaluate_expression(cond_expr, ctx) > 0.0
        except (ExpressionError, Exception):
            return True   # fail-open: admit by default


def compile_heuristic(
    heuristic: Any,
    *,
    extra_limits: Optional[Dict[str, Any]] = None,
) -> CompiledHeuristic:
    """Verify and compile a heuristic DSL document.

    Parameters
    ----------
    heuristic : dict — parsed JSON heuristic document.
    extra_limits : optional dict overriding DEFAULT_LIMITS entries.

    Returns
    -------
    CompiledHeuristic — ready for use in HeuristicPolicy.

    Raises
    ------
    CompilationError — if verification fails (includes all error messages).
    """
    vr: VerificationResult = verify_heuristic(heuristic, extra_limits=extra_limits)
    if not vr.valid:
        lines = [f"  [{code}] {msg}" for code, msg in vr.errors]
        raise CompilationError(
            f"Heuristic '{heuristic.get('name', '<unnamed>')}' failed verification "
            f"with {len(vr.errors)} error(s):\n" + "\n".join(lines)
        )

    limits = {**DEFAULT_LIMITS, **(extra_limits or {})}
    name = heuristic.get("name", "unnamed")
    tie_breaker = heuristic.get("tie_breaker", "arrival_order")
    description = heuristic.get("description", "")
    default_rule = heuristic.get("default", {})

    regimes: List[Tuple[Any, Dict[str, Any]]] = []
    for regime in heuristic.get("regimes", []):
        cond_expr = regime.get("condition")
        regimes.append((cond_expr, regime))

    return CompiledHeuristic(
        name=name,
        tie_breaker=tie_breaker,
        description=description,
        raw=heuristic,
        regimes=regimes,
        default_rule=default_rule,
        limits=limits,
    )

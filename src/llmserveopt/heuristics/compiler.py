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

CC3 additions
-------------
* {"primitive"/"primitive_gate"/"param"} leaf nodes in the verified,
  as-authored heuristic are lowered (via primitive_bridge.lower_expression)
  into ordinary {"var": "<reserved name>"} nodes before evaluation, so
  expressions.py itself never needs to know about primitives.py.
* An optional "fallback" declares (or, if absent, inherits) a reference to
  one of a small set of already-verified canonical safe policies
  (ALLOWED_FALLBACK_POLICIES). Any expression-evaluation failure during
  scoring/admission now explicitly delegates to the compiled fallback's same
  method instead of silently returning an ad hoc literal.
* Optional top-level "parameters", "placement", and "admission_budget"
  blocks are parsed into CompiledHeuristic fields; HeuristicPolicy (policy.py)
  is responsible for actually resolving/instantiating them against real
  ObservableRequest/ObservableState/ObservableGPUState objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import primitive_bridge as bridge
from .dsl_schema import (
    ALLOWED_FALLBACK_POLICIES,
    COMPILER_VERSION,
    DEFAULT_FALLBACK_POLICY,
    DEFAULT_LIMITS,
    DSL_SCHEMA_VERSION,
)
from .expressions import ExpressionError, evaluate_expression
from .verifier import VerificationResult, verify_heuristic


class CompilationError(Exception):
    """Raised when a heuristic fails verification or cannot be compiled."""


@dataclass(frozen=True)
class ParamDecl:
    """A declared, externally-supplied bounded DSL parameter (distinct from
    a primitive's own ParamBound -- see primitive_bridge.py)."""

    minimum: float
    maximum: float
    default: float

    def validate(self, value: float, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompilationError(f"parameter '{name}' override must be numeric, got {value!r}")
        if value < self.minimum or value > self.maximum:
            raise CompilationError(
                f"parameter '{name}' override {value!r} outside bounds [{self.minimum}, {self.maximum}]"
            )
        return float(value)


# Canonical fallback policies are compiled and cached once, lazily, to avoid
# recompiling fifo_like/edf_like on every compile_heuristic() call.
_FALLBACK_CACHE: Dict[str, "CompiledHeuristic"] = {}


def _get_fallback_compiled(name: str) -> "CompiledHeuristic":
    if name not in _FALLBACK_CACHE:
        from . import examples as _examples

        builder = {"fifo_like": _examples.fifo_like, "edf_like": _examples.edf_like}[name]
        _FALLBACK_CACHE[name] = compile_heuristic(builder())
    return _FALLBACK_CACHE[name]


@dataclass
class CompiledHeuristic:
    """Compiled, ready-to-use scheduling heuristic.

    Attributes
    ----------
    name : str — heuristic identifier.
    tie_breaker : str — one of ALLOWED_TIE_BREAKERS.
    description : str — optional human-readable description.
    raw : dict — original heuristic JSON (as-authored, pre-lowering).
    regimes : list — ordered list of (lowered_condition_expr, lowered_rule_block) tuples.
    default_rule : dict — lowered fallback rule block.
    limits : dict — effective limits used during compilation.
    param_declarations : dict — name -> ParamDecl for externally-supplied parameters.
    resolved_params : dict — name -> float, declared defaults possibly overridden at compile time.
    fallback_name : Optional[str] — which ALLOWED_FALLBACK_POLICIES entry this inherits/declares.
    fallback : Optional[CompiledHeuristic] — the compiled fallback policy (None only for the
        two canonical fallback policies themselves, which are their own terminal case).
    on_no_admits : Optional[str] — declared behavior when admission_condition rejects everyone.
    placement_keys : list — [(primitive_name, bound_params), ...], empty if "placement" absent.
    admission_budget_spec : Optional[Tuple[str, dict]] — (primitive_name, bound_params) for the
        one stateful primitive, or None if "admission_budget" absent.
    primitive_refs : list — distinct (kind, name, bound_params) primitive references collected
        from the raw (pre-lowering) document, for HeuristicPolicy to resolve per request/step.
    dsl_version, compiler_version : instrumentation metadata.
    """

    name: str
    tie_breaker: str
    description: str
    raw: Dict[str, Any]
    regimes: List[Tuple[Any, Dict[str, Any]]]
    default_rule: Dict[str, Any]
    limits: Dict[str, Any]
    param_declarations: Dict[str, ParamDecl] = field(default_factory=dict)
    resolved_params: Dict[str, float] = field(default_factory=dict)
    fallback_name: Optional[str] = None
    fallback: Optional["CompiledHeuristic"] = None
    on_no_admits: Optional[str] = None
    placement_keys: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)
    admission_budget_spec: Optional[Tuple[str, Dict[str, float]]] = None
    primitive_refs: List[Tuple[str, str, Dict[str, float]]] = field(default_factory=list)
    dsl_version: int = DSL_SCHEMA_VERSION
    compiler_version: str = COMPILER_VERSION

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

    def _with_fallback(self, method_name: str, trace: Optional[Dict[str, Any]], default_value: Any, *call_args):
        """Shared failure path for score_request/score_batch/check_admission:
        delegate to the compiled fallback on any evaluation failure instead
        of silently returning `default_value`. `default_value` is used only
        for the two terminal canonical-fallback policies themselves (which
        have no further fallback to delegate to)."""
        if self.fallback is not None:
            if trace is not None:
                trace["fallback_activated"] = True
            return getattr(self.fallback, method_name)(*call_args)
        if trace is not None:
            trace["fallback_activated"] = True
        return default_value

    def score_request(
        self,
        req_vars: Dict[str, float],
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
        *,
        trace: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate the active request_score expression for one request.

        On evaluation failure, delegates to the declared/inherited fallback
        policy instead of silently returning 0.0 (see class docstring).
        """
        regime_name, rule = self.active_rule(sys_vars, batch_vars)
        if trace is not None:
            trace["active_regime"] = regime_name
            trace["dsl_version"] = self.dsl_version
            trace["compiler_version"] = self.compiler_version
            trace.setdefault("fallback_activated", False)
            trace["active_primitives"] = [bridge.canonical_ref_key(n, p) for _, n, p in self.primitive_refs]
        score_expr = rule.get("request_score")
        if score_expr is None:
            return 0.0
        ctx = self._make_context(req_vars=req_vars, sys_vars=sys_vars, batch_vars=batch_vars)
        try:
            return float(evaluate_expression(score_expr, ctx))
        except ExpressionError:
            return self._with_fallback("score_request", trace, 0.0, req_vars, sys_vars, batch_vars)

    def score_batch(
        self,
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
        *,
        trace: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate the active batch_score expression (if defined).

        Returns 0.0 when the active rule has no batch_score (not a failure).
        """
        _, rule = self.active_rule(sys_vars, batch_vars)
        score_expr = rule.get("batch_score")
        if score_expr is None:
            return 0.0
        ctx = self._make_context(sys_vars=sys_vars, batch_vars=batch_vars)
        try:
            return float(evaluate_expression(score_expr, ctx))
        except ExpressionError:
            return self._with_fallback("score_batch", trace, 0.0, sys_vars, batch_vars)

    def check_admission(
        self,
        req_vars: Dict[str, float],
        sys_vars: Dict[str, float],
        batch_vars: Dict[str, float],
        *,
        trace: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluate the active admission_condition (if any).

        Returns True when no condition is defined (admit by default). On
        evaluation failure, delegates to the fallback policy's own
        check_admission (which is True for both canonical fallback policies,
        since neither declares an admission_condition) instead of an ad hoc
        literal.
        """
        _, rule = self.active_rule(sys_vars, batch_vars)
        cond_expr = rule.get("admission_condition")
        if cond_expr is None:
            return True
        ctx = self._make_context(req_vars=req_vars, sys_vars=sys_vars, batch_vars=batch_vars)
        try:
            return evaluate_expression(cond_expr, ctx) > 0.0
        except ExpressionError:
            return self._with_fallback("check_admission", trace, True, req_vars, sys_vars, batch_vars)


def _dedup_primitive_refs(refs: List[Tuple[str, str, Dict[str, float]]]) -> List[Tuple[str, str, Dict[str, float]]]:
    seen: Dict[str, Tuple[str, str, Dict[str, float]]] = {}
    for kind, name, params in refs:
        bound = bridge.validate_primitive_params(name, params)
        key = bridge.canonical_ref_key(name, bound)
        seen[key] = (kind, name, bound)
    return list(seen.values())


def compile_heuristic(
    heuristic: Any,
    *,
    extra_limits: Optional[Dict[str, Any]] = None,
    param_overrides: Optional[Dict[str, float]] = None,
) -> CompiledHeuristic:
    """Verify and compile a heuristic DSL document.

    Parameters
    ----------
    heuristic : dict — parsed JSON heuristic document.
    extra_limits : optional dict overriding DEFAULT_LIMITS entries.
    param_overrides : optional dict overriding declared "parameters" defaults.
        Unknown keys (not in the document's "parameters" declaration) are
        rejected -- no undeclared parameter is ever accepted.

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

    default_rule_raw = heuristic.get("default", {})
    default_rule = bridge.lower_expression(default_rule_raw) if default_rule_raw else {}

    regimes: List[Tuple[Any, Dict[str, Any]]] = []
    for regime in heuristic.get("regimes", []):
        lowered_regime = bridge.lower_expression(regime)
        cond_expr = lowered_regime.get("condition")
        regimes.append((cond_expr, lowered_regime))

    # --- parameters ---
    param_declarations: Dict[str, ParamDecl] = {}
    for p in heuristic.get("parameters", []) or []:
        param_declarations[p["name"]] = ParamDecl(
            minimum=float(p["min"]), maximum=float(p["max"]), default=float(p["default"])
        )
    resolved_params: Dict[str, float] = {n: d.default for n, d in param_declarations.items()}
    if param_overrides:
        unknown = set(param_overrides) - set(param_declarations)
        if unknown:
            raise CompilationError(
                f"Heuristic '{name}' received unsupported parameter override(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(param_declarations)}"
            )
        for pname, value in param_overrides.items():
            resolved_params[pname] = param_declarations[pname].validate(value, name=pname)

    # --- fallback (canonical fallback policies are their own terminal case) ---
    if name in ALLOWED_FALLBACK_POLICIES:
        fallback_name: Optional[str] = None
        fallback: Optional[CompiledHeuristic] = None
    else:
        fallback_block = heuristic.get("fallback") or {}
        fallback_name = fallback_block.get("policy", DEFAULT_FALLBACK_POLICY)
        fallback = _get_fallback_compiled(fallback_name)

    on_no_admits = heuristic.get("on_no_admits")

    # --- placement ---
    placement_keys: List[Tuple[str, Dict[str, float]]] = []
    placement_raw = heuristic.get("placement")
    if placement_raw:
        for k in placement_raw.get("keys", []):
            bound = bridge.validate_primitive_params(k["name"], k.get("params", {}) or {})
            placement_keys.append((k["name"], bound))

    # --- admission_budget ---
    admission_budget_spec: Optional[Tuple[str, Dict[str, float]]] = None
    ab_raw = heuristic.get("admission_budget")
    if ab_raw:
        bound = bridge.validate_primitive_params(ab_raw["primitive"], ab_raw.get("params", {}) or {})
        admission_budget_spec = (ab_raw["primitive"], bound)

    # --- distinct primitive refs (from the raw, pre-lowering expression trees only) ---
    raw_refs: List[Tuple[str, str, Dict[str, float]]] = []
    for expr_block in bridge.iter_expression_blocks(heuristic):
        raw_refs.extend(bridge.collect_primitive_refs(expr_block))
    primitive_refs = _dedup_primitive_refs(raw_refs)

    return CompiledHeuristic(
        name=name,
        tie_breaker=tie_breaker,
        description=description,
        raw=heuristic,
        regimes=regimes,
        default_rule=default_rule,
        limits=limits,
        param_declarations=param_declarations,
        resolved_params=resolved_params,
        fallback_name=fallback_name,
        fallback=fallback,
        on_no_admits=on_no_admits,
        placement_keys=placement_keys,
        admission_budget_spec=admission_budget_spec,
        primitive_refs=primitive_refs,
    )

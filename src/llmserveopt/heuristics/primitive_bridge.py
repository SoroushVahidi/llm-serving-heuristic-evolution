"""Read-only DSL-facing adapter over the CC2 canonical primitive registry.

This module is the "separate compiler-facing adapter module" anticipated by
docs/architecture/contextual_composition_primitives.md section 9. It imports
``policies/primitives.py`` read-only and never modifies it; CC3 owns
everything in this file, CC2 owns everything it imports from.

Three primitive "shapes" are bridged into the DSL:

* RANKING-family and ADMISSION-value primitives (``RankingPrimitive``)
  resolve to a float via ``.value(req, state, **params)``.
* ADMISSION gates (``AdmissionGate``) resolve to a bool via
  ``.passes(req, state, **params)``, exposed to the DSL as 1.0/0.0.
* PLACEMENT keys (``PlacementKeyPrimitive``) resolve to a tuple via
  ``.key(gpu, req, **params)`` and are only ever used through the top-level
  ``"placement"`` block, never inline in a request_score/admission_condition
  expression.

The one stateful primitive (``admission_credit_budget``) is deliberately not
resolved here as a value/gate/key -- it is only ever instantiated by
``HeuristicPolicy`` from the top-level ``"admission_budget"`` declaration.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..core.types import ObservableGPUState, ObservableRequest, ObservableState
from ..policies import primitives as prim
from ..policies.primitives import PrimitiveError, PrimitiveFamily

# ---------------------------------------------------------------------------
# Explicit per-shape registries (deliberately not built by reflecting over
# primitives.py module globals -- an explicit mapping keeps this adapter
# resilient to unrelated additions/renames inside primitives.py).
# ---------------------------------------------------------------------------

_VALUE_PRIMITIVES: Dict[str, Any] = {
    "deadline_urgency": prim.DEADLINE_URGENCY,
    "laxity": prim.LAXITY,
    "prompt_length": prim.PROMPT_LENGTH,
    "predicted_output_length": prim.PREDICTED_OUTPUT_LENGTH,
    "estimated_service_time": prim.ESTIMATED_SERVICE_TIME,
    "priority": prim.PRIORITY,
    "queue_age": prim.QUEUE_AGE,
    "fairness_starvation_bonus": prim.FAIRNESS_STARVATION_BONUS,
    "laxity_urgency": prim.LAXITY_URGENCY,
    "weighted_shortest_processing_score": prim.WEIGHTED_SHORTEST_PROCESSING_SCORE,
    "request_id_tiebreak": prim.REQUEST_ID_TIEBREAK,
    "admission_risk": prim.ADMISSION_RISK,
}

# System-level value primitives are plain state->float functions rather than
# RankingPrimitive instances; bridged the same way but ignoring `req`.
_SYSTEM_VALUE_PRIMITIVES: Dict[str, Any] = {
    "system_kv_pressure": prim.system_kv_pressure,
    "decode_pressure": prim.decode_pressure,
    "prefill_pressure": prim.prefill_pressure,
    "queue_pressure": prim.queue_pressure,
}

_GATE_PRIMITIVES: Dict[str, Any] = {
    "laxity_gate": prim.LAXITY_GATE,
    "ttft_slack_gate": prim.TTFT_SLACK_GATE,
}

_PLACEMENT_KEY_PRIMITIVES: Dict[str, Any] = {
    "projected_gpu_load": prim.PROJECTED_GPU_LOAD,
    "kv_pressure": prim.KV_PRESSURE_PLACEMENT,
    "tightest_kv_fit": prim.TIGHTEST_KV_FIT,
    "least_loaded": prim.LEAST_LOADED,
}

# The one stateful primitive; referencing it inline via {"primitive": ...}
# is rejected by the verifier (STATEFUL_PRIMITIVE_MISPLACED).
STATEFUL_PRIMITIVE_NAMES: frozenset[str] = frozenset({"admission_credit_budget"})

VALUE_PRIMITIVE_NAMES: frozenset[str] = frozenset(_VALUE_PRIMITIVES) | frozenset(_SYSTEM_VALUE_PRIMITIVES)
GATE_PRIMITIVE_NAMES: frozenset[str] = frozenset(_GATE_PRIMITIVES)
PLACEMENT_PRIMITIVE_NAMES: frozenset[str] = frozenset(_PLACEMENT_KEY_PRIMITIVES)

ALL_BRIDGED_PRIMITIVE_NAMES: frozenset[str] = (
    VALUE_PRIMITIVE_NAMES | GATE_PRIMITIVE_NAMES | PLACEMENT_PRIMITIVE_NAMES | STATEFUL_PRIMITIVE_NAMES
)


def is_known_primitive(name: str) -> bool:
    return name in ALL_BRIDGED_PRIMITIVE_NAMES


def primitive_kind(name: str) -> str:
    """Return one of 'value', 'system_value', 'gate', 'placement', 'stateful', 'unknown'."""
    if name in _VALUE_PRIMITIVES:
        return "value"
    if name in _SYSTEM_VALUE_PRIMITIVES:
        return "system_value"
    if name in _GATE_PRIMITIVES:
        return "gate"
    if name in _PLACEMENT_KEY_PRIMITIVES:
        return "placement"
    if name in STATEFUL_PRIMITIVE_NAMES:
        return "stateful"
    return "unknown"


def primitive_family(name: str) -> PrimitiveFamily:
    """Family lookup for any bridged name, delegating to the CC2 registry spec."""
    spec = prim.get_primitive_spec(name)
    return spec.family


def families_compatible(names: Sequence[str]) -> bool:
    """True iff every named primitive's compatible_families sets share a
    nonempty intersection (CC2's own cross-family-misuse rule, reused
    verbatim rather than re-derived)."""
    if not names:
        return True
    sets = [prim.get_primitive_spec(n).compatible_families for n in names]
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return len(common) > 0


def validate_primitive_params(name: str, params: Mapping[str, float]) -> Dict[str, float]:
    """Bind + bounds-check params against the CC2 registry's own ParamBound
    objects. Raises PrimitiveError (unknown param name, non-numeric, NaN, or
    out-of-bounds) -- the verifier wraps this into precise DSL error codes."""
    spec = prim.get_primitive_spec(name)
    unknown = set(params) - set(spec.param_bounds)
    if unknown:
        raise PrimitiveError(
            f"{name} received unsupported parameter(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(spec.param_bounds)}"
        )
    bound: Dict[str, float] = {}
    for pname, bound_spec in spec.param_bounds.items():
        value = params.get(pname, bound_spec.default)
        bound[pname] = bound_spec.validate(value, param_name=pname, primitive_name=name)
    return bound


def canonical_ref_key(name: str, params: Mapping[str, float]) -> str:
    """Deterministic string key identifying one (primitive, bound-params) pair."""
    payload = json.dumps(sorted(params.items()), sort_keys=True, separators=(",", ":"))
    return f"{name}::{payload}"


def resolve_value(name: str, params: Mapping[str, float], req: ObservableRequest, state: ObservableState) -> float:
    """Resolve a 'value' or 'system_value' primitive to a float."""
    kind = primitive_kind(name)
    if kind == "value":
        return float(_VALUE_PRIMITIVES[name].value(req, state, **params))
    if kind == "system_value":
        return float(_SYSTEM_VALUE_PRIMITIVES[name](state))
    raise PrimitiveError(f"{name!r} is not a value-shaped primitive (kind={kind})")


def resolve_gate(name: str, params: Mapping[str, float], req: ObservableRequest, state: ObservableState) -> float:
    """Resolve a gate primitive to 1.0 (pass) / 0.0 (reject)."""
    if primitive_kind(name) != "gate":
        raise PrimitiveError(f"{name!r} is not a gate primitive")
    return 1.0 if _GATE_PRIMITIVES[name].passes(req, state, **params) else 0.0


def resolve_placement_key(
    name: str,
    params: Mapping[str, float],
    gpu: ObservableGPUState,
    req: ObservableRequest,
) -> tuple:
    if primitive_kind(name) != "placement":
        raise PrimitiveError(f"{name!r} is not a placement-key primitive")
    return _PLACEMENT_KEY_PRIMITIVES[name].key(gpu, req, **params)


def build_composite_placement_key(
    keys_spec: Sequence[Tuple[str, Mapping[str, float]]],
):
    """Return a callable (gpu, req) -> tuple that concatenates each declared
    placement-key primitive's tuple in declared order, with gpu_id always
    appended last as the structural, non-configurable final tie-break."""

    def _key(gpu: ObservableGPUState, req: ObservableRequest) -> tuple:
        parts: list = []
        for name, params in keys_spec:
            parts.extend(resolve_placement_key(name, params, gpu, req))
        parts.append(gpu.gpu_id)
        return tuple(parts)

    return _key


def build_runtime_context(
    primitive_refs: Sequence[Tuple[str, str, Mapping[str, float]]],
    resolved_params: Mapping[str, float],
    req: ObservableRequest,
    state: ObservableState,
) -> Dict[str, float]:
    """Resolve every distinct primitive/param reference a compiled heuristic
    declared against the real (req, state) objects for one candidate request
    this step, keyed by the exact same reserved variable names
    compile_heuristic()'s lowering pass produced. Merge the result into the
    same flat float ctx already used for req.*/sys.*/batch.* variables."""
    from .dsl_schema import PARAM_VAR_PREFIX, PRIMITIVE_VAR_PREFIX

    ctx: Dict[str, float] = {}
    for kind, name, params in primitive_refs:
        key = PRIMITIVE_VAR_PREFIX + canonical_ref_key(name, params)
        if kind == "primitive":
            ctx[key] = resolve_value(name, params, req, state)
        else:  # "primitive_gate"
            ctx[key] = resolve_gate(name, params, req, state)
    for pname, value in resolved_params.items():
        ctx[PARAM_VAR_PREFIX + pname] = value
    return ctx


def iter_expression_blocks(heuristic: Mapping[str, Any]) -> List[Any]:
    """Return exactly the expression-tree fields of a heuristic document:
    default/regime request_score, batch_score, admission_condition, and
    regime condition. Deliberately excludes "admission_budget", "placement",
    and "parameters" -- those top-level blocks reuse the {"primitive": ...}
    JSON shape for their own declarations, not expression-tree references,
    so collect_primitive_refs() must never be run over the whole document."""
    blocks: List[Any] = []
    default = heuristic.get("default")
    if isinstance(default, dict):
        for f in ("request_score", "batch_score", "admission_condition"):
            if f in default:
                blocks.append(default[f])
    for regime in heuristic.get("regimes", []) or []:
        if isinstance(regime, dict):
            for f in ("condition", "request_score", "batch_score", "admission_condition"):
                if f in regime:
                    blocks.append(regime[f])
    return blocks


def build_system_context(
    primitive_refs: Sequence[Tuple[str, str, Mapping[str, float]]],
    state: ObservableState,
) -> Dict[str, float]:
    """Resolve every distinct 'system_value'-kind primitive reference (the
    only shape that needs no per-request ObservableRequest) against the
    current step's state, keyed the same way build_runtime_context() keys
    per-request primitives. Regime "condition" expressions are only ever
    evaluated against sys_vars/batch_vars (never req_vars, even pre-CC3), so
    a primitive reference inside a condition must be system-level to ever
    resolve; this is what makes that possible."""
    from .dsl_schema import PRIMITIVE_VAR_PREFIX

    ctx: Dict[str, float] = {}
    for kind, name, params in primitive_refs:
        if kind == "primitive" and primitive_kind(name) == "system_value":
            key = PRIMITIVE_VAR_PREFIX + canonical_ref_key(name, params)
            ctx[key] = resolve_value(name, params, None, state)  # type: ignore[arg-type]
    return ctx


def collect_primitive_refs(node: Any) -> List[Tuple[str, str, Dict[str, float]]]:
    """Walk an expression tree (or any nested dict/list) collecting every
    (kind, name, params) reference to {"primitive": ...} / {"primitive_gate": ...}
    leaf nodes. `kind` is "primitive" or "primitive_gate" (matches the JSON key).

    Callers must pass an actual expression tree (or use
    iter_expression_blocks() first) -- never the whole heuristic document,
    since "admission_budget"/"placement" reuse the same {"primitive": ...}
    JSON key for non-expression declarations."""
    found: List[Tuple[str, str, Dict[str, float]]] = []

    def _walk(n: Any) -> None:
        if isinstance(n, dict):
            for kind in ("primitive", "primitive_gate"):
                if kind in n:
                    name = n[kind]
                    params = n.get("params", {}) or {}
                    if isinstance(name, str) and isinstance(params, dict):
                        found.append((kind, name, dict(params)))
            for key, value in n.items():
                if key in ("primitive", "primitive_gate", "params"):
                    continue
                _walk(value)
        elif isinstance(n, (list, tuple)):
            for item in n:
                _walk(item)

    _walk(node)
    return found


def lower_expression(node: Any) -> Any:
    """Deep-copy `node`, replacing every {"primitive"/"primitive_gate": name,
    "params": {...}} leaf with {"var": "<PRIMITIVE_VAR_PREFIX><canonical key>"}
    and every {"param": name} leaf with {"var": "<PARAM_VAR_PREFIX>name"}.

    Uses CC2's own bound (fully-defaulted) params to compute the canonical
    key, so the same key is produced here and at HeuristicPolicy runtime
    regardless of which optional params the author omitted. Raises
    PrimitiveError if a referenced primitive name/params are invalid --
    callers that only want a best-effort lowering (e.g. the verifier's dry
    run, which has already reported such errors via its own checks) should
    catch and skip.
    """
    from .dsl_schema import PARAM_VAR_PREFIX, PRIMITIVE_VAR_PREFIX

    if isinstance(node, dict):
        for kind in ("primitive", "primitive_gate"):
            if kind in node:
                name = node[kind]
                params = node.get("params", {}) or {}
                bound = validate_primitive_params(name, params)
                return {"var": PRIMITIVE_VAR_PREFIX + canonical_ref_key(name, bound)}
        if "param" in node:
            return {"var": PARAM_VAR_PREFIX + node["param"]}
        return {k: lower_expression(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [lower_expression(v) for v in node]
    return node


def collect_param_refs(node: Any) -> List[str]:
    """Walk an expression tree collecting every {"param": name} leaf reference."""
    found: List[str] = []

    def _walk(n: Any) -> None:
        if isinstance(n, dict):
            if "param" in n and isinstance(n["param"], str):
                found.append(n["param"])
            for key, value in n.items():
                if key == "param":
                    continue
                _walk(value)
        elif isinstance(n, (list, tuple)):
            for item in n:
                _walk(item)

    _walk(node)
    return found

"""Canonical scheduling primitive interface (CC2).

This module defines small, typed, deterministic building blocks for
scheduling behavior, organized into five separate families that are never
collapsed into one scalar score:

* ``RANKING``        -- per-request ordering features (deadline urgency,
                        laxity, prompt/output length, estimated service
                        time, priority, queue age, fairness bonus, ...).
* ``ADMISSION``       -- gates and continuous risk scores that decide
                        whether a request is scheduling-eligible this step.
* ``PLACEMENT``       -- GPU-selection keys and placement engines.
* ``BATCHING``        -- token-budget and admission-credit parameters that
                        bound how much work enters a step.
* ``RESOURCE_GUARD``  -- feasibility and system-pressure guards.

Every primitive is a pure function (or a small explicitly-stateful class
for the credit-budget primitive) over ``ObservableRequest``/
``ObservableGPUState``/``ObservableState`` -- the same causal, oracle-free
surface every existing policy already uses (see ``core/types.py``).  No
primitive accepts or derives from ``Request.actual_output_tokens`` or any
other hidden ground-truth field.

This module intentionally reuses the existing scoring/placement helpers
(``scoring.py``, ``policy_library_v2_helpers.py``) rather than
re-deriving their formulas, and is meant to sit alongside
``composition.py`` (rank-expert aggregation) as a lower-level, more
atomic decomposition -- ``composition.py``'s named experts are mostly
reproducible as short compositions of these primitives.

See ``docs/architecture/contextual_composition_primitives.md`` for the
full taxonomy, causal-input notes, and representative-policy mappings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.action import Action
from ..core.types import ObservableGPUState, ObservableRequest, ObservableState
from .policy_library_v2_helpers import gpu_pressure as _gpu_pressure_fn
from .scoring import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    kv_fill_ratio,
    predicted_service_proxy,
    remaining_kv,
)

_EPS = 1e-9


class PrimitiveFamily(str, Enum):
    RANKING = "ranking"
    ADMISSION = "admission"
    PLACEMENT = "placement"
    BATCHING = "batching"
    RESOURCE_GUARD = "resource_guard"


class PrimitiveError(ValueError):
    """Raised for unknown primitive names, invalid/out-of-bounds parameters,
    or an attempt to register a non-causal (oracle/future-information)
    primitive."""


# ---------------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamBound:
    """Inclusive bounds and a default for one primitive parameter."""

    minimum: float
    maximum: float
    default: float

    def validate(self, value: float, *, param_name: str, primitive_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PrimitiveError(
                f"{primitive_name}.{param_name} must be numeric, got {value!r}"
            )
        if math.isnan(value):
            raise PrimitiveError(f"{primitive_name}.{param_name} must not be NaN")
        if value < self.minimum or value > self.maximum:
            raise PrimitiveError(
                f"{primitive_name}.{param_name}={value!r} outside bounds "
                f"[{self.minimum}, {self.maximum}]"
            )
        return float(value)


@dataclass(frozen=True)
class PrimitiveSpec:
    """Registry metadata for one canonical primitive.

    ``compatible_families`` records which other primitive families this
    primitive's output may be legally combined with (e.g. a RANKING
    primitive's value may feed an ADMISSION gate's threshold comparison,
    but a PLACEMENT key is never a valid ADMISSION predicate input).
    """

    name: str
    family: PrimitiveFamily
    input_type: str
    output_type: str
    param_bounds: Mapping[str, ParamBound]
    doc: str
    compatible_families: frozenset
    deterministic: bool = True
    causal: bool = True
    derived_from: Tuple[str, ...] = ()


_REGISTRY: Dict[str, PrimitiveSpec] = {}


def register_primitive(spec: PrimitiveSpec) -> PrimitiveSpec:
    if spec.name in _REGISTRY:
        raise PrimitiveError(f"Primitive {spec.name!r} is already registered")
    if not spec.causal:
        raise PrimitiveError(
            f"Primitive {spec.name!r} must be causal; oracle/future-information "
            "primitives are not supported by this registry"
        )
    if not spec.deterministic:
        raise PrimitiveError(f"Primitive {spec.name!r} must be deterministic")
    _REGISTRY[spec.name] = spec
    return spec


def get_primitive_spec(name: str) -> PrimitiveSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise PrimitiveError(
            f"Unknown primitive {name!r}; registered primitives: {sorted(_REGISTRY)}"
        ) from exc


def list_primitives(family: Optional[PrimitiveFamily] = None) -> List[PrimitiveSpec]:
    specs = list(_REGISTRY.values())
    if family is not None:
        specs = [s for s in specs if s.family == family]
    return sorted(specs, key=lambda s: s.name)


def _bind_params(spec: PrimitiveSpec, overrides: Mapping[str, float]) -> Dict[str, float]:
    unknown = set(overrides) - set(spec.param_bounds)
    if unknown:
        raise PrimitiveError(
            f"{spec.name} received unsupported parameter(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(spec.param_bounds)}"
        )
    bound: Dict[str, float] = {}
    for pname, bound_spec in spec.param_bounds.items():
        value = overrides.get(pname, bound_spec.default)
        bound[pname] = bound_spec.validate(value, param_name=pname, primitive_name=spec.name)
    return bound


# ---------------------------------------------------------------------------
# RANKING family: per-request scalar features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingPrimitive:
    """A per-request scalar feature usable as one component of a
    deterministic, lexicographic (tuple) sort key.

    ``higher_is_preferred`` documents the feature's natural direction;
    ``sort_key_component`` always returns a value where *ascending* tuple
    order means "scheduled first", negating the raw value when the
    natural direction is "higher is better".
    """

    spec: PrimitiveSpec
    higher_is_preferred: bool
    value_fn: Callable[[ObservableRequest, ObservableState, Mapping[str, float]], float]

    def value(self, req: ObservableRequest, state: ObservableState, **params: float) -> float:
        bound = _bind_params(self.spec, params)
        return self.value_fn(req, state, bound)

    def sort_key_component(self, req: ObservableRequest, state: ObservableState, **params: float) -> float:
        v = self.value(req, state, **params)
        return -v if self.higher_is_preferred else v


def _deadline_urgency_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return req.slo_deadline


DEADLINE_URGENCY = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="deadline_urgency",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float (seconds, absolute slo_deadline)",
        param_bounds={},
        doc=(
            "Earliest-deadline urgency: ranks requests by absolute SLO "
            "deadline; earlier deadlines are more urgent and preferred "
            "first. This is the literal earliest-deadline-first (EDF) "
            "notion of urgency and deliberately ignores estimated service "
            "time -- see `laxity` for the slack-aware refinement."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
    )),
    higher_is_preferred=False,
    value_fn=_deadline_urgency_value,
)


def _laxity_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    proxy = predicted_service_proxy(req, params["alpha"], params["beta"])
    return req.slo_deadline - state.time - proxy * params["step_size"]


LAXITY = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="laxity",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest, ObservableState",
        output_type="float (seconds, remaining slack after estimated service)",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "beta": ParamBound(0.0, 1e6, DEFAULT_BETA),
            "step_size": ParamBound(1e-9, 10.0, 0.001),
        },
        doc=(
            "Remaining time budget after estimated service: "
            "slo_deadline - now - step_size * (alpha*prompt_tokens + "
            "beta*predicted_output_tokens). Negative laxity means the "
            "request is already expected to miss its deadline; smaller "
            "(more negative) laxity is more urgent."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING, PrimitiveFamily.ADMISSION}),
    )),
    higher_is_preferred=False,
    value_fn=_laxity_value,
)


def _prompt_length_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return float(req.prompt_tokens)


PROMPT_LENGTH = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="prompt_length",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float (tokens)",
        param_bounds={},
        doc="Prompt token count; shorter prompts preferred first (SJF-style).",
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
    )),
    higher_is_preferred=False,
    value_fn=_prompt_length_value,
)


def _predicted_output_length_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return float(req.predicted_output_tokens)


PREDICTED_OUTPUT_LENGTH = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="predicted_output_length",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float (tokens)",
        param_bounds={},
        doc=(
            "Predicted output token count (never actual_output_tokens); "
            "shorter predicted decode work preferred first."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
    )),
    higher_is_preferred=False,
    value_fn=_predicted_output_length_value,
)


def _estimated_service_time_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return predicted_service_proxy(req, params["alpha"], params["beta"])


ESTIMATED_SERVICE_TIME = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="estimated_service_time",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float (decode-step proxy, dimensionless)",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "beta": ParamBound(0.0, 1e6, DEFAULT_BETA),
        },
        doc=(
            "Estimated total service time proxy: alpha*prompt_tokens + "
            "beta*predicted_output_tokens. Shorter estimated work "
            "preferred first (SJF/SPT-style)."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING, PrimitiveFamily.ADMISSION}),
    )),
    higher_is_preferred=False,
    value_fn=_estimated_service_time_value,
)


def _priority_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return float(req.priority)


PRIORITY = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="priority",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float",
        param_bounds={},
        doc="Request priority weight; higher priority preferred first.",
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
    )),
    higher_is_preferred=True,
    value_fn=_priority_value,
)


def _queue_age_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return max(0.0, state.time - req.arrival_time)


QUEUE_AGE = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="queue_age",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest, ObservableState",
        output_type="float (seconds)",
        param_bounds={},
        doc=(
            "Time since arrival (now - arrival_time); older-waiting "
            "requests preferred first (fairness/FIFO-equivalent within a "
            "single scheduling step, since `now` is constant across the "
            "waiting queue at that step)."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
    )),
    higher_is_preferred=True,
    value_fn=_queue_age_value,
)


def _fairness_starvation_bonus_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    age = max(0.0, state.time - req.arrival_time)
    return params["priority_weight"] * req.priority + params["age_bonus"] * age


FAIRNESS_STARVATION_BONUS = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="fairness_starvation_bonus",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest, ObservableState",
        output_type="float",
        param_bounds={
            "priority_weight": ParamBound(0.0, 1e6, 1.0),
            "age_bonus": ParamBound(0.0, 1e6, 0.05),
        },
        doc=(
            "Composite fairness/starvation-prevention bonus: "
            "priority_weight*priority + age_bonus*queue_age. Higher "
            "preferred first; guards against indefinite starvation of "
            "low-priority, long-waiting requests. Derived from `priority` "
            "and `queue_age`."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
        derived_from=("priority", "queue_age"),
    )),
    higher_is_preferred=True,
    value_fn=_fairness_starvation_bonus_value,
)


def _laxity_urgency_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    lax = _laxity_value(req, state, params)
    return 1.0 / max(lax, _EPS)


LAXITY_URGENCY = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="laxity_urgency",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest, ObservableState",
        output_type="float (1/seconds)",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "beta": ParamBound(0.0, 1e6, DEFAULT_BETA),
            "step_size": ParamBound(1e-9, 10.0, 0.001),
        },
        doc=(
            "Inverse-laxity urgency: 1 / max(laxity, eps). Higher is more "
            "urgent and preferred first. Derived from `laxity`; matches "
            "the composite urgency term used by SCORPIO-style guards."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
        derived_from=("laxity",),
    )),
    higher_is_preferred=True,
    value_fn=_laxity_urgency_value,
)


def _wsp_score_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    proxy = predicted_service_proxy(req, params["alpha"], params["beta"])
    return proxy / max(req.priority, _EPS)


WEIGHTED_SHORTEST_PROCESSING_SCORE = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="weighted_shortest_processing_score",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "beta": ParamBound(0.0, 1e6, DEFAULT_BETA),
        },
        doc=(
            "WSPT score: estimated_service_time / priority. Lower "
            "preferred first. Derived from `estimated_service_time` and "
            "`priority`."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
        derived_from=("estimated_service_time", "priority"),
    )),
    higher_is_preferred=False,
    value_fn=_wsp_score_value,
)


def _request_id_tiebreak_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    return float(req.request_id)


REQUEST_ID_TIEBREAK = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="request_id_tiebreak",
        family=PrimitiveFamily.RANKING,
        input_type="ObservableRequest",
        output_type="float",
        param_bounds={},
        doc=(
            "Final deterministic tie-break by request_id. Should be the "
            "last component of any composed ranking key so no two "
            "requests can ever tie."
        ),
        compatible_families=frozenset({PrimitiveFamily.RANKING}),
    )),
    higher_is_preferred=False,
    value_fn=_request_id_tiebreak_value,
)


RankingComponent = Tuple[RankingPrimitive, Mapping[str, float]]


def build_ranking_key(
    components: Sequence[RankingComponent],
    state: ObservableState,
) -> Callable[[ObservableRequest], tuple]:
    """Compose ranking primitives into one deterministic lexicographic sort key.

    ``components`` is an ordered sequence of ``(primitive, params)`` pairs;
    earlier components take precedence. Callers are responsible for
    appending ``REQUEST_ID_TIEBREAK`` (or another exhaustive tie-break) if
    full determinism across arbitrary inputs is required.
    """
    if not components:
        raise PrimitiveError("build_ranking_key requires at least one ranking component")

    def _key(req: ObservableRequest) -> tuple:
        return tuple(prim.sort_key_component(req, state, **params) for prim, params in components)

    return _key


def rank_requests(
    state: ObservableState,
    components: Sequence[RankingComponent],
) -> List[ObservableRequest]:
    """Sort ``state.waiting_queue`` by a composed ranking key."""
    key = build_ranking_key(components, state)
    return sorted(state.waiting_queue, key=key)


# ---------------------------------------------------------------------------
# ADMISSION family: gates and continuous risk scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionGate:
    """A boolean admission predicate over one request."""

    spec: PrimitiveSpec
    predicate_fn: Callable[[ObservableRequest, ObservableState, Mapping[str, float]], bool]

    def passes(self, req: ObservableRequest, state: ObservableState, **params: float) -> bool:
        bound = _bind_params(self.spec, params)
        return self.predicate_fn(req, state, bound)


def _laxity_gate_predicate(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> bool:
    lax = _laxity_value(req, state, params)
    return lax >= -params["laxity_threshold"]


LAXITY_GATE = AdmissionGate(
    spec=register_primitive(PrimitiveSpec(
        name="laxity_gate",
        family=PrimitiveFamily.ADMISSION,
        input_type="ObservableRequest, ObservableState",
        output_type="bool",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "beta": ParamBound(0.0, 1e6, DEFAULT_BETA),
            "step_size": ParamBound(1e-9, 10.0, 0.001),
            "laxity_threshold": ParamBound(0.0, float("inf"), float("inf")),
        },
        doc=(
            "Admission gate: keep a request only if laxity >= "
            "-laxity_threshold. laxity_threshold=inf (default) disables "
            "filtering; 0.0 admits only requests whose estimated service "
            "still fits within the remaining deadline."
        ),
        compatible_families=frozenset({PrimitiveFamily.ADMISSION}),
        derived_from=("laxity",),
    )),
    predicate_fn=_laxity_gate_predicate,
)


def _ttft_slack_gate_predicate(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> bool:
    prefill_proxy_seconds = params["alpha"] * req.prompt_tokens * params["step_size"]
    slack = req.slo_deadline - state.time - prefill_proxy_seconds
    return slack >= -params["ttft_slack_threshold"]


TTFT_SLACK_GATE = AdmissionGate(
    spec=register_primitive(PrimitiveSpec(
        name="ttft_slack_gate",
        family=PrimitiveFamily.ADMISSION,
        input_type="ObservableRequest, ObservableState",
        output_type="bool",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "step_size": ParamBound(1e-9, 10.0, 0.001),
            "ttft_slack_threshold": ParamBound(0.0, float("inf"), 0.0),
        },
        doc=(
            "Admission gate on predicted prefill (TTFT proxy) slack only: "
            "keep a request if slo_deadline - now - alpha*prompt_tokens*"
            "step_size >= -ttft_slack_threshold. Distinct from `laxity_gate`, "
            "which also accounts for predicted decode time."
        ),
        compatible_families=frozenset({PrimitiveFamily.ADMISSION}),
    )),
    predicate_fn=_ttft_slack_gate_predicate,
)


def _admission_risk_value(req: ObservableRequest, state: ObservableState, params: Mapping[str, float]) -> float:
    lax = _laxity_value(req, state, params)
    z = -lax / max(params["scale"], _EPS)
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


ADMISSION_RISK = RankingPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="admission_risk",
        family=PrimitiveFamily.ADMISSION,
        input_type="ObservableRequest, ObservableState",
        output_type="float in [0, 1]",
        param_bounds={
            "alpha": ParamBound(0.0, 1e6, DEFAULT_ALPHA),
            "beta": ParamBound(0.0, 1e6, DEFAULT_BETA),
            "step_size": ParamBound(1e-9, 10.0, 0.001),
            "scale": ParamBound(_EPS, 1e6, 1.0),
        },
        doc=(
            "Continuous SLO-miss risk score in [0, 1]: a logistic "
            "transform of `laxity` (risk=0.5 exactly at laxity=0, higher "
            "risk for more negative laxity). Provided as a smooth "
            "alternative to `laxity_gate`'s hard threshold; at threshold "
            "0.5 the two agree exactly by construction (see equivalence "
            "tests)."
        ),
        compatible_families=frozenset({PrimitiveFamily.ADMISSION, PrimitiveFamily.RANKING}),
        derived_from=("laxity",),
    )),
    higher_is_preferred=False,
    value_fn=_admission_risk_value,
)


# ---------------------------------------------------------------------------
# RESOURCE_GUARD family: feasibility and system-pressure guards
# ---------------------------------------------------------------------------

_FEASIBLE_ON_GPU_SPEC = register_primitive(PrimitiveSpec(
    name="feasible_on_gpu",
    family=PrimitiveFamily.RESOURCE_GUARD,
    input_type="ObservableGPUState, ObservableRequest",
    output_type="bool",
    param_bounds={},
    doc=(
        "Capacity feasibility guard: True if admitting req to gpu would "
        "not exceed max_active_sequences, max_kv_tokens, or "
        "max_batch_tokens. Raises PrimitiveError if the GPU's own "
        "capacity bounds are non-positive (unsupported/misconfigured "
        "state)."
    ),
    compatible_families=frozenset({PrimitiveFamily.RESOURCE_GUARD, PrimitiveFamily.PLACEMENT}),
))


def feasible_on_gpu(gpu: ObservableGPUState, req: ObservableRequest) -> bool:
    if gpu.max_active_sequences <= 0 or gpu.max_kv_tokens <= 0 or gpu.max_batch_tokens <= 0:
        raise PrimitiveError(
            f"feasible_on_gpu requires positive GPU capacity bounds, got "
            f"max_active_sequences={gpu.max_active_sequences}, "
            f"max_kv_tokens={gpu.max_kv_tokens}, max_batch_tokens={gpu.max_batch_tokens}"
        )
    new_count = len(gpu.active_request_ids) + 1
    new_kv = gpu.current_kv_tokens + req.prompt_tokens
    new_batch = new_count
    return (
        new_count <= gpu.max_active_sequences
        and new_kv <= gpu.max_kv_tokens
        and new_batch <= gpu.max_batch_tokens
    )


_SYSTEM_KV_PRESSURE_SPEC = register_primitive(PrimitiveSpec(
    name="system_kv_pressure",
    family=PrimitiveFamily.RESOURCE_GUARD,
    input_type="ObservableState",
    output_type="float in [0, 1]",
    param_bounds={},
    doc="Worst-case (max over GPUs) KV-cache fill ratio in the system; 0.0 if there are no GPUs.",
    compatible_families=frozenset({PrimitiveFamily.RESOURCE_GUARD, PrimitiveFamily.ADMISSION}),
))


def system_kv_pressure(state: ObservableState) -> float:
    if not state.gpu_states:
        return 0.0
    return max(kv_fill_ratio(g) for g in state.gpu_states)


_DECODE_PRESSURE_SPEC = register_primitive(PrimitiveSpec(
    name="decode_pressure",
    family=PrimitiveFamily.RESOURCE_GUARD,
    input_type="ObservableState",
    output_type="float in [0, 1]",
    param_bounds={},
    doc=(
        "Worst-case (max over GPUs) decode-phase load: "
        "decoding_count / max_active_sequences. One channel of "
        "'projected GPU load'; the other is the placement-time "
        "`projected_gpu_load` per-GPU key."
    ),
    compatible_families=frozenset({PrimitiveFamily.RESOURCE_GUARD, PrimitiveFamily.ADMISSION}),
))


def decode_pressure(state: ObservableState) -> float:
    if not state.gpu_states:
        return 0.0
    return max(g.decoding_count / max(g.max_active_sequences, 1) for g in state.gpu_states)


_PREFILL_PRESSURE_SPEC = register_primitive(PrimitiveSpec(
    name="prefill_pressure",
    family=PrimitiveFamily.RESOURCE_GUARD,
    input_type="ObservableState",
    output_type="float in [0, 1]",
    param_bounds={},
    doc="Worst-case (max over GPUs) prefill-phase load: prefilling_count / max_active_sequences.",
    compatible_families=frozenset({PrimitiveFamily.RESOURCE_GUARD, PrimitiveFamily.ADMISSION}),
))


def prefill_pressure(state: ObservableState) -> float:
    if not state.gpu_states:
        return 0.0
    return max(g.prefilling_count / max(g.max_active_sequences, 1) for g in state.gpu_states)


_QUEUE_PRESSURE_SPEC = register_primitive(PrimitiveSpec(
    name="queue_pressure",
    family=PrimitiveFamily.RESOURCE_GUARD,
    input_type="ObservableState",
    output_type="float >= 0",
    param_bounds={},
    doc="Waiting-queue length normalized by total sequence capacity across GPUs.",
    compatible_families=frozenset({PrimitiveFamily.RESOURCE_GUARD, PrimitiveFamily.ADMISSION}),
))


def queue_pressure(state: ObservableState) -> float:
    cap = sum(g.max_active_sequences for g in state.gpu_states)
    return len(state.waiting_queue) / max(cap, 1)


def system_overload_guard(
    *,
    kv_pressure: float,
    decode_pressure: float,
    queue_pressure: float,
    mean_laxity: float,
    kv_threshold: float,
    decode_threshold: float,
    queue_threshold: float,
) -> bool:
    """Composite overload guard: True if any pressure signal exceeds its
    threshold, or the candidate pool's mean laxity is already negative.

    Named-scalar inputs (rather than ``ObservableState`` directly) keep
    this primitive reusable with either the system-wide guard signals
    above or a caller's own bespoke pressure definitions.
    """
    return (
        kv_pressure >= kv_threshold
        or decode_pressure >= decode_threshold
        or queue_pressure >= queue_threshold
        or mean_laxity < 0.0
    )


register_primitive(PrimitiveSpec(
    name="system_overload_guard",
    family=PrimitiveFamily.RESOURCE_GUARD,
    input_type="named pressure scalars (kv_pressure, decode_pressure, queue_pressure, mean_laxity)",
    output_type="bool",
    param_bounds={},
    doc=(
        "Composite overload guard combining `system_kv_pressure`, "
        "`decode_pressure`, `queue_pressure`, and mean candidate `laxity` "
        "via OR-of-thresholds, matching the SCORPIO-style guard-activation "
        "rule."
    ),
    compatible_families=frozenset({PrimitiveFamily.RESOURCE_GUARD, PrimitiveFamily.ADMISSION, PrimitiveFamily.BATCHING}),
    derived_from=("system_kv_pressure", "decode_pressure", "queue_pressure", "laxity"),
))


# ---------------------------------------------------------------------------
# PLACEMENT family: GPU-selection keys and placement engines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementKeyPrimitive:
    """A GPU-selection key: lower is preferred, mirroring Python's `sorted`."""

    spec: PrimitiveSpec
    key_fn: Callable[[ObservableGPUState, ObservableRequest, Mapping[str, float]], tuple]

    def key(self, gpu: ObservableGPUState, req: ObservableRequest, **params: float) -> tuple:
        bound = _bind_params(self.spec, params)
        return self.key_fn(gpu, req, bound)


def _projected_gpu_load_key(gpu: ObservableGPUState, req: ObservableRequest, params: Mapping[str, float]) -> tuple:
    return (_gpu_pressure_fn(gpu), gpu.gpu_id)


PROJECTED_GPU_LOAD = PlacementKeyPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="projected_gpu_load",
        family=PrimitiveFamily.PLACEMENT,
        input_type="ObservableGPUState",
        output_type="tuple(float in [0,1], int) placement key, ascending preferred",
        param_bounds={},
        doc=(
            "Blended per-GPU load projection: 0.45*sequence fill + "
            "0.45*KV fill + 0.10*prefill-phase fill, tie-broken by "
            "gpu_id. Least-loaded GPU is preferred (matches "
            "`policy_library_v2_helpers.gpu_pressure`, the same key "
            "`composition.deterministic_place` uses by default)."
        ),
        compatible_families=frozenset({PrimitiveFamily.PLACEMENT}),
    )),
    key_fn=_projected_gpu_load_key,
)


def _kv_pressure_placement_key(gpu: ObservableGPUState, req: ObservableRequest, params: Mapping[str, float]) -> tuple:
    return (kv_fill_ratio(gpu), gpu.gpu_id)


KV_PRESSURE_PLACEMENT = PlacementKeyPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="kv_pressure",
        family=PrimitiveFamily.PLACEMENT,
        input_type="ObservableGPUState",
        output_type="tuple(float in [0,1], int) placement key, ascending preferred",
        param_bounds={},
        doc=(
            "Per-GPU KV-cache fill ratio as a placement key, tie-broken by "
            "gpu_id; the least KV-pressured GPU is preferred. See "
            "`system_kv_pressure` for the system-wide RESOURCE_GUARD "
            "channel of the same underlying signal."
        ),
        compatible_families=frozenset({PrimitiveFamily.PLACEMENT, PrimitiveFamily.RESOURCE_GUARD}),
    )),
    key_fn=_kv_pressure_placement_key,
)


def _tightest_kv_fit_key(gpu: ObservableGPUState, req: ObservableRequest, params: Mapping[str, float]) -> tuple:
    return (remaining_kv(gpu), gpu.gpu_id)


TIGHTEST_KV_FIT = PlacementKeyPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="tightest_kv_fit",
        family=PrimitiveFamily.PLACEMENT,
        input_type="ObservableGPUState",
        output_type="tuple(int, int) placement key, ascending preferred",
        param_bounds={},
        doc=(
            "Best-fit bin-packing key: smallest remaining KV capacity "
            "among feasible GPUs is preferred (tightest fit), tie-broken "
            "by gpu_id."
        ),
        compatible_families=frozenset({PrimitiveFamily.PLACEMENT}),
    )),
    key_fn=_tightest_kv_fit_key,
)


def _least_loaded_key(gpu: ObservableGPUState, req: ObservableRequest, params: Mapping[str, float]) -> tuple:
    kv_tie = gpu.current_kv_tokens / max(gpu.max_kv_tokens, 1)
    return (len(gpu.active_request_ids) + kv_tie * 0.01, gpu.gpu_id)


LEAST_LOADED = PlacementKeyPrimitive(
    spec=register_primitive(PrimitiveSpec(
        name="least_loaded",
        family=PrimitiveFamily.PLACEMENT,
        input_type="ObservableGPUState",
        output_type="tuple(float, int) placement key, ascending preferred",
        param_bounds={},
        doc=(
            "Fewest-active-sequences placement key with KV fill ratio as "
            "a light tiebreaker (weight 0.01), tie-broken by gpu_id."
        ),
        compatible_families=frozenset({PrimitiveFamily.PLACEMENT}),
    )),
    key_fn=_least_loaded_key,
)


register_primitive(PrimitiveSpec(
    name="round_robin_placement",
    family=PrimitiveFamily.PLACEMENT,
    input_type="ObservableState, ranked requests",
    output_type="Action",
    param_bounds={},
    doc=(
        "Placement engine (not a per-GPU key): cycles a rotating start "
        "index across GPUs in gpu_states order, admitting each ranked "
        "request to the first feasible GPU found starting from the "
        "current index. `advance_index_on_failure` controls whether the "
        "index still advances by one when no GPU is feasible for a "
        "request (True for admission_control/SCORPIO-style policies; "
        "False for fifo/edf/wsp/estf). See `place_round_robin`."
    ),
    compatible_families=frozenset({PrimitiveFamily.PLACEMENT}),
))

register_primitive(PrimitiveSpec(
    name="greedy_key_placement",
    family=PrimitiveFamily.PLACEMENT,
    input_type="ObservableState, ranked requests, PlacementKeyPrimitive",
    output_type="Action",
    param_bounds={},
    doc=(
        "Placement engine (not a per-GPU key): for each ranked request in "
        "order, admits to the feasible GPU with the smallest placement "
        "key (no rotation). See `place_greedy_key`."
    ),
    compatible_families=frozenset({PrimitiveFamily.PLACEMENT}),
))


def place_round_robin(
    state: ObservableState,
    ranked: Sequence[ObservableRequest],
    *,
    advance_index_on_failure: bool = False,
    max_admits: Optional[int] = None,
) -> Action:
    """Round-robin placement engine; see the `round_robin_placement` registry entry."""
    admit: Dict[int, List[int]] = {g.gpu_id: [] for g in state.gpu_states}
    n_gpus = len(state.gpu_states)
    if n_gpus == 0:
        return Action(admit=admit)
    gpu_idx = 0
    admitted = 0
    for req in ranked:
        if max_admits is not None and admitted >= max_admits:
            break
        placed = False
        for offset in range(n_gpus):
            gpu = state.gpu_states[(gpu_idx + offset) % n_gpus]
            if feasible_on_gpu(gpu, req):
                admit[gpu.gpu_id].append(req.request_id)
                gpu.active_request_ids.append(req.request_id)
                gpu.current_kv_tokens += req.prompt_tokens
                gpu_idx = (gpu_idx + offset + 1) % n_gpus
                placed = True
                admitted += 1
                break
        if not placed and advance_index_on_failure:
            gpu_idx = (gpu_idx + 1) % n_gpus
    return Action(admit=admit)


AdmitFilter = Callable[[ObservableRequest, ObservableGPUState], bool]


def place_greedy_key(
    state: ObservableState,
    ranked: Sequence[ObservableRequest],
    key_primitive: PlacementKeyPrimitive,
    *,
    key_params: Optional[Mapping[str, float]] = None,
    admit_filter: Optional[AdmitFilter] = None,
    max_admits: Optional[int] = None,
) -> Action:
    """Greedy best-key placement engine; see the `greedy_key_placement` registry entry."""
    admit: Dict[int, List[int]] = {g.gpu_id: [] for g in state.gpu_states}
    if not state.gpu_states:
        return Action(admit=admit)
    key_params = key_params or {}
    admitted = 0
    for req in ranked:
        if max_admits is not None and admitted >= max_admits:
            break
        feasible = [g for g in state.gpu_states if feasible_on_gpu(g, req)]
        if admit_filter is not None:
            feasible = [g for g in feasible if admit_filter(req, g)]
        if not feasible:
            continue
        best_gpu = min(feasible, key=lambda g: key_primitive.key(g, req, **key_params))
        admit[best_gpu.gpu_id].append(req.request_id)
        best_gpu.active_request_ids.append(req.request_id)
        best_gpu.current_kv_tokens += req.prompt_tokens
        admitted += 1
    return Action(admit=admit)


# ---------------------------------------------------------------------------
# BATCHING family: token-budget and admission-credit parameters
# ---------------------------------------------------------------------------

register_primitive(PrimitiveSpec(
    name="token_budget_remaining",
    family=PrimitiveFamily.BATCHING,
    input_type="ObservableGPUState",
    output_type="int",
    param_bounds={},
    doc=(
        "Batch-token slots available for new requests this step "
        "(max_batch_tokens - active_request_count); each Phase-1 request "
        "consumes exactly one slot."
    ),
    compatible_families=frozenset({PrimitiveFamily.BATCHING, PrimitiveFamily.RESOURCE_GUARD}),
))


def token_budget_remaining(gpu: ObservableGPUState) -> int:
    return gpu.max_batch_tokens - len(gpu.active_request_ids)


register_primitive(PrimitiveSpec(
    name="admission_credit_budget",
    family=PrimitiveFamily.BATCHING,
    input_type="stateful (refill/consume across steps)",
    output_type="int (max admits this step)",
    param_bounds={
        "refill_per_step": ParamBound(0.0, 1e6, 2.0),
        "max_budget": ParamBound(0.0, 1e6, 4.0),
        "cost_per_admit": ParamBound(0.0, 1e6, 1.0),
    },
    doc=(
        "Token-bucket admission-rate limiter: refills toward max_budget "
        "each step by refill_per_step, is consumed by cost_per_admit per "
        "admitted request, and caps max admits per step to "
        "max(1, int(remaining_budget)). This is the one explicitly "
        "stateful primitive in the registry (see `AdmissionCreditBudget`); "
        "every other primitive is a pure function of its inputs."
    ),
    compatible_families=frozenset({PrimitiveFamily.BATCHING, PrimitiveFamily.ADMISSION}),
))


@dataclass
class AdmissionCreditBudget:
    """Stateful token-bucket admission-rate limiter (the `admission_credit_budget` primitive)."""

    refill_per_step: float = 2.0
    max_budget: float = 4.0
    cost_per_admit: float = 1.0
    _budget: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.refill_per_step < 0 or self.max_budget < 0 or self.cost_per_admit < 0:
            raise PrimitiveError("AdmissionCreditBudget parameters must be non-negative")
        self._budget = self.max_budget

    def reset(self) -> None:
        self._budget = self.max_budget

    def refill(self) -> None:
        self._budget = min(self.max_budget, self._budget + self.refill_per_step)

    def remaining(self) -> float:
        return self._budget

    def max_admits(self) -> int:
        return max(1, int(self._budget))

    def consume(self, admitted_count: int) -> None:
        self._budget = max(0.0, self._budget - admitted_count * self.cost_per_admit)

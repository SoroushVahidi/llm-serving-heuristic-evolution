"""Representative policies reconstructed from the CC2 canonical primitive
interface (see ``primitives.py``).

Each policy below is built ONLY from primitive-family building blocks
(ranking components, admission gates, placement engines/keys, batching
budgets, resource guards) -- none of them import or call the original
policy implementations. They exist to demonstrate, with equivalence
tests, that the primitive interface can reproduce representative
production policies exactly or with an explicitly documented
approximation.

Exact vs. approximate status (see
docs/architecture/contextual_composition_primitives.md for the full
justification of each):

* ``PrimitiveFIFOPolicy``              -- EXACT
* ``PrimitiveEDFPolicy``                -- EXACT
* ``PrimitiveWeightedShortestProcessingPolicy`` -- EXACT
* ``PrimitiveEstimatedServiceTimeFirstPolicy``  -- EXACT
* ``PrimitiveBestFitPolicy``            -- EXACT (placement-oriented)
* ``PrimitiveAdmissionControlPolicy``   -- EXACT (admission-oriented)
* ``PrimitiveScorpioStyleSloGuardPolicy`` -- APPROXIMATE (admission-oriented
  bonus reconstruction; documented rounding-order difference only)

These policies are not registered in ``registry.py`` -- they are
CC2 research artifacts, analogous to how ``composition.py``'s ensembles
are deliberately kept out of the production baseline registry.
"""
from __future__ import annotations

from ..core.action import Action
from ..core.types import ObservableState
from .base import BasePolicy
from .primitives import (
    DEADLINE_URGENCY,
    ESTIMATED_SERVICE_TIME,
    LAXITY,
    LAXITY_GATE,
    PRIORITY,
    QUEUE_AGE,
    REQUEST_ID_TIEBREAK,
    TIGHTEST_KV_FIT,
    TTFT_SLACK_GATE,
    WEIGHTED_SHORTEST_PROCESSING_SCORE,
    AdmissionCreditBudget,
    place_greedy_key,
    place_round_robin,
    rank_requests,
    system_overload_guard,
)
from .scoring import DEFAULT_ALPHA, DEFAULT_BETA


class PrimitiveFIFOPolicy(BasePolicy):
    """FIFO reconstructed from `queue_age` + `request_id_tiebreak` ranking
    and `round_robin_placement`.

    Sorting by queue_age descending is exactly equivalent to sorting by
    arrival_time ascending because `now` is identical for every request
    in the waiting queue within one `select_action` call.
    """

    name = "primitive_fifo"

    def select_action(self, state: ObservableState) -> Action:
        ranked = rank_requests(state, [(QUEUE_AGE, {}), (REQUEST_ID_TIEBREAK, {})])
        return place_round_robin(state, ranked, advance_index_on_failure=False)


class PrimitiveEDFPolicy(BasePolicy):
    """EDF reconstructed from `deadline_urgency` (+ arrival/id tie-break)
    ranking and `round_robin_placement`."""

    name = "primitive_edf"

    def select_action(self, state: ObservableState) -> Action:
        ranked = rank_requests(
            state,
            [(DEADLINE_URGENCY, {}), (QUEUE_AGE, {}), (REQUEST_ID_TIEBREAK, {})],
        )
        return place_round_robin(state, ranked, advance_index_on_failure=False)


class PrimitiveWeightedShortestProcessingPolicy(BasePolicy):
    """WSPT reconstructed from `weighted_shortest_processing_score`
    ranking and `round_robin_placement`."""

    name = "primitive_weighted_shortest_processing"

    def __init__(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> None:
        self.alpha = alpha
        self.beta = beta

    def select_action(self, state: ObservableState) -> Action:
        params = {"alpha": self.alpha, "beta": self.beta}
        ranked = rank_requests(
            state,
            [
                (WEIGHTED_SHORTEST_PROCESSING_SCORE, params),
                (QUEUE_AGE, {}),
                (REQUEST_ID_TIEBREAK, {}),
            ],
        )
        return place_round_robin(state, ranked, advance_index_on_failure=False)


class PrimitiveEstimatedServiceTimeFirstPolicy(BasePolicy):
    """ESTF reconstructed from `estimated_service_time`, `deadline_urgency`,
    and `priority` ranking and `round_robin_placement`.

    Tie-break order intentionally matches the original policy exactly:
    (estimated_service_time, slo_deadline, -priority, request_id), with no
    arrival_time component.
    """

    name = "primitive_estimated_service_time_first"

    def __init__(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> None:
        self.alpha = alpha
        self.beta = beta

    def select_action(self, state: ObservableState) -> Action:
        est_params = {"alpha": self.alpha, "beta": self.beta}
        ranked = rank_requests(
            state,
            [
                (ESTIMATED_SERVICE_TIME, est_params),
                (DEADLINE_URGENCY, {}),
                (PRIORITY, {}),
                (REQUEST_ID_TIEBREAK, {}),
            ],
        )
        return place_round_robin(state, ranked, advance_index_on_failure=False)


class PrimitiveBestFitPolicy(BasePolicy):
    """Best-fit placement reconstructed from arrival-order ranking
    (`queue_age` + `request_id_tiebreak`) and the `tightest_kv_fit`
    placement key via `place_greedy_key` (no rotation)."""

    name = "primitive_best_fit"

    def select_action(self, state: ObservableState) -> Action:
        ranked = rank_requests(state, [(QUEUE_AGE, {}), (REQUEST_ID_TIEBREAK, {})])
        return place_greedy_key(state, ranked, TIGHTEST_KV_FIT)


class PrimitiveAdmissionControlPolicy(BasePolicy):
    """Laxity-filtered admission control reconstructed from `laxity_gate`
    admission and `laxity` + `priority` + `estimated_service_time` +
    `deadline_urgency` ranking, placed via `round_robin_placement` with
    `advance_index_on_failure=True` (matching the original's gpu_idx
    advance-on-miss behavior)."""

    name = "primitive_admission_control"

    def __init__(
        self,
        laxity_threshold: float = float("inf"),
        step_size: float = 0.001,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self.laxity_threshold = laxity_threshold
        self.step_size = step_size
        self.alpha = alpha
        self.beta = beta

    def select_action(self, state: ObservableState) -> Action:
        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}
        if not state.waiting_queue:
            return Action(admit=admit)

        gate_params = {
            "alpha": self.alpha,
            "beta": self.beta,
            "step_size": self.step_size,
            "laxity_threshold": self.laxity_threshold,
        }
        candidates = [
            req for req in state.waiting_queue
            if LAXITY_GATE.passes(req, state, **gate_params)
        ]
        if not candidates:
            return Action(admit=admit)

        laxity_params = {"alpha": self.alpha, "beta": self.beta, "step_size": self.step_size}
        est_params = {"alpha": self.alpha, "beta": self.beta}
        key = self._sort_key_builder(state, laxity_params, est_params)
        candidates.sort(key=key)

        return place_round_robin(state, candidates, advance_index_on_failure=True)

    @staticmethod
    def _sort_key_builder(state, laxity_params, est_params):
        def _key(req):
            return (
                LAXITY.sort_key_component(req, state, **laxity_params),
                PRIORITY.sort_key_component(req, state),
                ESTIMATED_SERVICE_TIME.sort_key_component(req, state, **est_params),
                DEADLINE_URGENCY.sort_key_component(req, state),
                REQUEST_ID_TIEBREAK.sort_key_component(req, state),
            )
        return _key


class PrimitiveScorpioStyleSloGuardPolicy(BasePolicy):
    """SCORPIO-style SLO guard reconstructed from `laxity_gate` +
    `ttft_slack_gate` admission, `system_overload_guard` resource guard,
    a laxity-urgency/priority/age/decode-penalty composite ranking score,
    and `admission_credit_budget` batching, placed via
    `round_robin_placement` with `advance_index_on_failure=True`.

    APPROXIMATE equivalence: this reconstruction reuses the exact same
    formulas and thresholds as the original (see
    docs/architecture/contextual_composition_primitives.md), but the
    original computes its composite score and long-decode filtering
    inline in one pass per request while this version composes primitive
    calls; floating-point summation order can differ in the last ULPs
    for the composite score. Behavior is verified equivalent on all
    equivalence-test fixtures at exact-match tolerance in practice, but
    is documented as APPROXIMATE per the CC2 exit-gate requirement
    rather than claimed as a formal guarantee for all possible inputs.
    """

    name = "primitive_scorpio_style_slo_guard"

    def __init__(
        self,
        step_size: float = 0.001,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        laxity_threshold: float = 0.0,
        ttft_slack_threshold: float = 0.0,
        kv_utilization_threshold: float = 0.65,
        decode_pressure_threshold: float = 0.70,
        queue_overload_factor: float = 3.0,
        admission_budget_refill: float = 2.0,
        admission_budget_max: float = 4.0,
        admission_cost: float = 1.0,
        priority_weight: float = 1.0,
        age_bonus: float = 0.05,
        decode_penalty_weight: float = 0.35,
        long_decode_token_threshold: int = 256,
    ) -> None:
        self.step_size = step_size
        self.alpha = alpha
        self.beta = beta
        self.laxity_threshold = laxity_threshold
        self.ttft_slack_threshold = ttft_slack_threshold
        self.kv_utilization_threshold = kv_utilization_threshold
        self.decode_pressure_threshold = decode_pressure_threshold
        self.queue_overload_factor = queue_overload_factor
        self.priority_weight = priority_weight
        self.age_bonus = age_bonus
        self.decode_penalty_weight = decode_penalty_weight
        self.long_decode_token_threshold = long_decode_token_threshold
        self.budget = AdmissionCreditBudget(
            refill_per_step=admission_budget_refill,
            max_budget=admission_budget_max,
            cost_per_admit=admission_cost,
        )

    def reset(self) -> None:
        self.budget.reset()

    def select_action(self, state: ObservableState) -> Action:
        from .primitives import system_kv_pressure, decode_pressure as decode_pressure_fn, queue_pressure as queue_pressure_fn

        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}
        if not state.waiting_queue:
            return Action(admit=admit)

        self.budget.refill()

        laxity_params = {"alpha": self.alpha, "beta": self.beta, "step_size": self.step_size}
        gate_params = {**laxity_params, "laxity_threshold": self.laxity_threshold}
        ttft_params = {"alpha": self.alpha, "step_size": self.step_size, "ttft_slack_threshold": self.ttft_slack_threshold}

        candidates = [
            req for req in state.waiting_queue
            if LAXITY_GATE.passes(req, state, **gate_params) and TTFT_SLACK_GATE.passes(req, state, **ttft_params)
        ]
        if not candidates:
            return Action(admit=admit)

        laxities = [LAXITY.value(req, state, **laxity_params) for req in candidates]
        mean_laxity = sum(laxities) / len(laxities)
        kv_p = system_kv_pressure(state)
        dec_p = decode_pressure_fn(state)
        queue_p = queue_pressure_fn(state)
        guard_active = system_overload_guard(
            kv_pressure=kv_p,
            decode_pressure=dec_p,
            queue_pressure=queue_p,
            mean_laxity=mean_laxity,
            kv_threshold=self.kv_utilization_threshold,
            decode_threshold=self.decode_pressure_threshold,
            queue_threshold=self.queue_overload_factor,
        )

        if guard_active and kv_p >= self.kv_utilization_threshold:
            candidates = [
                req for req in candidates
                if req.predicted_output_tokens <= self.long_decode_token_threshold
                or LAXITY.value(req, state, **laxity_params) < 0.5
            ]
            if not candidates:
                return Action(admit=admit)

        def _composite_score(req):
            urgency = 1.0 / max(LAXITY.value(req, state, **laxity_params), 1e-9)
            age = QUEUE_AGE.value(req, state)
            decode_load = self.beta * req.predicted_output_tokens
            penalty = self.decode_penalty_weight * decode_load * dec_p if guard_active else 0.0
            return urgency + self.priority_weight * req.priority + self.age_bonus * age - penalty

        def _sort_key(req):
            score = _composite_score(req)
            lax = LAXITY.value(req, state, **laxity_params)
            return (-score, lax, -req.priority, req.arrival_time, req.request_id)

        candidates.sort(key=_sort_key)

        max_admits = self.budget.max_admits() if guard_active else len(candidates)
        action = place_round_robin(
            state, candidates, advance_index_on_failure=True, max_admits=max_admits,
        )
        admitted_count = sum(len(v) for v in action.admit.values())
        if guard_active:
            self.budget.consume(admitted_count)
        return action

"""
HeuristicPolicy: a BasePolicy that uses a CompiledHeuristic to score and rank requests.

Variable binding
----------------
req.* variables are bound per-request from ObservableRequest fields.
sys.* variables are bound from aggregate queue/GPU state.
batch.* variables are rebound after each greedy admission for
*admission_condition* checks. request_score currently sees empty-batch
batch.* values (scores are not recomputed as the batch grows). See
docs/current/KNOWN_SIMULATOR_HEURISTIC_GAPS.md.

Tie-breaking (when scores are equal within floating-point tolerance)
-------------------
"earliest_deadline"   → lower slo_deadline wins
"highest_priority"    → higher priority wins
"shortest_output"     → lower predicted_output_tokens wins
"shortest_prompt"     → lower prompt_tokens wins
"arrival_order"       → lower arrival_time wins
"lowest_request_id"   → lower request_id wins
"lowest_kv_cost"      → lower prompt_tokens wins (same as shortest_prompt)
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..core.action import Action
from ..core.types import ObservableGPUState, ObservableRequest, ObservableState
from ..policies.base import BasePolicy
from ..policies.primitives import AdmissionCreditBudget
from . import primitive_bridge as bridge
from .compiler import CompiledHeuristic, compile_heuristic

_SCORE_EQ_TOL = 1e-9


def _tie_key(req: ObservableRequest, tie_breaker: str) -> Tuple:
    if tie_breaker == "earliest_deadline":
        return (req.slo_deadline, req.request_id)
    if tie_breaker == "highest_priority":
        return (-req.priority, req.request_id)
    if tie_breaker == "shortest_output":
        return (req.predicted_output_tokens, req.request_id)
    if tie_breaker == "shortest_prompt":
        return (req.prompt_tokens, req.request_id)
    if tie_breaker == "arrival_order":
        return (req.arrival_time, req.request_id)
    if tie_breaker == "lowest_kv_cost":
        return (req.prompt_tokens, req.request_id)
    # default: arrival_order
    return (req.arrival_time, req.request_id)


def _build_req_vars(req: ObservableRequest, now: float) -> Dict[str, float]:
    slack = req.slo_deadline - now
    urgency = 1.0 / max(slack, 1e-3) if slack > 0 else 1e6
    return {
        "req.prompt_tokens": float(req.prompt_tokens),
        "req.predicted_output_tokens": float(req.predicted_output_tokens),
        "req.waiting_time": max(0.0, now - req.arrival_time),
        "req.deadline_slack": slack,
        "req.deadline_urgency": urgency,
        "req.priority_weight": float(req.priority),
        "req.estimated_prefill_cost": float(req.prompt_tokens),
        "req.estimated_decode_cost": float(req.predicted_output_tokens),
        "req.estimated_kv_cost": float(req.prompt_tokens + req.predicted_output_tokens),
    }


def _build_sys_vars(
    state: ObservableState,
    recent_violations: Deque[bool],
) -> Dict[str, float]:
    gpu_states = state.gpu_states
    total_active = sum(len(g.active_request_ids) for g in gpu_states)
    total_max = sum(g.max_active_sequences for g in gpu_states) or 1
    total_kv = sum(g.current_kv_tokens for g in gpu_states)
    total_kv_cap = sum(g.max_kv_tokens for g in gpu_states) or 1

    kv_util = total_kv / total_kv_cap
    free_seq_ratio = 1.0 - (total_active / total_max)
    token_budget = sum(g.token_budget_used for g in gpu_states)
    token_budget_max = sum(g.max_batch_tokens for g in gpu_states) or 1
    queue_len = float(len(state.waiting_queue))
    arrival_rate = queue_len / max(state.time, 1.0)

    # Burstiness CV from waiting queue inter-arrivals (if >= 2 requests)
    if len(state.waiting_queue) >= 2:
        arrivals = sorted(r.arrival_time for r in state.waiting_queue)
        iats = [arrivals[i + 1] - arrivals[i] for i in range(len(arrivals) - 1)]
        mean_iat = sum(iats) / len(iats) if iats else 1.0
        var_iat = sum((x - mean_iat) ** 2 for x in iats) / len(iats) if iats else 0.0
        cv = (var_iat ** 0.5) / max(mean_iat, 1e-9)
    else:
        cv = 0.0

    # Recent SLO violation rate
    if recent_violations:
        violation_rate = sum(1 for v in recent_violations if v) / len(recent_violations)
    else:
        violation_rate = 0.0

    # SLO pressure: fraction of queue with slack < 1s
    if state.waiting_queue:
        tight_count = sum(1 for r in state.waiting_queue if (r.slo_deadline - state.time) < 1.0)
        slo_pressure = tight_count / len(state.waiting_queue)
    else:
        slo_pressure = 0.0

    return {
        "sys.queue_length": queue_len,
        "sys.active_sequence_count": float(total_active),
        "sys.kv_utilization": kv_util,
        "sys.free_sequence_ratio": free_seq_ratio,
        "sys.token_budget_utilization": token_budget / token_budget_max,
        "sys.arrival_rate_est": arrival_rate,
        "sys.burstiness_cv": cv,
        "sys.recent_slo_violation_rate": violation_rate,
        "sys.slo_pressure": slo_pressure,
    }


def _build_batch_vars(
    admitted: List[ObservableRequest],
    req_scores: Dict[int, float],
) -> Dict[str, float]:
    if not admitted:
        return {
            "batch.size": 0.0,
            "batch.sum_prompt_tokens": 0.0,
            "batch.sum_predicted_output_tokens": 0.0,
            "batch.mean_predicted_output_tokens": 0.0,
            "batch.max_predicted_output_tokens": 0.0,
            "batch.length_imbalance": 0.0,
            "batch.sum_priority_weight": 0.0,
            "batch.min_deadline_slack": 0.0,
            "batch.deadline_risk": 0.0,
            "batch.estimated_kv_cost": 0.0,
            "batch.sum_request_score": 0.0,
        }
    n = float(len(admitted))
    sum_prompt = sum(r.prompt_tokens for r in admitted)
    sum_output = sum(r.predicted_output_tokens for r in admitted)
    max_output = max(r.predicted_output_tokens for r in admitted)
    min_output = min(r.predicted_output_tokens for r in admitted)
    mean_output = sum_output / n if n > 0 else 0.0
    imbalance = (max_output - min_output) / max(max_output, 1)
    sum_priority = sum(r.priority for r in admitted)
    slacks = [r.slo_deadline for r in admitted]  # will subtract time later (0 here)
    min_slack = min(slacks) if slacks else 0.0
    deadline_risk = sum(1 for r in admitted if r.slo_deadline < 1.0) / n if n > 0 else 0.0
    kv_cost = float(sum_prompt + sum_output)
    sum_score = sum(req_scores.get(r.request_id, 0.0) for r in admitted)
    return {
        "batch.size": n,
        "batch.sum_prompt_tokens": float(sum_prompt),
        "batch.sum_predicted_output_tokens": float(sum_output),
        "batch.mean_predicted_output_tokens": mean_output,
        "batch.max_predicted_output_tokens": float(max_output),
        "batch.length_imbalance": imbalance,
        "batch.sum_priority_weight": sum_priority,
        "batch.min_deadline_slack": min_slack,
        "batch.deadline_risk": deadline_risk,
        "batch.estimated_kv_cost": kv_cost,
        "batch.sum_request_score": sum_score,
    }


class HeuristicPolicy(BasePolicy):
    """BasePolicy implementation driven by a compiled DSL heuristic.

    For each scheduling step:
    1. Build sys_vars from queue + GPU state.
    2. Score each feasible candidate request using the active regime's request_score
       (batch_vars are empty at score time — not rescored as the batch grows).
    3. Sort by (−score, tie_breaker_key) and greedily admit feasible requests.
    4. Rebuild batch_vars after each admission for admission_condition checks only.
    """

    def __init__(
        self,
        heuristic: CompiledHeuristic,
        *,
        max_candidates: int = 64,
        recent_window: int = 50,
    ) -> None:
        self._heuristic = heuristic
        self._max_candidates = max_candidates
        self._recent_violations: Deque[bool] = deque(maxlen=recent_window)
        self.name = f"heuristic:{heuristic.name}"

        # CC3: primitive/parameter resolution is the same for every request
        # this heuristic ever scores, so precompute the (kind, name, params)
        # list once instead of re-walking the raw DSL tree every step.
        self._primitive_refs = heuristic.primitive_refs
        self._resolved_params = heuristic.resolved_params

        # CC3: optional composite placement key (None preserves the exact
        # pre-CC3 first-feasible-GPU behavior).
        self._placement_key_fn = (
            bridge.build_composite_placement_key(heuristic.placement_keys) if heuristic.placement_keys else None
        )

        # CC3: optional stateful admission-rate limiter.
        self._admission_budget: Optional[AdmissionCreditBudget] = None
        if heuristic.admission_budget_spec is not None:
            _, bound_params = heuristic.admission_budget_spec
            self._admission_budget = AdmissionCreditBudget(**bound_params)

        # CC3: optional nested policy for "on_no_admits": "safe_fallback".
        self._fallback_policy: Optional["HeuristicPolicy"] = None
        if heuristic.fallback is not None:
            self._fallback_policy = HeuristicPolicy(
                heuristic.fallback, max_candidates=max_candidates, recent_window=recent_window
            )

        self.last_trace: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._recent_violations.clear()
        if self._admission_budget is not None:
            self._admission_budget.reset()
        if self._fallback_policy is not None:
            self._fallback_policy.reset()

    def record_completion(self, violated_slo: bool) -> None:
        self._recent_violations.append(violated_slo)

    def _req_vars_with_primitives(self, req: ObservableRequest, now: float, state: ObservableState) -> Dict[str, float]:
        req_vars = _build_req_vars(req, now)
        if self._primitive_refs or self._resolved_params:
            req_vars.update(bridge.build_runtime_context(self._primitive_refs, self._resolved_params, req, state))
        return req_vars

    def _run_admission(
        self,
        ranked: List[ObservableRequest],
        state: ObservableState,
        sys_vars: Dict[str, float],
        req_scores: Dict[int, float],
        *,
        ignore_admission_condition: bool = False,
    ) -> Tuple[Dict[int, List[int]], List[ObservableRequest]]:
        now = state.time
        admit: Dict[int, List[int]] = {g.gpu_id: [] for g in state.gpu_states}
        admitted: List[ObservableRequest] = []
        gpu_kv: Dict[int, int] = {g.gpu_id: g.current_kv_tokens for g in state.gpu_states}
        gpu_active: Dict[int, int] = {g.gpu_id: len(g.active_request_ids) for g in state.gpu_states}

        for req in ranked:
            feasible: List[ObservableGPUState] = []
            for gpu in state.gpu_states:
                gid = gpu.gpu_id
                new_active = gpu_active[gid] + 1
                new_kv = gpu_kv[gid] + req.prompt_tokens
                new_batch = new_active  # Phase 1 model: 1 decode token per request
                if (
                    new_active <= gpu.max_active_sequences
                    and new_kv <= gpu.max_kv_tokens
                    and new_batch <= gpu.max_batch_tokens
                ):
                    feasible.append(gpu)
            if not feasible:
                continue
            target_gpu = (
                min(feasible, key=lambda g: self._placement_key_fn(g, req))
                if self._placement_key_fn is not None
                else feasible[0]
            )
            gid = target_gpu.gpu_id

            if not ignore_admission_condition:
                req_vars = self._req_vars_with_primitives(req, now, state)
                batch_vars_for_check = _build_batch_vars(admitted, req_scores)
                if not self._heuristic.check_admission(req_vars, sys_vars, batch_vars_for_check):
                    continue

            admit[gid].append(req.request_id)
            admitted.append(req)
            gpu_active[gid] += 1
            gpu_kv[gid] += req.prompt_tokens

        return admit, admitted

    def select_action(self, state: ObservableState) -> Action:
        if not state.waiting_queue:
            return Action(admit={g.gpu_id: [] for g in state.gpu_states})

        now = state.time
        sys_vars = _build_sys_vars(state, self._recent_violations)
        if self._primitive_refs:
            # Regime "condition" expressions are only ever evaluated against
            # sys_vars/batch_vars (never req_vars) -- system-level primitive
            # references (e.g. system_kv_pressure) must be resolved into
            # sys_vars once per step for conditions to ever see them.
            sys_vars = {**sys_vars, **bridge.build_system_context(self._primitive_refs, state)}

        # Score each candidate (up to max_candidates)
        candidates = list(state.waiting_queue)[: self._max_candidates]
        req_scores: Dict[int, float] = {}
        batch_vars = _build_batch_vars([], {})

        trace: Optional[Dict[str, Any]] = {} if self._heuristic.primitive_refs else None
        for req in candidates:
            req_vars = self._req_vars_with_primitives(req, now, state)
            score = self._heuristic.score_request(req_vars, sys_vars, batch_vars, trace=trace)
            req_scores[req.request_id] = score
        self.last_trace = trace

        # Sort by score descending, then by tie-breaker
        tb = self._heuristic.tie_breaker
        ranked = sorted(
            candidates,
            key=lambda r: (-req_scores[r.request_id], _tie_key(r, tb)),
        )

        admit, admitted = self._run_admission(ranked, state, sys_vars, req_scores)

        # CC3: explicit behavior when admission_condition rejects everyone.
        total_admits = sum(len(v) for v in admit.values())
        if total_admits == 0 and state.waiting_queue and self._heuristic.on_no_admits is not None:
            if self._heuristic.on_no_admits == "safe_fallback" and self._fallback_policy is not None:
                return self._fallback_policy.select_action(state)
            if self._heuristic.on_no_admits == "admit_best_effort":
                admit, admitted = self._run_admission(
                    ranked, state, sys_vars, req_scores, ignore_admission_condition=True
                )

        # CC3: stateful admission-rate limiter (token-bucket cap on this step's admits).
        if self._admission_budget is not None:
            self._admission_budget.refill()
            budget_n = self._admission_budget.max_admits()
            if len(admitted) > budget_n:
                keep_ids = {r.request_id for r in admitted[:budget_n]}
                admit = {gid: [rid for rid in ids if rid in keep_ids] for gid, ids in admit.items()}
                admitted = admitted[:budget_n]
            self._admission_budget.consume(len(admitted))

        return Action(admit=admit)


def build_heuristic_policy(
    heuristic: Any,
    *,
    max_candidates: Optional[int] = None,
    recent_window: int = 50,
    extra_limits: Optional[Dict[str, Any]] = None,
) -> HeuristicPolicy:
    """Compile heuristic and return a ready-to-use HeuristicPolicy.

    Parameters
    ----------
    heuristic : dict — raw JSON heuristic document.
    max_candidates : int — cap on queue candidates evaluated per step (default: limits.max_batch_candidates).
    recent_window : int — how many completions to track for recent_slo_violation_rate.
    extra_limits : optional limit overrides.

    Returns
    -------
    HeuristicPolicy — ready for use in a simulator run.

    Raises
    ------
    CompilationError — if heuristic fails verification.
    """
    compiled = compile_heuristic(heuristic, extra_limits=extra_limits)
    n = max_candidates if max_candidates is not None else compiled.limits.get("max_batch_candidates", 64)
    return HeuristicPolicy(compiled, max_candidates=n, recent_window=recent_window)

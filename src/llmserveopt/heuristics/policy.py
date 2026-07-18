"""
HeuristicPolicy: a BasePolicy that uses a CompiledHeuristic to score and rank requests.

Variable binding
----------------
req.* variables are bound per-request from ObservableRequest fields.
sys.* variables are bound from aggregate queue/GPU state.
batch.* variables are updated incrementally as requests are added to the batch.

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
from ..core.types import ObservableRequest, ObservableState
from ..policies.base import BasePolicy
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
    2. Score each feasible candidate request using the active regime's request_score.
    3. Sort by (−score, tie_breaker_key) and greedily admit feasible requests.
    4. Update batch_vars after each admission (for incremental scoring if needed).
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

    def reset(self) -> None:
        self._recent_violations.clear()

    def record_completion(self, violated_slo: bool) -> None:
        self._recent_violations.append(violated_slo)

    def select_action(self, state: ObservableState) -> Action:
        if not state.waiting_queue:
            return Action(admit={g.gpu_id: [] for g in state.gpu_states})

        now = state.time
        sys_vars = _build_sys_vars(state, self._recent_violations)

        # Score each candidate (up to max_candidates)
        candidates = list(state.waiting_queue)[: self._max_candidates]
        req_scores: Dict[int, float] = {}
        batch_vars = _build_batch_vars([], {})

        for req in candidates:
            req_vars = _build_req_vars(req, now)
            score = self._heuristic.score_request(req_vars, sys_vars, batch_vars)
            req_scores[req.request_id] = score

        # Sort by score descending, then by tie-breaker
        tb = self._heuristic.tie_breaker
        ranked = sorted(
            candidates,
            key=lambda r: (-req_scores[r.request_id], _tie_key(r, tb)),
        )

        # Greedy admission
        admit: Dict[int, List[int]] = {g.gpu_id: [] for g in state.gpu_states}
        admitted: List[ObservableRequest] = []
        # Track per-GPU running state for feasibility checks
        gpu_kv: Dict[int, int] = {g.gpu_id: g.current_kv_tokens for g in state.gpu_states}
        gpu_active: Dict[int, int] = {g.gpu_id: len(g.active_request_ids) for g in state.gpu_states}

        for req in ranked:
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
                    # Check admission_condition if defined
                    req_vars = _build_req_vars(req, now)
                    batch_vars_for_check = _build_batch_vars(admitted, req_scores)
                    if self._heuristic.check_admission(req_vars, sys_vars, batch_vars_for_check):
                        admit[gid].append(req.request_id)
                        admitted.append(req)
                        gpu_active[gid] = new_active
                        gpu_kv[gid] = new_kv
                        break
            # (If no GPU can take the request, skip it)

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

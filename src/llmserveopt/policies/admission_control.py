"""
Admission Control baseline policy.

This is a simple, deployable admission-control baseline for SLO-aware LLM
serving research.  It filters waiting requests by a laxity threshold before
scheduling, dropping requests that are already too late to meet their SLO
with high probability.

IMPORTANT: This is NOT a reproduction of Tempo, JITServe, SCORPIO, or any
other published admission-control system.  It is a simple deterministic
baseline designed to isolate the admission-control effect in simulation.

Online deployable: YES
Uses future information: NO (uses only online-observable fields)
SLO-aware: YES (laxity-based filtering)
KV/token-budget aware: YES (respects GPU capacity)

Algorithm
---------
1. Estimate service time for each waiting request:
       est = alpha * prompt_tokens + beta * predicted_output_tokens
2. Compute laxity:
       laxity = slo_deadline - now - est
3. Filter: keep only requests with laxity >= -laxity_threshold
   (requests with laxity < -threshold are already unlikely to meet their SLO
   and are skipped this step; they remain in the queue but are never admitted
   unless conditions improve).
4. Sort survivors by:
       (a) laxity ascending (most urgent first)
       (b) priority descending
       (c) estimated service time ascending
       (d) slo_deadline ascending
       (e) request_id ascending (deterministic tie-break)
5. Greedily assign each request to any GPU with sufficient capacity.

Tie-breaking is fully deterministic.  The policy is stateless between steps.
"""
from __future__ import annotations

from ..core.action import Action
from ..core.types import ObservableRequest, ObservableState
from .base import BasePolicy
from .scoring import DEFAULT_ALPHA, DEFAULT_BETA, predicted_service_proxy


class AdmissionControlPolicy(BasePolicy):
    """Laxity-filtered admission-control scheduling baseline.

    Parameters
    ----------
    laxity_threshold : float
        Requests with laxity < -laxity_threshold are skipped (treated as
        already expired or infeasible).

        Default float("inf") = no filtering (all requests are candidates;
        policy acts as urgency-sorted admission).  Set to a finite value to
        enable the admission-control filter.

        NOTE: laxity mixes units — the service proxy (alpha*prompt + beta*output)
        is in decode steps while slo_deadline/now are in seconds.  Calibrate
        this threshold against your service model's step_size when using a
        finite value.  For the default synthetic service model (step_size=0.001),
        a threshold of ~500 would admit requests whose proxy is within 500
        steps of their deadline.
    alpha : float
        Prefill cost coefficient in service-time estimate.
    beta : float
        Decode cost coefficient in service-time estimate.
    """

    name = "admission_control"

    def __init__(
        self,
        laxity_threshold: float = float("inf"),
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
    ) -> None:
        self.laxity_threshold = laxity_threshold
        self.alpha = alpha
        self.beta = beta

    def _laxity(self, req: ObservableRequest, now: float) -> float:
        est = predicted_service_proxy(req, self.alpha, self.beta)
        return req.slo_deadline - now - est

    def select_action(self, state: ObservableState) -> Action:
        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}

        if not state.waiting_queue:
            return Action(admit=admit)

        now = state.time

        # Step 1-3: filter by laxity threshold
        min_laxity = -self.laxity_threshold
        candidates = [
            req for req in state.waiting_queue
            if self._laxity(req, now) >= min_laxity
        ]

        # Step 4: sort survivors deterministically
        def sort_key(r: ObservableRequest):
            lax = self._laxity(r, now)
            est = predicted_service_proxy(r, self.alpha, self.beta)
            return (
                lax,                  # ascending: most urgent first
                -r.priority,          # descending: higher priority first
                est,                  # ascending: shorter job first
                r.slo_deadline,       # ascending: earlier deadline first
                r.request_id,         # ascending: deterministic
            )

        candidates.sort(key=sort_key)

        # Step 5: greedy GPU assignment
        gpu_idx = 0
        n_gpus = len(state.gpu_states)

        for req in candidates:
            placed = False
            for offset in range(n_gpus):
                gpu = state.gpu_states[(gpu_idx + offset) % n_gpus]
                if self._feasible_on_gpu(gpu, req):
                    admit[gpu.gpu_id].append(req.request_id)
                    gpu.active_request_ids.append(req.request_id)
                    gpu.current_kv_tokens += req.prompt_tokens
                    gpu_idx = (gpu_idx + offset + 1) % n_gpus
                    placed = True
                    break
            if not placed:
                gpu_idx = (gpu_idx + 1) % n_gpus

        return Action(admit=admit)

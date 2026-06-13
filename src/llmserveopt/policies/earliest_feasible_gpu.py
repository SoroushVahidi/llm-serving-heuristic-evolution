"""
EarliestFeasibleGPU: dispatch each request to the first GPU that can admit it.

Requests are processed FIFO.  For each request, GPUs are tried in gpu_id order
(not round-robin); the request goes to the first feasible GPU.
Simulates a static dispatching policy (e.g., round-robin with first-fit fallback).
"""
from __future__ import annotations

from ..core.action import Action
from ..core.types import ObservableState
from .base import BasePolicy
from .tie_breaking import arrival_then_id


class EarliestFeasibleGPUPolicy(BasePolicy):
    name = "earliest_feasible_gpu"

    def select_action(self, state: ObservableState) -> Action:
        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}

        queue = sorted(state.waiting_queue, key=arrival_then_id)
        gpus = sorted(state.gpu_states, key=lambda g: g.gpu_id)

        for req in queue:
            for gpu in gpus:
                if self._feasible_on_gpu(gpu, req):
                    admit[gpu.gpu_id].append(req.request_id)
                    gpu.active_request_ids.append(req.request_id)
                    gpu.current_kv_tokens += req.prompt_tokens
                    break

        return Action(admit=admit)

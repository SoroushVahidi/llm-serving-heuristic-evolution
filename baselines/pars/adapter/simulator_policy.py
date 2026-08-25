"""Simulator policy wrapper for PARS ranking scores.

Mirrors ``baselines/vllm_ltr/adapter/simulator_policy.py``'s structure and
scope exactly (see that module's docstring for the full rationale of why
an offline-precomputed score map is required rather than a live per-step
call): this simulator's ``ObservableRequest`` carries only an integer
``prompt_tokens`` count, never raw text, so PARS's BERT-based text scorer
cannot run inside ``select_action()``. Scores are precomputed once, cached,
and this policy only ever reads the cache.

Deliberately **not** registered in ``src/llmserveopt/policies/registry.py``
or any selector-candidate set -- evaluation-only, per this task's explicit
scope boundary.
"""
from __future__ import annotations

from typing import Dict

from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy

from .ranking_adapter import order_by_pars_score

#: Never a selector candidate / never in the historical registries.
SELECTOR_ELIGIBLE = False


class PARSSemanticReferencePolicy(BasePolicy):
    """Admits requests in PARS-ranked order (ascending score = shortest
    predicted response first), given precomputed offline scores."""

    name = "pars_semantic_reference"

    def __init__(self, scores: Dict[int, float]):
        self._scores = dict(scores)

    def select_action(self, state: ObservableState) -> Action:
        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}
        ordered = order_by_pars_score(state.waiting_queue, self._scores)

        gpu_idx = 0
        n_gpus = len(state.gpu_states)
        for req in ordered:
            for offset in range(n_gpus):
                gpu = state.gpu_states[(gpu_idx + offset) % n_gpus]
                if self._feasible_on_gpu(gpu, req):
                    admit[gpu.gpu_id].append(req.request_id)
                    gpu.active_request_ids.append(req.request_id)
                    gpu.current_kv_tokens += req.prompt_tokens
                    gpu_idx = (gpu_idx + offset + 1) % n_gpus
                    break

        return Action(admit=admit)

"""Simulator policy wrapper for vLLM-LTR ranking scores.

**Not a live, self-contained policy.** See ``PROVENANCE.md``'s "Critical
structural finding": the official predictor's only input is tokenized
prompt *text*, and this project's ``ObservableRequest``
(``src/llmserveopt/core/types.py``) carries only an integer
``prompt_tokens`` *count* -- there is no text or token-ID field anywhere in
the simulator's data model for this policy to read at ``select_action()``
time. Wiring the real predictor into the simulator's live per-step loop is
therefore not possible without first extending the simulator's request data
model (out of scope for this baseline-integration scaffold).

What this class actually does: it accepts a precomputed
``{request_id: score}`` mapping -- produced *offline*, before the simulator
run, by tokenizing each request's real prompt text and running it through
``adapter.checkpoint_loader.OPTPredictorHandle.score()`` -- and applies the
official ranking rule (``adapter.ranking_adapter.order_by_ltr_score``) to
admit requests in official-predictor-ranked order. This mirrors how the
official system is actually used in practice (score once per request,
cache it, per ``need_aux_model_score()``/``obtain_aux_scores()`` in the
pinned source) rather than recomputing a score every step.

No fallback: a request missing from ``scores`` raises
``MissingScoreError`` (via ``ranking_adapter.order_by_ltr_score``) rather
than silently being deprioritized/admitted by some other rule. This
guarantees the policy never uses ``actual_output_tokens`` or any other
oracle information -- it only ever consults the externally-supplied score
map and each GPU's feasibility state.

Deliberately **not** registered in ``src/llmserveopt/policies/registry.py``,
``external_baselines_registry.py``, or any selector candidate set (per the
task's explicit "do not add vLLM-LTR to the main selector candidate set
yet" instruction) -- it cannot be evaluated on the same footing as the
existing policies until every request in a trace has a real, offline
precomputed score, which the simulator's synthetic workload generators do
not currently produce (they don't carry prompt text either).
"""
from __future__ import annotations

from typing import Dict

from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy

from .ranking_adapter import order_by_ltr_score

#: Never a selector candidate / never in the historical registries -- see
#: module docstring and docs/audits/vllm_ltr_baseline_audit_20260804.md.
SELECTOR_ELIGIBLE = False


class VLLMLTRSemanticReferencePolicy(BasePolicy):
    """Admits requests in official vLLM-LTR ranked order, given precomputed
    offline scores. See module docstring for why this requires externally
    supplied scores rather than reading them from ``ObservableState``."""

    name = "vllm_ltr_semantic_reference"

    def __init__(self, scores: Dict[int, float]):
        self._scores = dict(scores)

    def select_action(self, state: ObservableState) -> Action:
        admit: dict[int, list[int]] = {g.gpu_id: [] for g in state.gpu_states}
        ordered = order_by_ltr_score(state.waiting_queue, self._scores)

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

"""Variant B ("matched-admission FIFO") for the VTC fairness-benchmark
repair: FIFO ordering under EXACTLY the same admission-feasibility rule
official VTC uses.

Rather than hand-writing a "matched" feasibility check and risking a
subtle mismatch from the real one, this reuses the official, unmodified
``ReqQueue`` base class directly -- `VTCReqQueue` (see
``simulator_policy.py``) is itself a subclass of ``ReqQueue`` that
overrides only ``generate_new_batch``'s ORDERING (min-served-first instead
of arrival order) while inheriting its admission gate
(``_can_add_new_req``) and batch-token-budget check verbatim.
``ReqQueue.generate_new_batch`` on its own, with no VTC subclassing at
all, IS a plain FCFS-by-arrival scheduler using that exact same gate --
the official repository's own FCFS reference implementation, real code
its own experiments run as the ``slora`` baseline scheduler (see
``fair_bench/README.md``'s "FCFS scheduler" instructions). Using it here
means variant B's admission semantics are bit-for-bit identical to
variant A's by construction, not merely "the same formula copied
elsewhere" -- there is no way for the two to drift apart.

Does NOT use ``self.served``/``self.fairw``/tenant fair-weights at all --
``ReqQueue`` has no such concept; ordering is pure FIFO on
``self.waiting_req_list``, exactly as ``append()`` (base, unmodified)
appends to it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.base import BasePolicy

from .errors import MissingTenantIdError, UnsupportedTopologyError
from .official_loader import load_vtc_official_classes
from .simulator_policy import default_tenant_of

SELECTOR_ELIGIBLE = False


class MatchedAdmissionFIFOPolicy(BasePolicy):
    """FIFO ordering (official, unmodified ``ReqQueue.generate_new_batch``)
    under the same feasibility gate official VTC uses. See module
    docstring. Tenant labeling is for REPORTING/per-tenant-metrics
    purposes only -- FIFO itself never reorders by tenant."""

    name = "matched_admission_fifo"

    def __init__(
        self,
        tenant_of: Optional[callable] = None,
        clone_path: Optional[str] = None,
        batch_token_budget_override: Optional[int] = None,
    ):
        official = load_vtc_official_classes(clone_path)
        self._Req = official.Req
        self._Batch = official.Batch
        self._SamplingParams = official.SamplingParams
        self._ReqQueue = official.ReqQueue
        self._tenant_of = tenant_of or default_tenant_of
        self._batch_token_budget_override = batch_token_budget_override

        self._q = None
        self._known_request_ids: set = set()
        self._req_tenant: Dict[int, str] = {}
        self.reset()

    def reset(self) -> None:
        self._q = self._ReqQueue(max_total_tokens=1, batch_max_tokens=1, running_max_req_size=1)
        self._known_request_ids = set()
        self._req_tenant = {}

    def _make_req(self, request_id: int, tenant: str, prompt_tokens: int, max_new_tokens: int):
        sp = self._SamplingParams(max_new_tokens=max(1, max_new_tokens))
        return self._Req(tenant, request_id, [0] * max(1, prompt_tokens), sp)

    def _register_new_arrivals(self, waiting_queue: Sequence[ObservableRequest]) -> None:
        for req in waiting_queue:
            if req.request_id in self._known_request_ids:
                continue
            tenant = self._tenant_of(req)
            if tenant is None:
                raise MissingTenantIdError(
                    f"Request {req.request_id} has no tenant id under the "
                    "configured tenant-mapping function."
                )
            self._req_tenant[req.request_id] = tenant
            official_req = self._make_req(
                req.request_id, tenant, req.prompt_tokens, req.predicted_output_tokens
            )
            self._q.append(official_req)
            self._known_request_ids.add(req.request_id)

    def _build_current_batch(self, gpu: ObservableGPUState):
        if not gpu.active_requests_info:
            return None
        reqs = []
        for info in gpu.active_requests_info:
            tenant = self._req_tenant.get(info.request_id)
            if tenant is None:
                continue
            stub = self._make_req(info.request_id, tenant, info.prompt_tokens, info.predicted_output_tokens)
            decoded = gpu.tokens_decoded_per_request.get(info.request_id, 0)
            stub.output_ids = [0] * decoded
            reqs.append(stub)
        return self._Batch("current", reqs) if reqs else None

    def select_action(self, state: ObservableState) -> Action:
        if len(state.gpu_states) != 1:
            raise UnsupportedTopologyError(
                f"MatchedAdmissionFIFOPolicy supports exactly 1 GPU (matching "
                f"the VTCFairnessPolicy comparison point); got {len(state.gpu_states)}."
            )
        gpu = state.gpu_states[0]

        self._register_new_arrivals(state.waiting_queue)

        self._q.max_total_tokens = gpu.max_kv_tokens
        self._q.batch_max_tokens = (
            self._batch_token_budget_override
            if self._batch_token_budget_override is not None
            else gpu.max_batch_tokens
        )
        self._q.running_max_req_size = gpu.max_active_sequences

        current_batch = self._build_current_batch(gpu)
        lora_ranks = {t: 0 for t in set(self._req_tenant.values())}
        new_batch = self._q.generate_new_batch(current_batch, lora_ranks)

        admit_ids: List[int] = []
        if new_batch is not None:
            admit_ids = [r.request_id for r in new_batch.reqs]

        return Action(admit={gpu.gpu_id: admit_ids})

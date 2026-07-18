"""
GPU-level state management within the simulator.

Phase 1.5 changes
-----------------
* admit() now accepts a ServiceModel to initialise prefill_remaining.
* step() now handles prefill / decode phases when enable_prefill_modeling=True.
* to_observable() exposes prefilling_count and decoding_count so that
  serving-style policies (Sarathi, SplitFuse) can reason about phase.
* KV tracking uses InternalRequest.kv_tokens, which grows during prefill.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

from ..core.types import CompletedRequest, GPUConfig, ObservableGPUState, ObservableRequest
from .constraints import check_admission, incremental_feasible
from .request import InternalRequest, RequestPhase
from .service_model import ServiceModel


class GPUState:
    def __init__(self, config: GPUConfig) -> None:
        self.config = config
        self._active: Dict[int, InternalRequest] = {}
        self.step_active_counts: List[int] = []
        self.step_kv_used: List[int] = []

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def gpu_id(self) -> int:
        return self.config.gpu_id

    @property
    def active_requests(self) -> List[InternalRequest]:
        return list(self._active.values())

    @property
    def num_active(self) -> int:
        return len(self._active)

    @property
    def current_kv_tokens(self) -> int:
        return sum(r.kv_tokens for r in self._active.values())

    @property
    def current_batch_tokens(self) -> int:
        return len(self._active)

    @property
    def utilization(self) -> float:
        if self.config.max_active_sequences == 0:
            return 0.0
        return self.num_active / self.config.max_active_sequences

    @property
    def prefilling_count(self) -> int:
        return sum(1 for r in self._active.values() if r.is_prefilling)

    @property
    def decoding_count(self) -> int:
        return sum(1 for r in self._active.values() if r.is_decoding)

    # ------------------------------------------------------------------ #
    # Admission
    # ------------------------------------------------------------------ #

    def can_admit(self, req: InternalRequest) -> bool:
        return incremental_feasible(self.config, self.active_requests, req)

    def admit(
        self,
        req: InternalRequest,
        admission_time: float,
        service_model: Optional[ServiceModel] = None,
    ) -> bool:
        """Admit a request.  Returns False and warns if constraints violated.

        When service_model is provided, initialises prefill_remaining per the
        configured prefill cost.  When service_model is None (or Phase 1 mode),
        prefill_remaining stays 0 (instant prefill).
        """
        violations = check_admission(self.config, self.active_requests, [req])
        if violations:
            warnings.warn(
                f"GPU {self.gpu_id}: admission of request {req.request_id} rejected: "
                + "; ".join(violations)
            )
            return False
        req.phase = RequestPhase.ACTIVE
        req.gpu_id = self.gpu_id
        req.admission_time = admission_time
        if service_model is not None:
            req.prefill_remaining = service_model.compute_prefill_tokens(
                req.request.prompt_tokens
            )
        else:
            req.prefill_remaining = 0   # Phase 1: instant prefill
        self._active[req.request_id] = req
        return True

    # ------------------------------------------------------------------ #
    # Eviction (preemption)
    # ------------------------------------------------------------------ #

    def evict(self, request_id: int) -> Optional[InternalRequest]:
        """Forcibly remove an ACTIVE request and reset it to a clean
        WAITING-equivalent state (recompute-on-resume semantics: all decode/
        prefill progress is discarded, matching vLLM's recompute preemption
        mode -- see docs/vllm_faithful_scheduler_reference.md).

        Returns the InternalRequest (for the caller to re-enqueue) or None
        if request_id is not currently active on this GPU. Backward
        compatible: only invoked when an Action's `preempt` field is
        non-empty, which no existing policy ever sets.
        """
        req = self._active.pop(request_id, None)
        if req is None:
            return None
        req.phase = RequestPhase.WAITING
        req.gpu_id = -1
        req.admission_time = -1.0
        req.tokens_decoded = 0
        req.prefill_remaining = 0
        req.first_token_time = -1.0
        return req

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(
        self,
        current_time: float,
        service_model: Optional[ServiceModel] = None,
    ) -> List[CompletedRequest]:
        """Advance one simulation step.  Returns newly completed requests.

        When service_model is None or enable_prefill_modeling=False:
            behaves identically to Phase 1 (all active requests advance by
            one decode token each step).
        When enable_prefill_modeling=True:
            prefilling requests consume token budget;
            decoding requests each produce one output token.
        """
        if service_model is None or not service_model.enable_prefill_modeling:
            return self._step_phase1(current_time)
        return self._step_phase15(current_time, service_model)

    # ------------------------------------------------------------------ #
    # Internal step implementations
    # ------------------------------------------------------------------ #

    def _step_phase1(self, current_time: float) -> List[CompletedRequest]:
        """Original Phase 1 step: every active request advances by 1 decode token."""
        completed: List[CompletedRequest] = []
        to_remove: List[int] = []

        for rid, req in self._active.items():
            done = req.advance_decode(current_time)
            if done:
                req.phase = RequestPhase.COMPLETED
                req.completion_time = current_time
                completed.append(
                    CompletedRequest(
                        request=req.request,
                        admission_time=req.admission_time,
                        completion_time=current_time,
                        first_token_time=req.first_token_time,
                        gpu_id=self.gpu_id,
                    )
                )
                to_remove.append(rid)

        for rid in to_remove:
            del self._active[rid]

        self.step_active_counts.append(self.num_active)
        self.step_kv_used.append(self.current_kv_tokens)
        return completed

    def _step_phase15(
        self, current_time: float, service_model: ServiceModel
    ) -> List[CompletedRequest]:
        """Phase 1.5 step: separate prefill / decode phases with token budget."""
        completed: List[CompletedRequest] = []
        to_remove: List[int] = []

        prefilling = [r for r in self._active.values() if r.is_prefilling]
        decoding   = [r for r in self._active.values() if r.is_decoding]

        # Budget accounting
        budget = service_model.step_token_budget
        if service_model.decode_first:
            # Guarantee full decode budget before any prefill
            budget -= len(decoding)   # each decode request uses 1 token
            prefill_budget = max(0, budget)
        else:
            budget -= len(decoding)
            prefill_budget = max(0, budget)

        # --- Advance decode ---
        for req in decoding:
            done = req.advance_decode(current_time)
            if done:
                req.phase = RequestPhase.COMPLETED
                req.completion_time = current_time
                completed.append(
                    CompletedRequest(
                        request=req.request,
                        admission_time=req.admission_time,
                        completion_time=current_time,
                        first_token_time=req.first_token_time,
                        gpu_id=self.gpu_id,
                    )
                )
                to_remove.append(req.request_id)

        # --- Advance prefill with remaining budget ---
        for req in prefilling:
            if prefill_budget <= 0:
                break  # no budget left for prefill this step
            chunk = min(
                service_model.max_prefill_chunk_tokens,
                req.prefill_remaining,
                prefill_budget,
            )
            if chunk <= 0:
                continue
            req.advance_prefill(chunk)
            prefill_budget -= chunk
            # Prefill just completed → first decode token will be next step
            # (first_token_time is set in advance_decode on the following step)

        for rid in to_remove:
            del self._active[rid]

        self.step_active_counts.append(self.num_active)
        self.step_kv_used.append(self.current_kv_tokens)
        return completed

    # ------------------------------------------------------------------ #
    # Observable state
    # ------------------------------------------------------------------ #

    def to_observable(self) -> ObservableGPUState:
        obs_requests = [
            ObservableRequest.from_request(ir.request)
            for ir in self._active.values()
        ]
        return ObservableGPUState(
            gpu_id=self.gpu_id,
            max_active_sequences=self.config.max_active_sequences,
            max_batch_tokens=self.config.max_batch_tokens,
            max_kv_tokens=self.config.max_kv_tokens,
            active_request_ids=list(self._active.keys()),
            active_requests_info=obs_requests,
            current_kv_tokens=self.current_kv_tokens,
            tokens_decoded_per_request={
                rid: ir.tokens_decoded for rid, ir in self._active.items()
            },
            prefilling_count=self.prefilling_count,
            decoding_count=self.decoding_count,
        )

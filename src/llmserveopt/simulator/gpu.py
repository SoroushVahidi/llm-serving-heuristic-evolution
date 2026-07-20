"""
GPU-level state management within the simulator.

Phase 1.5 changes
-----------------
* admit() now accepts a ServiceModel to initialise prefill_remaining.
* step() now handles prefill / decode phases when enable_prefill_modeling=True.
* to_observable() exposes prefilling_count and decoding_count so that
  serving-style policies (Sarathi, SplitFuse) can reason about phase.
* KV tracking uses InternalRequest.kv_tokens, which grows during prefill.

Disaggregated prefill/decode addition (opt-in; see
docs/distserve_faithful_scheduler_reference.md)
-------------------------------------------------
* admit() skips prefill entirely when this GPU's role is "decode" (the
  request is assumed already-prefilled elsewhere -- the only correct
  interpretation for that role).
* When service_model.enable_disaggregation is set and this GPU's role is
  "prefill", a request whose prefill JUST completed this step is handed
  off (removed from _active, phase set to MIGRATING) instead of
  continuing to decode in place. Buffered in _pending_handoff for the
  Simulator to drain via pop_pending_handoff().
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
        # Disaggregated prefill/decode: requests handed off this step,
        # awaiting collection by the Simulator (always empty otherwise).
        self._pending_handoff: List[InternalRequest] = []

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
        is_relocation: bool = False,
    ) -> bool:
        """Admit a request.  Returns False and warns if constraints violated.

        When service_model is provided, initialises prefill_remaining per the
        configured prefill cost.  When service_model is None (or Phase 1 mode),
        prefill_remaining stays 0 (instant prefill).

        is_relocation=True (added for llumnix_faithful; see
        docs/llumnix_faithful_scheduler_reference.md): this call is resuming
        an already-in-service request that a policy live-migrated here via
        Action.migrate, not admitting it for the first time. `admission_time`
        (the overall service-start anchor used for queuing-delay/TTFT/latency
        metrics) and `prefill_remaining` are left exactly as they already are
        (preserved across the relocation by GPUState.evict(preserve_progress=True)
        on the source GPU) rather than being recomputed/reset, so a live
        migration is invisible to end-to-end service-time metrics -- exactly
        the paper's own intent ("live migration... near-zero overhead").
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
        req.transfer_ready_time = -1.0   # no longer migrating, if it was
        req.migration_destination_gpu_id = -1   # no longer relocating, if it was
        if not is_relocation:
            req.admission_time = admission_time
            if self.config.role == "decode":
                # Disaggregated decode-side GPU: prefill already happened on a
                # role="prefill" GPU before this request was handed off here.
                req.prefill_remaining = 0
            elif service_model is not None:
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

    def evict(self, request_id: int, preserve_progress: bool = False) -> Optional[InternalRequest]:
        """Forcibly remove an ACTIVE request.

        preserve_progress=False (default, unchanged): reset to a clean
        WAITING-equivalent state (recompute-on-resume semantics: all decode/
        prefill progress is discarded, matching vLLM's recompute preemption
        mode -- see docs/vllm_faithful_scheduler_reference.md). Invoked via
        an Action's `preempt` field.

        preserve_progress=True (added for distserve_faithful; see
        docs/distserve_faithful_scheduler_reference.md): swap semantics --
        tokens_decoded/first_token_time are left untouched, matching
        DistServe's decode-stage swap-out (progress is not discarded,
        since by the time a request is decoding its prefill/migration is
        already a sunk cost not worth re-paying). Invoked via an Action's
        `swap` field; the caller (Simulator._apply_action) is responsible
        for re-queuing the returned request into the bridge queue as
        immediately transfer-ready, not the ordinary waiting queue.

        preserve_progress=True is also used by llumnix_faithful's live
        migration (Action's `migrate` field; see
        docs/llumnix_faithful_scheduler_reference.md): in that case
        `admission_time` (the overall service-start anchor -- see
        GPUState.admit's `is_relocation` parameter) is ALSO preserved
        (unlike a plain preempt/swap re-admission, which always overwrites
        it with the current time regardless of what evict() leaves it as),
        so a live migration is invisible to end-to-end service-time
        metrics, matching the paper's own "near-zero overhead" claim.

        Returns the InternalRequest (for the caller to re-enqueue) or None
        if request_id is not currently active on this GPU. Backward
        compatible: only invoked when an Action's `preempt`/`swap`/`migrate`
        field is non-empty, which no existing policy other than
        distserve_faithful (for `swap`), llumnix_faithful (for `migrate`),
        or vllm_faithful/sarathi_faithful (for `preempt`) ever sets.
        """
        req = self._active.pop(request_id, None)
        if req is None:
            return None
        req.phase = RequestPhase.WAITING
        req.gpu_id = -1
        if not preserve_progress:
            req.admission_time = -1.0
            req.tokens_decoded = 0
            req.prefill_remaining = 0
            req.first_token_time = -1.0
        req.transfer_ready_time = -1.0
        return req

    # ------------------------------------------------------------------ #
    # Disaggregated prefill/decode handoff
    # ------------------------------------------------------------------ #

    def pop_pending_handoff(self) -> List[InternalRequest]:
        """Drain and return requests handed off (prefill just finished on
        this role="prefill" GPU) during the most recent step() call.
        Always empty unless ServiceModel.enable_disaggregation is set and
        this GPU has role="prefill" -- see docs/distserve_faithful_scheduler_reference.md."""
        handed_off = self._pending_handoff
        self._pending_handoff = []
        return handed_off

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
        """Phase 1.5 step: separate prefill / decode phases with token budget.

        Dispatches to one of two execution models (see
        docs/decode_prefill_contention_execution_model.md for the full
        design rationale):

        * `enable_decode_prefill_contention=False` (default): the
          historical decode-protected model. `decode_first` has no
          observable effect -- decode always receives its budget first,
          exactly reproducing this method's pre-fix behavior bit-for-bit.
          Kept as the default because a large body of existing experiment
          configs rely on this (accidentally-consistent) behavior.
        * `enable_decode_prefill_contention=True` (opt-in): `decode_first`
          becomes genuinely load-bearing -- True still uses the
          decode-protected formula above (Sarathi-style stall-free
          execution); False switches to a single shared per-step budget
          consumed by decode and prefill together in FCFS-by-arrival-time
          order (vLLM v0.4.2 `_schedule_running`-style contention, no
          decode-priority phase).
        """
        prefilling = [r for r in self._active.values() if r.is_prefilling]
        decoding = [r for r in self._active.values() if r.is_decoding]

        if service_model.enable_decode_prefill_contention and not service_model.decode_first:
            completed, to_remove, handoff_ids = self._advance_shared_contention(
                current_time, service_model, prefilling, decoding
            )
        else:
            completed, to_remove, handoff_ids = self._advance_decode_protected(
                current_time, service_model, prefilling, decoding
            )

        for rid in handoff_ids:
            self._pending_handoff.append(self._active.pop(rid))

        for rid in to_remove:
            del self._active[rid]

        self.step_active_counts.append(self.num_active)
        self.step_kv_used.append(self.current_kv_tokens)
        return completed

    def _advance_decode_protected(
        self,
        current_time: float,
        service_model: ServiceModel,
        prefilling: List[InternalRequest],
        decoding: List[InternalRequest],
    ) -> tuple:
        """Decode-protected execution (Sarathi-style stall-free principle):
        decode always receives its full budget first, unconditionally;
        prefill gets whatever is left. This is the historical Phase-1.5
        formula, UNCHANGED -- it is both the default execution model
        (`enable_decode_prefill_contention=False`, any `decode_first`
        value) and the `decode_first=True` behavior once contention mode
        is enabled. Iteration order over `prefilling` is deliberately left
        as-is (insertion/admission order into `self._active`), not
        re-sorted, to keep this path bit-identical to the pre-fix code for
        every existing caller.
        """
        completed: List[CompletedRequest] = []
        to_remove: List[int] = []
        handoff_ids: List[int] = []

        budget = service_model.step_token_budget - len(decoding)
        prefill_budget = max(0, budget)

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
            prefill_just_finished = req.advance_prefill(chunk)
            prefill_budget -= chunk
            if (
                prefill_just_finished
                and service_model.enable_disaggregation
                and self.config.role == "prefill"
            ):
                # Disaggregated mode: hand off to the "migrating" pool
                # instead of continuing to decode in place (see
                # docs/distserve_faithful_scheduler_reference.md).
                req.phase = RequestPhase.MIGRATING
                req.transfer_ready_time = current_time + service_model.migration_transfer_delay
                req.gpu_id = -1
                handoff_ids.append(req.request_id)
            # Otherwise: prefill just completed → first decode token will be
            # next step (first_token_time is set in advance_decode then).

        return completed, to_remove, handoff_ids

    def _advance_shared_contention(
        self,
        current_time: float,
        service_model: ServiceModel,
        prefilling: List[InternalRequest],
        decoding: List[InternalRequest],
    ) -> tuple:
        """vLLM-v0.4.2-chunked-prefill-style shared execution: decode and
        prefill requests compete for ONE combined per-step token budget,
        consumed in a single FCFS-by-arrival-time pass -- no decode
        priority phase (see docs/decode_prefill_contention_execution_model.md
        and docs/vllm_chunked_prefill_faithful_scheduler_reference.md's
        `_schedule_running` algorithm section). A request later in that
        order can receive zero progress this step if the budget runs out
        on an earlier one -- matching the pinned reference's own
        `_get_num_new_tokens(...) == 0: break` behavior (stop scheduling
        the rest of the running queue entirely this step, not
        skip-and-continue).

        Only reached when `enable_decode_prefill_contention=True` and
        `decode_first=False`.
        """
        completed: List[CompletedRequest] = []
        to_remove: List[int] = []
        handoff_ids: List[int] = []

        combined = sorted(
            list(decoding) + list(prefilling),
            key=lambda r: (r.request.arrival_time, r.request_id),
        )
        budget = service_model.step_token_budget

        for req in combined:
            if budget <= 0:
                break
            if req.is_decoding:
                num_new_tokens = 1
            else:
                num_new_tokens = min(
                    service_model.max_prefill_chunk_tokens,
                    req.prefill_remaining,
                    budget,
                )
            if num_new_tokens <= 0:
                break  # budget exhausted: stop scheduling the rest this step

            if req.is_decoding:
                budget -= num_new_tokens
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
            else:
                budget -= num_new_tokens
                prefill_just_finished = req.advance_prefill(num_new_tokens)
                if (
                    prefill_just_finished
                    and service_model.enable_disaggregation
                    and self.config.role == "prefill"
                ):
                    req.phase = RequestPhase.MIGRATING
                    req.transfer_ready_time = current_time + service_model.migration_transfer_delay
                    req.gpu_id = -1
                    handoff_ids.append(req.request_id)

        return completed, to_remove, handoff_ids

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
            role=self.config.role,
        )

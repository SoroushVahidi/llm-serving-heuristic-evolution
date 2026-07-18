"""
Deterministic iteration-level LLM serving simulator.

Design overview
---------------
Time is measured in seconds (float).  The simulator advances in discrete
decode steps of `step_size` seconds each.  At every step:

  1. Enqueue all requests whose arrival_time ≤ current_time.
  2. Build an ObservableState (no ground-truth leakage).
  3. Call the policy to obtain an Action.
  4. Validate and apply the Action: admitted requests join GPU active batches.
  5. Advance every active request by one decode token.
  6. Remove completed requests and record CompletedRequest objects.
  7. Record per-step utilization history.

Phase 1 simplifications (see docs/simulator_design.md):
  - Prefill is instantaneous (no separate prefill step).
  - All GPUs are identical.
  - One output token produced per active request per step.
  - No preemption / eviction.
  - No speculative decoding.
"""
from __future__ import annotations

import time as _time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..core.action import Action
from ..core.metrics import RunMetrics, compute_metrics
from ..core.types import (
    CompletedRequest,
    GPUConfig,
    ObservableRequest,
    ObservableState,
    Request,
)
from .gpu import GPUState
from .request import InternalRequest, RequestPhase
from .service_model import ServiceModel


@dataclass
class SimulatorConfig:
    gpu_configs: List[GPUConfig]
    service_model: ServiceModel = field(default_factory=ServiceModel)
    # Maximum simulation steps; if None, run until all requests complete or
    # trace is exhausted plus a drain window.
    max_steps: Optional[int] = None
    # After last arrival, continue for this many extra steps to drain queues.
    drain_steps: int = 50_000
    # Warn (but do not fail) when a policy tries to admit a non-existent request.
    warn_on_invalid_action: bool = True


class Simulator:
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self._gpus: List[GPUState] = [
            GPUState(gc) for gc in config.gpu_configs
        ]
        self._gpu_map: Dict[int, GPUState] = {g.gpu_id: g for g in self._gpus}
        self._waiting: deque[InternalRequest] = deque()
        self._waiting_map: Dict[int, InternalRequest] = {}
        # Disaggregated prefill/decode "bridge queue" (opt-in; see
        # docs/distserve_faithful_scheduler_reference.md): requests that
        # finished prefill on a role="prefill" GPU and are awaiting transfer
        # completion. Always empty unless ServiceModel.enable_disaggregation
        # is set.
        self._migrating: deque[InternalRequest] = deque()
        self._migrating_map: Dict[int, InternalRequest] = {}
        self._pending_arrivals: List[InternalRequest] = []   # sorted by arrival_time
        self._completed: List[CompletedRequest] = []
        self._step: int = 0
        self._time: float = 0.0

        # Per-step history for metrics
        self._util_history: List[float] = []
        self._batch_history: List[float] = []
        self._policy_times: List[float] = []
        # Steps skipped during idle-period fast-forwarding
        self._idle_skipped: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_trace(self, requests: Sequence[Request]) -> None:
        """Load a sorted request trace.  Must be called before run()."""
        sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
        self._pending_arrivals = [InternalRequest(request=r) for r in sorted_reqs]

    def run(self, policy, workload_tag: str = "unknown", seed: int = 0) -> RunMetrics:
        """Run simulation with `policy` and return metrics.

        Parameters
        ----------
        policy : BasePolicy
            Must implement select_action(ObservableState) -> Action.
        workload_tag : str
            Label used in metrics output.
        seed : int
            Seed used for workload generation (recorded in metrics only).
        """
        wall_start = _time.perf_counter()
        self._reset()

        arrival_idx = 0
        n_arrivals = len(self._pending_arrivals)

        step_size = self.config.service_model.step_size
        max_steps = self.config.max_steps
        drain_steps = self.config.drain_steps

        steps_since_last_arrival = 0

        while True:
            self._time = self._step * step_size

            # --- 1. Enqueue newly arrived requests ---
            while (
                arrival_idx < n_arrivals
                and self._pending_arrivals[arrival_idx].request.arrival_time
                <= self._time
            ):
                ir = self._pending_arrivals[arrival_idx]
                self._waiting.append(ir)
                self._waiting_map[ir.request_id] = ir
                arrival_idx += 1

            # --- 2. Build observable state ---
            state = self._build_observable_state()

            # --- 3. Call policy ---
            t0 = _time.perf_counter()
            action = policy.select_action(state)
            self._policy_times.append(_time.perf_counter() - t0)

            # --- 4. Apply action ---
            self._apply_action(action)

            # --- 5. Advance decode ---
            step_completed = self._advance_decode()
            self._completed.extend(step_completed)

            # --- 5b. Collect any disaggregated prefill->decode handoffs
            # produced this step (always a no-op unless
            # ServiceModel.enable_disaggregation is set).
            self._collect_handoffs()

            # --- 6. Record per-step metrics ---
            total_active = sum(g.num_active for g in self._gpus)
            n_gpus = len(self._gpus)
            mean_util = (
                sum(g.utilization for g in self._gpus) / n_gpus if n_gpus else 0.0
            )
            self._util_history.append(mean_util)
            self._batch_history.append(total_active)

            # --- 7. Termination check ---
            all_arrivals_done = arrival_idx >= n_arrivals
            all_active_done = total_active == 0
            # A request mid-transfer (in the bridge queue) is neither
            # "active" on any GPU nor in the ordinary waiting queue -- it
            # must still block termination/idle-fast-forward, or the
            # simulation could end (or skip ahead) while a request is
            # genuinely in flight.
            queue_empty = len(self._waiting) == 0 and len(self._migrating) == 0

            if all_arrivals_done:
                steps_since_last_arrival += 1
            else:
                steps_since_last_arrival = 0

            if max_steps is not None and self._step >= max_steps:
                break
            if all_arrivals_done and queue_empty and all_active_done:
                break
            if all_arrivals_done and steps_since_last_arrival >= drain_steps:
                break

            # Fast-forward over idle periods: when the queue is empty, the GPU
            # is empty, and the next arrival is far away, jump to just before
            # that arrival instead of stepping through empty time one-by-one.
            if not all_arrivals_done and queue_empty and all_active_done:
                next_arr_time = self._pending_arrivals[arrival_idx].request.arrival_time
                skip_to = int(next_arr_time / step_size)
                idle_gap = skip_to - (self._step + 1)
                if idle_gap > 0:
                    self._idle_skipped += idle_gap
                    self._step = skip_to - 1  # += 1 below makes it skip_to

            self._step += 1

        sim_duration = self._time
        wall_elapsed = _time.perf_counter() - wall_start

        # Requests still in waiting queue (or, in disaggregated mode,
        # mid-transfer -- should not normally happen given the
        # termination-check fix above, but included defensively) at end
        # = dropped.
        dropped = [ir.request for ir in self._waiting] + [
            ir.request for ir in self._migrating
        ]

        return compute_metrics(
            completed=self._completed,
            dropped=dropped,
            sim_duration=sim_duration,
            gpu_utilization_history=self._util_history,
            active_batch_history=self._batch_history,
            policy_name=policy.name,
            workload_tag=workload_tag,
            seed=seed,
            policy_decision_times=self._policy_times,
            wall_clock_s=wall_elapsed,
            idle_steps_skipped=self._idle_skipped,
            num_total=n_arrivals,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        for g in self._gpus:
            g._active.clear()
            g.step_active_counts.clear()
            g.step_kv_used.clear()
            g._pending_handoff.clear()
        self._waiting.clear()
        self._waiting_map.clear()
        self._migrating.clear()
        self._migrating_map.clear()
        self._completed.clear()
        self._step = 0
        self._time = 0.0
        self._util_history.clear()
        self._batch_history.clear()
        self._policy_times.clear()
        self._idle_skipped = 0
        # Reset internal request states
        for ir in self._pending_arrivals:
            ir.phase = RequestPhase.WAITING
            ir.gpu_id = -1
            ir.admission_time = -1.0
            ir.completion_time = -1.0
            ir.tokens_decoded = 0
            ir.prefill_remaining = 0
            ir.first_token_time = -1.0
            ir.transfer_ready_time = -1.0

    def _build_observable_state(self) -> ObservableState:
        waiting_obs = [
            ObservableRequest.from_request(ir.request)
            for ir in self._waiting
        ]
        # Disaggregated prefill/decode: only requests whose transfer delay
        # has already elapsed are exposed as admission-eligible (still-in-
        # transit ones are invisible to policies, matching "decode begins
        # only after handoff/transfer completion"). Always empty unless
        # ServiceModel.enable_disaggregation is set.
        migrating_obs = [
            ObservableRequest.from_request(ir.request)
            for ir in self._migrating
            if ir.transfer_ready_time <= self._time
        ]
        gpu_obs = [g.to_observable() for g in self._gpus]
        return ObservableState(
            time=self._time,
            waiting_queue=waiting_obs,
            gpu_states=gpu_obs,
            completed_count=len(self._completed),
            step=self._step,
            migrating_queue=migrating_obs,
        )

    def _apply_action(self, action: Action) -> None:
        # Preemptions and swaps are applied first (matching the pinned
        # vllm_faithful/distserve_faithful references: eviction decisions
        # are made before new admissions from `waiting`/the bridge queue in
        # the same scheduling iteration). No existing policy other than
        # vllm_faithful/sarathi_faithful (preempt) or distserve_faithful
        # (swap) ever sets these, so this is a no-op for all legacy behavior.
        preempted_ids = self._apply_preemptions(action)
        swapped_ids = self._apply_swaps(action)
        evicted_ids = preempted_ids | swapped_ids

        admitted_ids = set()

        for gpu_id, req_ids in action.admit.items():
            if gpu_id not in self._gpu_map:
                if self.config.warn_on_invalid_action:
                    warnings.warn(f"Action references unknown gpu_id={gpu_id}; skipped.")
                continue

            gpu = self._gpu_map[gpu_id]

            for rid in req_ids:
                if rid in admitted_ids:
                    # Prevent double-admission (same request to two GPUs)
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Request {rid} appears in multiple GPUs in the same action; skipped."
                        )
                    continue

                if rid in evicted_ids:
                    # A request cannot be preempted/swapped and re-admitted
                    # within the same step's Action (mirrors the pinned
                    # reference's own "if seq_group in preempted: break" guard).
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Request {rid} was preempted/swapped and admitted in the "
                            "same action; skipping the admit (it is back in a queue)."
                        )
                    continue

                # Resolve the request from wherever it is currently
                # eligible: the ordinary waiting queue, or (disaggregated
                # mode only) the bridge queue of transfer-ready requests.
                from_migrating = False
                if rid in self._waiting_map:
                    ir = self._waiting_map[rid]
                elif rid in self._migrating_map:
                    ir = self._migrating_map[rid]
                    from_migrating = True
                else:
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Action references request {rid} not in waiting queue; skipped."
                        )
                    continue

                if from_migrating:
                    # Transfer-ready check: a still-in-transit request must
                    # never be admitted early (mirrors the arrival-time
                    # check below for ordinary waiting requests).
                    if ir.transfer_ready_time > self._time:
                        if self.config.warn_on_invalid_action:
                            warnings.warn(
                                f"Request {rid} is still mid-transfer "
                                f"(ready at {ir.transfer_ready_time}, now {self._time}); skipped."
                            )
                        continue
                else:
                    # Validate arrival: request must have arrived by now
                    if ir.request.arrival_time > self._time:
                        if self.config.warn_on_invalid_action:
                            warnings.warn(
                                f"Request {rid} has arrival_time={ir.request.arrival_time} "
                                f"> current time={self._time}; skipped."
                            )
                        continue

                ok = gpu.admit(
                    ir,
                    admission_time=self._time,
                    service_model=self.config.service_model,
                )
                if ok:
                    if from_migrating:
                        self._migrating_map.pop(rid)
                    else:
                        self._waiting_map.pop(rid)
                    admitted_ids.add(rid)

        # Remove admitted requests from the waiting/migrating deques
        if admitted_ids:
            self._waiting = deque(
                ir for ir in self._waiting if ir.request_id not in admitted_ids
            )
            self._migrating = deque(
                ir for ir in self._migrating if ir.request_id not in admitted_ids
            )

    def _apply_preemptions(self, action: Action) -> set:
        """Evict each request named in action.preempt back to the FRONT of
        the waiting queue, discarding its progress (recompute-on-resume;
        see docs/vllm_faithful_scheduler_reference.md). Returns the set of
        request IDs actually preempted. A no-op whenever action.preempt is
        empty -- true for every policy except vllm_faithful."""
        if not action.preempt:
            return set()

        preempted_ids: set = set()
        for gpu_id, req_ids in action.preempt.items():
            if gpu_id not in self._gpu_map:
                if self.config.warn_on_invalid_action:
                    warnings.warn(
                        f"Action references unknown gpu_id={gpu_id} in preempt; skipped."
                    )
                continue

            gpu = self._gpu_map[gpu_id]

            for rid in req_ids:
                if rid in preempted_ids:
                    # Prevent double-preemption (same request listed twice)
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Request {rid} appears more than once in preempt; skipped."
                        )
                    continue

                ir = gpu.evict(rid)
                if ir is None:
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Action references request {rid} for preemption but it is "
                            f"not active on gpu_id={gpu_id}; skipped."
                        )
                    continue

                preempted_ids.add(rid)
                # Recompute-preempted requests re-enter at the FRONT of the
                # waiting queue, in the order the policy listed them (matches
                # the pinned reference's own `waiting.insert(0, seq_group)`
                # called once per victim).
                self._waiting.appendleft(ir)
                self._waiting_map[rid] = ir

        return preempted_ids

    def _apply_swaps(self, action: Action) -> set:
        """Evict each request named in action.swap WITHOUT discarding
        progress (see docs/distserve_faithful_scheduler_reference.md),
        re-queuing it into the bridge queue as immediately transfer-ready
        (so it is admission-eligible again as soon as the policy chooses,
        with no additional transfer delay -- it never actually left the
        decode side's memory pool in the real system this models). Returns
        the set of request IDs actually swapped. A no-op whenever
        action.swap is empty -- true for every policy except
        distserve_faithful."""
        if not action.swap:
            return set()

        swapped_ids: set = set()
        for gpu_id, req_ids in action.swap.items():
            if gpu_id not in self._gpu_map:
                if self.config.warn_on_invalid_action:
                    warnings.warn(
                        f"Action references unknown gpu_id={gpu_id} in swap; skipped."
                    )
                continue

            gpu = self._gpu_map[gpu_id]

            for rid in req_ids:
                if rid in swapped_ids:
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Request {rid} appears more than once in swap; skipped."
                        )
                    continue

                ir = gpu.evict(rid, preserve_progress=True)
                if ir is None:
                    if self.config.warn_on_invalid_action:
                        warnings.warn(
                            f"Action references request {rid} for swap but it is "
                            f"not active on gpu_id={gpu_id}; skipped."
                        )
                    continue

                swapped_ids.add(rid)
                ir.transfer_ready_time = self._time  # immediately re-admission-eligible
                self._migrating.appendleft(ir)
                self._migrating_map[rid] = ir

        return swapped_ids

    def _advance_decode(self) -> List[CompletedRequest]:
        completed: List[CompletedRequest] = []
        step_end_time = (self._step + 1) * self.config.service_model.step_size
        for g in self._gpus:
            completed.extend(
                g.step(
                    current_time=step_end_time,
                    service_model=self.config.service_model,
                )
            )
        return completed

    def _collect_handoffs(self) -> None:
        """Move requests handed off this step (prefill just finished on a
        role="prefill" GPU, disaggregated mode) into the bridge queue.
        Always a no-op unless ServiceModel.enable_disaggregation is set --
        every GPU's pop_pending_handoff() returns an empty list otherwise."""
        for g in self._gpus:
            for ir in g.pop_pending_handoff():
                self._migrating.append(ir)
                self._migrating_map[ir.request_id] = ir

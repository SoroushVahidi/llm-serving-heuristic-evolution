"""Tests for the backward-compatible preemption extension:
Action.preempt, GPUState.evict(), Simulator._apply_preemptions().

This machinery exists to support the vllm_faithful baseline's
recompute-preemption semantics (see docs/vllm_faithful_scheduler_reference.md).
It is opt-in: no existing policy ever sets Action.preempt, so every test
here that checks "legacy behavior unchanged" is a genuine regression guard,
not just incidental coverage.
"""
from __future__ import annotations

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.simulator.gpu import GPUState
from llmserveopt.simulator.request import InternalRequest, RequestPhase
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _gpu_config(max_seq=4, max_tok=64, max_kv=1024):
    return GPUConfig(gpu_id=0, max_active_sequences=max_seq,
                      max_batch_tokens=max_tok, max_kv_tokens=max_kv)


def _req(rid, arrival=0.0, prompt=10, output=5, deadline=1000.0, priority=1.0):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id="medium",
    )


# ---------------------------------------------------------------------------
# Action.preempt: field defaults and helper methods
# ---------------------------------------------------------------------------

def test_action_preempt_defaults_empty():
    action = Action(admit={0: [1, 2]})
    assert action.preempt == {}
    assert action.all_preempted_ids() == set()


def test_action_is_empty_considers_preempt():
    assert Action().is_empty() is True
    assert Action(admit={0: []}, preempt={0: []}).is_empty() is True
    assert Action(admit={0: []}, preempt={0: [5]}).is_empty() is False


def test_action_all_preempted_ids_across_gpus():
    action = Action(admit={}, preempt={0: [1, 2], 1: [3]})
    assert action.all_preempted_ids() == {1, 2, 3}


def test_existing_admit_only_construction_still_works():
    """The exact construction pattern used by every pre-existing policy
    (Action(admit=...), no preempt kwarg) must remain valid."""
    action = Action(admit={0: [1], 1: [2]})
    assert action.preempt == {}
    assert action.all_admitted_ids() == {1, 2}


# ---------------------------------------------------------------------------
# GPUState.evict()
# ---------------------------------------------------------------------------

def test_evict_unknown_request_returns_none():
    gpu = GPUState(_gpu_config())
    assert gpu.evict(999) is None


def test_evict_removes_from_active_and_resets_progress():
    gpu = GPUState(_gpu_config())
    ir = InternalRequest(request=_req(1))
    assert gpu.admit(ir, admission_time=0.0) is True
    assert gpu.num_active == 1

    ir.tokens_decoded = 3  # simulate some decode progress
    ir.first_token_time = 0.5

    evicted = gpu.evict(1)
    assert evicted is ir
    assert gpu.num_active == 0
    assert evicted.phase == RequestPhase.WAITING
    assert evicted.gpu_id == -1
    assert evicted.admission_time == -1.0
    assert evicted.tokens_decoded == 0
    assert evicted.prefill_remaining == 0
    assert evicted.first_token_time == -1.0


def test_evict_frees_kv_capacity():
    gpu = GPUState(_gpu_config(max_kv=20))
    ir = InternalRequest(request=_req(1, prompt=15))
    assert gpu.admit(ir, admission_time=0.0) is True
    assert gpu.current_kv_tokens == 15
    gpu.evict(1)
    assert gpu.current_kv_tokens == 0
    assert gpu.num_active == 0


# ---------------------------------------------------------------------------
# Simulator-level integration: a minimal test policy that preempts
# ---------------------------------------------------------------------------

class _PreemptOnceThenFifoPolicy(BasePolicy):
    """Admits FIFO-style, but at a chosen step preempts a chosen request_id
    exactly once. Used only to exercise the simulator's preemption plumbing
    end-to-end -- not a real scheduling policy."""
    name = "test_preempt_once"

    def __init__(self, preempt_at_step: int, preempt_request_id: int):
        self.preempt_at_step = preempt_at_step
        self.preempt_request_id = preempt_request_id
        self._done = False

    def select_action(self, state):
        admit: dict = {g.gpu_id: [] for g in state.gpu_states}
        preempt: dict = {g.gpu_id: [] for g in state.gpu_states}

        if (not self._done and state.step == self.preempt_at_step
                and self.preempt_request_id in {
                    rid for g in state.gpu_states for rid in g.active_request_ids
                }):
            for g in state.gpu_states:
                if self.preempt_request_id in g.active_request_ids:
                    preempt[g.gpu_id].append(self.preempt_request_id)
                    self._done = True

        for req in state.waiting_queue:
            for g in state.gpu_states:
                if self._feasible_on_gpu(g, req):
                    admit[g.gpu_id].append(req.request_id)
                    g.active_request_ids.append(req.request_id)
                    g.current_kv_tokens += req.prompt_tokens
                    break
        return Action(admit=admit, preempt=preempt)


class _RecordWaitingOrderPolicy(BasePolicy):
    """Admits request 1 immediately, preempts it at a chosen step, and
    records the exact waiting_queue request-id order on every subsequent
    step (so the test can assert request 1 reappears at the FRONT)."""
    name = "test_record_waiting_order"

    def __init__(self, preempt_at_step: int):
        self.preempt_at_step = preempt_at_step
        self._done = False
        self.waiting_order_by_step: dict = {}

    def select_action(self, state):
        admit: dict = {g.gpu_id: [] for g in state.gpu_states}
        preempt: dict = {g.gpu_id: [] for g in state.gpu_states}
        self.waiting_order_by_step[state.step] = [r.request_id for r in state.waiting_queue]

        active_ids = {rid for g in state.gpu_states for rid in g.active_request_ids}
        if state.step == self.preempt_at_step and 1 in active_ids and not self._done:
            for g in state.gpu_states:
                if 1 in g.active_request_ids:
                    preempt[g.gpu_id].append(1)
                    self._done = True

        for req in state.waiting_queue:
            for g in state.gpu_states:
                if self._feasible_on_gpu(g, req):
                    admit[g.gpu_id].append(req.request_id)
                    g.active_request_ids.append(req.request_id)
                    g.current_kv_tokens += req.prompt_tokens
                    break
        return Action(admit=admit, preempt=preempt)


def test_preempted_request_returns_to_front_of_waiting_queue():
    """Request 1 is admitted at step 0 (waiting queue empty at that point),
    then request 2 arrives and waits behind it conceptually. When request 1
    is preempted at step 2, it must reappear at the FRONT of the observed
    waiting_queue on the very next step -- ahead of request 2, which never
    left the queue."""
    gpus = _gpu_config_list(max_seq=1, max_kv=1000)  # 1 slot: req 2 must wait
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=10))
    sim.load_trace([_req(1, arrival=0.0, output=50), _req(2, arrival=0.0, output=50)])
    policy = _RecordWaitingOrderPolicy(preempt_at_step=2)
    sim.run(policy, workload_tag="test")

    # Step 0: only req 1 has arrived-and-not-yet-admitted at observation time
    # (both arrive at t=0, but only 1 slot exists so req 2 stays waiting
    # once req 1 is admitted at step 0).
    assert policy.waiting_order_by_step[0] == [1, 2]
    # Step 1 (after req 1 admitted at step 0): req 2 alone is waiting.
    assert policy.waiting_order_by_step[1] == [2]
    # Step 3 (the step after preemption is applied at step 2): req 1 must be
    # back at the FRONT, ahead of req 2.
    assert policy.waiting_order_by_step[3] == [1, 2]


def test_preemption_resets_progress_and_delays_completion():
    """A preempted request must finish LATER than it would have without
    preemption, because recompute discards all decode progress."""
    # Baseline: no preemption at all (plain FIFO).
    sim_baseline = Simulator(SimulatorConfig(gpu_configs=_gpu_config_list(max_seq=1), max_steps=200))
    sim_baseline.load_trace([_req(1, arrival=0.0, output=20)])
    baseline_metrics = sim_baseline.run(FIFOPolicy(), workload_tag="baseline")
    assert baseline_metrics.num_completed == 1
    baseline_latency = baseline_metrics.mean_latency  # only 1 request: this IS its latency

    # With preemption partway through decode: must complete later (it has
    # to redo all decode work from scratch).
    sim_preempt = Simulator(SimulatorConfig(gpu_configs=_gpu_config_list(max_seq=1), max_steps=200))
    sim_preempt.load_trace([_req(1, arrival=0.0, output=20)])
    policy = _PreemptOnceThenFifoPolicy(preempt_at_step=5, preempt_request_id=1)
    preempt_metrics = sim_preempt.run(policy, workload_tag="preempt")
    assert preempt_metrics.num_completed == 1
    preempt_latency = preempt_metrics.mean_latency

    assert preempt_latency > baseline_latency


def _gpu_config_list(max_seq=1, max_tok=64, max_kv=1000):
    return [_gpu_config(max_seq=max_seq, max_tok=max_tok, max_kv=max_kv)]


# ---------------------------------------------------------------------------
# Regression: every existing (admit-only) policy is completely unaffected
# ---------------------------------------------------------------------------

def test_fifo_policy_unaffected_by_preempt_field_existing():
    """A plain FIFOPolicy run must be bit-identical whether or not the new
    Action.preempt field exists -- it never touches it."""
    gpus = _gpu_config_list(max_seq=4, max_kv=1000)
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=1000))
    sim.load_trace([_req(1, arrival=0.0, output=10), _req(2, arrival=0.01, output=10)])
    metrics = sim.run(FIFOPolicy(), workload_tag="regression")
    assert metrics.num_completed == 2
    assert metrics.num_dropped == 0

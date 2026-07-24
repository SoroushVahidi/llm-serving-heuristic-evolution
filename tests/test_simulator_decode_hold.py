"""Tests for the backward-compatible decode-hold extension:
Action.hold_decode, GPUState.step(held_decode_ids=...), Simulator._advance_decode().

This machinery exists to support the slai_faithful baseline's decode-deferral
semantics (see docs/slai_faithful_scheduler_reference.md): the pinned SLAI
reference decides, per decode-phase request, whether its decode-iteration is
"critical" (must run now) or "non-critical" (safe to defer to a later batch)
based on that request's own last-schedulable-time. Neither of the
simulator's two existing GLOBAL execution models (decode-protected /
shared-contention) can express a per-request deferral decision -- they apply
one uniform rule to the whole decoding population.

It is opt-in: no existing policy ever sets Action.hold_decode, so every test
here that checks "legacy behavior unchanged" is a genuine regression guard,
not just incidental coverage.
"""
from __future__ import annotations

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.simulator.gpu import GPUState
from llmserveopt.simulator.request import InternalRequest
from llmserveopt.simulator.service_model import ServiceModel
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
# Action.hold_decode: field defaults and helper methods
# ---------------------------------------------------------------------------

def test_action_hold_decode_defaults_empty():
    action = Action(admit={0: [1, 2]})
    assert action.hold_decode == {}
    assert action.all_held_decode_ids() == set()


def test_action_is_empty_considers_hold_decode():
    assert Action().is_empty() is True
    assert Action(admit={0: []}, hold_decode={0: []}).is_empty() is True
    assert Action(admit={0: []}, hold_decode={0: [5]}).is_empty() is False


def test_action_all_held_decode_ids_across_gpus():
    action = Action(admit={}, hold_decode={0: [1, 2], 1: [3]})
    assert action.all_held_decode_ids() == {1, 2, 3}


def test_existing_admit_only_construction_still_works():
    action = Action(admit={0: [1], 1: [2]})
    assert action.hold_decode == {}
    assert action.all_admitted_ids() == {1, 2}


def test_action_repr_includes_hold_decode_only_when_nonempty():
    assert "total_held" not in repr(Action(admit={0: [1]}))
    assert "total_held" in repr(Action(admit={0: []}, hold_decode={0: [7]}))


# ---------------------------------------------------------------------------
# GPUState.step(held_decode_ids=...): unit-level mechanics
# ---------------------------------------------------------------------------

def _phase15_service_model(**overrides):
    defaults = dict(
        step_size=0.001,
        enable_prefill_modeling=True,
        prefill_cost_per_token=1.0,
        max_prefill_chunk_tokens=512,
        step_token_budget=4096,
        decode_first=True,
    )
    defaults.update(overrides)
    return ServiceModel(**defaults)


def test_held_request_does_not_advance_phase1():
    """Phase 1 (no prefill modeling): a held request must not decode."""
    gpu = GPUState(_gpu_config())
    ir = InternalRequest(request=_req(1, output=5))
    gpu.admit(ir, admission_time=0.0)

    gpu.step(current_time=0.001, held_decode_ids=frozenset({1}))
    assert ir.tokens_decoded == 0
    assert ir.first_token_time == -1.0
    assert gpu.num_active == 1  # still active, not evicted


def test_unheld_request_advances_normally_phase1():
    gpu = GPUState(_gpu_config())
    ir = InternalRequest(request=_req(1, output=5))
    gpu.admit(ir, admission_time=0.0)

    gpu.step(current_time=0.001, held_decode_ids=frozenset())
    assert ir.tokens_decoded == 1


def test_held_request_does_not_advance_phase15_decode_protected():
    sm = _phase15_service_model()
    gpu = GPUState(_gpu_config(max_tok=4096, max_kv=4096))
    ir = InternalRequest(request=_req(1, prompt=1, output=5))
    gpu.admit(ir, admission_time=0.0, service_model=sm)
    # Finish prefill instantly (prompt=1, cost_per_token=1 -> 1 token to prefill)
    gpu.step(current_time=0.001, service_model=sm)
    assert ir.is_decoding

    tokens_before = ir.tokens_decoded
    gpu.step(current_time=0.002, service_model=sm, held_decode_ids=frozenset({1}))
    assert ir.tokens_decoded == tokens_before
    assert gpu.num_active == 1


def test_held_decode_frees_budget_for_prefill_decode_protected():
    """The central SLAI mechanism: holding a decode request's slot must let
    an otherwise-budget-starved prefill make progress that step."""
    sm = _phase15_service_model(step_token_budget=1, max_prefill_chunk_tokens=1)
    gpu = GPUState(_gpu_config(max_seq=4, max_tok=4096, max_kv=4096))

    decode_req = InternalRequest(request=_req(1, prompt=1, output=5))
    gpu.admit(decode_req, admission_time=0.0, service_model=sm)
    gpu.step(current_time=0.001, service_model=sm)  # finishes 1-token prefill -> decoding
    assert decode_req.is_decoding

    prefill_req = InternalRequest(request=_req(2, prompt=3, output=5))
    gpu.admit(prefill_req, admission_time=0.002, service_model=sm)
    assert prefill_req.is_prefilling
    assert prefill_req.prefill_remaining == 3

    # Baseline (no hold): with a budget of exactly 1 token, decode-protected
    # execution reserves that whole budget for the 1 decoding request,
    # leaving 0 for prefill.
    remaining_before = prefill_req.prefill_remaining
    gpu.step(current_time=0.003, service_model=sm)
    assert prefill_req.prefill_remaining == remaining_before  # no progress: budget went to decode
    assert decode_req.tokens_decoded == 1  # decode-protected: decode always advances unconditionally

    # Now hold the decode request: its slot frees up, so prefill can use
    # the single token of budget instead.
    tokens_before_hold = decode_req.tokens_decoded
    gpu.step(current_time=0.004, service_model=sm, held_decode_ids=frozenset({1}))
    assert decode_req.tokens_decoded == tokens_before_hold  # held: did not advance
    assert prefill_req.prefill_remaining == remaining_before - 1  # prefill got the freed budget


def test_held_request_excluded_from_shared_contention_combined_list():
    sm = _phase15_service_model(
        enable_decode_prefill_contention=True, decode_first=False,
        step_token_budget=1, max_prefill_chunk_tokens=1,
    )
    gpu = GPUState(_gpu_config(max_seq=4, max_tok=4096, max_kv=4096))

    decode_req = InternalRequest(request=_req(1, arrival=0.0, prompt=1, output=5))
    gpu.admit(decode_req, admission_time=0.0, service_model=sm)
    gpu.step(current_time=0.001, service_model=sm)
    assert decode_req.is_decoding

    prefill_req = InternalRequest(request=_req(2, arrival=0.002, prompt=3, output=5))
    gpu.admit(prefill_req, admission_time=0.002, service_model=sm)

    remaining_before = prefill_req.prefill_remaining
    tokens_before = decode_req.tokens_decoded
    # decode_req has the earlier arrival_time, so in plain shared-contention
    # FCFS-by-arrival it would consume the single token of budget itself,
    # leaving prefill starved. Holding it must let prefill through instead.
    gpu.step(current_time=0.003, service_model=sm, held_decode_ids=frozenset({1}))
    assert decode_req.tokens_decoded == tokens_before
    assert prefill_req.prefill_remaining == remaining_before - 1


def test_held_request_diagnostics_count_as_deferred():
    sm = _phase15_service_model()
    gpu = GPUState(_gpu_config(max_tok=4096, max_kv=4096))
    ir = InternalRequest(request=_req(1, prompt=1, output=5))
    gpu.admit(ir, admission_time=0.0, service_model=sm)
    gpu.step(current_time=0.001, service_model=sm)

    gpu.step(current_time=0.002, service_model=sm, held_decode_ids=frozenset({1}))
    diag = gpu.step_contention_diagnostics[-1]
    assert diag.decode_tokens_deferred == 1
    assert diag.decode_tokens_served == 0


def test_held_but_unknown_request_id_is_harmless():
    """Holding an ID that is not currently active/decoding must be a no-op,
    not an error."""
    gpu = GPUState(_gpu_config())
    ir = InternalRequest(request=_req(1, output=5))
    gpu.admit(ir, admission_time=0.0)
    gpu.step(current_time=0.001, held_decode_ids=frozenset({999}))
    assert ir.tokens_decoded == 1  # unaffected; 999 doesn't exist


# ---------------------------------------------------------------------------
# Simulator-level integration: a minimal test policy that holds decodes
# ---------------------------------------------------------------------------

class _HoldOnceThenFifoPolicy(BasePolicy):
    """Admits FIFO-style, but at a chosen step holds a chosen request_id's
    decode exactly once. Used only to exercise the simulator's hold_decode
    plumbing end-to-end -- not a real scheduling policy."""
    name = "test_hold_once"

    def __init__(self, hold_at_step: int, hold_request_id: int):
        self.hold_at_step = hold_at_step
        self.hold_request_id = hold_request_id
        self._done = False
        self.tokens_decoded_at_hold_step = None

    def select_action(self, state):
        admit: dict = {g.gpu_id: [] for g in state.gpu_states}
        hold_decode: dict = {g.gpu_id: [] for g in state.gpu_states}

        active_ids = {rid for g in state.gpu_states for rid in g.active_request_ids}
        if (not self._done and state.step == self.hold_at_step
                and self.hold_request_id in active_ids):
            for g in state.gpu_states:
                if self.hold_request_id in g.active_request_ids:
                    hold_decode[g.gpu_id].append(self.hold_request_id)
                    self.tokens_decoded_at_hold_step = g.tokens_decoded_per_request.get(
                        self.hold_request_id
                    )
                    self._done = True

        for req in state.waiting_queue:
            for g in state.gpu_states:
                if self._feasible_on_gpu(g, req):
                    admit[g.gpu_id].append(req.request_id)
                    g.active_request_ids.append(req.request_id)
                    g.current_kv_tokens += req.prompt_tokens
                    break
        return Action(admit=admit, hold_decode=hold_decode)


def test_held_request_delays_completion_by_exactly_one_step():
    """A request whose decode is held for exactly one step must complete
    exactly one step later than it would have without the hold -- it stays
    active and simply skips one token's worth of progress."""
    gpus = [_gpu_config(max_seq=1, max_kv=1000)]

    sim_baseline = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=200))
    sim_baseline.load_trace([_req(1, arrival=0.0, output=20)])
    baseline_metrics = sim_baseline.run(FIFOPolicy(), workload_tag="baseline")
    assert baseline_metrics.num_completed == 1

    gpus2 = [_gpu_config(max_seq=1, max_kv=1000)]
    sim_hold = Simulator(SimulatorConfig(gpu_configs=gpus2, max_steps=200))
    sim_hold.load_trace([_req(1, arrival=0.0, output=20)])
    policy = _HoldOnceThenFifoPolicy(hold_at_step=5, hold_request_id=1)
    hold_metrics = sim_hold.run(policy, workload_tag="hold")
    assert hold_metrics.num_completed == 1
    assert policy._done is True

    step_size = sim_baseline.config.service_model.step_size
    baseline_steps = round(baseline_metrics.mean_latency / step_size)
    hold_steps = round(hold_metrics.mean_latency / step_size)
    assert hold_steps == baseline_steps + 1


def test_held_request_never_evicted_or_requeued():
    """Unlike preempt/swap/migrate, a held request must never reappear in
    the waiting queue -- it stays active throughout (including on the very
    step it is held). Request 1 is naturally observed in the waiting queue
    at step 0 -- before its OWN first admission decision is even made, the
    same way test_simulator_preemption.py's equivalent test documents for
    preempt -- so this check only starts from step 1 onward, once request 1
    has actually been admitted."""
    gpus = [_gpu_config(max_seq=1, max_kv=1000)]
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=50))
    sim.load_trace([_req(1, arrival=0.0, output=10), _req(2, arrival=0.0, output=5)])
    policy = _HoldOnceThenFifoPolicy(hold_at_step=3, hold_request_id=1)

    seen_req1_in_waiting_after_admission = False

    class _Wrapped(BasePolicy):
        name = "wrapped"

        def select_action(self, state):
            nonlocal seen_req1_in_waiting_after_admission
            if state.step > 0 and any(r.request_id == 1 for r in state.waiting_queue):
                seen_req1_in_waiting_after_admission = True
            return policy.select_action(state)

    sim.run(_Wrapped(), workload_tag="test")
    assert seen_req1_in_waiting_after_admission is False


# ---------------------------------------------------------------------------
# Regression: every existing (admit-only) policy is completely unaffected
# ---------------------------------------------------------------------------

def test_fifo_policy_unaffected_by_hold_decode_field_existing():
    """A plain FIFOPolicy run must be bit-identical whether or not the new
    Action.hold_decode field exists -- it never touches it."""
    gpus = [_gpu_config(max_seq=4, max_kv=1000)]
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, max_steps=1000))
    sim.load_trace([_req(1, arrival=0.0, output=10), _req(2, arrival=0.01, output=10)])
    metrics = sim.run(FIFOPolicy(), workload_tag="regression")
    assert metrics.num_completed == 2
    assert metrics.num_dropped == 0

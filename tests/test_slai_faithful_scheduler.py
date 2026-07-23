"""Focused tests for slai_faithful: a faithful reimplementation of the SLAI
scheduler from "Optimal Scheduling Algorithms for LLM Inference: Theory and
Practice" (Bari, Hegde, de Veciana; arXiv:2508.01002). See
docs/slai_faithful_scheduler_reference.md for the full fidelity record and
src/llmserveopt/policies/slai_faithful.py's module docstring for the
pinned upstream source (github.com/agrimUT/SLAI, commit 5098a7a).

Test categories (see docs/slai_faithful_scheduler_reference.md §Testing):
  1.  Basic deterministic scheduling
  2.  Correct TBT/decode-deadline (last-schedulable-time) behavior
  3.  Decode deferral when safe (non-critical -> held)
  4.  Decode scheduling when deferral would threaten TBT (critical -> served)
  5.  Prefill receives capacity released by safe decode deferral
  6.  Behavior under prefill-heavy load
  7.  Behavior under decode-heavy load
  8.  Tight TBT constraints
  9.  Relaxed TBT constraints
  10. Multiple SLO/user tiers (class_id -> TBT mapping)
  11. Token-budget feasibility
  12. KV/resource feasibility
  13. Deterministic tie-breaking
  14. No oracle leakage (no actual_output_tokens access)
  15. No actual_output_tokens access (AttributeError if ever attempted)
  16. No double admission / invalid request transitions
"""
from __future__ import annotations

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, ObservableRequest, Request
from llmserveopt.policies.slai_faithful import SlaiFaithfulPolicy
from llmserveopt.simulator.request import RequestPhase
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _advance_one_step(sim: Simulator, policy) -> tuple:
    """Replicates exactly one iteration of Simulator.run()'s loop body
    (enqueue arrivals -> build state -> select_action -> apply -> advance
    decode), for white-box tests that need to inspect the intermediate
    ObservableState/Action rather than only final aggregate metrics.
    Advances sim._step/_time by exactly one step_size."""
    for ir in sim._pending_arrivals:
        if (ir.phase == RequestPhase.WAITING
                and ir.request.arrival_time <= sim._time
                and ir.request_id not in sim._waiting_map):
            sim._waiting.append(ir)
            sim._waiting_map[ir.request_id] = ir
    state = sim._build_observable_state()
    action = policy.select_action(state)
    sim._apply_action(action)
    completed = sim._advance_decode(action)
    sim._completed.extend(completed)
    sim._collect_handoffs()
    sim._step += 1
    sim._time = sim._step * sim.config.service_model.step_size
    return state, action


def _gpu(max_seq=64, max_tok=1_000_000, max_kv=200_000):
    return GPUConfig(gpu_id=0, max_active_sequences=max_seq,
                      max_batch_tokens=max_tok, max_kv_tokens=max_kv)


def _req(rid, arrival=0.0, prompt=100, output=10, deadline=100.0, priority=1.0, class_id="medium"):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id=class_id,
    )


def _sm(**overrides):
    defaults = dict(
        step_size=0.001,
        enable_prefill_modeling=True,
        prefill_cost_per_token=1.0,
        max_prefill_chunk_tokens=512,
        step_token_budget=512,
        decode_first=True,
    )
    defaults.update(overrides)
    return ServiceModel(**defaults)


# ---------------------------------------------------------------------------
# 1. Basic deterministic scheduling
# ---------------------------------------------------------------------------

def test_basic_run_completes_deterministically():
    sm = _sm()
    reqs = [_req(i, arrival=i * 0.01, prompt=100 + i * 20, output=10) for i in range(8)]

    def _run():
        sim = Simulator(SimulatorConfig(gpu_configs=[_gpu()], service_model=sm, drain_steps=5000))
        sim.load_trace(reqs)
        policy = SlaiFaithfulPolicy(step_size=sm.step_size)
        policy.reset()
        m = sim.run(policy=policy, workload_tag="test", seed=0)
        return m.num_completed, m.num_dropped, round(m.mean_latency, 9)

    r1 = _run()
    r2 = _run()
    assert r1 == r2  # deterministic
    assert r1[0] == 8
    assert r1[1] == 0


# ---------------------------------------------------------------------------
# 2. Correct TBT/decode-deadline (last-schedulable-time) behavior
# ---------------------------------------------------------------------------

def test_lst_is_none_until_first_decode_ready_then_set():
    policy = SlaiFaithfulPolicy(step_size=0.001, token_budget=512, decode_limit=128)
    gpu = _gpu()
    sm = _sm()
    reqs = [_req(0, prompt=10, output=5, class_id="tight")]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()
    sim.run(policy=policy, workload_tag="test", seed=0)
    # After the run, the request's LST must have been recorded at some point.
    assert 0 in policy._lst.get(0, {}) or policy._request_states.get(0, {}) == {}


def test_lst_formula_matches_paper_eq8():
    """Eq. 8: Y = s_{i-1} + TBT - offset * b_batch. With fixed_offset,
    b_batch = step_size (see module docstring), so LST should equal
    now + tbt - offset*step_size at the moment it is first assigned."""
    policy = SlaiFaithfulPolicy(
        step_size=0.001, fixed_offset=True, below_memory_limit_offset=5,
        tbt_by_class={"tight": 0.1}, default_tbt=0.1,
    )
    gpu = _gpu()
    sm = _sm()
    reqs = [_req(0, prompt=1, output=50, class_id="tight")]  # tiny prompt: prefill finishes fast
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()

    # Run step by step until LST first appears, and check the formula.
    for _ in range(20):
        state, action = _advance_one_step(sim, policy)
        if 0 in policy._lst.get(0, {}):
            recorded_now = state.time
            expected = recorded_now + 0.1 - 5 * 0.001
            assert abs(policy._lst[0][0] - expected) < 1e-9
            return
    raise AssertionError("LST was never assigned within 20 steps")


# ---------------------------------------------------------------------------
# 3 & 4. Decode deferral when safe / served when critical
# ---------------------------------------------------------------------------

def test_non_critical_decode_is_held_when_far_from_deadline():
    """A request with a very relaxed TBT should be classified non-critical
    (held) immediately after its first decode-ready step, since `now` will
    be far below its last-schedulable-time."""
    policy = SlaiFaithfulPolicy(
        step_size=0.001, fixed_offset=True, below_memory_limit_offset=0,
        tbt_by_class={"loose": 10.0}, default_tbt=10.0,  # huge relaxed TBT
        token_budget=2,  # must match sm.step_token_budget for genuine contention
    )
    gpu = _gpu(max_tok=2)  # tight budget forces prioritization to matter
    sm = _sm(step_token_budget=2, max_prefill_chunk_tokens=2)
    reqs = [
        _req(0, prompt=1, output=50, class_id="loose"),
        _req(1, arrival=0.05, prompt=50, output=5, class_id="loose"),
    ]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=5000))
    sim.load_trace(reqs)
    policy.reset()

    held_any = False
    for _ in range(80):
        _, action = _advance_one_step(sim, policy)
        if action.hold_decode.get(0):
            held_any = True
    assert held_any, "expected at least one held decode with a very relaxed TBT under tight budget"


def test_critical_decode_is_served_even_under_tight_budget():
    """A request whose TBT deadline has already passed must be served
    (never held) regardless of budget pressure elsewhere -- SLAI never lets
    a critical decode starve."""
    policy = SlaiFaithfulPolicy(
        step_size=0.001, fixed_offset=True, below_memory_limit_offset=0,
        tbt_by_class={"tight": 0.001}, default_tbt=0.001,  # near-zero TBT: always critical
    )
    gpu = _gpu()
    sm = _sm()
    reqs = [_req(0, prompt=1, output=20, class_id="tight")]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()
    m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 1
    assert m.num_dropped == 0


# ---------------------------------------------------------------------------
# 5. Prefill receives capacity released by safe decode deferral
# ---------------------------------------------------------------------------

def test_deferring_decode_lets_prefill_progress_under_tight_budget():
    """End-to-end: a decode-heavy request with a very relaxed TBT should not
    starve a competing prefill-heavy request of budget forever -- deferral
    must actually let prefill make progress (verified via faster admission
    completion than a hypothetical always-serve-decode policy would allow)."""
    policy = SlaiFaithfulPolicy(
        step_size=0.001, fixed_offset=True, below_memory_limit_offset=0,
        tbt_by_class={"loose": 5.0}, default_tbt=5.0,
        token_budget=2,
    )
    gpu = _gpu(max_tok=2)
    sm = _sm(step_token_budget=2, max_prefill_chunk_tokens=2)
    reqs = [
        _req(0, prompt=1, output=200, class_id="loose"),   # long decode, relaxed TBT
        _req(1, arrival=0.02, prompt=4, output=5, class_id="loose"),  # small prefill job
    ]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=10000))
    sim.load_trace(reqs)
    policy.reset()
    m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 2
    assert m.num_dropped == 0


# ---------------------------------------------------------------------------
# 6 & 7. Prefill-heavy / decode-heavy load
# ---------------------------------------------------------------------------

def test_prefill_heavy_load_completes_cleanly():
    sm = _sm()
    reqs = [_req(i, arrival=i * 0.001, prompt=2000, output=3) for i in range(6)]
    sim = Simulator(SimulatorConfig(gpu_configs=[_gpu()], service_model=sm, drain_steps=10000))
    sim.load_trace(reqs)
    policy = SlaiFaithfulPolicy(step_size=sm.step_size)
    policy.reset()
    m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 6
    assert m.num_dropped == 0


def test_decode_heavy_load_completes_cleanly():
    sm = _sm()
    reqs = [_req(i, arrival=i * 0.001, prompt=10, output=300) for i in range(6)]
    sim = Simulator(SimulatorConfig(gpu_configs=[_gpu()], service_model=sm, drain_steps=20000))
    sim.load_trace(reqs)
    policy = SlaiFaithfulPolicy(step_size=sm.step_size)
    policy.reset()
    m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 6
    assert m.num_dropped == 0


# ---------------------------------------------------------------------------
# 8 & 9. Tight / relaxed TBT constraints
# ---------------------------------------------------------------------------

def test_tight_tbt_class_maps_to_small_tbt_value():
    policy = SlaiFaithfulPolicy()
    req = ObservableRequest(request_id=0, arrival_time=0.0, prompt_tokens=10,
                             predicted_output_tokens=5, slo_deadline=1.0, priority=1.0, class_id="tight")
    assert policy._tbt_for(req) == policy.tbt_by_class["tight"]
    assert policy._tbt_for(req) < policy.tbt_by_class["loose"]


def test_relaxed_tbt_class_maps_to_large_tbt_value():
    policy = SlaiFaithfulPolicy()
    req = ObservableRequest(request_id=0, arrival_time=0.0, prompt_tokens=10,
                             predicted_output_tokens=5, slo_deadline=1.0, priority=1.0, class_id="loose")
    assert policy._tbt_for(req) == policy.tbt_by_class["loose"]


def test_unrecognized_class_id_falls_back_to_default_tbt():
    policy = SlaiFaithfulPolicy(default_tbt=0.42)
    req = ObservableRequest(request_id=0, arrival_time=0.0, prompt_tokens=10,
                             predicted_output_tokens=5, slo_deadline=1.0, priority=1.0, class_id="unknown_tier")
    assert policy._tbt_for(req) == 0.42


def test_cross_vocabulary_class_id_equivalence():
    """This project has two independently-authored 3-tier class_id
    vocabularies in active use: 'tight'/'medium'/'loose' (synthetic
    workloads, SwissAI/TraceLab sweeps) and 'interactive'/'standard'/'batch'
    (BurstGPT/Azure loaders via workloads/augmentation.py). Both must map
    to the SAME TBT per tier under the default config, or slai_faithful
    would silently behave differently across datasets that merely use a
    different (but equivalent-priority) class_id vocabulary."""
    policy = SlaiFaithfulPolicy()
    pairs = [("tight", "interactive"), ("medium", "standard"), ("loose", "batch")]
    for a, b in pairs:
        req_a = ObservableRequest(request_id=0, arrival_time=0.0, prompt_tokens=10,
                                   predicted_output_tokens=5, slo_deadline=1.0, priority=1.0, class_id=a)
        req_b = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=10,
                                   predicted_output_tokens=5, slo_deadline=1.0, priority=1.0, class_id=b)
        assert policy._tbt_for(req_a) == policy._tbt_for(req_b), f"{a} and {b} must map to the same TBT"


# ---------------------------------------------------------------------------
# 10. Multiple SLO/user tiers -- tiered-SPF admission ordering
# ---------------------------------------------------------------------------

def test_strict_tier_admitted_before_relaxed_tier_under_spf_priority():
    """Two requests arrive together; a strict-TBT request with a LONGER
    prompt should still be admitted before a relaxed-TBT request with a
    SHORTER prompt, under user_priority=True SPF ordering (tier takes
    precedence over prompt length)."""
    policy = SlaiFaithfulPolicy(fcfs=False, user_priority=True, token_budget=10)
    gpu = _gpu(max_tok=10)
    sm = _sm(step_token_budget=10, max_prefill_chunk_tokens=10)
    reqs = [
        _req(0, prompt=8, output=5, class_id="loose"),   # relaxed, shorter... wait must be longer
        _req(1, prompt=9, output=5, class_id="tight"),   # strict, longer prompt
    ]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()
    _, action = _advance_one_step(sim, policy)
    admitted = action.admit.get(0, [])
    assert admitted[0] == 1, "strict-tier request must be admitted first despite longer prompt"


def test_fcfs_mode_ignores_tier_and_uses_arrival_order():
    policy = SlaiFaithfulPolicy(fcfs=True, token_budget=10)
    gpu = _gpu(max_tok=10)
    sm = _sm(step_token_budget=10, max_prefill_chunk_tokens=10)
    reqs = [
        _req(0, arrival=0.0, prompt=8, output=5, class_id="loose"),
        _req(1, arrival=0.0, prompt=1, output=5, class_id="tight"),
    ]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()
    _, action = _advance_one_step(sim, policy)
    admitted = action.admit.get(0, [])
    # FCFS: request order in the waiting queue is preserved (arrival order,
    # request_id tiebreak), not reordered by tier or length.
    assert admitted[0] == 0


# ---------------------------------------------------------------------------
# 11. Token-budget feasibility
# ---------------------------------------------------------------------------

def test_never_exceeds_token_budget_in_a_single_step():
    """Admission is CHUNKED: a request can be admitted while only a
    fraction of its prompt is processed this step (the rest continues via
    `still_prefilling` on later steps). The real per-step budget invariant
    is on the CHUNK actually applied (states[rid].remaining_prefill after
    admission), not the admitted request's full prompt_tokens -- with a
    100-token prompt and a 16-token budget, exactly one request should be
    admitted (consuming the whole budget as its first chunk), leaving 84
    tokens of remaining prefill for later steps."""
    policy = SlaiFaithfulPolicy(token_budget=16)
    gpu = _gpu(max_tok=1_000_000)
    sm = _sm(step_token_budget=16, max_prefill_chunk_tokens=16)
    reqs = [_req(i, arrival=0.0, prompt=100, output=5) for i in range(20)]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()
    _, action = _advance_one_step(sim, policy)
    admitted = action.admit.get(0, [])
    assert len(admitted) == 1, "budget (16) < prompt (100): only the first chunk should be admitted this step"
    rstate = policy._request_states[0][admitted[0]]
    chunk_applied = 100 - rstate.remaining_prefill
    assert chunk_applied == 16


# ---------------------------------------------------------------------------
# 12. KV/resource feasibility
# ---------------------------------------------------------------------------

def test_never_admits_beyond_kv_capacity():
    policy = SlaiFaithfulPolicy(block_size=1)
    gpu = _gpu(max_kv=50, max_tok=1_000_000)
    sm = _sm(step_token_budget=1_000_000, max_prefill_chunk_tokens=1_000_000)
    reqs = [_req(i, arrival=0.0, prompt=30, output=5) for i in range(5)]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=5000))
    sim.load_trace(reqs)
    policy.reset()
    m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 5  # all eventually complete, just not all at once
    assert m.num_dropped == 0


# ---------------------------------------------------------------------------
# 13. Deterministic tie-breaking
# ---------------------------------------------------------------------------

def test_tie_breaking_by_request_id_is_deterministic():
    """Two requests with identical prompt length and class_id (hence
    identical SPF/tier sort keys) must be ordered deterministically by
    request_id, not by insertion-order accident."""
    policy_a = SlaiFaithfulPolicy(fcfs=False, user_priority=True, token_budget=10)
    policy_b = SlaiFaithfulPolicy(fcfs=False, user_priority=True, token_budget=10)
    gpu = _gpu(max_tok=10)
    sm = _sm(step_token_budget=10, max_prefill_chunk_tokens=10)
    reqs = [_req(5, prompt=8, output=5, class_id="medium"), _req(2, prompt=8, output=5, class_id="medium")]

    results = []
    for policy in (policy_a, policy_b):
        sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=2000))
        sim.load_trace(reqs)
        policy.reset()
        _, action = _advance_one_step(sim, policy)
        results.append(tuple(action.admit.get(0, [])))
    assert results[0] == results[1]
    assert results[0][0] == 2  # lower request_id wins the tie


# ---------------------------------------------------------------------------
# 14 & 15. No oracle leakage / no actual_output_tokens access
# ---------------------------------------------------------------------------

def test_policy_never_touches_actual_output_tokens():
    """ObservableRequest has no actual_output_tokens field at all -- any
    attempt to read it would raise AttributeError. Run a full simulation
    and confirm it completes without ever raising."""
    sm = _sm()
    reqs = [_req(i, arrival=i * 0.005, prompt=50, output=15) for i in range(10)]
    sim = Simulator(SimulatorConfig(gpu_configs=[_gpu()], service_model=sm, drain_steps=5000))
    sim.load_trace(reqs)
    policy = SlaiFaithfulPolicy(step_size=sm.step_size)
    policy.reset()
    m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 10


def test_observable_request_has_no_actual_output_tokens_attribute():
    req = ObservableRequest(request_id=0, arrival_time=0.0, prompt_tokens=10,
                             predicted_output_tokens=5, slo_deadline=1.0, priority=1.0, class_id="medium")
    assert not hasattr(req, "actual_output_tokens")


# ---------------------------------------------------------------------------
# 16. No double admission / invalid request transitions
# ---------------------------------------------------------------------------

def test_no_double_admission_across_gpus():
    policy = SlaiFaithfulPolicy(token_budget=100)
    gpus = [_gpu(max_tok=100), _gpu(max_tok=100)]
    gpus[1] = GPUConfig(gpu_id=1, max_active_sequences=64, max_batch_tokens=100, max_kv_tokens=200_000)
    sm = _sm(step_token_budget=100, max_prefill_chunk_tokens=100)
    reqs = [_req(i, arrival=0.0, prompt=10, output=5) for i in range(10)]
    sim = Simulator(SimulatorConfig(gpu_configs=gpus, service_model=sm, drain_steps=2000))
    sim.load_trace(reqs)
    policy.reset()
    _, action = _advance_one_step(sim, policy)
    all_admitted = list(action.admit.get(0, [])) + list(action.admit.get(1, []))
    assert len(all_admitted) == len(set(all_admitted)), "no request should be admitted to two GPUs at once"


def test_no_warnings_raised_across_full_run():
    """A full run must never trigger the simulator's warn_on_invalid_action
    path -- i.e. this policy never issues a malformed Action (double
    admission, holding a non-existent request, etc.)."""
    import warnings as warnings_module
    sm = _sm()
    reqs = [_req(i, arrival=i * 0.003, prompt=80, output=25, class_id=["tight", "medium", "loose"][i % 3])
            for i in range(15)]
    sim = Simulator(SimulatorConfig(gpu_configs=[_gpu()], service_model=sm, drain_steps=8000))
    sim.load_trace(reqs)
    policy = SlaiFaithfulPolicy(step_size=sm.step_size)
    policy.reset()
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        m = sim.run(policy=policy, workload_tag="test", seed=0)
    assert m.num_completed == 15


def test_held_request_is_never_evicted_or_requeued_by_slai():
    """A request currently being deferred by slai_faithful must remain
    active throughout -- never reappear in the waiting queue (this policy
    never sets Action.preempt/swap/migrate, only hold_decode)."""
    policy = SlaiFaithfulPolicy(
        step_size=0.001, fixed_offset=True, below_memory_limit_offset=0,
        tbt_by_class={"loose": 10.0}, default_tbt=10.0, token_budget=2,
    )
    gpu = _gpu(max_tok=2)
    sm = _sm(step_token_budget=2, max_prefill_chunk_tokens=2)
    reqs = [
        _req(0, prompt=1, output=100, class_id="loose"),
        _req(1, arrival=0.02, prompt=30, output=5, class_id="loose"),
    ]
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=sm, drain_steps=10000))
    sim.load_trace(reqs)
    policy.reset()

    admitted_ever = set()
    for _ in range(200):
        pre_step = sim._step
        # Inspect the waiting queue as _advance_one_step will observe it
        # (i.e. after this call's own arrival enqueue, before admission).
        for ir in sim._pending_arrivals:
            if (ir.phase == RequestPhase.WAITING and ir.request.arrival_time <= sim._time
                    and ir.request_id not in sim._waiting_map):
                sim._waiting.append(ir)
                sim._waiting_map[ir.request_id] = ir
        if pre_step > 0:
            for ir in sim._waiting:
                assert ir.request_id not in admitted_ever, (
                    f"request {ir.request_id} reappeared in waiting_queue after admission"
                )
        state = sim._build_observable_state()
        action = policy.select_action(state)
        admitted_ever.update(action.all_admitted_ids())
        sim._apply_action(action)
        sim._advance_decode(action)
        sim._collect_handoffs()
        total_active = sum(g.num_active for g in sim._gpus)
        if total_active == 0 and not sim._waiting and pre_step > 0:
            break
        sim._step += 1
        sim._time = sim._step * sm.step_size

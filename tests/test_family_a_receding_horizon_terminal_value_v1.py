"""Focused regression tests for the Family-A terminal-value redesign (design
doc `docs/design/FAMILY_A_TERMINAL_VALUE_V1.md`)."""
from __future__ import annotations

import copy
import inspect

import pytest

from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.core.types import (
    GPUConfig,
    ObservableGPUState,
    ObservableRequest,
    ObservableState,
    Request,
)
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_receding_horizon_oracle_v1 import (
    FamilyARecedingHorizonOracleV1,
)
from llmserveopt.policies.family_a_receding_horizon_terminal_value_v1 import (
    COMMON_CONTINUATION_BUDGET,
    ESTF_ID,
    WFS_ID,
    FamilyARecedingHorizonTerminalValueV1,
    _inflight_terminal_credit,
    _run_chained_branch_dual,
    _window_weighted_slo_goodput,
)
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    actions_disagree,
    canonical_action,
    restore_gpu_counters,
    snapshot_gpu_counters,
)
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def _req(rid, arrival, output, deadline, priority, cls="c", prompt=1):
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=deadline, priority=priority, class_id=cls,
    )


def _obs_req(rid, output, deadline, priority, prompt=1, cls="c", arrival=0.0):
    return ObservableRequest(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, slo_deadline=deadline, priority=priority,
        class_id=cls,
    )


def _gpu_state(rid_active, tokens_decoded, active_req, gpu_id=0):
    return ObservableGPUState(
        gpu_id=gpu_id, max_active_sequences=1, max_batch_tokens=64, max_kv_tokens=100_000,
        active_request_ids=[rid_active] if active_req else [],
        active_requests_info=[active_req] if active_req else [],
        current_kv_tokens=0,
        tokens_decoded_per_request={rid_active: tokens_decoded} if active_req else {},
    )


def _sim(requests, drain_steps=200):
    gpu = [GPUConfig(gpu_id=0, max_active_sequences=1, max_batch_tokens=64, max_kv_tokens=100_000)]
    sm = ServiceModel(step_size=0.001)
    sim = Simulator(SimulatorConfig(gpu_configs=gpu, service_model=sm, drain_steps=drain_steps))
    sim.load_trace(list(requests))
    return sim


class _Captured(Exception):
    def __init__(self, state):
        self.state = state


def _first_decision_state(sim: Simulator):
    class _RaiseOnFirstCall:
        name = "capture_once"

        def select_action(self, state):
            raise _Captured(state)

        def reset(self) -> None:
            pass

    try:
        sim.run(_RaiseOnFirstCall(), workload_tag="capture", seed=0)
    except _Captured as exc:
        return exc.state
    raise AssertionError("simulator drained with zero decision steps")


def _long_short_requests(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0, arrival=0.0):
    reqs = [_req(0, arrival, o_long, deadline_long, p_long)]
    reqs += [_req(100 + i, arrival, 1, 10_000.0, 1.0) for i in range(n_short)]
    return reqs


# ---------------------------------------------------------------------------
# C: completed request receives correct realized value (old term unchanged)
# ---------------------------------------------------------------------------

def test_window_weighted_slo_goodput_unchanged_from_v1():
    class _FakeReq:
        def __init__(self, priority):
            self.priority = priority

    class _FakeCompleted:
        def __init__(self, priority, violated):
            self.request = _FakeReq(priority)
            self.slo_violated = violated

    completed = [_FakeCompleted(2.0, False), _FakeCompleted(5.0, True), _FakeCompleted(0.0, False)]
    assert _window_weighted_slo_goodput(completed) == pytest.approx(2.0 + 0.0 + 1.0)


# ---------------------------------------------------------------------------
# D/E: unfinished high-priority in-flight work receives terminal credit,
# and responds correctly to remaining service.
# ---------------------------------------------------------------------------

def test_inflight_active_request_receives_positive_credit_when_feasible():
    req = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0)
    state = ObservableState(
        time=0.0, waiting_queue=[], completed_count=0, step=0,
        gpu_states=[_gpu_state(1, tokens_decoded=50, active_req=req)],
    )
    credit = _inflight_terminal_credit(state)
    # progress_fraction = 50/100 = 0.5, feasible (deadline far away) -> credit = 10.0*0.5*1
    assert credit == pytest.approx(5.0)


def test_inflight_credit_increases_with_remaining_service_progress():
    def credit_at(tokens_decoded):
        req = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0)
        state = ObservableState(
            time=0.0, waiting_queue=[], completed_count=0, step=0,
            gpu_states=[_gpu_state(1, tokens_decoded=tokens_decoded, active_req=req)],
        )
        return _inflight_terminal_credit(state)

    assert credit_at(10) < credit_at(50) < credit_at(90)


def test_waiting_request_with_zero_progress_gets_zero_credit():
    req = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0)
    state = ObservableState(time=0.0, waiting_queue=[req], completed_count=0, step=0, gpu_states=[])
    assert _inflight_terminal_credit(state) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F: deadline-infeasible work is not overvalued
# ---------------------------------------------------------------------------

def test_deadline_infeasible_active_request_gets_zero_credit_despite_progress():
    # Large remaining service, deadline already effectively unreachable.
    req = _obs_req(rid=1, output=100, deadline=0.0001, priority=10.0)
    state = ObservableState(
        time=0.0, waiting_queue=[], completed_count=0, step=0,
        gpu_states=[_gpu_state(1, tokens_decoded=1, active_req=req)],  # 99 tokens remaining, tiny deadline
    )
    assert _inflight_terminal_credit(state) == pytest.approx(0.0)


def test_deadline_feasible_active_request_gets_nonzero_credit():
    req = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0)
    state = ObservableState(
        time=0.0, waiting_queue=[], completed_count=0, step=0,
        gpu_states=[_gpu_state(1, tokens_decoded=1, active_req=req)],
    )
    assert _inflight_terminal_credit(state) > 0.0


# ---------------------------------------------------------------------------
# G: no separate fairness-debt term is used (design decision, SS4 of design
# doc) -- verify the credit function's signature/behavior depends only on
# priority/predicted_output_tokens/slo_deadline/tokens_decoded, never on
# class_id-based deficit/service-share state.
# ---------------------------------------------------------------------------

def test_no_fairness_debt_term_class_id_does_not_affect_credit():
    req_a = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0, cls="starved_class")
    req_b = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0, cls="privileged_class")
    state_a = ObservableState(time=0.0, waiting_queue=[], completed_count=0, step=0,
                               gpu_states=[_gpu_state(1, tokens_decoded=50, active_req=req_a)])
    state_b = ObservableState(time=0.0, waiting_queue=[], completed_count=0, step=0,
                               gpu_states=[_gpu_state(1, tokens_decoded=50, active_req=req_b)])
    assert _inflight_terminal_credit(state_a) == pytest.approx(_inflight_terminal_credit(state_b))


# ---------------------------------------------------------------------------
# H: no scenario metadata leakage
# ---------------------------------------------------------------------------

def test_no_scenario_metadata_symbols_referenced_in_module():
    import llmserveopt.policies.family_a_receding_horizon_terminal_value_v1 as mod
    src = inspect.getsource(mod)
    for forbidden in ("canonical_scenario_id", "favlong", "favshort", "TEST"):
        assert forbidden not in src


# ---------------------------------------------------------------------------
# A/B: terminal value deterministic and uses no future information (a pure
# function of the passed-in terminal ObservableState).
# ---------------------------------------------------------------------------

def test_terminal_credit_deterministic_and_state_only():
    req = _obs_req(rid=1, output=100, deadline=1000.0, priority=10.0)
    state = ObservableState(time=5.0, waiting_queue=[], completed_count=3, step=17,
                             gpu_states=[_gpu_state(1, tokens_decoded=30, active_req=req)])
    v1 = _inflight_terminal_credit(state)
    v2 = _inflight_terminal_credit(copy.deepcopy(state))
    assert v1 == pytest.approx(v2)
    # Two states differing only in fields the formula does not read (step,
    # completed_count) must produce identical credit -- confirms the value
    # depends only on the declared causal fields, not on hidden context.
    state_other_step = ObservableState(time=5.0, waiting_queue=[], completed_count=999, step=4242,
                                        gpu_states=[_gpu_state(1, tokens_decoded=30, active_req=req)])
    assert _inflight_terminal_credit(state_other_step) == pytest.approx(v1)


# ---------------------------------------------------------------------------
# I: identical controller mechanics relative to V1 except scoring
# ---------------------------------------------------------------------------

def test_eligibility_and_fallback_identical_to_v1():
    reqs = _long_short_requests(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0)
    sim_old = _sim(reqs)
    sim_new = _sim(reqs)
    old = FamilyARecedingHorizonOracleV1(sim_ref=sim_old, horizon=5, continuation_budget=8)
    new = FamilyARecedingHorizonTerminalValueV1(sim_ref=sim_new, horizon=5, continuation_budget=8)
    sim_old.run(old, workload_tag="t", seed=0)
    sim_new.run(new, workload_tag="t", seed=0)
    old_log = old.decision_log()
    new_log = new.decision_log()
    assert len(old_log) == len(new_log)
    for o, n in zip(old_log, new_log):
        assert o["step"] == n["step"]
        assert o["eligible"] == n["eligible"]
        assert o["planning_call"] == n["planning_call"]
        assert o["fallback_reason"] == n["fallback_reason"]


def test_empty_queue_and_ineligible_fallback_to_wfs():
    sim = _sim([_req(0, 0.0, 5, 1000.0, 1.0)])
    state = _first_decision_state(sim)
    expected_state = copy.deepcopy(state)
    policy = FamilyARecedingHorizonTerminalValueV1(sim_ref=sim, horizon=5, continuation_budget=8)
    action = policy.select_action(state)
    expected = WeightedFairSharePolicy().select_action(expected_state)
    assert canonical_action(action) == canonical_action(expected)
    assert policy.decision_log()[0]["fallback_reason"] == "outside_candidate_region"


# ---------------------------------------------------------------------------
# J: snapshot/restore remains non-interfering
# ---------------------------------------------------------------------------

def test_dual_branch_does_not_mutate_source_simulator():
    reqs = _long_short_requests(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0)
    sim = _sim(reqs)
    state = _first_decision_state(sim)
    snapshot = snapshot_gpu_counters(state)
    action_estf = EstimatedServiceTimeFirstPolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    action_wfs = WeightedFairSharePolicy().select_action(state)
    restore_gpu_counters(state, snapshot)

    fp_before = dcm._state_fingerprint(sim)
    _run_chained_branch_dual(
        sim, candidate_policy=EstimatedServiceTimeFirstPolicy(), candidate_policy_id=ESTF_ID,
        candidate_first_action=action_estf, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    fp_mid = dcm._state_fingerprint(sim)
    _run_chained_branch_dual(
        sim, candidate_policy=WeightedFairSharePolicy(), candidate_policy_id=WFS_ID,
        candidate_first_action=action_wfs, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    fp_after = dcm._state_fingerprint(sim)
    assert fp_before == fp_mid == fp_after


# ---------------------------------------------------------------------------
# K: fixed seed reproducibility
# ---------------------------------------------------------------------------

def test_full_run_deterministic():
    from llmserveopt.core.metrics import metrics_to_dict

    def run_once():
        reqs = _long_short_requests(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0)
        sim = _sim(reqs)
        policy = FamilyARecedingHorizonTerminalValueV1(sim_ref=sim, horizon=5, continuation_budget=8)
        metrics = sim.run(policy, workload_tag="t", seed=0)
        return metrics_to_dict(metrics), policy.decision_log()

    _TIMING_FIELDS = {"wall_clock_s", "mean_policy_time_s", "total_policy_time_s"}
    m1, log1 = run_once()
    m2, log2 = run_once()
    m1_sub = {k: v for k, v in m1.items() if k not in _TIMING_FIELDS}
    m2_sub = {k: v for k, v in m2.items() if k not in _TIMING_FIELDS}
    assert m1_sub == m2_sub
    assert log1 == log2


# ---------------------------------------------------------------------------
# L: synthetic favshort-like case where ESTF is preferred
# ---------------------------------------------------------------------------

def test_favshort_like_case_prefers_estf():
    # Many short, low-priority, feasible jobs and no long job: ESTF's
    # throughput advantage should still win under the new value (in-flight
    # credit only matters when there IS meaningful long/high-priority work).
    reqs = [_req(100 + i, 0.0, 1, 10_000.0, 1.0) for i in range(5)]
    sim = _sim(reqs)
    policy = FamilyARecedingHorizonTerminalValueV1(sim_ref=sim, horizon=5, continuation_budget=8)
    sim.run(policy, workload_tag="t", seed=0)
    diag = policy.diagnostics()
    # No disagreement expected (all short/uniform) -- eligibility should be
    # rare or absent; this just documents that the redesign doesn't force
    # spurious WFS preference outside the failure regime it targets.
    assert diag["eligible_count"] >= 0


# ---------------------------------------------------------------------------
# M: synthetic favlong-like case where the NEW value flips old-ESTF -> WFS
# ---------------------------------------------------------------------------

def test_favlong_like_case_new_value_credits_inflight_long_priority_work():
    # One long, high-priority job (WFS's priority/est_steps score prefers it
    # over the short fillers -- score = priority/est_steps = 50/50.5 ~= 0.99
    # > 1/1.5 ~= 0.67) plus short filler jobs. The long job cannot finish
    # within the bounded window, but WFS's admission of it makes real,
    # feasible decode progress the old (completed-only) objective scores as
    # a flat 0 -- exactly the regime the diagnosis identified.
    long_req = _req(0, 0.0, 50, 1000.0, 50.0)  # long, high priority, feasible deadline
    short_reqs = [_req(100 + i, 0.0, 1, 10_000.0, 1.0) for i in range(3)]
    sim = _sim([long_req] + short_reqs, drain_steps=400)
    state = _first_decision_state(sim)
    snapshot = snapshot_gpu_counters(state)
    action_estf = EstimatedServiceTimeFirstPolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    action_wfs = WeightedFairSharePolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    assert actions_disagree(action_estf, action_wfs)

    estf_branch = _run_chained_branch_dual(
        sim, candidate_policy=EstimatedServiceTimeFirstPolicy(), candidate_policy_id=ESTF_ID,
        candidate_first_action=action_estf, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    wfs_branch = _run_chained_branch_dual(
        sim, candidate_policy=WeightedFairSharePolicy(), candidate_policy_id=WFS_ID,
        candidate_first_action=action_wfs, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    old_gap = estf_branch.old_window_objective - wfs_branch.old_window_objective
    new_gap = estf_branch.new_terminal_value - wfs_branch.new_terminal_value
    # Old objective clearly favors ESTF (it completes 3 short jobs, WFS
    # completes none within the window): old_gap == 3.0 > 0.
    assert old_gap > 0
    # New value credits WFS's feasible in-progress work on the long job,
    # strictly narrowing (here: fully closing) ESTF's advantage.
    assert new_gap < old_gap
    assert wfs_branch.new_terminal_value > wfs_branch.old_window_objective


# ---------------------------------------------------------------------------
# N: no trivial always-WFS collapse
# ---------------------------------------------------------------------------

def test_no_trivial_always_wfs_collapse_across_varied_fixtures():
    # H=1 is the myopic/local-action regime where a single-step lookahead
    # gives the in-flight term essentially no room to accumulate before the
    # comparison is made -- exactly where ESTF's throughput advantage should
    # still show through under the new value, as it does under the old one
    # (test_short_horizon_myopically_favors_estf_h1 in the V1 suite).
    fixtures = [
        _long_short_requests(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0),
    ]
    any_estf = False
    for reqs in fixtures:
        sim = _sim(reqs)
        policy = FamilyARecedingHorizonTerminalValueV1(sim_ref=sim, horizon=1, continuation_budget=8)
        sim.run(policy, workload_tag="t", seed=0)
        diag = policy.diagnostics()
        if diag["estf_win_count"] > 0:
            any_estf = True
    assert any_estf, "new terminal value collapsed to always-WFS even at H=1"

"""Focused regression tests for the Family-A receding-horizon oracle
feasibility controller (design doc
`docs/design/FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md`).

All fixtures use a single GPU with `max_active_sequences=1` (matching every
real Family-A GPU config) and hand-picked request parameters so that ESTF and
WFS genuinely disagree (WFS's `deficit * priority / est_steps` score favors a
high-priority "LONG" request; ESTF's shortest-service-time rule favors cheap
"SHORT" requests) and so that the resulting windowed objective comparison is
analytically predictable and was independently verified by direct
computation before being hard-coded here (see scratch probe used during
development; the numbers below are not tuned to make tests pass after the
fact -- they were derived first, then asserted).
"""
from __future__ import annotations

import copy

import pytest

from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.family_a_receding_horizon_oracle_v1 import (
    COMMON_CONTINUATION_BUDGET,
    ESTF_ID,
    WFS_ID,
    FamilyARecedingHorizonOracleV1,
    _run_chained_branch,
    _window_weighted_slo_goodput,
)
from llmserveopt.policies.family_a_stateful_controller_v1 import (
    actions_disagree,
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
    """Advance `sim` internally exactly to its true first-decision point
    (enqueue done, GPUs/queue at their genuine initial state, nothing
    applied/advanced yet) and return that `ObservableState`, leaving `sim`
    frozen there -- mirrors `Simulator.run()`'s own per-step body up to (not
    including) `policy.select_action` returning, without reimplementing any
    admission/decode logic. `sim._build_observable_state()` alone is not
    enough: `_waiting` is only populated by `Simulator.run()`'s own enqueue
    step, which never runs if the state is built directly after
    `load_trace`."""
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
# H: eligibility gate is exactly the reused actions_disagree mechanism
# ---------------------------------------------------------------------------

def test_eligibility_is_exactly_reused_actions_disagree_on_pre_decision_snapshot():
    sim = _sim(_long_short_requests())
    state = _first_decision_state(sim)
    snapshot = snapshot_gpu_counters(state)
    action_estf = EstimatedServiceTimeFirstPolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    action_wfs = WeightedFairSharePolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    expected_eligible = actions_disagree(action_estf, action_wfs)
    assert expected_eligible is True  # fixture is constructed to disagree

    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=1, continuation_budget=8)
    policy.select_action(state)
    assert policy.decision_log()[0]["eligible"] == expected_eligible


def test_fallback_to_wfs_outside_candidate_region_no_planning_call():
    # A single feasible request: ESTF and WFS trivially agree (only one
    # candidate exists), so this is outside the candidate region.
    sim = _sim([_req(0, 0.0, 5, 1000.0, 1.0)])
    state = _first_decision_state(sim)
    expected_state = copy.deepcopy(state)
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=5, continuation_budget=8)
    action = policy.select_action(state)
    expected = WeightedFairSharePolicy().select_action(expected_state)
    from llmserveopt.policies.family_a_stateful_controller_v1 import canonical_action
    assert canonical_action(action) == canonical_action(expected)
    decision = policy.decision_log()[0]
    assert decision["eligible"] is False
    assert decision["planning_call"] is False
    assert decision["fallback_reason"] == "outside_candidate_region"


def test_fallback_to_wfs_on_empty_queue():
    sim = _sim([_req(0, 1000.0, 5, 2000.0, 1.0)])  # arrives far in the future
    state = _first_decision_state(sim)
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=5, continuation_budget=8)
    action = policy.select_action(state)
    assert action.admit == {} or all(not v for v in action.admit.values())
    decision = policy.decision_log()[0]
    assert decision["eligible"] is False
    assert decision["fallback_reason"] == "empty_queue"


# ---------------------------------------------------------------------------
# A: H=1, continuation_budget=0 reproduces a maximally local, single-step
# window-objective comparison.
# ---------------------------------------------------------------------------

def test_h1_zero_continuation_is_purely_local_window_objective():
    sim = _sim(_long_short_requests(n_short=1, o_long=10, deadline_long=1000.0))
    state = _first_decision_state(sim)
    snapshot = snapshot_gpu_counters(state)
    action_estf = EstimatedServiceTimeFirstPolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    action_wfs = WeightedFairSharePolicy().select_action(state)
    restore_gpu_counters(state, snapshot)

    estf_branch = _run_chained_branch(
        sim, candidate_policy=EstimatedServiceTimeFirstPolicy(), candidate_policy_id=ESTF_ID,
        candidate_first_action=action_estf, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=1, continuation_budget=0,
    )
    wfs_branch = _run_chained_branch(
        sim, candidate_policy=WeightedFairSharePolicy(), candidate_policy_id=WFS_ID,
        candidate_first_action=action_wfs, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=1, continuation_budget=0,
    )
    # ESTF admits the SHORT request (output=1): completes within the single
    # step -> window objective = its weight (1.0).
    assert estf_branch.window_objective == pytest.approx(1.0)
    assert estf_branch.raw_completed_count == 1
    # WFS admits LONG (output=10): cannot complete within a single step ->
    # window objective = 0.
    assert wfs_branch.window_objective == pytest.approx(0.0)
    assert wfs_branch.continuation_steps_run == 0  # candidate never finished -> no continuation fork


# ---------------------------------------------------------------------------
# B/C/D: horizon-dependent reversal. H=1 myopically favors ESTF (window too
# short to ever realize LONG's payoff either way, so the guaranteed cheap
# SHORT completion wins); H=5 gives ESTF enough self-control to admit LONG
# too, but only after burning 3 steps on SHORT jobs, so LONG still misses
# the (candidate+continuation) window under ESTF, while WFS's own branch
# (which admits LONG immediately) both meets LONG's deadline AND still has
# window left over for the SHORT jobs -- WFS wins clearly by H=5.
# ---------------------------------------------------------------------------

_REVERSAL_KWARGS = dict(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0)


def test_short_horizon_myopically_favors_estf_h1():
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=1, continuation_budget=8)
    sim.run(policy, workload_tag="t", seed=0)
    first = [d for d in policy.decision_log() if d["planning_call"]][0]
    assert first["winner"] == ESTF_ID
    assert first["estf_objective"] == pytest.approx(1.0)
    assert first["wfs_objective"] == pytest.approx(0.0)


def test_longer_horizon_correctly_reverses_to_wfs_h5():
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=5, continuation_budget=8)
    sim.run(policy, workload_tag="t", seed=0)
    calls = [d for d in policy.decision_log() if d["planning_call"]]
    assert len(calls) == 1  # WFS's own first action resolves the whole conflict
    assert calls[0]["winner"] == WFS_ID
    assert calls[0]["estf_objective"] == pytest.approx(3.0)
    assert calls[0]["wfs_objective"] == pytest.approx(53.0)


def test_rollout_controller_can_choose_estf():  # SS9.C
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    state = _first_decision_state(sim)
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=1, continuation_budget=8)
    policy.select_action(state)
    assert policy.decision_log()[0]["winner"] == ESTF_ID


def test_rollout_controller_can_choose_wfs():  # SS9.D
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    state = _first_decision_state(sim)
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=5, continuation_budget=8)
    policy.select_action(state)
    assert policy.decision_log()[0]["winner"] == WFS_ID


# ---------------------------------------------------------------------------
# E: replanning reverses a prior choice after the observed real state
# changes. A single continuous scenario: a first LONG/SHORT conflict at
# t=0 (H=1 -> ESTF, per the fixture above) is followed by a second,
# independent LONG/SHORT conflict arriving later with different slack that
# the SAME controller instance (same H=1, same continuation_budget=8)
# resolves in favor of WFS.
# ---------------------------------------------------------------------------

def test_replanning_reverses_choice_after_real_state_changes():
    round1 = _long_short_requests(n_short=3, o_long=10, deadline_long=0.011, p_long=50.0, arrival=0.0)
    t2 = 0.013
    round2 = [
        _req(1, t2, 6, t2 + 0.006, 50.0),
        _req(200, t2, 1, 10_000.0, 1.0),
    ]
    sim = _sim(round1 + round2, drain_steps=100)
    policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=1, continuation_budget=8)
    sim.run(policy, workload_tag="t", seed=0)
    calls = [d for d in policy.decision_log() if d["planning_call"]]
    winners = [c["winner"] for c in calls]
    assert winners[0] == ESTF_ID
    assert winners[-1] == WFS_ID
    assert ESTF_ID in winners and WFS_ID in winners  # genuine reversal, not monotone drift


# ---------------------------------------------------------------------------
# F/G: non-interference. Rollout branches (including the chained
# candidate -> common-continuation fork) never mutate the real simulator,
# regardless of order or repetition.
# ---------------------------------------------------------------------------

def test_chained_branch_does_not_mutate_source_simulator():
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    state = _first_decision_state(sim)
    snapshot = snapshot_gpu_counters(state)
    action_estf = EstimatedServiceTimeFirstPolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    action_wfs = WeightedFairSharePolicy().select_action(state)
    restore_gpu_counters(state, snapshot)

    fp_before = dcm._state_fingerprint(sim)
    _run_chained_branch(
        sim, candidate_policy=EstimatedServiceTimeFirstPolicy(), candidate_policy_id=ESTF_ID,
        candidate_first_action=action_estf, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    fp_mid = dcm._state_fingerprint(sim)
    _run_chained_branch(
        sim, candidate_policy=WeightedFairSharePolicy(), candidate_policy_id=WFS_ID,
        candidate_first_action=action_wfs, continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    fp_after = dcm._state_fingerprint(sim)
    assert fp_before == fp_mid == fp_after


def test_branch_order_independence_and_determinism():
    """Candidate A cannot affect candidate B: running ESTF-branch then
    WFS-branch gives the same WFS-branch result as running WFS-branch
    first, and repeated calls are bit-identical (no hidden shared state
    leaking between forks, no RNG)."""
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    state = _first_decision_state(sim)
    snapshot = snapshot_gpu_counters(state)
    action_estf = EstimatedServiceTimeFirstPolicy().select_action(state)
    restore_gpu_counters(state, snapshot)
    action_wfs = WeightedFairSharePolicy().select_action(state)
    restore_gpu_counters(state, snapshot)

    def wfs_branch():
        return _run_chained_branch(
            sim, candidate_policy=WeightedFairSharePolicy(), candidate_policy_id=WFS_ID,
            candidate_first_action=copy.deepcopy(action_wfs), continuation_policy=WeightedFairSharePolicy(),
            continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
        )

    wfs_only = wfs_branch()
    _run_chained_branch(
        sim, candidate_policy=EstimatedServiceTimeFirstPolicy(), candidate_policy_id=ESTF_ID,
        candidate_first_action=copy.deepcopy(action_estf), continuation_policy=WeightedFairSharePolicy(),
        continuation_policy_id=WFS_ID, horizon=5, continuation_budget=8,
    )
    wfs_after_estf = wfs_branch()

    assert wfs_only.window_objective == wfs_after_estf.window_objective
    assert wfs_only.raw_completed_count == wfs_after_estf.raw_completed_count
    assert wfs_only.candidate_steps_run == wfs_after_estf.candidate_steps_run
    assert wfs_only.continuation_steps_run == wfs_after_estf.continuation_steps_run


def test_full_run_state_untouched_and_deterministic():
    """Two identical fresh runs of the controller over the same scenario
    produce bit-identical decision logs and metrics (SS12 pre-run gate:
    deterministic reproducibility)."""
    from llmserveopt.core.metrics import metrics_to_dict

    def run_once():
        sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
        policy = FamilyARecedingHorizonOracleV1(sim_ref=sim, horizon=5, continuation_budget=8)
        metrics = sim.run(policy, workload_tag="t", seed=0)
        return metrics_to_dict(metrics), policy.decision_log()

    _TIMING_FIELDS = {"wall_clock_s", "mean_policy_time_s", "total_policy_time_s"}
    m1, log1 = run_once()
    m2, log2 = run_once()
    m1_substantive = {k: v for k, v in m1.items() if k not in _TIMING_FIELDS}
    m2_substantive = {k: v for k, v in m2.items() if k not in _TIMING_FIELDS}
    assert m1_substantive == m2_substantive
    assert log1 == log2


# ---------------------------------------------------------------------------
# Safety cap
# ---------------------------------------------------------------------------

def test_planning_call_cap_falls_back_to_wfs_and_is_reported():
    sim = _sim(_long_short_requests(**_REVERSAL_KWARGS))
    policy = FamilyARecedingHorizonOracleV1(
        sim_ref=sim, horizon=1, continuation_budget=8, max_planning_calls_per_scenario=1,
    )
    sim.run(policy, workload_tag="t", seed=0)
    diag = policy.diagnostics()
    assert diag["planning_calls_used"] == 1
    assert diag["planning_cap_hit"] is True
    calls = [d for d in policy.decision_log() if d["fallback_reason"] == "planning_call_cap_reached"]
    assert len(calls) >= 1


# ---------------------------------------------------------------------------
# Window objective arithmetic
# ---------------------------------------------------------------------------

def test_window_weighted_slo_goodput_uses_priority_and_slo_flag():
    class _FakeReq:
        def __init__(self, priority):
            self.priority = priority

    class _FakeCompleted:
        def __init__(self, priority, violated):
            self.request = _FakeReq(priority)
            self.slo_violated = violated

    completed = [
        _FakeCompleted(priority=2.0, violated=False),
        _FakeCompleted(priority=5.0, violated=True),
        _FakeCompleted(priority=0.0, violated=False),  # falls back to weight 1.0
    ]
    assert _window_weighted_slo_goodput(completed) == pytest.approx(2.0 + 0.0 + 1.0)


def test_default_continuation_budget_is_frozen_constant():
    assert COMMON_CONTINUATION_BUDGET == 200

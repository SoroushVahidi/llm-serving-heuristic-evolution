"""Family-A receding-horizon oracle feasibility controller V1.

Implements the design frozen by
`docs/design/FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md`. At each eligible
Family-A decision (ESTF and WFS disagree on the identical pre-decision
state), this controller forks the LIVE simulator into two independent,
non-interfering counterfactual branches -- "ESTF drives the next H steps,
then WFS" vs. "WFS drives the next H steps, then WFS" -- and executes only
the FIRST action of whichever branch realizes the higher windowed weighted
objective. The next real scheduler step re-plans from the true observed
state (receding horizon), with no extra bookkeeping beyond the ordinary
`Simulator.run()` -> `select_action` loop.

ORACLE FEASIBILITY ONLY. Uses the simulator itself as a perfect short-horizon
transition model via `fork_from_live_simulator`/`LiveFork`
(`decision_criticality_timescale_trainval_v1`), never a learned world model.
Reuses (imports, never reimplements) the eligibility-gate mechanism already
used and tested by `family_a_stateful_controller_v1`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.action import Action
from ..core.types import ObservableState
from ..simulator.simulator import Simulator
from .base import BasePolicy
from .estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from .family_a_stateful_controller_v1 import (
    actions_disagree,
    restore_gpu_counters,
    snapshot_gpu_counters,
)
from .weighted_fair_share import WeightedFairSharePolicy

from ..analysis import decision_criticality_timescale_trainval_v1 as dcm

ESTF_ID = "estimated_service_time_first"
WFS_ID = "weighted_fair_share"

#: Frozen (design doc SS6): common continuation used identically by every
#: candidate at every horizon, so terminal-handling never itself creates an
#: ESTF/WFS preference.
COMMON_CONTINUATION_BUDGET: int = 200

#: Default fallback outside the candidate region (design doc SS4): the
#: strongest fixed Family-A parent in the prior closed-loop evaluation.
FALLBACK_POLICY_ID = WFS_ID


def _window_weighted_slo_goodput(completed_slice: List[Any]) -> float:
    """Design doc SS5: sum(weight_i * 1[not slo_violated_i]) over a bounded
    rollout window's newly-completed requests. `completed_slice` is a list of
    `CompletedRequest` (exactly what `shell._completed` accumulates)."""
    total = 0.0
    for c in completed_slice:
        weight = c.request.priority if c.request.priority > 0 else 1.0
        if not c.slo_violated:
            total += weight
    return total


@dataclass
class BranchResult:
    policy_id: str
    window_objective: float
    raw_completed_count: int
    candidate_steps_run: int
    continuation_steps_run: int


def _run_chained_branch(
    sim: Simulator,
    *,
    candidate_policy: BasePolicy,
    candidate_policy_id: str,
    candidate_first_action: Action,
    continuation_policy: BasePolicy,
    continuation_policy_id: str,
    horizon: int,
    continuation_budget: int = COMMON_CONTINUATION_BUDGET,
) -> BranchResult:
    """Design doc SS6: fork `sim` at its current step, drive it for up to
    `horizon` steps under `candidate_policy` (starting from the already-
    computed `candidate_first_action`), then fork again from the resulting
    shell and drive it for up to `continuation_budget` further steps under
    `continuation_policy`. `sim` itself is never mutated (only
    `dcm.fork_from_live_simulator`'s own deep-copied shells are advanced)."""
    candidate_fork = dcm.fork_from_live_simulator(
        sim,
        policy=candidate_policy,
        policy_id=candidate_policy_id,
        first_action=candidate_first_action,
    )
    candidate_steps_run = 1
    while not candidate_fork.finished and candidate_steps_run < horizon:
        candidate_fork.advance_one_step()
        candidate_steps_run += 1

    continuation_steps_run = 0
    continuation_window: List[Any] = []
    if not candidate_fork.finished and continuation_budget > 0:
        cont_state = candidate_fork.shell._build_observable_state()
        cont_first_action = continuation_policy.select_action(cont_state)
        continuation_start_len = len(candidate_fork.shell._completed)
        continuation_fork = dcm.fork_from_live_simulator(
            candidate_fork.shell,
            policy=continuation_policy,
            policy_id=continuation_policy_id,
            first_action=cont_first_action,
        )
        continuation_steps_run = 1
        while not continuation_fork.finished and continuation_steps_run < continuation_budget:
            continuation_fork.advance_one_step()
            continuation_steps_run += 1
        continuation_window = continuation_fork.shell._completed[continuation_start_len:]

    candidate_window = candidate_fork.shell._completed[-candidate_fork.completed_in_window:] \
        if candidate_fork.completed_in_window else []
    objective = _window_weighted_slo_goodput(candidate_window) + _window_weighted_slo_goodput(continuation_window)
    raw_count = candidate_fork.completed_in_window + len(continuation_window)
    return BranchResult(
        policy_id=candidate_policy_id,
        window_objective=objective,
        raw_completed_count=raw_count,
        candidate_steps_run=candidate_steps_run,
        continuation_steps_run=continuation_steps_run,
    )


@dataclass
class PlanningDecision:
    step: int
    eligible: bool
    winner: Optional[str]
    estf_objective: Optional[float]
    wfs_objective: Optional[float]
    planning_call: bool
    fallback_reason: Optional[str] = None


@dataclass
class FamilyARecedingHorizonOracleV1(BasePolicy):
    """Oracle short-horizon receding-horizon Family-A controller.

    `sim_ref` MUST be the exact `Simulator` instance this policy is about to
    be run with via `sim.run(policy, ...)` -- rollout branches fork from it
    every eligible step, never mutating it (design doc SS3)."""

    sim_ref: Simulator
    horizon: int
    continuation_budget: int = COMMON_CONTINUATION_BUDGET
    max_planning_calls_per_scenario: Optional[int] = None

    estf_policy: EstimatedServiceTimeFirstPolicy = field(default_factory=EstimatedServiceTimeFirstPolicy)
    wfs_policy: WeightedFairSharePolicy = field(default_factory=WeightedFairSharePolicy)
    continuation_policy: WeightedFairSharePolicy = field(default_factory=WeightedFairSharePolicy)

    name: str = "family_a_receding_horizon_oracle_v1"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.reset()

    def reset(self) -> None:
        self.estf_policy.reset()
        self.wfs_policy.reset()
        self.continuation_policy.reset()
        self._decisions: List[PlanningDecision] = []
        self._planning_calls_used = 0
        self._planning_cap_hit = False

    def select_action(self, state: ObservableState) -> Action:
        if not state.waiting_queue and not state.migrating_queue:
            self._decisions.append(
                PlanningDecision(
                    step=int(state.step), eligible=False, winner=None,
                    estf_objective=None, wfs_objective=None, planning_call=False,
                    fallback_reason="empty_queue",
                )
            )
            return self.wfs_policy.select_action(state)

        snapshot = snapshot_gpu_counters(state)
        action_estf = self.estf_policy.select_action(state)
        restore_gpu_counters(state, snapshot)
        action_wfs = self.wfs_policy.select_action(state)
        restore_gpu_counters(state, snapshot)

        eligible = actions_disagree(action_estf, action_wfs)
        if not eligible:
            self._decisions.append(
                PlanningDecision(
                    step=int(state.step), eligible=False, winner=None,
                    estf_objective=None, wfs_objective=None, planning_call=False,
                    fallback_reason="outside_candidate_region",
                )
            )
            return action_wfs

        cap = self.max_planning_calls_per_scenario
        if cap is not None and self._planning_calls_used >= cap:
            self._planning_cap_hit = True
            self._decisions.append(
                PlanningDecision(
                    step=int(state.step), eligible=True, winner=None,
                    estf_objective=None, wfs_objective=None, planning_call=False,
                    fallback_reason="planning_call_cap_reached",
                )
            )
            return action_wfs

        self._planning_calls_used += 1
        estf_branch = _run_chained_branch(
            self.sim_ref,
            candidate_policy=self.estf_policy,
            candidate_policy_id=ESTF_ID,
            candidate_first_action=action_estf,
            continuation_policy=self.continuation_policy,
            continuation_policy_id=WFS_ID,
            horizon=self.horizon,
            continuation_budget=self.continuation_budget,
        )
        wfs_branch = _run_chained_branch(
            self.sim_ref,
            candidate_policy=self.wfs_policy,
            candidate_policy_id=WFS_ID,
            candidate_first_action=action_wfs,
            continuation_policy=self.continuation_policy,
            continuation_policy_id=WFS_ID,
            horizon=self.horizon,
            continuation_budget=self.continuation_budget,
        )

        if estf_branch.window_objective > wfs_branch.window_objective:
            winner = ESTF_ID
            chosen_action = action_estf
        else:
            winner = WFS_ID
            chosen_action = action_wfs

        self._decisions.append(
            PlanningDecision(
                step=int(state.step), eligible=True, winner=winner,
                estf_objective=estf_branch.window_objective,
                wfs_objective=wfs_branch.window_objective,
                planning_call=True,
            )
        )
        return chosen_action

    def diagnostics(self) -> Dict[str, Any]:
        total = len(self._decisions)
        eligible = [d for d in self._decisions if d.eligible]
        planned = [d for d in self._decisions if d.planning_call]
        estf_wins = sum(1 for d in planned if d.winner == ESTF_ID)
        wfs_wins = sum(1 for d in planned if d.winner == WFS_ID)
        return {
            "horizon": self.horizon,
            "continuation_budget": self.continuation_budget,
            "total_decisions": total,
            "eligible_count": len(eligible),
            "planning_calls_used": self._planning_calls_used,
            "planning_cap_hit": self._planning_cap_hit,
            "estf_win_count": estf_wins,
            "wfs_win_count": wfs_wins,
            "estf_win_fraction": (estf_wins / len(planned)) if planned else 0.0,
            "wfs_win_fraction": (wfs_wins / len(planned)) if planned else 0.0,
        }

    def decision_log(self) -> List[Dict[str, Any]]:
        return [d.__dict__.copy() for d in self._decisions]

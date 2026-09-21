"""Family-A receding-horizon terminal-value redesign V1.

Implements the design frozen by
`docs/design/FAMILY_A_TERMINAL_VALUE_V1.md`. Identical receding-horizon
controller mechanics to `family_a_receding_horizon_oracle_v1` (same
eligibility gate, same WFS fallback, same ESTF/WFS candidates, same
horizons, same common continuation, same execute-first-action/replan
semantics) -- the ONLY change is the branch-scoring rule: rollout branches
are scored by `V(branch) = W_completed(branch) + V_inflight(branch)`
instead of `W_completed(branch)` alone, crediting feasible, service-invested
but not-yet-completed work at the branch's terminal state, not just fully
completed requests.

This module never imports from, or modifies,
`family_a_receding_horizon_oracle_v1.py` -- it is additive, reusing only the
same underlying simulator/fork primitives
(`decision_criticality_timescale_trainval_v1.fork_from_live_simulator`) and
the same eligibility-gate helpers
(`family_a_stateful_controller_v1.snapshot_gpu_counters` /
`restore_gpu_counters` / `actions_disagree`) both prior modules already use
unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
from .scoring import DEFAULT_BETA, deadline_slack
from .weighted_fair_share import WeightedFairSharePolicy

from ..analysis import decision_criticality_timescale_trainval_v1 as dcm

ESTF_ID = "estimated_service_time_first"
WFS_ID = "weighted_fair_share"

#: Frozen, unchanged from V1 (design doc FAMILY_A_TERMINAL_VALUE_V1.md SS6).
COMMON_CONTINUATION_BUDGET: int = 200
FALLBACK_POLICY_ID = WFS_ID


def _weight_of(request: Any) -> float:
    """Design doc SS4: identical weight definition to the old objective's
    `weight_i` -- keeps V_inflight in completion-equivalent units."""
    return request.priority if request.priority > 0 else 1.0


def _window_weighted_slo_goodput(completed_slice: List[Any]) -> float:
    """Unchanged from V1: sum(weight_i * 1[not slo_violated_i]) over a
    bounded rollout window's newly-completed requests."""
    total = 0.0
    for c in completed_slice:
        if not c.slo_violated:
            total += _weight_of(c.request)
    return total


def _inflight_terminal_credit(state: ObservableState) -> float:
    """Design doc SS4: `V_inflight(branch)` computed from the branch's own
    terminal `ObservableState` alone -- no future information, no scenario
    identity, no scenario-regime metadata. Sums, over every
    not-yet-completed request visible in `state` (waiting_queue + every
    GPU's active_requests_info), `weight_i * progress_fraction_i *
    feasible_i`. Reads no scenario-identity or generator-parameter fields --
    only per-request/per-GPU quantities every online policy already sees."""
    total = 0.0
    decoded_by_gpu = [gpu.tokens_decoded_per_request for gpu in state.gpu_states]

    def credit(req: Any, tokens_decoded: int) -> float:
        predicted = max(int(req.predicted_output_tokens), 1)
        progress_fraction = min(max(tokens_decoded / predicted, 0.0), 1.0)
        if progress_fraction <= 0.0:
            return 0.0
        remaining_tokens = max(req.predicted_output_tokens - tokens_decoded, 0)
        remaining_service = DEFAULT_BETA * remaining_tokens
        slack = deadline_slack(req, now=state.time, service_proxy=remaining_service)
        feasible = 1.0 if slack >= 0 else 0.0
        return _weight_of(req) * progress_fraction * feasible

    for req in state.waiting_queue:
        total += credit(req, tokens_decoded=0)
    for req in state.migrating_queue:
        total += credit(req, tokens_decoded=0)
    for gpu, decoded in zip(state.gpu_states, decoded_by_gpu):
        for req in gpu.active_requests_info:
            total += credit(req, tokens_decoded=int(decoded.get(req.request_id, 0)))
    return total


@dataclass
class DualBranchResult:
    policy_id: str
    old_window_objective: float
    new_terminal_value: float
    candidate_steps_run: int
    continuation_steps_run: int


def _run_chained_branch_dual(
    sim: Simulator,
    *,
    candidate_policy: BasePolicy,
    candidate_policy_id: str,
    candidate_first_action: Action,
    continuation_policy: BasePolicy,
    continuation_policy_id: str,
    horizon: int,
    continuation_budget: int = COMMON_CONTINUATION_BUDGET,
) -> DualBranchResult:
    """Identical rollout mechanics to
    `family_a_receding_horizon_oracle_v1._run_chained_branch` (same fork
    primitives, same candidate-then-common-continuation chain), but scores
    the resulting branch under BOTH the old window objective and the new
    terminal value, so both can be compared side by side without any change
    to which branch actually gets executed by callers of this function."""
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
    terminal_shell = candidate_fork.shell
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
        terminal_shell = continuation_fork.shell

    candidate_window = candidate_fork.shell._completed[-candidate_fork.completed_in_window:] \
        if candidate_fork.completed_in_window else []
    old_objective = _window_weighted_slo_goodput(candidate_window) + _window_weighted_slo_goodput(continuation_window)

    terminal_state = terminal_shell._build_observable_state()
    new_value = old_objective + _inflight_terminal_credit(terminal_state)

    return DualBranchResult(
        policy_id=candidate_policy_id,
        old_window_objective=old_objective,
        new_terminal_value=new_value,
        candidate_steps_run=candidate_steps_run,
        continuation_steps_run=continuation_steps_run,
    )


@dataclass
class PlanningDecision:
    step: int
    eligible: bool
    winner: Optional[str]
    estf_old_objective: Optional[float]
    wfs_old_objective: Optional[float]
    estf_new_value: Optional[float]
    wfs_new_value: Optional[float]
    planning_call: bool
    fallback_reason: Optional[str] = None


@dataclass
class FamilyARecedingHorizonTerminalValueV1(BasePolicy):
    """Same receding-horizon mechanics as `FamilyARecedingHorizonOracleV1`,
    scored by the new terminal value (design doc
    `FAMILY_A_TERMINAL_VALUE_V1.md`) instead of the old window objective.

    `sim_ref` MUST be the exact `Simulator` instance this policy is about to
    be run with via `sim.run(policy, ...)`."""

    sim_ref: Simulator
    horizon: int
    continuation_budget: int = COMMON_CONTINUATION_BUDGET
    max_planning_calls_per_scenario: Optional[int] = None

    estf_policy: EstimatedServiceTimeFirstPolicy = field(default_factory=EstimatedServiceTimeFirstPolicy)
    wfs_policy: WeightedFairSharePolicy = field(default_factory=WeightedFairSharePolicy)
    continuation_policy: WeightedFairSharePolicy = field(default_factory=WeightedFairSharePolicy)

    name: str = "family_a_receding_horizon_terminal_value_v1"

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
                    estf_old_objective=None, wfs_old_objective=None,
                    estf_new_value=None, wfs_new_value=None, planning_call=False,
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
                    estf_old_objective=None, wfs_old_objective=None,
                    estf_new_value=None, wfs_new_value=None, planning_call=False,
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
                    estf_old_objective=None, wfs_old_objective=None,
                    estf_new_value=None, wfs_new_value=None, planning_call=False,
                    fallback_reason="planning_call_cap_reached",
                )
            )
            return action_wfs

        self._planning_calls_used += 1
        estf_branch = _run_chained_branch_dual(
            self.sim_ref,
            candidate_policy=self.estf_policy, candidate_policy_id=ESTF_ID,
            candidate_first_action=action_estf,
            continuation_policy=self.continuation_policy, continuation_policy_id=WFS_ID,
            horizon=self.horizon, continuation_budget=self.continuation_budget,
        )
        wfs_branch = _run_chained_branch_dual(
            self.sim_ref,
            candidate_policy=self.wfs_policy, candidate_policy_id=WFS_ID,
            candidate_first_action=action_wfs,
            continuation_policy=self.continuation_policy, continuation_policy_id=WFS_ID,
            horizon=self.horizon, continuation_budget=self.continuation_budget,
        )

        if estf_branch.new_terminal_value > wfs_branch.new_terminal_value:
            winner = ESTF_ID
            chosen_action = action_estf
        else:
            winner = WFS_ID
            chosen_action = action_wfs

        self._decisions.append(
            PlanningDecision(
                step=int(state.step), eligible=True, winner=winner,
                estf_old_objective=estf_branch.old_window_objective,
                wfs_old_objective=wfs_branch.old_window_objective,
                estf_new_value=estf_branch.new_terminal_value,
                wfs_new_value=wfs_branch.new_terminal_value,
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

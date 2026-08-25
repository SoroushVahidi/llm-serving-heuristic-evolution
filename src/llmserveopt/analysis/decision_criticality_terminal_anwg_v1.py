"""Terminal-ANWG one-step counterfactual decision criticality v1.

Implements the estimand frozen in
`docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_V1.md`.

Reuses (does not reimplement) parent fork machinery from
`decision_criticality_timescale_trainval_v1`:
`fork_from_live_simulator`, `LiveFork`, `actions_disagree`,
`canonical_action`, `alternative_policy_id`, TRAIN/VAL guards, scenario rebuild,
frozen model fit.

Primary estimand:
  C_t = ANWG(CF) - ANWG(reference)
where CF applies one alternative native action at t, then continues with a
clone of the live hierarchical router to natural termination.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.action import Action
from ..core.metrics import RunMetrics, compute_metrics
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policy_separation.hierarchical_regime_router_v1 import ACTIVE_REGIMES
from ..policy_separation.hierarchical_router_live_harness_v1 import (
    LiveHierarchicalRouterPolicy,
    build_feature_rows_by_regime,
    build_native_policy_instances,
)
from ..policy_separation.schema import PolicySeparationScenario
from ..selector.hierarchical_stage2_selectors_v1 import Stage2Selector
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from ..policy_separation.hierarchical_regime_router_v1 import Stage1Router

from . import decision_criticality_timescale_trainval_v1 as dcm

SCHEMA_VERSION = "decision_criticality_terminal_anwg_v1.0.0"

MAX_DISAGREEMENT_PER_SCENARIO = 5
MAX_AGREEMENT_CONTROL_PER_SCENARIO = 3
CONTROL_SEED = 20260824
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 1000
ANWG_EQ_ATOL = 1e-12
PRACTICAL_THRESHOLDS = (0.001, 0.005, 0.01)


def clone_live_router(router: LiveHierarchicalRouterPolicy) -> LiveHierarchicalRouterPolicy:
    """Clone continuation state after step-t select_action.

    Shares frozen Stage-1/Stage-2 model objects (read-only at inference).
    Deep-copies FSM, native policy instances, and counters so CF/REF-replay
    continuations do not mutate the live reference router.
    """
    cloned = LiveHierarchicalRouterPolicy(
        scenario_id=router.scenario_id,
        stage1=router.stage1,
        stage2_selectors=router.stage2_selectors,
        feature_rows_by_regime=router.feature_rows_by_regime,
        dwell_steps=router.dwell_steps,
        forced_expert=router.forced_expert,
        record_trajectory=False,
    )
    cloned.native_policies = copy.deepcopy(router.native_policies)
    cloned._fsm = copy.deepcopy(router._fsm)
    cloned.stage2_call_count = dict(router.stage2_call_count)
    cloned.selected_policy_step_counts = dict(router.selected_policy_step_counts)
    cloned.last_dwell_diagnostics = copy.deepcopy(router.last_dwell_diagnostics)
    cloned._stage2_policy_cache = dict(getattr(router, "_stage2_policy_cache", {}))
    return cloned


def finalize_shell_anwg(
    shell: Simulator,
    *,
    all_requests: Sequence,
    policy_name: str,
    workload_tag: str,
    seed: int,
) -> RunMetrics:
    dropped = (
        [ir.request for ir in shell._waiting]
        + [ir.request for ir in shell._migrating]
        + [ir.request for ir in shell._relocating.values()]
    )
    return compute_metrics(
        completed=list(shell._completed),
        dropped=dropped,
        sim_duration=float(shell._time),
        gpu_utilization_history=list(shell._util_history),
        active_batch_history=list(shell._batch_history),
        policy_name=policy_name,
        workload_tag=workload_tag,
        seed=seed,
        num_total=len(all_requests),
        all_requests=list(all_requests),
    )


def advance_fork_to_termination(fork: dcm.LiveFork, *, max_steps: int = 500_000) -> int:
    """Continue a LiveFork to natural completion with Simulator-equivalent idle skip."""
    steps = 0
    shell = fork.shell
    step_size = fork.step_size
    while not fork.finished and steps < max_steps:
        # Idle fast-forward (mirrors Simulator.run)
        pending = shell._pending_arrivals
        n = len(pending)
        all_arrivals_done = fork.arrival_idx >= n
        all_active_done = sum(g.num_active for g in shell._gpus) == 0
        queue_empty = (
            len(shell._waiting) == 0
            and len(shell._migrating) == 0
            and len(shell._relocating) == 0
        )
        if (not all_arrivals_done) and queue_empty and all_active_done:
            next_arr_time = pending[fork.arrival_idx].request.arrival_time
            skip_to = int(next_arr_time / step_size)
            idle_gap = skip_to - (shell._step + 1)
            if idle_gap > 0:
                shell._idle_skipped = getattr(shell, "_idle_skipped", 0) + idle_gap
                shell._step = skip_to - 1
                shell._time = shell._step * step_size

        fork.advance_one_step()
        steps += 1
    return steps


def run_one_step_then_router_terminal(
    sim: Simulator,
    *,
    first_action: Action,
    continuation_router: LiveHierarchicalRouterPolicy,
    all_requests: Sequence,
    workload_tag: str,
    seed: int,
    branch_label: str,
) -> Dict[str, Any]:
    """Fork sim, force first_action, continue with continuation_router to terminal ANWG.

    Continuation uses `Simulator.continue_run` (same idle/drain/handoff semantics as
    the reference `Simulator.run`) rather than stepping `LiveFork` to termination,
    which is orders of magnitude slower on long Family-A trajectories.
    """
    fp_before = dcm._state_fingerprint(sim)
    step_before = int(sim._step)
    fork = dcm.fork_from_live_simulator(
        sim,
        policy=continuation_router,
        policy_id="live_hierarchical_router_continuation",
        first_action=copy.deepcopy(first_action),
    )
    # First action already applied inside fork_from_live_simulator.
    continuation_router.name = branch_label  # type: ignore[attr-defined]
    metrics = fork.shell.continue_run(
        continuation_router,
        workload_tag=workload_tag,
        seed=seed,
        num_total=len(all_requests),
        all_requests=all_requests,
    )
    fp_after = dcm._state_fingerprint(sim)
    return {
        "anwg": float(metrics.arrival_normalized_weighted_goodput),
        "num_completed": int(metrics.num_completed),
        "num_dropped": int(metrics.num_dropped),
        "sim_duration": float(metrics.sim_duration),
        "extra_steps": int(fork.shell._step - step_before),
        "finished": True,
        "live_fingerprint_unchanged": fp_before == fp_after,
        "pre_fork_fingerprint": fp_before,
        "idle_steps_skipped": int(getattr(fork.shell, "_idle_skipped", 0)),
    }


def _state_features(state: ObservableState) -> Dict[str, Any]:
    q = len(state.waiting_queue)
    active = sum(len(g.active_request_ids) for g in state.gpu_states)
    kv_utils = [
        (g.current_kv_tokens / g.max_kv_tokens) if g.max_kv_tokens else 0.0
        for g in state.gpu_states
    ]
    kv = float(np.mean(kv_utils)) if kv_utils else 0.0
    return {
        "queue_size": int(q),
        "active_sequences": int(active),
        "kv_utilization_mean": kv,
        "completed_count_pre": int(state.completed_count),
        "sim_time": float(state.time),
    }


def _acquisition_priority(scenario_id: str, step: int, seed: int) -> float:
    """Deterministic priority in [0,1) for agreement-control sampling."""
    h = hashlib.sha256(f"{seed}:{scenario_id}:{step}".encode()).hexdigest()
    return int(h[:16], 16) / float(16**16)


@dataclass
class TerminalANWGObserver(BasePolicy):
    """Shadow observer: never alters reference actions; evaluates terminal-ANWG CFs."""

    name = "decision_criticality_terminal_anwg_observer_v1"

    sim_ref: Simulator
    inner_router: LiveHierarchicalRouterPolicy
    shadow_policies: Dict[str, BasePolicy]
    all_requests: Sequence
    canonical_scenario_id: str
    mechanism_family: str
    split: str
    seed: int
    max_disagreement: int = MAX_DISAGREEMENT_PER_SCENARIO
    max_agreement_control: int = MAX_AGREEMENT_CONTROL_PER_SCENARIO
    control_seed: int = CONTROL_SEED
    run_ref_replay_once: bool = True

    n_disagreement_kept: int = 0
    n_agreement_kept: int = 0
    ref_replay_done: bool = False
    branch_rows: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.inner_router.reset()
        for p in self.shadow_policies.values():
            p.reset()
        self.n_disagreement_kept = 0
        self.n_agreement_kept = 0
        self.ref_replay_done = False
        self.branch_rows = []

    def select_action(self, state: ObservableState) -> Action:
        real_action = self.inner_router.select_action(state)
        if not self.inner_router.trajectory:
            return real_action
        row = self.inner_router.trajectory[-1]
        regime = row.effective_regime
        chosen_id = row.selected_policy
        if regime not in ACTIVE_REGIMES:
            return real_action

        alt_id = dcm.alternative_policy_id(regime, chosen_id)
        if alt_id is None:
            return real_action

        shadow_state = copy.deepcopy(state)
        alt_action = self.shadow_policies[alt_id].select_action(shadow_state)
        if alt_id in dcm._PREFILL_CHUNK_BY_POLICY:
            dcm._apply_prefill_chunk_override(alt_action, shadow_state, alt_id)
        if chosen_id in dcm._PREFILL_CHUNK_BY_POLICY:
            # real_action already had override applied inside router
            pass

        disagree = dcm.actions_disagree(real_action, alt_action)
        feats = _state_features(state)
        meta = {
            "step": int(state.step),
            "regime": regime,
            "chosen_policy_id": chosen_id,
            "alt_policy_id": alt_id,
            "disagree": bool(disagree),
            "canonical_ref_action": json.dumps(dcm.canonical_action(real_action), sort_keys=True),
            "canonical_alt_action": json.dumps(dcm.canonical_action(alt_action), sort_keys=True),
            **feats,
            "a_active": bool(row.a_active),
            "b_active_v2": bool(row.b_active_v2),
            "c_active": bool(row.c_active),
            "contention_score_v2": float(row.contention_score_v2),
            "priority_skew": float(row.priority_skew),
            "kv_pressure": float(row.kv_pressure),
            "queue_length_signal": float(row.queue_length),
        }

        if disagree:
            if self.n_disagreement_kept < self.max_disagreement:
                self._evaluate_intervention(
                    state=state,
                    real_action=real_action,
                    alt_action=alt_action,
                    acquisition="DISAGREEMENT",
                    meta=meta,
                )
                self.n_disagreement_kept += 1
        elif self.n_agreement_kept < self.max_agreement_control:
            # First-M agreement controls (trajectory order; outcome-blind).
            meta["acquisition_priority"] = _acquisition_priority(
                self.canonical_scenario_id, int(state.step), self.control_seed
            )
            self._evaluate_intervention(
                state=state,
                real_action=real_action,
                alt_action=alt_action,
                acquisition="AGREEMENT_CONTROL",
                meta=meta,
            )
            self.n_agreement_kept += 1

        return real_action

    def _evaluate_intervention(
        self,
        *,
        state: ObservableState,
        real_action: Action,
        alt_action: Action,
        acquisition: str,
        meta: Dict[str, Any],
    ) -> None:
        cont_cf = clone_live_router(self.inner_router)
        cf = run_one_step_then_router_terminal(
            self.sim_ref,
            first_action=alt_action,
            continuation_router=cont_cf,
            all_requests=self.all_requests,
            workload_tag=self.canonical_scenario_id,
            seed=self.seed,
            branch_label="cf_one_step_alt",
        )

        ref_replay = None
        if self.run_ref_replay_once and not self.ref_replay_done:
            cont_rr = clone_live_router(self.inner_router)
            ref_replay = run_one_step_then_router_terminal(
                self.sim_ref,
                first_action=real_action,
                continuation_router=cont_rr,
                all_requests=self.all_requests,
                workload_tag=self.canonical_scenario_id,
                seed=self.seed,
                branch_label="ref_action_replay",
            )
            self.ref_replay_done = True

        # Compare trajectories lightly: CF vs live fingerprint at creation only;
        # post-hoc divergence vs reference path requires storing CF shell states.
        # Record whether CF finished and basic deltas vs reference will be filled
        # after the reference run completes (reference_anwg attached later).
        row = {
            "canonical_scenario_id": self.canonical_scenario_id,
            "mechanism_family": self.mechanism_family,
            "split": self.split,
            "seed": int(self.seed),
            "acquisition_type": acquisition,
            "reference_policy": "hierarchical_router_live_v1",
            "alternative_policy": meta["alt_policy_id"],
            "chosen_native_policy": meta["chosen_policy_id"],
            **meta,
            "cf_anwg": cf["anwg"],
            "cf_num_completed": cf["num_completed"],
            "cf_num_dropped": cf["num_dropped"],
            "cf_sim_duration": cf["sim_duration"],
            "cf_extra_steps": cf["extra_steps"],
            "cf_finished": cf["finished"],
            "live_fingerprint_unchanged": cf["live_fingerprint_unchanged"],
            "ref_replay_anwg": None if ref_replay is None else ref_replay["anwg"],
            "ref_replay_live_fingerprint_unchanged": (
                None if ref_replay is None else ref_replay["live_fingerprint_unchanged"]
            ),
        }
        self.branch_rows.append(row)


def run_scenario_terminal_anwg(
    row: pd.Series,
    *,
    stage1: Stage1Router,
    stage2_selectors: Dict[str, Stage2Selector],
    max_disagreement: int = MAX_DISAGREEMENT_PER_SCENARIO,
    max_agreement_control: int = MAX_AGREEMENT_CONTROL_PER_SCENARIO,
) -> Dict[str, Any]:
    dcm.assert_trainval_only(row["split"])
    scenario = dcm.rebuild_scenario_from_row(row)
    cid = str(row["canonical_scenario_id"])
    feature_rows = build_feature_rows_by_regime(scenario, cid)
    inner = LiveHierarchicalRouterPolicy(
        scenario_id=cid,
        stage1=stage1,
        stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows,
        record_trajectory=True,
    )
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**scenario.service_model_kwargs),
            max_steps=500_000,
            drain_steps=50_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    observer = TerminalANWGObserver(
        sim_ref=sim,
        inner_router=inner,
        shadow_policies=build_native_policy_instances(),
        all_requests=list(scenario.requests),
        canonical_scenario_id=cid,
        mechanism_family=str(row["mechanism_family"]),
        split=str(row["split"]),
        seed=int(row["seed"]),
        max_disagreement=max_disagreement,
        max_agreement_control=max_agreement_control,
    )
    observer.sim_ref = sim
    ref_metrics = sim.run(observer, workload_tag=cid, seed=int(row["seed"]))
    ref_anwg = float(ref_metrics.arrival_normalized_weighted_goodput)

    # Attach reference ANWG / deltas
    for br in observer.branch_rows:
        br["reference_anwg"] = ref_anwg
        br["reference_num_completed"] = int(ref_metrics.num_completed)
        br["reference_sim_duration"] = float(ref_metrics.sim_duration)
        br["delta_anwg"] = float(br["cf_anwg"] - ref_anwg)
        br["abs_delta_anwg"] = abs(float(br["delta_anwg"]))
        br["completion_count_delta"] = int(br["cf_num_completed"] - int(ref_metrics.num_completed))
        br["sim_duration_delta"] = float(br["cf_sim_duration"] - float(ref_metrics.sim_duration))
        br["terminal_utility_effect"] = bool(br["abs_delta_anwg"] > ANWG_EQ_ATOL)
        br["subsequent_trajectory_diverged"] = bool(
            br["terminal_utility_effect"]
            or br["completion_count_delta"] != 0
            or abs(br["sim_duration_delta"]) > 1e-12
        )
        if br.get("ref_replay_anwg") is not None:
            br["ref_replay_minus_reference_anwg"] = float(br["ref_replay_anwg"] - ref_anwg)
            br["ref_replay_matches_reference"] = bool(
                abs(br["ref_replay_minus_reference_anwg"]) <= ANWG_EQ_ATOL
            )

    # Agreement fill-up: if priority cutoff yielded too few, accept first remaining
    # from pool is impossible without rewind — record shortfall only.
    return {
        "canonical_scenario_id": cid,
        "mechanism_family": str(row["mechanism_family"]),
        "split": str(row["split"]),
        "seed": int(row["seed"]),
        "reference_anwg": ref_anwg,
        "reference_num_completed": int(ref_metrics.num_completed),
        "n_steps": int(ref_metrics.sim_duration / scenario.service_model_kwargs.get("step_size", 0.001))
        if scenario.service_model_kwargs.get("step_size")
        else None,
        "n_branch_rows": len(observer.branch_rows),
        "n_disagreement_evaluated": observer.n_disagreement_kept,
        "n_agreement_evaluated": observer.n_agreement_kept,
        "branch_rows": observer.branch_rows,
    }

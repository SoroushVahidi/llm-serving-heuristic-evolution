"""Family-A Observability / Continuation-Dependence Diagnostic v1 -- TRAIN/VAL-ONLY.

Implements the diagnostic frozen by
`docs/design/FAMILY_A_OBSERVABILITY_CONTINUATION_DIAGNOSTIC_V1.md`. For the
Family-A (`RANKING_FAIRNESS`) native pair (`estimated_service_time_first` /
`weighted_fair_share`), at a sample of real policy-disagreement states, asks
whether the local advantage of one candidate over the other is:

  - explained by an immediate-action effect that is stable across which
    policy keeps driving afterward (same-continuation counterfactual,
    `Delta_same`), and/or
  - predictable from richer decision-time observable state than the parent
    decision-criticality diagnostic retained, and/or
  - primarily an artifact of which policy controls subsequent decisions
    (native-continuation counterfactual, `Delta_native`, vs. same-
    continuation -> continuation-dependence `C`).

DIAGNOSTIC / METHODOLOGY ONLY. Family A only -- never touches Family B/C.
Never modifies, retrains, or re-thresholds anything frozen by
`hierarchical_regime_router_v1.py` / `hierarchical_stage2_selectors_v1.py` /
`hierarchical_router_live_harness_v1.py`. Never reads a TEST-split scenario
or telemetry row (`assert_trainval_only`, reused from the parent decision-
criticality module). Never modifies the completed decision-criticality or
public-trace-replay canonical result artifacts.

Reuses (imports, never reimplements) the parent diagnostic's fork machinery:
`fork_from_live_simulator`, `run_bounded_rollout`, `LiveFork`,
`canonical_action`/`actions_disagree`, `assert_trainval_only`,
`load_trainval_scenario_table`, `rebuild_scenario_from_row`,
`fit_frozen_models`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.action import Action
from ..core.types import ObservableRequest, ObservableState
from ..policies.base import BasePolicy
from ..policies.scoring import predicted_service_proxy
from ..policies.policy_library_v2_helpers import queue_class_counts
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from ..policy_separation.hierarchical_regime_router_v1 import REGIME_A, STAGE2_CANDIDATES
from ..policy_separation.hierarchical_router_live_harness_v1 import (
    LiveHierarchicalRouterPolicy,
    build_native_policy_instances,
    build_feature_rows_by_regime,
)
from ..policy_separation.schema import PolicySeparationScenario
from ..selector.hierarchical_stage2_selectors_v1 import Stage2Selector

from . import decision_criticality_timescale_trainval_v1 as dcm

SCHEMA_VERSION = "family_a_observability_continuation_v1.1.0.0"

# ---------------------------------------------------------------------------
# Frozen constants (design doc "Frozen constants" table -- fixed BEFORE any
# scoring; reused where the parent diagnostic already froze the value)
# ---------------------------------------------------------------------------

ESTF_ID = "estimated_service_time_first"
WFS_ID = "weighted_fair_share"
assert STAGE2_CANDIDATES[REGIME_A] == (ESTF_ID, WFS_ID)

#: Bounded rollout cap for each of the 4 continuation branches (design doc
#: SS_G): half the parent diagnostic's 3000-step bound, chosen for
#: computational tractability given 4 branches/event vs. the parent's 2,
#: while remaining >=2x Family A's own p90 episode length (662 steps).
FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS: int = 1500

#: Reused verbatim from the parent diagnostic's own preregistered,
#: outcome-blind sampling rule (design doc SS_G).
FULL_TRAJECTORY_BRANCHES_PER_SCENARIO: int = dcm.FULL_TRAJECTORY_MAX_BRANCHES_PER_SCENARIO
assert FULL_TRAJECTORY_BRANCHES_PER_SCENARIO == 3

#: History window for Group-G partial-observability features (design doc
#: SS_C Group G); matches the parent diagnostic's own HORIZON_H.
HISTORY_WINDOW: int = dcm.HORIZON_H
assert HISTORY_WINDOW == 10

FAMILY_A = dcm.FAMILY_A

TRAIN = dcm.TRAIN
VAL = dcm.VAL
TEST = dcm.TEST

# Re-exported for the runner script / tests -- identical guards to the
# parent diagnostic, never redefined here.
assert_trainval_only = dcm.assert_trainval_only
assert_no_replication_module_imported = dcm.assert_no_replication_module_imported
TestSplitAccessError = dcm.TestSplitAccessError
fit_frozen_models = dcm.fit_frozen_models
canonical_action = dcm.canonical_action
actions_disagree = dcm.actions_disagree
fork_from_live_simulator = dcm.fork_from_live_simulator
run_bounded_rollout = dcm.run_bounded_rollout


def load_family_a_trainval_scenario_table() -> pd.DataFrame:
    """Family-A-only TRAIN/VAL rows of the exact same MF-PSD population the
    completed decision-criticality study used (design doc SS_A). Never
    returns a TEST row (delegates to the parent's own guarded loader)."""
    table = dcm.load_trainval_scenario_table()
    fam_a = table[table["mechanism_family"] == FAMILY_A].reset_index(drop=True)
    assert not (fam_a["split"] == TEST).any(), "internal error: TEST row leaked into Family-A trainval table"
    return fam_a


def rebuild_scenario_from_row(row: pd.Series) -> PolicySeparationScenario:
    return dcm.rebuild_scenario_from_row(row)


# ---------------------------------------------------------------------------
# Causal observable-state feature extraction (design doc SS_C)
# ---------------------------------------------------------------------------

def _quantile_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if len(values) == 0:
        return {f"{prefix}_p10": float("nan"), f"{prefix}_p50": float("nan"),
                f"{prefix}_p90": float("nan"), f"{prefix}_mean": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_mean": float(arr.mean()),
    }


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    df = pd.DataFrame({"a": a, "b": b})
    if df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return float("nan")
    return float(df["a"].corr(df["b"], method="spearman"))


def extract_causal_features(
    state: ObservableState,
    *,
    step_size: float,
    estf_policy: BasePolicy,
    wfs_policy: BasePolicy,
    admit_ids_estf: Sequence[int],
    admit_ids_wfs: Sequence[int],
    history_rows: pd.DataFrame,
) -> Dict[str, Any]:
    """All features come from `state` as observable AT this exact decision
    step (design doc SS_C: no future arrivals/completions/realized output
    length/final outcome are ever read here) plus REAL-trajectory history up
    to and including this step (Group G). `state` is never mutated by this
    function (`estf_policy`/`wfs_policy`'s scoring helpers are read-only;
    `select_action` is never called here -- callers pass already-computed
    admitted-id sets from their own, separately deep-copied, state
    snapshots)."""
    out: Dict[str, Any] = {}

    queue = list(state.waiting_queue)
    n_gpus = len(state.gpu_states)

    # -- Group A: snapshot workload ----------------------------------------
    out["queue_length"] = len(queue)
    out["active_count"] = sum(len(g.active_request_ids) for g in state.gpu_states)
    out["completed_count"] = int(state.completed_count)
    out["step"] = int(state.step)
    out["n_gpus"] = n_gpus

    # -- Group B: request-distribution --------------------------------------
    ages = [state.time - r.arrival_time for r in queue]
    predicted_output = [float(r.predicted_output_tokens) for r in queue]
    prompt = [float(r.prompt_tokens) for r in queue]
    est_service = [predicted_service_proxy(r) for r in queue]
    out.update(_quantile_stats(ages, "queue_age"))
    out.update(_quantile_stats(predicted_output, "predicted_output_tokens"))
    out.update(_quantile_stats(prompt, "prompt_tokens"))
    out.update(_quantile_stats(est_service, "est_service_time"))

    # -- Group C: fairness/starvation ---------------------------------------
    demand = queue_class_counts(queue)
    active_by_class = queue_class_counts(r for g in state.gpu_states for r in g.active_requests_info)
    if demand:
        deficits = [demand[c] / max(1, active_by_class[c] + 1) for c in demand]
        out["max_class_deficit_ratio"] = float(max(deficits))
    else:
        out["max_class_deficit_ratio"] = float("nan")
    out["longest_waiting_age"] = float(max(ages)) if ages else 0.0
    out["n_distinct_classes_in_queue"] = int(len(demand))

    # -- Group D: urgency/slack ----------------------------------------------
    laxity = [r.slo_deadline - state.time for r in queue]
    out.update(_quantile_stats(laxity, "laxity"))
    if laxity:
        near_deadline_cutoff = HISTORY_WINDOW * step_size
        out["fraction_laxity_negative"] = float(np.mean(np.asarray(laxity) < 0.0))
        out["fraction_laxity_near_deadline"] = float(np.mean(np.asarray(laxity) < near_deadline_cutoff))
    else:
        out["fraction_laxity_negative"] = float("nan")
        out["fraction_laxity_near_deadline"] = float("nan")

    # -- Group E: resource/KV ------------------------------------------------
    kv_utils = [
        (g.current_kv_tokens / g.max_kv_tokens) if g.max_kv_tokens else 0.0
        for g in state.gpu_states
    ]
    out["mean_kv_utilization"] = float(np.mean(kv_utils)) if kv_utils else 0.0
    out["max_kv_utilization"] = float(np.max(kv_utils)) if kv_utils else 0.0
    out["free_kv_capacity"] = float(sum(g.max_kv_tokens - g.current_kv_tokens for g in state.gpu_states))
    out["prefilling_count"] = int(sum(g.prefilling_count for g in state.gpu_states))
    out["decoding_count"] = int(sum(g.decoding_count for g in state.gpu_states))

    # -- Group F: pair-specific disagreement geometry ------------------------
    admit_estf = set(admit_ids_estf)
    admit_wfs = set(admit_ids_wfs)
    sym_diff = admit_estf ^ admit_wfs
    out["n_admit_estf"] = len(admit_estf)
    out["n_admit_wfs"] = len(admit_wfs)
    out["admit_symmetric_diff_size"] = len(sym_diff)
    out["is_shallow_disagreement"] = bool(len(sym_diff) <= 2)

    k = max(len(admit_estf), len(admit_wfs), 1)
    est_scores_by_id = {r.request_id: predicted_service_proxy(r) for r in queue}
    admitted_counts_empty: Dict[str, int] = {}
    from collections import Counter as _Counter
    wfs_scores_by_id = {
        r.request_id: wfs_policy._score(r, state, _Counter())  # read-only scoring call
        for r in queue
    }
    top_estf_ids = [rid for rid, _ in sorted(est_scores_by_id.items(), key=lambda kv: kv[1])[:k]]
    top_wfs_ids = [rid for rid, _ in sorted(wfs_scores_by_id.items(), key=lambda kv: -kv[1])[:k]]
    relevant_ids = sorted(set(top_estf_ids) | set(top_wfs_ids))
    est_vec = [est_scores_by_id[rid] for rid in relevant_ids]
    wfs_vec = [wfs_scores_by_id[rid] for rid in relevant_ids]
    out["pair_rank_spearman_topk"] = _spearman(est_vec, wfs_vec)
    out["pair_topk_n"] = len(relevant_ids)

    # -- Group G: short history (real trajectory only) -----------------------
    if len(history_rows) >= 2:
        window = history_rows.tail(HISTORY_WINDOW + 1)
        x = np.arange(len(window), dtype=float)
        out["history_queue_len_slope"] = float(np.polyfit(x, window["queue_len_after_admission"].to_numpy(dtype=float), 1)[0])
        out["history_kv_util_slope"] = float(np.polyfit(x, window["mean_kv_utilization_after_admission"].to_numpy(dtype=float), 1)[0])
        out["history_admitted_count_slope"] = float(np.polyfit(x, window["admitted_count"].to_numpy(dtype=float), 1)[0])
        out["history_window_truncated"] = bool(len(window) < HISTORY_WINDOW + 1)
    else:
        out["history_queue_len_slope"] = float("nan")
        out["history_kv_util_slope"] = float("nan")
        out["history_admitted_count_slope"] = float("nan")
        out["history_window_truncated"] = True

    return out


# ---------------------------------------------------------------------------
# Per-event branch execution (design doc SS_E)
# ---------------------------------------------------------------------------

def _admitted_ids(action: Action) -> List[int]:
    return [rid for ids in action.admit.values() for rid in ids]


# ---------------------------------------------------------------------------
# GPU-counter snapshot/restore (repair 2026-08-20: see
# docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md).
# Every native policy's `select_action` (ESTF's own admit loop,
# `deterministic_place` used by WFS) mutates ONLY these two per-GPU fields
# in place, as same-call admission-planning bookkeeping --
# `g.active_request_ids` (list, appended) and `g.current_kv_tokens` (scalar,
# incremented). `g.active_requests_info` is a separate list built once by
# `GPUState.to_observable()` and is never mutated by any native policy's
# `select_action`, so it needs no snapshot/restore. Verified directly against
# `estimated_service_time_first.py`, `policy_library_v2_helpers.py`
# (`deterministic_place`), and `weighted_fair_share.py`.
# ---------------------------------------------------------------------------

GpuCounterSnapshot = List[Tuple[List[int], int]]


def snapshot_gpu_counters(state: ObservableState) -> GpuCounterSnapshot:
    """Captures the two per-GPU fields every native policy mutates in place
    during `select_action` (admission-planning bookkeeping). Must be called
    BEFORE any policy's `select_action(state)` whose mutation should not be
    observed by a later caller of this function -- in particular, before
    `inner_router.select_action(state)`, so this is the TRUE pre-decision
    baseline, not a post-admission one."""
    return [(list(g.active_request_ids), g.current_kv_tokens) for g in state.gpu_states]


def restore_gpu_counters(state: ObservableState, snapshot: GpuCounterSnapshot) -> None:
    """Restores exactly the two fields `snapshot_gpu_counters` captured, by
    value (a fresh list is written into each `active_request_ids`, never
    aliased to the snapshot's own list), so repeated restores never leak a
    shared mutable reference between calls."""
    for g, (ids, kv) in zip(state.gpu_states, snapshot):
        g.active_request_ids[:] = ids
        g.current_kv_tokens = kv


@dataclass
class FamilyAEvent:
    canonical_scenario_id: str
    split: str
    step: int
    router_chosen_policy_id: str
    features: Dict[str, Any]
    br_estf_estf_completed: int
    br_wfs_wfs_completed: int
    br_wfs_estf_completed: int
    br_estf_wfs_completed: int
    delta_native: float
    delta_same_common_estf: float
    delta_same_common_wfs: float
    delta_same: float
    continuation_dependence: float
    sign_same_eq_native: bool

    def to_row(self) -> Dict[str, Any]:
        row = {
            "canonical_scenario_id": self.canonical_scenario_id,
            "split": self.split,
            "step": self.step,
            "router_chosen_policy_id": self.router_chosen_policy_id,
            "br_estf_estf_completed": self.br_estf_estf_completed,
            "br_wfs_wfs_completed": self.br_wfs_wfs_completed,
            "br_wfs_estf_completed": self.br_wfs_estf_completed,
            "br_estf_wfs_completed": self.br_estf_wfs_completed,
            "delta_native": self.delta_native,
            "delta_same_common_estf": self.delta_same_common_estf,
            "delta_same_common_wfs": self.delta_same_common_wfs,
            "delta_same": self.delta_same,
            "continuation_dependence": self.continuation_dependence,
            "sign_same_eq_native": self.sign_same_eq_native,
        }
        row.update(self.features)
        return row


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def run_four_branches(
    sim: Simulator,
    *,
    action_estf: Action,
    action_wfs: Action,
    shadow_policies: Dict[str, BasePolicy],
    max_extra_steps: int = FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS,
) -> Dict[str, int]:
    """The 4 bounded rollouts of design doc SS_E, each an independent fork
    of `sim`'s CURRENT (pre-this-step) state (`sim` itself never mutated --
    identical isolation guarantee as `dcm.fork_from_live_simulator`, reused
    here unmodified via `run_bounded_rollout`)."""
    br_estf_estf = run_bounded_rollout(
        sim, policy=shadow_policies[ESTF_ID], policy_id=ESTF_ID,
        first_action=copy.deepcopy(action_estf), max_extra_steps=max_extra_steps,
    )
    br_wfs_wfs = run_bounded_rollout(
        sim, policy=shadow_policies[WFS_ID], policy_id=WFS_ID,
        first_action=copy.deepcopy(action_wfs), max_extra_steps=max_extra_steps,
    )
    br_wfs_estf = run_bounded_rollout(
        sim, policy=shadow_policies[ESTF_ID], policy_id=ESTF_ID,
        first_action=copy.deepcopy(action_wfs), max_extra_steps=max_extra_steps,
    )
    br_estf_wfs = run_bounded_rollout(
        sim, policy=shadow_policies[WFS_ID], policy_id=WFS_ID,
        first_action=copy.deepcopy(action_estf), max_extra_steps=max_extra_steps,
    )
    return {
        "br_estf_estf": br_estf_estf["completed_count"],
        "br_wfs_wfs": br_wfs_wfs["completed_count"],
        "br_wfs_estf": br_wfs_estf["completed_count"],
        "br_estf_wfs": br_estf_wfs["completed_count"],
    }


def compute_deltas(branches: Dict[str, int]) -> Dict[str, float]:
    """Design doc SS_E, sign convention ESTF - WFS throughout."""
    delta_native = branches["br_estf_estf"] - branches["br_wfs_wfs"]
    delta_same_common_estf = branches["br_estf_estf"] - branches["br_wfs_estf"]
    delta_same_common_wfs = branches["br_estf_wfs"] - branches["br_wfs_wfs"]
    delta_same = float(np.mean([delta_same_common_estf, delta_same_common_wfs]))
    c = delta_native - delta_same
    return {
        "delta_native": float(delta_native),
        "delta_same_common_estf": float(delta_same_common_estf),
        "delta_same_common_wfs": float(delta_same_common_wfs),
        "delta_same": delta_same,
        "continuation_dependence": float(c),
        "sign_same_eq_native": bool(_sign(delta_same) == _sign(delta_native)),
    }


# ---------------------------------------------------------------------------
# Per-scenario driver (design doc SS_B/SS_G)
# ---------------------------------------------------------------------------

class FamilyAObservabilityObserver(BasePolicy):
    """Wraps the frozen `LiveHierarchicalRouterPolicy` (never alters its
    decisions -- the action this policy returns to `Simulator.run()` is
    always exactly the inner router's real action, identical invariant to
    the parent diagnostic's `ForkingObserverPolicy`). At each step where
    `effective_regime == REGIME_A`, computes BOTH native candidates' own
    actions from independent deep copies of that step's `ObservableState`
    (read-only shadow computation), and -- for at most
    `FULL_TRAJECTORY_BRANCHES_PER_SCENARIO` genuine-disagreement steps per
    scenario, the first encountered in trajectory order (frozen, outcome-
    blind rule, design doc SS_G) -- extracts causal features and runs the
    four bounded continuation branches."""

    name = "family_a_observability_continuation_shadow_v1"

    def __init__(
        self,
        *,
        sim_ref: Simulator,
        inner_router: LiveHierarchicalRouterPolicy,
        shadow_policies: Dict[str, BasePolicy],
        canonical_scenario_id: str,
        split: str,
        step_size: float,
        max_branches: int = FULL_TRAJECTORY_BRANCHES_PER_SCENARIO,
    ) -> None:
        self.sim_ref = sim_ref
        self.inner_router = inner_router
        self.shadow_policies = shadow_policies
        self.canonical_scenario_id = canonical_scenario_id
        self.split = split
        self.step_size = step_size
        self.max_branches = max_branches

        self.branches_used = 0
        self.events: List[FamilyAEvent] = []

    def reset(self) -> None:
        self.inner_router.reset()
        for p in self.shadow_policies.values():
            p.reset()
        self.branches_used = 0
        self.events = []

    def select_action(self, state: ObservableState) -> Action:
        # TRUE pre-decision snapshot -- captured BEFORE the real router
        # delegates to whichever ONE native policy Stage-2 selected (that
        # delegation mutates `state.gpu_states` in place as same-call
        # admission-planning bookkeeping -- see e.g.
        # `EstimatedServiceTimeFirstPolicy.select_action`'s direct
        # `gpu.active_request_ids.append(...)` / `gpu.current_kv_tokens +=
        # ...`). This is the repair for the 2026-08-20 n_events_total=0
        # defect: the pre-repair code snapshotted AFTER this mutation,
        # so both shadow candidates below were evaluated against an
        # already-capacity-consumed baseline instead of the true open
        # decision point, which suppressed essentially all detectable
        # disagreement. See
        # docs/current/family_a_observability_continuation_v1_repair_audit_20260820.md.
        pre_decision_gpu_state = snapshot_gpu_counters(state)

        real_action = self.inner_router.select_action(state)
        row = self.inner_router.trajectory[-1] if self.inner_router.trajectory else None

        # Snapshot of the state as the real router left it. `_log_step`
        # (inside `inner_router.select_action`, above) already fully
        # consumed the post-admission mutation for its own trajectory-row
        # logging, and `Simulator.run()` never reads `state` again after
        # `policy.select_action` returns (verified against
        # `simulator.py::run`) -- so nothing downstream actually depends on
        # `state.gpu_states` being left in this form. This snapshot/final
        # restore (bottom of this method) is purely defensive: it
        # guarantees this observer is a no-op with respect to anything a
        # future caller might read from `state`, identical to what a
        # non-instrumented `inner_router.select_action(state)` call alone
        # would have left behind.
        post_real_admission_gpu_state = snapshot_gpu_counters(state)

        if (
            row is not None
            and row.effective_regime == REGIME_A
            and self.branches_used < self.max_branches
        ):
            # Score ESTF and WFS against the SAME true pre-decision
            # snapshot (design doc SS_D: "both actions are already fully
            # determined [by] the snapshotted ObservableState") -- far
            # cheaper than a full `copy.deepcopy(state)` on every step
            # (this branch is reached on every Family-A-active step, not
            # just genuine disagreements).
            restore_gpu_counters(state, pre_decision_gpu_state)
            action_estf = self.shadow_policies[ESTF_ID].select_action(state)
            restore_gpu_counters(state, pre_decision_gpu_state)
            action_wfs = self.shadow_policies[WFS_ID].select_action(state)
            restore_gpu_counters(state, pre_decision_gpu_state)

            if actions_disagree(action_estf, action_wfs):
                # `state` is currently restored to the true pre-decision
                # baseline (the restore immediately above), so this deep
                # copy correctly captures decision-time-observable state
                # (design doc SS_C), not a post-admission-contaminated one.
                feature_state = copy.deepcopy(state)
                history_rows = self.inner_router.trajectory_df()
                features = extract_causal_features(
                    feature_state,
                    step_size=self.step_size,
                    estf_policy=self.shadow_policies[ESTF_ID],
                    wfs_policy=self.shadow_policies[WFS_ID],
                    admit_ids_estf=_admitted_ids(action_estf),
                    admit_ids_wfs=_admitted_ids(action_wfs),
                    history_rows=history_rows,
                )
                branches = run_four_branches(
                    self.sim_ref,
                    action_estf=action_estf,
                    action_wfs=action_wfs,
                    shadow_policies=self.shadow_policies,
                )
                deltas = compute_deltas(branches)
                self.events.append(FamilyAEvent(
                    canonical_scenario_id=self.canonical_scenario_id,
                    split=self.split,
                    step=int(state.step),
                    router_chosen_policy_id=row.selected_policy,
                    features=features,
                    br_estf_estf_completed=branches["br_estf_estf"],
                    br_wfs_wfs_completed=branches["br_wfs_wfs"],
                    br_wfs_estf_completed=branches["br_wfs_estf"],
                    br_estf_wfs_completed=branches["br_estf_wfs"],
                    delta_native=deltas["delta_native"],
                    delta_same_common_estf=deltas["delta_same_common_estf"],
                    delta_same_common_wfs=deltas["delta_same_common_wfs"],
                    delta_same=deltas["delta_same"],
                    continuation_dependence=deltas["continuation_dependence"],
                    sign_same_eq_native=deltas["sign_same_eq_native"],
                ))
                self.branches_used += 1

        # Defensive final restore -- see comment above
        # `post_real_admission_gpu_state`. Leaves `state` byte-identical to
        # what a non-instrumented `inner_router.select_action(state)` call
        # would have left, regardless of anything the shadow computation
        # above did.
        restore_gpu_counters(state, post_real_admission_gpu_state)

        return real_action


@dataclass
class ScenarioFamilyAResult:
    canonical_scenario_id: str
    split: str
    n_steps: int
    n_family_a_active_steps: int
    events: List[FamilyAEvent]


def run_family_a_scenario_diagnostic(
    scenario: PolicySeparationScenario,
    *,
    canonical_scenario_id: str,
    stage1,
    stage2_selectors: Dict[str, Stage2Selector],
    seed: int,
    split: str,
) -> ScenarioFamilyAResult:
    feature_rows = build_feature_rows_by_regime(scenario, canonical_scenario_id)
    inner_router = LiveHierarchicalRouterPolicy(
        scenario_id=canonical_scenario_id,
        stage1=stage1,
        stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows,
        record_trajectory=True,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    step_size = sim.config.service_model.step_size

    shadow_policies = build_native_policy_instances()
    observer = FamilyAObservabilityObserver(
        sim_ref=sim,
        inner_router=inner_router,
        shadow_policies=shadow_policies,
        canonical_scenario_id=canonical_scenario_id,
        split=split,
        step_size=step_size,
    )
    sim.run(observer, workload_tag=canonical_scenario_id, seed=seed)

    traj = inner_router.trajectory_df()
    n_active_a = int((traj["effective_regime"] == REGIME_A).sum()) if len(traj) else 0

    return ScenarioFamilyAResult(
        canonical_scenario_id=canonical_scenario_id,
        split=split,
        n_steps=len(traj),
        n_family_a_active_steps=n_active_a,
        events=observer.events,
    )


def run_family_a_row_diagnostic(
    row: pd.Series,
    *,
    stage1,
    stage2_selectors: Dict[str, Stage2Selector],
) -> ScenarioFamilyAResult:
    assert_trainval_only(row["split"])
    assert row["mechanism_family"] == FAMILY_A
    assert_no_replication_module_imported()

    canonical_scenario_id = row["canonical_scenario_id"]
    scenario = rebuild_scenario_from_row(row)
    return run_family_a_scenario_diagnostic(
        scenario,
        canonical_scenario_id=canonical_scenario_id,
        stage1=stage1,
        stage2_selectors=stage2_selectors,
        seed=int(row["seed"]),
        split=row["split"],
    )

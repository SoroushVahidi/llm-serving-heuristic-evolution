"""Decision-Criticality & Regime-Timescale Diagnostic v1 -- TRAIN/VAL-ONLY.

Implements the diagnostic frozen by
`docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`. Quantifies, on
TRAIN/VAL scenarios only:

  1. how often the two native policies in an active regime's pair actually
     propose different actions (`ACTION_DISAGREEMENT`);
  2. how often that disagreement causally matters downstream, via bounded
     counterfactual simulator forks (`IMMEDIATE_STATE_DIVERGENCE`,
     `SHORT_HORIZON_CAUSAL_DIVERGENCE`, bounded `FULL_TRAJECTORY_OPPORTUNITY`);
  3. how long raw A/B/C regime episodes last, vs. the frozen `dwell=20`
     reaction floor (episode-timescale + dwell-latency diagnostics).

DIAGNOSTIC / METHODOLOGY ONLY. Never modifies, retrains, or re-thresholds
anything frozen by `hierarchical_regime_router_v1.py` /
`hierarchical_stage2_selectors_v1.py` / `hierarchical_router_live_harness_v1.py`
/ `configs/hierarchical_regime_router_v1_gates.json`. Never reads a TEST-split
scenario or telemetry row (`assert_trainval_only` raises immediately on any
"test" value). Never imports or reads
`family_b_balanced_replication_v1` / `experiments/family_b_balanced_replication_v1/`
-- the preregistered held-out Family-B replication is out of scope for this
diagnostic (design doc S0). Computes no new project-level scientific verdict.

Counterfactual forking never mutates the real, currently-running reference
`Simulator` -- every fork is an isolated `copy.deepcopy` of only the mutable
per-step state (see `_fork_from_live_simulator`), driven forward exclusively
via the simulator's own unmodified `_apply_action`/`_advance_decode`/
`_build_observable_state` methods (zero reimplementation of admission/decode
logic).
"""
from __future__ import annotations

import copy
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from ..policy_separation.hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    DWELL_MINIMUM_STEPS,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    STAGE2_CANDIDATES,
    Stage1Router,
    build_splits,
)
from ..policy_separation.hierarchical_router_live_harness_v1 import (
    _PREFILL_CHUNK_BY_POLICY,
    _apply_prefill_chunk_override,
    LiveHierarchicalRouterPolicy,
    build_native_policy_instances,
    build_feature_rows_by_regime,
)
from ..policy_separation.schema import PolicySeparationScenario
from ..selector.hierarchical_stage2_selectors_v1 import Stage2Selector

SCHEMA_VERSION = "decision_criticality_timescale_trainval_v1.1.0.0"

# ---------------------------------------------------------------------------
# Frozen constants (design doc SS5F/SS7/SS5D -- fixed BEFORE any scoring)
# ---------------------------------------------------------------------------

#: Short-horizon counterfactual window, in scheduling steps. Frozen before
#: running any diagnostic; never chosen based on an observed result (design
#: doc SS5F).
HORIZON_H: int = 10

#: Read-only reference to the frozen FSM dwell minimum -- never modified or
#: swept by this module.
DWELL_REFERENCE: int = DWELL_MINIMUM_STEPS
assert DWELL_REFERENCE == 20

#: Bounded full-trajectory branch cap (design doc SS5D): at most this many
#: extra steps are run past a fork point when probing "opportunity" toward
#: full-scenario completion. A fixed, preregistered cap -- not tuned on any
#: observed result.
FULL_TRAJECTORY_MAX_EXTRA_STEPS: int = 3000

#: At most this many full-trajectory branch attempts per scenario (design
#: doc SS5D), taken as the first N disagreement steps encountered in that
#: scenario's reference trajectory (a fixed, outcome-blind selection rule).
FULL_TRAJECTORY_MAX_BRANCHES_PER_SCENARIO: int = 3

TRAIN = "train"
VAL = "val"
TEST = "test"
TRAINVAL_SPLITS = frozenset({TRAIN, VAL})

ROOT = Path(__file__).resolve().parents[3]
MF_PSD_SCENARIOS_CSV = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
TELEMETRY_CSV = ROOT / "experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv"
DATASETS_ROOT = ROOT / ".local_data"

FORBIDDEN_REPLICATION_MODULE = "family_b_balanced_replication_v1"


# ---------------------------------------------------------------------------
# Split guards (design doc SS3 / SS12.1-2)
# ---------------------------------------------------------------------------

class TestSplitAccessError(RuntimeError):
    """Raised whenever this diagnostic is asked to touch a TEST-split row."""


def assert_trainval_only(split_value: str) -> None:
    """Raises `TestSplitAccessError` unless `split_value` is 'train' or
    'val'. The single split guard every scenario-id/telemetry-row entry
    point in this module routes through."""
    if split_value not in TRAINVAL_SPLITS:
        raise TestSplitAccessError(
            f"decision_criticality_timescale_trainval_v1 is TRAIN/VAL-only; "
            f"got split={split_value!r} (forbidden: TEST access is categorically "
            f"disallowed by design doc SS0/SS3)."
        )


def assert_no_replication_module_imported() -> None:
    """Structural guard (design doc SS0/SS12.2): this diagnostic must never
    import or reference the preregistered-but-not-authorized Family-B
    replication module. Checked by source-text inspection in the test
    suite, and callable here for a runtime self-check."""
    import sys

    for name in sys.modules:
        if name.endswith(FORBIDDEN_REPLICATION_MODULE):
            raise RuntimeError(
                f"forbidden module {name!r} is imported; the Family-B held-out "
                f"replication is out of scope for this TRAIN/VAL diagnostic."
            )


# ---------------------------------------------------------------------------
# Scenario loading (TRAIN/VAL only)
# ---------------------------------------------------------------------------

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"

FAMILY_TO_REGIME = {FAMILY_A: REGIME_A, FAMILY_B: REGIME_B, FAMILY_C: REGIME_C}


def load_trainval_scenario_table() -> pd.DataFrame:
    """The frozen MF-PSD scenario table, `split`-annotated via the frozen
    `build_splits`, filtered to TRAIN/VAL rows only. Raises via
    `assert_trainval_only` if (defensively) any non-TRAIN/VAL row were ever
    passed downstream -- this function itself never returns a TEST row."""
    scen = pd.read_csv(MF_PSD_SCENARIOS_CSV)
    split_map = build_splits(scen)
    scen = scen.copy()
    scen["split"] = scen["canonical_scenario_id"].map(split_map)
    trainval = scen[scen["split"].isin(TRAINVAL_SPLITS)].reset_index(drop=True)
    assert not (trainval["split"] == TEST).any(), "internal error: TEST row leaked into trainval table"
    return trainval


def rebuild_scenario_from_row(row: pd.Series) -> PolicySeparationScenario:
    """Rebuild the real `PolicySeparationScenario` for one TRAIN/VAL MF-PSD
    scenario row, using the exact same frozen template functions
    `run_hierarchical_regime_router_live_reeval_v1.py::rebuild_scenario`
    already uses -- Family B additionally passes `datasets_root=DATASETS_ROOT`
    (that script never exercised Family B: its primary TEST split has 0
    Family-B scenarios) so BurstGPT-backed non-synthetic token lengths
    resolve correctly for the real TRAIN/VAL Family-B rows this diagnostic
    does need (design doc SS3)."""
    assert_trainval_only(row["split"])
    family = row["mechanism_family"]
    if family == FAMILY_A:
        from ..policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2
        return case_fairness_vs_size_v2(
            target_utilization=row["feat_A__target_utilization"],
            tenant_weight_skew=row["feat_A__tenant_weight_skew"],
            favored_tenant_size=row["feat_A__favored_tenant_size"],
            prediction_noise_sigma=row["feat_A__prediction_noise_sigma"],
            seed=int(row["seed"]),
            datasets_root=DATASETS_ROOT,
        )
    if family == FAMILY_B:
        from ..policy_separation.templates_prefill_decode_v2 import case_prefill_decode_ttft_contention
        return case_prefill_decode_ttft_contention(
            hog_count=row["feat_B__hog_count"],
            late_pressure=row["feat_B__late_pressure"],
            slo_emphasis=row["feat_B__slo_emphasis"],
            seed=int(row["seed"]),
            datasets_root=DATASETS_ROOT,
        )
    if family == FAMILY_C:
        from ..policy_separation.templates_kv_pressure_v2 import case_kv_pressure_reserve_contention_v2
        return case_kv_pressure_reserve_contention_v2(
            bulk_pressure=row["feat_C__bulk_pressure"],
            urgent_arrival_phase=row["feat_C__urgent_arrival_phase"],
            urgent_tightness=row["feat_C__urgent_tightness"],
            seed=int(row["seed"]),
            datasets_root=DATASETS_ROOT,
        )
    raise ValueError(f"unknown mechanism_family {family!r}")


# ---------------------------------------------------------------------------
# Model fitting (TRAIN only -- no tuning, exact frozen recipe reuse)
# ---------------------------------------------------------------------------

def fit_frozen_models() -> Tuple[Stage1Router, Dict[str, Stage2Selector]]:
    """Fits Stage-1/Stage-2 with the exact frozen recipe, TRAIN-only, no
    hyperparameter changes -- identical fitting code path to
    `run_hierarchical_regime_router_live_reeval_v1.py`."""
    from ..policy_separation.hierarchical_regime_router_v1 import add_regime_labels
    from ..policy_separation.hierarchical_router_evaluation_v1 import load_scenario_level_dataset
    from ..selector.hierarchical_stage2_selectors_v1 import fit_all_stage2_selectors

    scen = pd.read_csv(MF_PSD_SCENARIOS_CSV)
    split_map = build_splits(scen)

    telemetry = pd.read_csv(TELEMETRY_CSV)
    telemetry = add_regime_labels(telemetry)
    telemetry["split"] = telemetry["canonical_scenario_id"].map(split_map)
    train_tel = telemetry[telemetry["split"] == TRAIN]
    stage1 = Stage1Router().fit(train_tel)

    scenario_df = load_scenario_level_dataset()
    train_df = scenario_df[scenario_df["split"] == TRAIN]
    train_by_regime = {r: train_df[train_df["regime_ground_truth"] == r] for r in ACTIVE_REGIMES}
    stage2_selectors = fit_all_stage2_selectors(train_by_regime)
    return stage1, stage2_selectors


# ---------------------------------------------------------------------------
# Canonical action comparison (design doc SS5B)
# ---------------------------------------------------------------------------

def canonical_action(action: Action) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """`{gpu_id: sorted(admit_ids)}`, as a hashable/orderable tuple. Complete
    (non-lossy) for all six frozen native policies used here -- verified by
    `assert_action_has_no_non_admit_verbs`, not assumed."""
    return tuple(sorted((gid, tuple(sorted(ids))) for gid, ids in action.admit.items() if ids))


def assert_action_has_no_non_admit_verbs(action: Action) -> None:
    """Design doc SS5B / SS12.4: the six frozen native policies in the three
    frozen pairs never use preempt/swap/migrate/hold_decode -- checked
    directly on every action this module canonicalizes, not merely assumed."""
    if action.preempt or action.swap or action.migrate or action.hold_decode:
        raise AssertionError(
            "canonical_action() assumes admit-only actions for the frozen "
            "native-pair policies; got a non-empty preempt/swap/migrate/"
            "hold_decode verb -- canonicalization would be lossy."
        )


def actions_disagree(action_a: Action, action_b: Action) -> bool:
    assert_action_has_no_non_admit_verbs(action_a)
    assert_action_has_no_non_admit_verbs(action_b)
    return canonical_action(action_a) != canonical_action(action_b)


# ---------------------------------------------------------------------------
# Lightweight simulator forking (design doc SS5C)
# ---------------------------------------------------------------------------

def _state_fingerprint(sim: Simulator) -> Tuple[Any, ...]:
    """Cheap structural fingerprint of a simulator's mutable per-step state,
    used only by tests to assert a fork never mutated the original."""
    gpu_fp = tuple(
        (g.gpu_id, tuple(sorted(g._active.keys())), g.current_kv_tokens)
        for g in sim._gpus
    )
    waiting_fp = tuple(sorted(sim._waiting_map.keys()))
    return (gpu_fp, waiting_fp, sim._step, sim._time, len(sim._completed))


def alternative_policy_id(regime: str, chosen_policy_id: str) -> Optional[str]:
    """The OTHER native-pair candidate for `regime`, or None if
    `chosen_policy_id` is a fallback dispatch (`weighted_fair_share` when
    `regime` itself is not one of the two candidates for that regime --
    i.e. the FSM was in a fallback state, not truly routing within the
    pair)."""
    if regime not in STAGE2_CANDIDATES:
        return None
    p0, p1 = STAGE2_CANDIDATES[regime]
    if chosen_policy_id == p0:
        return p1
    if chosen_policy_id == p1:
        return p0
    return None


@dataclass
class LiveFork:
    """An isolated, deep-copied continuation of a live `Simulator`, driven
    forward only via the simulator's own unmodified `_apply_action`/
    `_advance_decode`/`_build_observable_state` methods. Never shares any
    mutable container with the simulator it was forked from."""

    shell: Simulator
    policy: BasePolicy
    policy_id: str
    arrival_times: np.ndarray  # sorted, fork-local (relative to the future-arrivals suffix)
    arrival_idx: int  # index into shell._pending_arrivals (fork-local slice)
    step_size: float
    created_at_step: int
    target_step_h: int  # created_at_step + HORIZON_H
    regime: str = ""
    chosen_policy_id: str = ""
    completed_at_creation: int = 0
    completed_in_window: int = 0
    immediate_recorded: bool = False
    immediate_finalized: bool = False
    immediate_state: Optional[ObservableState] = None
    finished: bool = False

    def advance_one_step(self, *, forced_action: Optional[Action] = None) -> Tuple[ObservableState, Action]:
        """Advance the fork exactly one scheduling step, mirroring
        `Simulator.run()`'s per-step body (enqueue -> build state -> action
        -> apply -> advance), using the simulator's own unmodified methods.
        `forced_action`, when given, is used instead of calling
        `self.policy.select_action` (used only for the fork's very first
        step, where the already-computed alternative-candidate action is
        reused rather than recomputed)."""
        shell = self.shell
        pending = shell._pending_arrivals
        n = len(pending)
        while self.arrival_idx < n and pending[self.arrival_idx].request.arrival_time <= shell._time:
            ir = pending[self.arrival_idx]
            shell._waiting.append(ir)
            shell._waiting_map[ir.request_id] = ir
            self.arrival_idx += 1

        state = shell._build_observable_state()
        action = forced_action if forced_action is not None else self.policy.select_action(state)
        if self.policy_id in _PREFILL_CHUNK_BY_POLICY:
            _apply_prefill_chunk_override(action, state, self.policy_id)

        shell._apply_action(action)
        completed = shell._advance_decode(action)
        shell._completed.extend(completed)
        self.completed_in_window += len(completed)

        shell._step += 1
        shell._time = shell._step * self.step_size

        all_active_done = sum(g.num_active for g in shell._gpus) == 0
        queue_empty = (
            len(shell._waiting) == 0
            and len(shell._migrating) == 0
            and len(shell._relocating) == 0
        )
        all_arrivals_done = self.arrival_idx >= n
        if all_arrivals_done and queue_empty and all_active_done:
            self.finished = True
        return state, action


def _clone_gpu_map(gpus: List[Any]) -> Dict[int, Any]:
    return {g.gpu_id: g for g in gpus}


def fork_from_live_simulator(
    sim: Simulator,
    *,
    policy: BasePolicy,
    policy_id: str,
    first_action: Action,
    regime: str = "",
    chosen_policy_id: str = "",
    completed_at_creation: int = 0,
) -> LiveFork:
    """Builds an isolated fork of `sim`'s CURRENT (pre-this-step) mutable
    state and immediately applies `first_action` to it (design doc SS5C).
    `sim` itself is never mutated: `_gpus`/`_waiting`*/`_migrating`*/
    `_relocating`/the not-yet-enqueued suffix of `_pending_arrivals` are all
    `copy.deepcopy`'d into fresh, independent objects before any mutation is
    ever applied to the fork; `config` and the already-consumed
    `_pending_arrivals` prefix and `_completed` are shared by reference
    (verified never mutated by `_apply_action`/`_advance_decode` -- see
    module docstring) or shallow-copied where a fresh *list* (not fresh
    *elements*) is needed."""
    step_size = sim.config.service_model.step_size

    shell = Simulator.__new__(Simulator)
    shell.config = sim.config
    shell._gpus = copy.deepcopy(sim._gpus)
    shell._gpu_map = _clone_gpu_map(shell._gpus)
    shell._waiting = copy.deepcopy(sim._waiting)
    shell._waiting_map = {ir.request_id: ir for ir in shell._waiting}
    shell._migrating = copy.deepcopy(sim._migrating)
    shell._migrating_map = {ir.request_id: ir for ir in shell._migrating}
    shell._relocating = copy.deepcopy(sim._relocating)

    # arrival_idx-at-fork == count of pending arrivals whose arrival_time
    # has already elapsed (see design doc SS5C: this equals the real,
    # externally-untracked `arrival_idx` local variable inside
    # `Simulator.run()` at this exact point, since enqueueing is strictly
    # monotonic and exhaustive up to `sim._time` every step).
    all_arrival_times = np.array([ir.request.arrival_time for ir in sim._pending_arrivals])
    consumed = int(bisect_right(all_arrival_times, sim._time))
    # Only the NOT-YET-ENQUUED suffix is deep-copied -- the already-consumed
    # prefix's InternalRequest objects are already faithfully represented
    # (in whatever state they've reached) by the just-deep-copied
    # `_gpus`/`_waiting`/`_migrating`/`_relocating` above; sharing the
    # prefix objects would be fine too (they're never touched again via
    # `_pending_arrivals` indexing), but slicing them out entirely avoids
    # any possibility of accidental future aliasing.
    future_originals = sim._pending_arrivals[consumed:]
    shell._pending_arrivals = copy.deepcopy(future_originals)

    shell._completed = list(sim._completed)  # fresh list container; elements are write-once, safe to share
    shell._step = sim._step
    shell._time = sim._time
    shell._util_history = []
    shell._batch_history = []
    shell._waiting_queue_history = []
    shell._policy_times = []
    shell._idle_skipped = 0

    fork = LiveFork(
        shell=shell,
        policy=policy,
        policy_id=policy_id,
        arrival_times=np.array([ir.request.arrival_time for ir in shell._pending_arrivals]),
        arrival_idx=0,
        step_size=step_size,
        created_at_step=sim._step,
        target_step_h=sim._step + HORIZON_H,
        regime=regime,
        chosen_policy_id=chosen_policy_id,
        completed_at_creation=completed_at_creation,
    )
    # First step of the fork: apply the ALREADY-COMPUTED alternative action
    # (computed from a deep-copied ObservableState at the disagreement step
    # -- see `run_reference_trajectory_with_shadow_forks`) rather than
    # recomputing it, so the exact action compared in ACTION_DISAGREEMENT is
    # the one whose consequences get measured.
    fork.advance_one_step(forced_action=first_action)
    fork.immediate_recorded = True
    fork.immediate_state = fork.shell._build_observable_state()
    return fork


# ---------------------------------------------------------------------------
# Divergence metrics (design doc SS5G)
# ---------------------------------------------------------------------------

def _queue_length(state: ObservableState) -> int:
    return len(state.waiting_queue)


def _active_count(state: ObservableState) -> int:
    return sum(len(g.active_request_ids) for g in state.gpu_states)


def _mean_kv_utilization(state: ObservableState) -> float:
    utils = [
        (g.current_kv_tokens / g.max_kv_tokens) if g.max_kv_tokens else 0.0
        for g in state.gpu_states
    ]
    return float(np.mean(utils)) if utils else 0.0


@dataclass
class DivergenceSnapshot:
    queue_length_abs_diff: int
    active_count_abs_diff: int
    kv_utilization_abs_diff: float
    completed_count_abs_diff: int
    any_nonzero: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def compare_states(
    real_state: ObservableState, fork_state: ObservableState, fork_completed_count: int, real_completed_delta: int
) -> DivergenceSnapshot:
    q = abs(_queue_length(real_state) - _queue_length(fork_state))
    a = abs(_active_count(real_state) - _active_count(fork_state))
    kv = abs(_mean_kv_utilization(real_state) - _mean_kv_utilization(fork_state))
    c = abs(real_completed_delta - fork_completed_count)
    return DivergenceSnapshot(
        queue_length_abs_diff=int(q),
        active_count_abs_diff=int(a),
        kv_utilization_abs_diff=float(kv),
        completed_count_abs_diff=int(c),
        any_nonzero=bool(q or a or (kv > 1e-9) or c),
    )


# ---------------------------------------------------------------------------
# Episode segmentation (design doc SS6)
# ---------------------------------------------------------------------------

REGIME_NONE_LABEL = "NONE"
REGIME_OVERLAP_LABEL = "OVERLAP"
REGIME_A_ACTIVE_LABEL = "A_active"
REGIME_B_ACTIVE_LABEL = "B_active_v2"
REGIME_C_ACTIVE_LABEL = "C_active"


def classify_raw_activity_state(a_active: bool, b_active_v2: bool, c_active: bool) -> str:
    n = int(a_active) + int(b_active_v2) + int(c_active)
    if n == 0:
        return REGIME_NONE_LABEL
    if n > 1:
        return REGIME_OVERLAP_LABEL
    if a_active:
        return REGIME_A_ACTIVE_LABEL
    if b_active_v2:
        return REGIME_B_ACTIVE_LABEL
    return REGIME_C_ACTIVE_LABEL


def segment_episodes(labels: Sequence[str]) -> pd.DataFrame:
    """Contiguous runs of `labels` -> one row per episode: label, start_idx
    (inclusive), end_idx (inclusive), length."""
    if len(labels) == 0:
        return pd.DataFrame(columns=["label", "start_idx", "end_idx", "length"])
    s = pd.Series(list(labels))
    group_id = (s != s.shift()).cumsum()
    rows = []
    for _, grp in s.groupby(group_id):
        idx = grp.index
        rows.append({
            "label": grp.iloc[0],
            "start_idx": int(idx.min()),
            "end_idx": int(idx.max()),
            "length": int(len(grp)),
        })
    return pd.DataFrame(rows)


def episode_length_distribution(lengths: Sequence[int]) -> Dict[str, Any]:
    if len(lengths) == 0:
        return {"count": 0}
    arr = np.asarray(list(lengths), dtype=float)
    return {
        "count": int(len(arr)),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "fraction_lt_5": float((arr < 5).mean()),
        "fraction_lt_10": float((arr < 10).mean()),
        "fraction_lt_20": float((arr < 20).mean()),
        "fraction_eq_20": float((arr == 20).mean()),
        "fraction_gt_20": float((arr > 20).mean()),
        "fraction_gt_40": float((arr > 40).mean()),
    }


def fraction_active_steps_in_short_episodes(episodes: pd.DataFrame, dwell: int = DWELL_REFERENCE) -> Optional[float]:
    """Design doc SS6: fraction of TOTAL ACTIVE STEPS (not episode count)
    contained in episodes shorter than `dwell`."""
    if len(episodes) == 0:
        return None
    total = int(episodes["length"].sum())
    if total == 0:
        return None
    short = int(episodes.loc[episodes["length"] < dwell, "length"].sum())
    return short / total


# ---------------------------------------------------------------------------
# Dwell-latency diagnostic (design doc SS7)
# ---------------------------------------------------------------------------

FULLY_REACTABLE = "FULLY_REACTABLE"
PARTIALLY_REACTABLE = "PARTIALLY_REACTABLE"
UNREACHABLE_UNDER_DWELL20 = "UNREACHABLE_UNDER_DWELL20"


def classify_dwell_reactability(length: int, dwell: int = DWELL_REFERENCE) -> str:
    if length < dwell:
        return UNREACHABLE_UNDER_DWELL20
    if length < 2 * dwell:
        return PARTIALLY_REACTABLE
    return FULLY_REACTABLE


def dwell_latency_diagnostic(effective_regime_episodes: pd.DataFrame, dwell: int = DWELL_REFERENCE) -> pd.DataFrame:
    """`effective_regime_episodes` must be the output of `segment_episodes`
    applied to a trajectory's `effective_regime` column, already filtered to
    active-regime (A/B/C) rows only."""
    out = effective_regime_episodes.copy()
    out["earliest_switch_eligible_step"] = out["start_idx"] + dwell
    out["ends_before_switch_eligible"] = out["end_idx"] < out["earliest_switch_eligible_step"]
    out["useful_active_steps_remaining_after_eligibility"] = np.where(
        out["ends_before_switch_eligible"],
        0,
        (out["end_idx"] - out["earliest_switch_eligible_step"] + 1).clip(lower=0),
    )
    out["reactability_class"] = out["length"].apply(lambda L: classify_dwell_reactability(int(L), dwell))
    return out


# ---------------------------------------------------------------------------
# Full-trajectory (bounded) branch rollout (design doc SS5D)
# ---------------------------------------------------------------------------

def run_bounded_rollout(
    sim: Simulator,
    *,
    policy: BasePolicy,
    policy_id: str,
    first_action: Action,
    max_extra_steps: int = FULL_TRAJECTORY_MAX_EXTRA_STEPS,
) -> Dict[str, Any]:
    """Forks `sim` at its current step and drives the fork forward, alone
    (not in lockstep with anything), for up to `max_extra_steps` further
    steps or until its own queue+active-set empties with no more arrivals
    -- a BOUNDED proxy for "continue this native policy to scenario
    completion" (design doc SS5D), never a literal full-scenario oracle for
    scenarios longer than the bound. `sim` itself is never mutated (see
    `fork_from_live_simulator`)."""
    fork = fork_from_live_simulator(
        sim, policy=policy, policy_id=policy_id, first_action=first_action,
    )
    steps_run = 1
    while not fork.finished and steps_run < max_extra_steps:
        fork.advance_one_step()
        steps_run += 1
    return {
        "policy_id": policy_id,
        "steps_run": steps_run,
        "bounded_horizon_steps": max_extra_steps,
        "ran_to_natural_completion": bool(fork.finished),
        "completed_count": int(fork.completed_in_window),
    }


# ---------------------------------------------------------------------------
# Reference-trajectory driver with shadow disagreement detection + forking
# (design doc SS5A-D)
# ---------------------------------------------------------------------------

class ForkingObserverPolicy(BasePolicy):
    """Wraps the frozen `LiveHierarchicalRouterPolicy` (never alters its
    decisions) and, purely as a read-only shadow computation, detects
    per-step `ACTION_DISAGREEMENT` against the OTHER native-pair candidate
    and manages bounded counterfactual forks (design doc SS5). The action
    this policy returns to `Simulator.run()` is always exactly the inner
    router's real action -- the reference trajectory this diagnostic
    observes is byte-for-byte what `hierarchical_router_live_harness_v1`
    would have produced on its own."""

    name = "decision_criticality_shadow_observer_v1"

    def __init__(
        self,
        *,
        sim_ref: Simulator,
        inner_router: LiveHierarchicalRouterPolicy,
        shadow_policies: Dict[str, BasePolicy],
        full_trajectory_budget: int = FULL_TRAJECTORY_MAX_BRANCHES_PER_SCENARIO,
        enable_full_trajectory_branches: bool = True,
    ) -> None:
        self.sim_ref = sim_ref
        self.inner_router = inner_router
        self.shadow_policies = shadow_policies
        self.full_trajectory_budget = full_trajectory_budget
        self.enable_full_trajectory_branches = enable_full_trajectory_branches

        self.in_flight_forks: List[LiveFork] = []
        self.full_trajectory_forks_used = 0
        self.disagreement_rows: List[Dict[str, Any]] = []
        self.full_trajectory_results: List[Dict[str, Any]] = []
        self._last_state: Optional[ObservableState] = None

    def reset(self) -> None:
        self.inner_router.reset()
        for p in self.shadow_policies.values():
            p.reset()
        self.in_flight_forks = []
        self.full_trajectory_forks_used = 0
        self.disagreement_rows = []
        self.full_trajectory_results = []
        self._last_state = None

    def select_action(self, state: ObservableState) -> Action:
        step = state.step
        self._last_state = state

        # -- (1) finalize any forks due at this exact step -------------
        still_in_flight = []
        for fork in self.in_flight_forks:
            if not fork.immediate_finalized and step == fork.created_at_step + 1:
                real_delta = state.completed_count - fork.completed_at_creation
                snap = compare_states(state, fork.immediate_state, fork.completed_in_window, real_delta)
                self._record_disagreement_result(fork, horizon=1, snapshot=snap)
                fork.immediate_finalized = True
            if step >= fork.target_step_h:
                fork_state = fork.shell._build_observable_state()
                real_delta = state.completed_count - fork.completed_at_creation
                snap = compare_states(state, fork_state, fork.completed_in_window, real_delta)
                self._record_disagreement_result(fork, horizon=HORIZON_H, snapshot=snap)
                continue  # fully finalized -- drop from in-flight list
            still_in_flight.append(fork)
        self.in_flight_forks = still_in_flight

        # -- (2) advance every still-in-flight fork by exactly one more
        #        step (forks created in THIS call are handled separately
        #        below, at their own creation) -----------------------
        for fork in self.in_flight_forks:
            if not fork.finished:
                fork.advance_one_step()

        # -- (3) the REAL router decision (never altered by anything
        #        below) ------------------------------------------------
        real_action = self.inner_router.select_action(state)
        row = self.inner_router.trajectory[-1] if self.inner_router.trajectory else None

        # -- (4) shadow disagreement check + fork creation --------------
        if row is not None and row.effective_regime in ACTIVE_REGIMES:
            regime = row.effective_regime
            chosen_id = row.selected_policy
            alt_id = alternative_policy_id(regime, chosen_id)
            if alt_id is not None:
                shadow_state = copy.deepcopy(state)
                alt_policy = self.shadow_policies[alt_id]
                alt_action = alt_policy.select_action(shadow_state)
                if alt_id in _PREFILL_CHUNK_BY_POLICY:
                    _apply_prefill_chunk_override(alt_action, shadow_state, alt_id)
                disagree = actions_disagree(real_action, alt_action)
                self.disagreement_rows.append({
                    "step": step,
                    "regime": regime,
                    "chosen_policy_id": chosen_id,
                    "alt_policy_id": alt_id,
                    "disagree": disagree,
                })
                if disagree:
                    fork = fork_from_live_simulator(
                        self.sim_ref,
                        policy=self.shadow_policies[alt_id],
                        policy_id=alt_id,
                        first_action=copy.deepcopy(alt_action),
                        regime=regime,
                        chosen_policy_id=chosen_id,
                        completed_at_creation=state.completed_count,
                    )
                    self.in_flight_forks.append(fork)

                    if (
                        self.enable_full_trajectory_branches
                        and self.full_trajectory_forks_used < self.full_trajectory_budget
                    ):
                        self.full_trajectory_forks_used += 1
                        chosen_shadow_action = copy.deepcopy(real_action)
                        chosen_result = run_bounded_rollout(
                            self.sim_ref,
                            policy=self.shadow_policies[chosen_id],
                            policy_id=chosen_id,
                            first_action=chosen_shadow_action,
                        )
                        alt_result = run_bounded_rollout(
                            self.sim_ref,
                            policy=self.shadow_policies[alt_id],
                            policy_id=alt_id,
                            first_action=copy.deepcopy(alt_action),
                        )
                        self.full_trajectory_results.append({
                            "step": step,
                            "regime": regime,
                            "chosen_policy_id": chosen_id,
                            "alt_policy_id": alt_id,
                            "chosen_rollout": chosen_result,
                            "alt_rollout": alt_result,
                            "alt_minus_chosen_completed_count": (
                                alt_result["completed_count"] - chosen_result["completed_count"]
                            ),
                        })

        return real_action

    def _record_disagreement_result(self, fork: LiveFork, *, horizon: int, snapshot: DivergenceSnapshot) -> None:
        self.disagreement_rows.append({
            "step": fork.created_at_step,
            "regime": fork.regime,
            "chosen_policy_id": fork.chosen_policy_id,
            "alt_policy_id": fork.policy_id,
            "horizon": horizon,
            **snapshot.to_dict(),
        })

    def finalize_incomplete_forks(self) -> None:
        """Called once after `Simulator.run()` returns (scenario ended
        before some in-flight forks reached their target step): finalizes
        each using the LAST real state observed, tagged as
        horizon-truncated."""
        last_state = self._last_state
        if last_state is None:
            return
        for fork in self.in_flight_forks:
            fork_state = fork.shell._build_observable_state()
            real_delta = last_state.completed_count - fork.completed_at_creation
            snap = compare_states(last_state, fork_state, fork.completed_in_window, real_delta)
            actual_horizon = last_state.step - fork.created_at_step
            self.disagreement_rows.append({
                "step": fork.created_at_step,
                "regime": fork.regime,
                "chosen_policy_id": fork.chosen_policy_id,
                "alt_policy_id": fork.policy_id,
                "horizon": actual_horizon,
                "horizon_truncated_by_scenario_end": True,
                **snap.to_dict(),
            })
        self.in_flight_forks = []


@dataclass
class ScenarioDiagnosticResult:
    canonical_scenario_id: str
    mechanism_family: str
    split: str
    n_steps: int
    trajectory: pd.DataFrame
    disagreement_rows: pd.DataFrame
    full_trajectory_results: List[Dict[str, Any]]


def run_scenario_diagnostic_from_scenario(
    scenario: PolicySeparationScenario,
    *,
    canonical_scenario_id: str,
    stage1: Stage1Router,
    stage2_selectors: Dict[str, Stage2Selector],
    seed: int = 0,
    mechanism_family: str = "",
    split: str = "",
    enable_full_trajectory_branches: bool = True,
) -> ScenarioDiagnosticResult:
    """Core per-scenario diagnostic driver (design doc SS5): drives an
    already-built `PolicySeparationScenario` through the real `Simulator`
    with the frozen live router wrapped in a read-only shadow-forking
    observer. Split-agnostic at this layer (the TRAIN/VAL/TEST split guard
    lives in `run_scenario_diagnostic`, the row-based entry point real
    TRAIN/VAL sweeps use) so small synthetic fixture scenarios can exercise
    this same code path directly in tests, exactly as
    `hierarchical_router_live_harness_v1.run_live_scenario` is tested."""
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

    shadow_policies = build_native_policy_instances()
    observer = ForkingObserverPolicy(
        sim_ref=sim,
        inner_router=inner_router,
        shadow_policies=shadow_policies,
        enable_full_trajectory_branches=enable_full_trajectory_branches,
    )

    sim.run(observer, workload_tag=canonical_scenario_id, seed=seed)
    observer.finalize_incomplete_forks()

    traj = inner_router.trajectory_df()
    if len(traj) > 0:
        traj["raw_activity_state"] = [
            classify_raw_activity_state(a, b, c)
            for a, b, c in zip(traj["a_active"], traj["b_active_v2"], traj["c_active"])
        ]

    disagreement_df = pd.DataFrame(observer.disagreement_rows)

    return ScenarioDiagnosticResult(
        canonical_scenario_id=canonical_scenario_id,
        mechanism_family=mechanism_family,
        split=split,
        n_steps=len(traj),
        trajectory=traj,
        disagreement_rows=disagreement_df,
        full_trajectory_results=observer.full_trajectory_results,
    )


def run_scenario_diagnostic(
    row: pd.Series,
    *,
    stage1: Stage1Router,
    stage2_selectors: Dict[str, Stage2Selector],
    enable_full_trajectory_branches: bool = True,
) -> ScenarioDiagnosticResult:
    """Row-based entry point for a real TRAIN/VAL MF-PSD scenario row
    (design doc SS3): enforces the TRAIN/VAL-only split guard, rebuilds the
    real scenario, then delegates to `run_scenario_diagnostic_from_scenario`."""
    assert_trainval_only(row["split"])
    assert_no_replication_module_imported()

    canonical_scenario_id = row["canonical_scenario_id"]
    scenario = rebuild_scenario_from_row(row)
    return run_scenario_diagnostic_from_scenario(
        scenario,
        canonical_scenario_id=canonical_scenario_id,
        stage1=stage1,
        stage2_selectors=stage2_selectors,
        seed=int(row["seed"]),
        mechanism_family=row["mechanism_family"],
        split=row["split"],
        enable_full_trajectory_branches=enable_full_trajectory_branches,
    )


# ---------------------------------------------------------------------------
# Aggregation across scenarios (design doc SS6-SS11)
# ---------------------------------------------------------------------------

def aggregate_episode_timescales(trajectories: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Design doc SS6: raw-activity-state episode length distributions, per
    family label and pooled, plus fraction of total active steps in
    episodes shorter than `dwell=20`."""
    per_label_lengths: Dict[str, List[int]] = {
        REGIME_A_ACTIVE_LABEL: [], REGIME_B_ACTIVE_LABEL: [], REGIME_C_ACTIVE_LABEL: [],
        REGIME_NONE_LABEL: [], REGIME_OVERLAP_LABEL: [],
    }
    per_label_episodes: Dict[str, List[pd.DataFrame]] = {k: [] for k in per_label_lengths}
    for sid, traj in trajectories.items():
        if len(traj) == 0:
            continue
        episodes = segment_episodes(traj["raw_activity_state"].tolist())
        for label, grp in episodes.groupby("label"):
            if label in per_label_lengths:
                per_label_lengths[label].extend(grp["length"].tolist())
                per_label_episodes[label].append(grp)

    out: Dict[str, Any] = {}
    for label, lengths in per_label_lengths.items():
        entry = episode_length_distribution(lengths)
        episodes_concat = (
            pd.concat(per_label_episodes[label], ignore_index=True)
            if per_label_episodes[label] else pd.DataFrame(columns=["length"])
        )
        entry["fraction_active_steps_in_episodes_shorter_than_dwell"] = (
            fraction_active_steps_in_short_episodes(episodes_concat) if label != REGIME_NONE_LABEL else None
        )
        out[label] = entry
    return out


def aggregate_dwell_latency(trajectories: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Design doc SS7: dwell-latency reactability classification over the
    FSM-resolved `effective_regime` column, active regimes only."""
    all_rows = []
    for sid, traj in trajectories.items():
        if len(traj) == 0:
            continue
        episodes = segment_episodes(traj["effective_regime"].tolist())
        active_eps = episodes[episodes["label"].isin(ACTIVE_REGIMES)].copy()
        if len(active_eps) == 0:
            continue
        active_eps["canonical_scenario_id"] = sid
        classified = dwell_latency_diagnostic(active_eps)
        all_rows.append(classified)
    if not all_rows:
        return {"episode_count": 0}
    combined = pd.concat(all_rows, ignore_index=True)
    out: Dict[str, Any] = {"episode_count": int(len(combined))}
    for regime, grp in combined.groupby("label"):
        counts = grp["reactability_class"].value_counts().to_dict()
        out[regime] = {
            "episode_count": int(len(grp)),
            "fully_reactable": int(counts.get(FULLY_REACTABLE, 0)),
            "partially_reactable": int(counts.get(PARTIALLY_REACTABLE, 0)),
            "unreachable_under_dwell20": int(counts.get(UNREACHABLE_UNDER_DWELL20, 0)),
        }
    return out


def aggregate_disagreement_rates(
    disagreement_frames: Dict[str, pd.DataFrame], trajectories: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """Design doc SS8: total/identical/different-action steps and
    disagreement fraction per regime, plus conditioned on raw-activity-label
    active/inactive."""
    label_by_regime = {
        REGIME_A: REGIME_A_ACTIVE_LABEL, REGIME_B: REGIME_B_ACTIVE_LABEL, REGIME_C: REGIME_C_ACTIVE_LABEL,
    }
    out: Dict[str, Any] = {}
    for regime in ACTIVE_REGIMES:
        total = identical = different = 0
        cond_active_total = cond_active_disagree = 0
        cond_inactive_total = cond_inactive_disagree = 0
        for sid, df in disagreement_frames.items():
            if len(df) == 0 or "regime" not in df.columns:
                continue
            sub = df[(df["regime"] == regime) & df["disagree"].notna()] if "disagree" in df.columns else df.iloc[0:0]
            if len(sub) == 0:
                continue
            traj = trajectories.get(sid)
            # `traj`'s positional row index does NOT necessarily equal
            # `state.step` -- idle-period fast-forwarding
            # (`Simulator.run()`) can skip step numbers entirely without a
            # `select_action` call, so a step-value lookup (not positional
            # `.iloc`) is required here.
            step_lookup = (
                traj.set_index("step")["raw_activity_state"].to_dict() if traj is not None and len(traj) else {}
            )
            raw_label = label_by_regime[regime]
            for _, r in sub.iterrows():
                total += 1
                if bool(r["disagree"]):
                    different += 1
                else:
                    identical += 1
                if int(r["step"]) in step_lookup:
                    is_raw_active = step_lookup[int(r["step"])] == raw_label
                    if is_raw_active:
                        cond_active_total += 1
                        cond_active_disagree += int(bool(r["disagree"]))
                    else:
                        cond_inactive_total += 1
                        cond_inactive_disagree += int(bool(r["disagree"]))
        out[regime] = {
            "total_evaluated_steps": total,
            "identical_action_steps": identical,
            "different_action_steps": different,
            "disagreement_fraction": (different / total) if total else None,
            "disagreement_fraction_given_raw_active": (
                cond_active_disagree / cond_active_total if cond_active_total else None
            ),
            "disagreement_fraction_given_raw_inactive": (
                cond_inactive_disagree / cond_inactive_total if cond_inactive_total else None
            ),
        }
    return out


def aggregate_causal_importance(disagreement_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Design doc SS8 causal-importance figures: one-step and H-step
    divergence rates and mean absolute per-metric divergence, over
    disagreement steps that were actually forked (i.e. rows carrying a
    `horizon` column)."""
    frames = [df for df in disagreement_frames.values() if len(df) and "horizon" in df.columns]
    if not frames:
        return {"forked_disagreement_steps": 0}
    combined = pd.concat(frames, ignore_index=True)
    out: Dict[str, Any] = {"forked_disagreement_steps": int(combined["step"].nunique())}
    for h, grp in combined.groupby("horizon"):
        out[f"horizon_{h}"] = {
            "n": int(len(grp)),
            "any_nonzero_divergence_rate": float(grp["any_nonzero"].mean()),
            "mean_queue_length_abs_diff": float(grp["queue_length_abs_diff"].mean()),
            "mean_active_count_abs_diff": float(grp["active_count_abs_diff"].mean()),
            "mean_kv_utilization_abs_diff": float(grp["kv_utilization_abs_diff"].mean()),
            "mean_completed_count_abs_diff": float(grp["completed_count_abs_diff"].mean()),
        }
    return out


def find_minority_critical_episodes(
    trajectories: Dict[str, pd.DataFrame], disagreement_frames: Dict[str, pd.DataFrame]
) -> Dict[str, List[Dict[str, Any]]]:
    """Design doc SS9: raw-activity episodes shorter than that family's own
    median episode length which also contain >=1 disagreement step. Median
    is computed from the length distribution alone (outcome-blind); episode
    selection never consults any TEST or Family-B-replication outcome."""
    label_by_regime = {
        REGIME_A: REGIME_A_ACTIVE_LABEL, REGIME_B: REGIME_B_ACTIVE_LABEL, REGIME_C: REGIME_C_ACTIVE_LABEL,
    }
    all_episodes: Dict[str, List[Tuple[str, pd.Series]]] = {r: [] for r in ACTIVE_REGIMES}
    for sid, traj in trajectories.items():
        if len(traj) == 0:
            continue
        episodes = segment_episodes(traj["raw_activity_state"].tolist())
        for regime, label in label_by_regime.items():
            for _, ep in episodes[episodes["label"] == label].iterrows():
                all_episodes[regime].append((sid, ep))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for regime in ACTIVE_REGIMES:
        eps = all_episodes[regime]
        if not eps:
            out[regime] = []
            continue
        median_len = float(np.median([ep["length"] for _, ep in eps]))
        result = []
        for sid, ep in eps:
            if ep["length"] >= median_len:
                continue
            dis = disagreement_frames.get(sid)
            if dis is None or len(dis) == 0 or "disagree" not in dis.columns:
                continue
            in_range = dis[(dis["step"] >= ep["start_idx"]) & (dis["step"] <= ep["end_idx"]) & (dis["regime"] == regime)]
            if in_range["disagree"].any():
                result.append({
                    "canonical_scenario_id": sid,
                    "episode_start_step": int(ep["start_idx"]),
                    "episode_end_step": int(ep["end_idx"]),
                    "episode_length": int(ep["length"]),
                    "family_median_length": median_len,
                })
        out[regime] = result
    return out


def aggregate_ceiling_diagnostic(
    disagreement_rate_summary: Dict[str, Any], full_trajectory_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Design doc SS10/SS11: policy-library ceiling per regime, combining
    the disagreement-rate summary with bounded full-trajectory branch
    outcomes (explicitly labeled bounded-horizon, never a full-scenario
    oracle)."""
    out: Dict[str, Any] = {}
    for regime in ACTIVE_REGIMES:
        base = disagreement_rate_summary.get(regime, {})
        branches = full_trajectory_results.get(regime, [])
        deltas = [b["alt_minus_chosen_completed_count"] for b in branches]
        out[regime] = {
            "fraction_identical_action": (
                1.0 - base["disagreement_fraction"] if base.get("disagreement_fraction") is not None else None
            ),
            "fraction_differing_action": base.get("disagreement_fraction"),
            "bounded_horizon_branches_attempted": len(branches),
            "bounded_horizon_max_alt_minus_chosen_completed_count": max(deltas) if deltas else None,
            "bounded_horizon_mean_alt_minus_chosen_completed_count": (
                float(np.mean(deltas)) if deltas else None
            ),
        }
    return out

"""Hierarchical Regime Router v1 -- LIVE closed-loop per-step evaluation
harness.

HARNESS DESIGN + IMPLEMENTATION + VALIDATION ONLY. Does not modify, retrain,
or re-threshold anything frozen by `hierarchical_regime_router_v1.py` /
`hierarchical_stage2_selectors_v1.py` / `online_regime_signals_v1.py` /
`configs/hierarchical_regime_router_v1_gates.json`. Computes no scientific
TEST verdict -- see docs/audits/hierarchical_router_live_harness_validation_v1_20260818.md
for the full audit this module implements.

Why this module exists
-----------------------
`hierarchical_router_evaluation_v1.py`'s own docstring documents its
approximation exactly: "a scenario's end-to-end outcome is approximated by
the MAJORITY effective regime over its per-step online telemetry... an
offline scenario-level approximation." The first held-out TEST evaluation
(`docs/audits/hierarchical_regime_router_v1_20260818.md`) traced its
`HIERARCHICAL_ROUTER_NO_GO` verdict to exactly that approximation point:
`hierarchical_router_evaluation_v1.scenario_regime_from_telemetry` computes
per-step effective regimes correctly, then collapses an entire scenario's
outcome to ONE majority-vote regime, dispatched to ONE precomputed
whole-scenario ANWG column -- so any regime that is only a MINORITY of a
scenario's steps (KV_MEMORY_PRESSURE: 8-25% of steps, per that audit) is
mechanically un-electable no matter how well Stage-1/Stage-2 perform at the
steps where it genuinely is active.

This module replaces that dispatch with a real per-step closed loop:
`ObservableState -> Stage-1 -> dwell/fallback FSM -> Stage-2 -> one native
policy's real select_action(state) -> Action -> Simulator applies it ->
next ObservableState`, by wrapping the six frozen native policies inside
one `BasePolicy` (`LiveHierarchicalRouterPolicy`) that the *existing,
unmodified* `Simulator.run()` loop drives exactly like it drives any other
policy (`simulator.py` SS3: `action = policy.select_action(state)` every
step, `self._apply_action(action)` immediately after). No Simulator change
is needed or made -- the closed loop falls directly out of `Simulator.run`
already calling `select_action` fresh every step and applying whatever
policy the wrapper delegates to that step.

Explicit regression guard (design doc S5): this module NEVER imports
`hierarchical_router_evaluation_v1` (where the majority-vote helper
`scenario_regime_from_telemetry` lives) and never computes `np.argmax` over
a `np.unique(..., return_counts=True)` regime histogram anywhere in its own
routing path. `tests/test_hierarchical_router_live_harness_v1.py` asserts
both of these structurally (source-text guard), not just behaviorally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.action import Action
from ..core.metrics import RunMetrics
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from ..policies.kv_constrained_online import KVConstrainedOnlinePolicy
from ..policies.least_laxity_first import LeastLaxityFirstPolicy
from ..policies.prefill_control_variants import (
    DEFAULT_CHUNK_SMALL,
    UNLIMITED_PREFILL_CHUNK,
    GreedyArrivalPrefillControlPolicy,
)
from ..policies.weighted_fair_share import WeightedFairSharePolicy
from ..selector.hierarchical_stage2_selectors_v1 import Stage2Selector
from ..selector.multifamily_contextual_selector_v1 import FEATURE_COLUMNS
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from .hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    DWELL_MINIMUM_STEPS,
    FALLBACK_POLICY,
    FALLBACK_REGIMES,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_CLASSES,
    STAGE1_INPUT_COLUMNS,
    STAGE2_CANDIDATES,
    DwellDiagnostics,
    Stage1Router,
    count_dwell_violations,
)
from .online_regime_signals_v1 import compute_activity_labels, compute_regime_signals
from .schema import PolicySeparationScenario

SCHEMA_VERSION = "hierarchical_router_live_harness_v1.0.0"

ROOT = Path(__file__).resolve().parents[3]
MF_PSD_SCENARIOS_CSV = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"

#: The Regime-B mechanism (design doc SS G) is NOT distinguished by any
#: Action.admit difference -- both `full_prefill` and `chunked_prefill_small`
#: use the identical arrival-ordered `GreedyArrivalPrefillControlPolicy`
#: admission logic (prefill_control_variants.py). Their entire mechanism
#: difference is a ServiceModel-level execution-budget config
#: (`max_prefill_chunk_tokens`), fixed at Simulator construction and
#: therefore NOT switchable mid-run by any Action verb the two policies'
#: own `select_action` returns.
#:
#: This harness reuses `Action.prefill_chunk_override` -- an EXISTING,
#: already-frozen, already-authorized per-step verb (added for
#: `composition.prefill_control_policy.PrefillControlChildPolicy`, see
#: `core/action.py`'s own docstring and `gpu.py` SS `step`) that overrides
#: `ServiceModel.max_prefill_chunk_tokens` for one GPU on one step only,
#: without ever mutating the shared frozen `ServiceModel` object. This is
#: the "existing canonical adapter" design doc S4 requires before any
#: cross-mechanism action translation -- nothing new is invented here, the
#: harness only decides WHEN to attach it (Regime B routing steps) and
#: WHICH chunk value (design doc SS G's own frozen native-pair constants,
#: identical to what `unified_utility_matrix.py::_build_policy`'s
#: `sm_override` already uses for these exact two policies).
_PREFILL_CHUNK_BY_POLICY: Dict[str, int] = {
    "full_prefill": UNLIMITED_PREFILL_CHUNK,
    "chunked_prefill_small": DEFAULT_CHUNK_SMALL,
}

#: SS3 policy-state audit (see module docstring / validation doc SS3):
#: every one of the six frozen native policies is a pure function of its
#: constructor hyperparameters (fixed at __init__, never mutated) and the
#: `ObservableState` argument passed to `select_action` each call -- none
#: reads or writes any other instance attribute across calls. Verified by
#: direct source inspection of all six policy classes (see validation doc
#: SS3 table) and enforced here by a single-instantiate-per-scenario-run
#: policy (no meaningful "internal state" exists to preserve or reset, so
#: single-instantiate and always-fresh-instantiate are behaviorally
#: identical; single-instantiate is chosen only to avoid rebuilding
#: `GreedyArrivalPrefillControlPolicy` objects every step).
def build_native_policy_instances() -> Dict[str, BasePolicy]:
    return {
        "estimated_service_time_first": EstimatedServiceTimeFirstPolicy(),
        "weighted_fair_share": WeightedFairSharePolicy(),
        "least_laxity_first": LeastLaxityFirstPolicy(),
        "kv_constrained_online": KVConstrainedOnlinePolicy(),
        "full_prefill": GreedyArrivalPrefillControlPolicy(),
        "chunked_prefill_small": GreedyArrivalPrefillControlPolicy(),
    }


def _apply_prefill_chunk_override(action: Action, state: ObservableState, policy_id: str) -> None:
    chunk = _PREFILL_CHUNK_BY_POLICY[policy_id]
    action.prefill_chunk_override = {g.gpu_id: chunk for g in state.gpu_states}


def stage1_input_row(signals) -> pd.DataFrame:
    """One-row DataFrame in the exact frozen column order Stage1Router
    requires (design doc SS C, `STAGE1_INPUT_COLUMNS`)."""
    row = {c: getattr(signals, c) for c in STAGE1_INPUT_COLUMNS}
    return pd.DataFrame([row])[list(STAGE1_INPUT_COLUMNS)]


# ---------------------------------------------------------------------------
# Stage-2 feature rows -- SS9 (validation doc) temporal-leakage-safe sourcing
# ---------------------------------------------------------------------------
#
# Stage2Selector (hierarchical_stage2_selectors_v1.py) was frozen fitting on
# `multifamily_contextual_selector_v1.FEATURE_COLUMNS` -- 33 SCENARIO-LEVEL
# columns (`feat_A__*`/`feat_B__*`/`feat_C__*`, `mf_psd_schema_v1.json`
# `learnable_feature_allowlist`). These are frozen scenario-GENERATION
# parameters (target_utilization, tenant_weight_skew, bulk_pressure, ...),
# fixed at scenario construction (t=0) and constant for the scenario's
# entire trajectory -- NOT a function of any later simulator state, so
# reading them at any step (not just once per scenario) introduces no new
# temporal leakage versus the frozen offline evaluation, which already read
# the identical values (`hierarchical_router_evaluation_v1.load_scenario_level_dataset`
# joins them in once per scenario row). Two sourcing paths:
#
#  1. EXACT -- for any scenario whose `canonical_scenario_id` is already a
#     row in the frozen `mf_psd_scenarios_v1.csv` (real TRAIN/VAL/TEST MF-PSD
#     scenarios), the real, already-audited feature row is looked up
#     directly. Zero approximation.
#  2. BEST-EFFORT -- for a scenario NOT in that table (e.g. a freshly-built
#     fixture or one of the frozen SS P blended microcases), the row is
#     derived from `scenario.params` (which the frozen `PolicySeparationScenario`
#     docstring itself defines as "the exact keyword arguments the template
#     function was called with") plus `scenario.stress_control_relationship`,
#     restricted to keys that are a byte-exact name match to a
#     `feat_<FAMILY>__<key>` column suffix. Every other FEATURE_COLUMNS entry
#     is left NaN, which is not a fabrication: `build_X` (imported, not
#     reimplemented) already has an explicit, frozen missing-value contract
#     for exactly this shape of gap (numeric NaN -> 0.0 + `__missing`
#     indicator; categorical NaN -> `"__NONE__"` category) -- the same
#     contract every real cross-family training row already relies on
#     (e.g. a real Family-A row has NaN for every feat_B__/feat_C__ column).
#     Fields this cannot honestly derive (request-trace aggregates like
#     `feat_B__hog_prompt_median`) are left NaN rather than invented.

_FAMILY_PREFIX = {
    REGIME_A: "feat_A__",
    REGIME_B: "feat_B__",
    REGIME_C: "feat_C__",
}


def _mf_psd_scenarios_df() -> pd.DataFrame:
    if not hasattr(_mf_psd_scenarios_df, "_cache"):
        _mf_psd_scenarios_df._cache = pd.read_csv(MF_PSD_SCENARIOS_CSV)  # type: ignore[attr-defined]
    return _mf_psd_scenarios_df._cache  # type: ignore[attr-defined]


def feature_row_for_canonical_scenario_id(canonical_scenario_id: str) -> Optional[pd.DataFrame]:
    """EXACT sourcing path: look up a real MF-PSD scenario's own frozen
    feature row by id. Returns None if not present (caller falls back to
    best-effort or to no Stage-2 call at all)."""
    df = _mf_psd_scenarios_df()
    hit = df[df["canonical_scenario_id"] == canonical_scenario_id]
    if len(hit) == 0:
        return None
    return hit[list(FEATURE_COLUMNS)].head(1).reset_index(drop=True)


def feature_row_best_effort(scenario: PolicySeparationScenario, regime: str) -> pd.DataFrame:
    """BEST-EFFORT sourcing path (see module-level note above) for a
    scenario that has no row in `mf_psd_scenarios_v1.csv`."""
    prefix = _FAMILY_PREFIX[regime]
    row: Dict[str, Any] = {c: np.nan for c in FEATURE_COLUMNS}
    for key, value in scenario.params.items():
        col = f"{prefix}{key}"
        if col in row:
            row[col] = value
    stress_col = f"{prefix}stress_control_relationship"
    if stress_col in row and scenario.stress_control_relationship is not None:
        row[stress_col] = scenario.stress_control_relationship
    return pd.DataFrame([row])[list(FEATURE_COLUMNS)]


def resolve_feature_row(
    scenario: PolicySeparationScenario, canonical_scenario_id: str, regime: str
) -> pd.DataFrame:
    exact = feature_row_for_canonical_scenario_id(canonical_scenario_id)
    if exact is not None:
        return exact
    return feature_row_best_effort(scenario, regime)


def build_feature_rows_by_regime(
    scenario: PolicySeparationScenario, canonical_scenario_id: str
) -> Dict[str, pd.DataFrame]:
    """One feature row per active regime the harness might route this
    scenario to -- built ONCE before the run starts (design doc SS9: these
    are t=0 scenario-construction constants, not per-step quantities)."""
    return {
        regime: resolve_feature_row(scenario, canonical_scenario_id, regime)
        for regime in ACTIVE_REGIMES
    }


# ---------------------------------------------------------------------------
# Incremental dwell/fallback FSM -- O(1)-per-step, verified-equivalent
# reimplementation of the frozen batch `apply_dwell_and_fallback`
# ---------------------------------------------------------------------------

class IncrementalDwellFallbackFSM:
    """O(1)-per-step dwell/fallback FSM carrying exactly the same state
    (`effective`, `steps_since_change`) the frozen batch
    `apply_dwell_and_fallback` (design doc SS K) computes by iterating its
    full input list -- this class is that same per-step transition rule,
    called once per real step instead of replayed over the whole history
    every step.

    Why not literally call `apply_dwell_and_fallback(history_so_far)` every
    step (byte-for-byte reuse, zero risk of behavioral drift): that is
    O(steps) per call / O(steps^2) per scenario, which is correct but
    prohibitively slow on real MF-PSD trajectories -- some scenarios have
    >10,000 steps (`docs/audits/hierarchical_regime_router_v1_20260818.md`
    SS7, B+C microcase). This class is instead PROVEN behaviorally
    identical to the batch function by a dedicated equivalence test
    (`tests/test_hierarchical_router_live_harness_v1.py`,
    `test_incremental_fsm_matches_frozen_batch_fsm_on_random_sequences`):
    many random raw-regime sequences, batch vs incremental, effective
    sequence and every diagnostic field asserted identical. The dwell
    minimum is never applied on entry into NONE/OVERLAP (transitions into
    a fallback regime are always instant), matching SS K exactly.

    `dwell_violation_count` is intentionally NOT tracked incrementally:
    the design doc calls it "should be exactly 0 by construction -- a
    correctness check, not a tunable outcome," and by this FSM's own
    construction (an active-regime transition is only ever taken when
    `steps_since_change >= dwell_steps` already held) it is always 0 for
    any sequence this class produces, exactly like the batch function.
    The harness verifies this once per run with a single O(steps) pass
    over the realized trajectory using the frozen `count_dwell_violations`
    check (see `run_live_scenario`), not per-step.
    """

    def __init__(self, dwell_steps: int = DWELL_MINIMUM_STEPS) -> None:
        self.dwell_steps = dwell_steps
        self._effective: Optional[str] = None
        self._prev_effective: Optional[str] = None
        self._steps_since_change = 0
        self._transitions = 0
        self._switches_per_regime: Dict[str, int] = {r: 0 for r in REGIME_CLASSES}
        self._n_steps = 0
        self._n_fallback_steps = 0

    def step(self, raw_regime: str) -> str:
        if raw_regime not in REGIME_CLASSES:
            raise ValueError(f"unknown raw regime {raw_regime!r}; must be one of {REGIME_CLASSES}")
        self._prev_effective = self._effective
        if self._effective is None:
            # First step: identical to apply_dwell_and_fallback's
            # `effective = raw_regimes[0]`.
            self._effective = raw_regime
            self._steps_since_change = 0
        elif raw_regime == self._effective:
            self._steps_since_change += 1
        elif raw_regime in FALLBACK_REGIMES:
            self._effective = raw_regime
            self._steps_since_change = 0
            self._transitions += 1
            self._switches_per_regime[raw_regime] += 1
        elif self._steps_since_change >= self.dwell_steps:
            self._effective = raw_regime
            self._steps_since_change = 0
            self._transitions += 1
            self._switches_per_regime[raw_regime] += 1
        else:
            self._steps_since_change += 1
        self._n_steps += 1
        if self._effective in FALLBACK_REGIMES:
            self._n_fallback_steps += 1
        return self._effective

    @property
    def effective(self) -> str:
        assert self._effective is not None, "step() must be called at least once"
        return self._effective

    @property
    def switched_this_step(self) -> bool:
        return self._prev_effective is not None and self._effective != self._prev_effective

    def diagnostics(self) -> DwellDiagnostics:
        n = max(1, self._n_steps)
        return DwellDiagnostics(
            total_transitions=self._transitions,
            switches_per_regime=dict(self._switches_per_regime),
            switching_rate_per_1000_steps=1000.0 * self._transitions / n,
            dwell_violation_count=0,
            fallback_rate=self._n_fallback_steps / n,
        )


# ---------------------------------------------------------------------------
# Live per-step trajectory log row
# ---------------------------------------------------------------------------

@dataclass
class LiveTrajectoryRow:
    scenario_id: str
    step: int
    sim_time: float
    contention_score_v2: float
    priority_skew: float
    kv_pressure: float
    queue_length: float
    stage1_raw_regime: str
    a_active: bool
    b_active_v2: bool
    c_active: bool
    effective_regime: str
    dwell_switched_this_step: bool
    fallback_active: bool
    stage2_regime: Optional[str]
    selected_policy: str
    admitted_count: int
    admitted_request_ids: List[int]
    prefill_chunk_override_active: bool
    queue_len_after_admission: int
    active_count_after_admission: int
    mean_kv_utilization_after_admission: float

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["admitted_request_ids"] = list(self.admitted_request_ids)
        return out


# ---------------------------------------------------------------------------
# The live per-step router policy
# ---------------------------------------------------------------------------

class LiveHierarchicalRouterPolicy(BasePolicy):
    """Per-step closed-loop hierarchical router (design doc SS2 contract).

    Every `select_action` call: reads the CURRENT `ObservableState` only,
    computes the four frozen Stage-1 inputs, runs the frozen `Stage1Router`,
    advances an `IncrementalDwellFallbackFSM` step (verified bit-identical
    to the frozen batch `apply_dwell_and_fallback` -- see that class's
    docstring and SS8 of the validation doc for why an incremental,
    equivalence-tested reimplementation was chosen over literally replaying
    the batch function every step), resolves exactly one native policy id, calls
    that policy's REAL `select_action(state)`, and returns its `Action`
    unmodified except for the Regime-B `prefill_chunk_override` attachment
    (SS4 of the validation doc). The returned Action is what the Simulator
    actually applies -- causally determining `state` at the next step.

    `forced_expert`, when set to one of the six native policy ids, bypasses
    Stage-1/Stage-2/dwell entirely and always delegates to that one policy
    -- used only by the forced-parent equivalence tests (design doc S6),
    never during real router evaluation.
    """

    name = "hierarchical_router_live_v1"

    def __init__(
        self,
        *,
        scenario_id: str,
        stage1: Stage1Router,
        stage2_selectors: Dict[str, Stage2Selector],
        feature_rows_by_regime: Dict[str, pd.DataFrame],
        dwell_steps: int = DWELL_MINIMUM_STEPS,
        forced_expert: Optional[str] = None,
        record_trajectory: bool = True,
    ) -> None:
        if forced_expert is not None and forced_expert not in build_native_policy_instances():
            raise ValueError(f"unknown forced_expert policy id {forced_expert!r}")
        self.scenario_id = scenario_id
        self.stage1 = stage1
        self.stage2_selectors = stage2_selectors
        self.feature_rows_by_regime = feature_rows_by_regime
        self.dwell_steps = dwell_steps
        self.forced_expert = forced_expert
        self.record_trajectory = record_trajectory

        self.native_policies = build_native_policy_instances()
        self._fsm = IncrementalDwellFallbackFSM(dwell_steps)
        self.trajectory: List[LiveTrajectoryRow] = []
        self.last_dwell_diagnostics = None
        self.stage2_call_count: Dict[str, int] = {r: 0 for r in ACTIVE_REGIMES}
        self.selected_policy_step_counts: Dict[str, int] = {}
        # Stage-2 inputs are t=0 scenario-level feature rows (constant for the
        # whole trajectory). Caching the predicted native policy id per regime
        # is therefore bit-identical to re-predicting every step, and avoids
        # repeating an expensive sklearn call ~30k times on long Family-A runs.
        self._stage2_policy_cache: Dict[str, str] = {}

    def reset(self) -> None:
        self._fsm = IncrementalDwellFallbackFSM(self.dwell_steps)
        self.trajectory = []
        self.last_dwell_diagnostics = None
        self.stage2_call_count = {r: 0 for r in ACTIVE_REGIMES}
        self.selected_policy_step_counts = {}
        self._stage2_policy_cache = {}
        for p in self.native_policies.values():
            p.reset()

    # -- core dispatch -----------------------------------------------------

    def select_action(self, state: ObservableState) -> Action:
        if self.forced_expert is not None:
            return self._select_forced(state)
        return self._select_routed(state)

    def _select_forced(self, state: ObservableState) -> Action:
        policy_id = self.forced_expert
        assert policy_id is not None
        policy = self.native_policies[policy_id]
        action = policy.select_action(state)
        if policy_id in STAGE2_CANDIDATES[REGIME_B]:
            _apply_prefill_chunk_override(action, state, policy_id)
        self.selected_policy_step_counts[policy_id] = self.selected_policy_step_counts.get(policy_id, 0) + 1
        if self.record_trajectory:
            self._log_step(
                state=state,
                signals=None,
                labels=None,
                raw_regime="FORCED",
                effective_regime="FORCED",
                switched=False,
                fallback_active=False,
                stage2_regime=None,
                policy_id=policy_id,
                action=action,
            )
        return action

    def _select_routed(self, state: ObservableState) -> Action:
        signals = compute_regime_signals(state)
        labels = compute_activity_labels(signals)

        raw_regime = str(self.stage1.predict(stage1_input_row(signals))[0])

        # O(1)-per-step FSM, verified bit-identical to the frozen batch
        # `apply_dwell_and_fallback` -- see `IncrementalDwellFallbackFSM`'s
        # docstring and the dedicated equivalence test.
        effective_regime = self._fsm.step(raw_regime)
        switched = self._fsm.switched_this_step
        self.last_dwell_diagnostics = self._fsm.diagnostics()

        stage2_regime: Optional[str] = None
        if effective_regime in ACTIVE_REGIMES:
            selector = self.stage2_selectors.get(effective_regime)
            feat_row = self.feature_rows_by_regime.get(effective_regime)
            if selector is not None and feat_row is not None:
                stage2_regime = effective_regime
                self.stage2_call_count[effective_regime] += 1
                cached = self._stage2_policy_cache.get(effective_regime)
                if cached is not None:
                    policy_id = cached
                else:
                    policy_id = str(selector.predict(feat_row)[0])
                    self._stage2_policy_cache[effective_regime] = policy_id
            else:
                # No trained selector / no feature row reachable for this
                # regime on this scenario -- safe default, same fallback
                # `hierarchical_router_evaluation_v1.baseline_d_anwg` already
                # uses for "no trained selector for this regime" (reused
                # convention, not a new invention).
                policy_id = FALLBACK_POLICY
        else:
            policy_id = FALLBACK_POLICY

        policy = self.native_policies[policy_id]
        action = policy.select_action(state)
        if policy_id in STAGE2_CANDIDATES[REGIME_B]:
            _apply_prefill_chunk_override(action, state, policy_id)

        self.selected_policy_step_counts[policy_id] = self.selected_policy_step_counts.get(policy_id, 0) + 1

        if self.record_trajectory:
            self._log_step(
                state=state,
                signals=signals,
                labels=labels,
                raw_regime=raw_regime,
                effective_regime=effective_regime,
                switched=switched,
                fallback_active=effective_regime in FALLBACK_REGIMES,
                stage2_regime=stage2_regime,
                policy_id=policy_id,
                action=action,
            )
        return action

    # -- logging -------------------------------------------------------

    def _log_step(
        self,
        *,
        state: ObservableState,
        signals,
        labels,
        raw_regime: str,
        effective_regime: str,
        switched: bool,
        fallback_active: bool,
        stage2_regime: Optional[str],
        policy_id: str,
        action: Action,
    ) -> None:
        admitted_ids = [rid for ids in action.admit.values() for rid in ids]
        prefill_override_active = bool(action.prefill_chunk_override)
        # `state.gpu_states` are mutated IN PLACE by every one of the six
        # native policies' `select_action` (via `deterministic_place` or the
        # equivalent manual admit loop in ESTF/LLF) -- reading them here,
        # immediately after delegation, reflects post-admission,
        # pre-decode-advance state. This is computed entirely from the
        # current decision the harness itself just made; it is not a peek
        # at any future simulator state.
        active_after = sum(len(g.active_request_ids) for g in state.gpu_states)
        kv_utils = [
            (g.current_kv_tokens / g.max_kv_tokens) if g.max_kv_tokens else 0.0
            for g in state.gpu_states
        ]
        mean_kv_after = float(np.mean(kv_utils)) if kv_utils else 0.0
        queue_after = max(0, len(state.waiting_queue) - len(admitted_ids))

        row = LiveTrajectoryRow(
            scenario_id=self.scenario_id,
            step=state.step,
            sim_time=state.time,
            contention_score_v2=float(signals.contention_score_v2) if signals is not None else float("nan"),
            priority_skew=float(signals.priority_skew) if signals is not None else float("nan"),
            kv_pressure=float(signals.kv_pressure) if signals is not None else float("nan"),
            queue_length=float(signals.queue_length) if signals is not None else float("nan"),
            stage1_raw_regime=raw_regime,
            a_active=bool(labels.a_active) if labels is not None else False,
            b_active_v2=bool(labels.b_active_v2) if labels is not None else False,
            c_active=bool(labels.c_active) if labels is not None else False,
            effective_regime=effective_regime,
            dwell_switched_this_step=switched,
            fallback_active=fallback_active,
            stage2_regime=stage2_regime,
            selected_policy=policy_id,
            admitted_count=len(admitted_ids),
            admitted_request_ids=admitted_ids,
            prefill_chunk_override_active=prefill_override_active,
            queue_len_after_admission=queue_after,
            active_count_after_admission=active_after,
            mean_kv_utilization_after_admission=mean_kv_after,
        )
        self.trajectory.append(row)

    def trajectory_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.trajectory])


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

@dataclass
class LiveRunResult:
    metrics: RunMetrics
    trajectory: pd.DataFrame
    dwell_diagnostics: Optional[Dict[str, Any]]
    selected_policy_step_counts: Dict[str, int]
    stage2_call_count: Dict[str, int]


def run_live_scenario(
    scenario: PolicySeparationScenario,
    *,
    canonical_scenario_id: str,
    stage1: Stage1Router,
    stage2_selectors: Dict[str, Stage2Selector],
    dwell_steps: int = DWELL_MINIMUM_STEPS,
    forced_expert: Optional[str] = None,
    record_trajectory: bool = True,
    max_steps: Optional[int] = None,
) -> LiveRunResult:
    """Run one scenario through the real `Simulator`, driven end-to-end by
    `LiveHierarchicalRouterPolicy` -- the actual closed loop this module
    exists to provide. `service_model_kwargs` are taken from the scenario
    itself, UNCHANGED (design doc: no scenario/service-model construction
    logic is altered by this harness)."""
    feature_rows = (
        {} if forced_expert is not None else build_feature_rows_by_regime(scenario, canonical_scenario_id)
    )
    router = LiveHierarchicalRouterPolicy(
        scenario_id=canonical_scenario_id,
        stage1=stage1,
        stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows,
        dwell_steps=dwell_steps,
        forced_expert=forced_expert,
        record_trajectory=record_trajectory,
    )
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
        max_steps=max_steps,
    ))
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(router, workload_tag=canonical_scenario_id, seed=scenario.seed)
    diag = router.last_dwell_diagnostics.to_dict() if router.last_dwell_diagnostics is not None else None
    if diag is not None and forced_expert is None and len(router.trajectory) > 0:
        # Independent correctness check (design doc SS K: "should be
        # exactly 0 by construction"), run ONCE as a single O(steps) pass
        # over the realized trajectory -- not per-step (see
        # `IncrementalDwellFallbackFSM`'s docstring for why).
        effective_sequence = [r.effective_regime for r in router.trajectory]
        diag["dwell_violation_count"] = count_dwell_violations(effective_sequence, dwell_steps)
    return LiveRunResult(
        metrics=metrics,
        trajectory=router.trajectory_df(),
        dwell_diagnostics=diag,
        selected_policy_step_counts=dict(router.selected_policy_step_counts),
        stage2_call_count=dict(router.stage2_call_count),
    )


def run_reference_single_policy(
    scenario: PolicySeparationScenario, policy_id: str, *, max_steps: Optional[int] = None
) -> RunMetrics:
    """Reference baseline for forced-parent equivalence (design doc S6):
    runs the named native policy directly through the plain `Simulator`,
    with the SAME `sm_override` merge `unified_utility_matrix.py::_build_policy`
    already uses for these exact policies -- so this is a faithful
    reproduction of the existing non-router evaluation path, not a new
    methodology."""
    sm_override: Dict[str, Any] = {}
    if policy_id in _PREFILL_CHUNK_BY_POLICY:
        sm_override = {
            "max_prefill_chunk_tokens": _PREFILL_CHUNK_BY_POLICY[policy_id],
            "decode_first": False,
        }
    merged_sm = dict(scenario.service_model_kwargs)
    merged_sm.update(sm_override)
    policy = build_native_policy_instances()[policy_id]
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**merged_sm),
        max_steps=max_steps,
    ))
    sim.load_trace(list(scenario.requests))
    return sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)

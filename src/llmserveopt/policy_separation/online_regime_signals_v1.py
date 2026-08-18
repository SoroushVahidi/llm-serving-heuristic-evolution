"""Online regime-signal formulas and telemetry-recording policy wrapper.

FEASIBILITY STUDY ONLY. See
docs/audits/online_regime_signal_feasibility_v1_20260817.md. This module
computes, from `ObservableState` alone (the exact snapshot every
`BasePolicy.select_action` already receives before choosing an action --
`src/llmserveopt/simulator/simulator.py` line ~168-172), candidate
per-step signals for three operational scheduling regimes:

  A. RANKING / FAIRNESS PRESSURE
  B. PREFILL / DECODE CONTENTION
  C. KV / MEMORY PRESSURE

Every pressure primitive (`_prefill_pressure`, `_decode_pressure`,
`_kv_pressure`, `_queue_pressure`, `causal_context_features`) is REUSED,
not reimplemented, from `llmserveopt.policies.composition` -- that module's
own functions are already used live inside real policies'
`select_action()` (`composition/estf_wfs_policies.py`,
`policies/composition.py`'s `ContextualRankEnsemblePolicy`), which is
direct, pre-existing evidence that these quantities are genuinely causal
(computed from state available at decision time, already load-bearing for
real admission/ranking decisions), not retrospective analysis invented for
this study.

Nothing here reads `Request.actual_output_tokens`, `mechanism_family`,
`scenario_id`, or any post-run metric. `TelemetryRecordingPolicy` forwards
its wrapped policy's action unmodified -- recording never changes what any
policy decides (same non-invasive pattern as
`llmserveopt.policies.instrumentation.InstrumentedPolicy`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policies.composition import (
    _decode_pressure,
    _kv_pressure,
    _prefill_pressure,
    _queue_pressure,
    causal_context_features,
)

SCHEMA_VERSION = "online_regime_signals_v1.0.0"

#: Preregistered activity thresholds. Chosen from physical/system semantics
#: or pre-existing project constants -- NOT fit to maximize any diagnostic
#: accuracy metric (task requirement, section 4/13).
#:
#: - PRIORITY_SKEW_THRESHOLD: 1.05 (>5% priority ratio spread) -- any
#:   measurable priority heterogeneity in the waiting queue; a fairness/
#:   ranking mechanism cannot matter when every waiting request has equal
#:   priority (skew == 1.0 exactly, which is Family A's OWN generator's
#:   definition of a "control" -- not-stressed -- scenario, per
#:   `templates_fairness_starvation_v2.py`'s `role = "control" if
#:   tenant_weight_skew == 1.0 else "stress"`; 1.05 gives a small margin
#:   above that exact boundary rather than reusing the boundary itself).
#: - MIN_CONFLICT_QUEUE: 2 -- a ranking choice requires at least 2
#:   candidates to choose between.
#: - CONTENTION_SCORE_THRESHOLD: 0.05 -- both prefill and decode cohorts
#:   must be simultaneously non-trivial (>=5% of capacity each) for their
#:   product to exceed this; physically motivated as "both phases
#:   meaningfully co-occupying the GPU," not accuracy-fit.
#: - KV_PRESSURE_THRESHOLD: 0.82 -- REUSED VERBATIM from
#:   `KVConstrainedOnlinePolicy.target_kv_utilization`'s own default
#:   (`src/llmserveopt/policies/kv_constrained_online.py`), the project's
#:   own pre-existing, already-deployed KV-admission-control threshold --
#:   not invented for this study.
PRIORITY_SKEW_THRESHOLD = 1.05
MIN_CONFLICT_QUEUE = 2
CONTENTION_SCORE_THRESHOLD = 0.05
KV_PRESSURE_THRESHOLD = 0.82

#: CONTENTION_SCORE_V2_THRESHOLD: 0.20 -- for the active-fraction-based
#: alternative contention score (`contention_score_v2`, §F of the audit),
#: which is maximized at 0.5 when prefill/decode phases exactly split the
#: currently-active cohort. 0.20 means "the minority phase is at least 40%
#: as large as the majority phase" -- a meaningfully large minority
#: presence, not a bare non-zero. Chosen from the formula's own [0, 0.5]
#: range structure, not fit to any accuracy outcome; added mid-study after
#: discovering CONTENTION_SCORE_THRESHOLD structurally never fires under
#: `max_active_sequences`-normalization at Family B's small scenario scale
#: (§F/§N) -- a normalization-denominator mismatch, not a retuned
#: threshold on the same formula.
CONTENTION_SCORE_V2_THRESHOLD = 0.20


@dataclass(frozen=True)
class RegimeSignals:
    # A. ranking/fairness
    priority_skew: float
    class_imbalance: float
    queue_length: float
    urgent_deadline_fraction: float
    # B. prefill/decode contention -- capacity-normalized (v1)
    prefill_pressure: float
    decode_pressure: float
    contention_score_product: float
    contention_score_min: float
    contention_score_sum: float
    # B. prefill/decode contention -- active-fraction-normalized (v2, see
    # CONTENTION_SCORE_V2_THRESHOLD docstring for why this was added)
    prefill_fraction_of_active: float
    decode_fraction_of_active: float
    contention_score_v2: float
    # C. KV/memory pressure
    kv_pressure: float
    queue_pressure: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "priority_skew": self.priority_skew,
            "class_imbalance": self.class_imbalance,
            "queue_length": self.queue_length,
            "urgent_deadline_fraction": self.urgent_deadline_fraction,
            "prefill_pressure": self.prefill_pressure,
            "decode_pressure": self.decode_pressure,
            "contention_score_product": self.contention_score_product,
            "contention_score_min": self.contention_score_min,
            "contention_score_sum": self.contention_score_sum,
            "prefill_fraction_of_active": self.prefill_fraction_of_active,
            "decode_fraction_of_active": self.decode_fraction_of_active,
            "contention_score_v2": self.contention_score_v2,
            "kv_pressure": self.kv_pressure,
            "queue_pressure": self.queue_pressure,
        }


def compute_regime_signals(state: ObservableState) -> RegimeSignals:
    """Pure function of `ObservableState` alone -- every value here is
    computable from information already available to a policy at the
    moment it must choose `select_action(state)`."""
    causal = causal_context_features(state)
    prefill_p = _prefill_pressure(state)
    decode_p = _decode_pressure(state)

    prefilling_total = sum(g.prefilling_count for g in state.gpu_states)
    decoding_total = sum(g.decoding_count for g in state.gpu_states)
    active_total = max(1, prefilling_total + decoding_total)
    prefill_frac = prefilling_total / active_total
    decode_frac = decoding_total / active_total

    return RegimeSignals(
        priority_skew=causal["priority_skew"],
        class_imbalance=causal["class_imbalance"],
        queue_length=causal["queue_length"],
        urgent_deadline_fraction=causal["urgent_deadline_fraction"],
        prefill_pressure=prefill_p,
        decode_pressure=decode_p,
        contention_score_product=prefill_p * decode_p,
        contention_score_min=min(prefill_p, decode_p),
        contention_score_sum=prefill_p + decode_p,
        prefill_fraction_of_active=prefill_frac,
        decode_fraction_of_active=decode_frac,
        contention_score_v2=min(prefill_frac, decode_frac),
        kv_pressure=_kv_pressure(state),
        queue_pressure=_queue_pressure(state),
    )


@dataclass(frozen=True)
class ActivityLabels:
    a_active: bool
    b_active: bool
    b_active_v2: bool
    c_active: bool

    def to_dict(self) -> Dict[str, bool]:
        return {
            "a_active": self.a_active,
            "b_active": self.b_active,
            "b_active_v2": self.b_active_v2,
            "c_active": self.c_active,
        }


def compute_activity_labels(signals: RegimeSignals) -> ActivityLabels:
    """Three (four, counting both B contention-score variants) independent
    binary activity labels (may overlap) -- see module docstring for why
    each threshold was chosen the way it was. `b_active` uses the
    capacity-normalized contention_score_product; `b_active_v2` uses the
    active-fraction-normalized contention_score_v2 (added after `b_active`
    was found to structurally never fire at Family B's scenario scale --
    both are reported, neither is silently dropped)."""
    a_active = signals.priority_skew > PRIORITY_SKEW_THRESHOLD and signals.queue_length >= MIN_CONFLICT_QUEUE
    b_active = signals.contention_score_product > CONTENTION_SCORE_THRESHOLD
    b_active_v2 = signals.contention_score_v2 > CONTENTION_SCORE_V2_THRESHOLD
    c_active = signals.kv_pressure > KV_PRESSURE_THRESHOLD
    return ActivityLabels(a_active=a_active, b_active=b_active, b_active_v2=b_active_v2, c_active=c_active)


@dataclass
class TelemetryRow:
    # Audit-only identity metadata -- never a learnable feature.
    canonical_scenario_id: str
    mechanism_family: str
    step: int
    sim_time: float
    # Learnable online signals.
    signals: RegimeSignals
    labels: ActivityLabels

    def to_flat_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "canonical_scenario_id": self.canonical_scenario_id,
            "mechanism_family": self.mechanism_family,
            "step": self.step,
            "sim_time": self.sim_time,
        }
        out.update(self.signals.to_dict())
        out.update(self.labels.to_dict())
        return out


TELEMETRY_IDENTITY_COLUMNS = ("canonical_scenario_id", "mechanism_family", "step", "sim_time")
TELEMETRY_LEARNABLE_SIGNAL_COLUMNS = (
    "priority_skew",
    "class_imbalance",
    "queue_length",
    "urgent_deadline_fraction",
    "prefill_pressure",
    "decode_pressure",
    "contention_score_product",
    "contention_score_min",
    "contention_score_sum",
    "prefill_fraction_of_active",
    "decode_fraction_of_active",
    "contention_score_v2",
    "kv_pressure",
    "queue_pressure",
)
TELEMETRY_LABEL_COLUMNS = ("a_active", "b_active", "b_active_v2", "c_active")
TELEMETRY_COLUMNS = (
    tuple(TELEMETRY_IDENTITY_COLUMNS) + TELEMETRY_LEARNABLE_SIGNAL_COLUMNS + TELEMETRY_LABEL_COLUMNS
)


class TelemetryRecordingPolicy(BasePolicy):
    """Wraps any `BasePolicy`; records one `TelemetryRow` per step from the
    `ObservableState` the wrapped policy is about to see, THEN delegates.
    The wrapped policy's returned `Action` is forwarded completely
    unmodified -- recording cannot change what is decided (same contract
    as `llmserveopt.policies.instrumentation.InstrumentedPolicy`)."""

    def __init__(
        self,
        wrapped: BasePolicy,
        canonical_scenario_id: str,
        mechanism_family: str,
        *,
        sample_stride_steps: int = 20,
    ) -> None:
        """`sample_stride_steps`: record telemetry at least once every this
        many raw simulator steps (recording economy for scenarios with very
        long trajectories, e.g. Family A's `max_active_sequences=1` design
        -- see build_online_regime_telemetry_v1.py's audit §D). Every
        activity-label TRANSITION (any of a_active/b_active/c_active
        flipping) is recorded exactly regardless of stride -- the study's
        within-trajectory activation/deactivation timing (audit §K) must
        never be approximated by the sampling cadence, only the steady-
        state stretches between transitions are thinned."""
        self.wrapped = wrapped
        self.canonical_scenario_id = canonical_scenario_id
        self.mechanism_family = mechanism_family
        self.sample_stride_steps = sample_stride_steps
        self.name = f"telemetry:{wrapped.name}"
        self.rows: List[TelemetryRow] = []
        self.n_steps_observed = 0
        self._last_labels: Optional[tuple] = None
        self._steps_since_recorded = sample_stride_steps  # force recording of step 0

    def reset(self) -> None:
        self.wrapped.reset()
        self.rows.clear()
        self.n_steps_observed = 0
        self._last_labels = None
        self._steps_since_recorded = self.sample_stride_steps

    def select_action(self, state: ObservableState) -> Action:
        self.n_steps_observed += 1
        signals = compute_regime_signals(state)
        labels = compute_activity_labels(signals)
        label_tuple = (labels.a_active, labels.b_active, labels.b_active_v2, labels.c_active)
        self._steps_since_recorded += 1
        is_transition = label_tuple != self._last_labels
        due_by_cadence = self._steps_since_recorded >= self.sample_stride_steps
        if is_transition or due_by_cadence:
            self._last_labels = label_tuple
            self._steps_since_recorded = 0
            self.rows.append(
                TelemetryRow(
                    canonical_scenario_id=self.canonical_scenario_id,
                    mechanism_family=self.mechanism_family,
                    step=state.step,
                    sim_time=state.time,
                    signals=signals,
                    labels=labels,
                )
            )
        return self.wrapped.select_action(state)

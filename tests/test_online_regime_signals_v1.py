"""Focused tests for the online regime-signal feasibility study.

See docs/audits/online_regime_signal_feasibility_v1_20260817.md and
src/llmserveopt/policy_separation/online_regime_signals_v1.py. Covers
temporal causality, formula correctness, activity-label formulas, overlap
counting, deterministic replay, exact row alignment, and frozen-source
immutability -- per the task's section 15 requirements.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.core.action import Action  # noqa: E402
from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState  # noqa: E402
from llmserveopt.policies.base import BasePolicy  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policy_separation.online_regime_signals_v1 import (  # noqa: E402
    CONTENTION_SCORE_THRESHOLD,
    CONTENTION_SCORE_V2_THRESHOLD,
    KV_PRESSURE_THRESHOLD,
    MIN_CONFLICT_QUEUE,
    PRIORITY_SKEW_THRESHOLD,
    TELEMETRY_COLUMNS,
    TELEMETRY_IDENTITY_COLUMNS,
    TELEMETRY_LABEL_COLUMNS,
    TELEMETRY_LEARNABLE_SIGNAL_COLUMNS,
    ActivityLabels,
    RegimeSignals,
    TelemetryRecordingPolicy,
    compute_activity_labels,
    compute_regime_signals,
)

MF_PSD_LONG = REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_long_v1.csv"
TELEMETRY_DIR = REPO_ROOT / "experiments" / "online_regime_signal_feasibility_v1"
TELEMETRY_CSV = TELEMETRY_DIR / "online_regime_telemetry_v1.csv"
MANIFEST_PATH = TELEMETRY_DIR / "online_regime_telemetry_v1_manifest.json"


def _gpu(gpu_id=0, max_active=64, max_batch=64, max_kv=1000, current_kv=0, active_ids=None,
         prefilling=0, decoding=0):
    return ObservableGPUState(
        gpu_id=gpu_id,
        max_active_sequences=max_active,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=active_ids or [],
        active_requests_info=[],
        current_kv_tokens=current_kv,
        tokens_decoded_per_request={},
        prefilling_count=prefilling,
        decoding_count=decoding,
    )


def _req(rid, arrival=0.0, prompt=100, pred_out=10, deadline=100.0, priority=1.0, class_id="c"):
    return ObservableRequest(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=pred_out, slo_deadline=deadline, priority=priority, class_id=class_id,
    )


def _state(waiting=None, gpus=None, time=0.0, step=0):
    return ObservableState(
        time=time, waiting_queue=waiting or [], gpu_states=gpus or [_gpu()], completed_count=0, step=step,
    )


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


def test_kv_pressure_formula_and_threshold_match_project_precedent():
    gpu = _gpu(max_kv=1000, current_kv=850)
    state = _state(gpus=[gpu])
    signals = compute_regime_signals(state)
    assert signals.kv_pressure == pytest.approx(0.85)
    labels = compute_activity_labels(signals)
    assert labels.c_active is True
    assert KV_PRESSURE_THRESHOLD == 0.82  # reused verbatim from KVConstrainedOnlinePolicy default


def test_contention_score_v2_zero_when_prefill_modeling_disabled():
    """Families A/C run with enable_prefill_modeling=False -> prefilling_count
    and decoding_count are always 0 on ObservableGPUState -- contention_score_v2
    must be a clean, mechanistic zero there, not a near-miss."""
    gpu = _gpu(prefilling=0, decoding=0)
    state = _state(gpus=[gpu])
    signals = compute_regime_signals(state)
    assert signals.prefill_fraction_of_active == 0.0
    assert signals.decode_fraction_of_active == 0.0
    assert signals.contention_score_v2 == 0.0


def test_contention_score_v2_maximized_at_balanced_split():
    gpu = _gpu(prefilling=5, decoding=5)
    state = _state(gpus=[gpu])
    signals = compute_regime_signals(state)
    assert signals.prefill_fraction_of_active == pytest.approx(0.5)
    assert signals.decode_fraction_of_active == pytest.approx(0.5)
    assert signals.contention_score_v2 == pytest.approx(0.5)


def test_contention_score_product_never_fires_at_small_scale_regression():
    """Regression guard for the audit's central finding: capacity-normalized
    contention_score_product structurally cannot cross CONTENTION_SCORE_THRESHOLD
    when max_active_sequences is generous relative to actual concurrent
    requests (Family B's own scale). If this ever flips, re-examine the
    audit's §F finding."""
    gpu = _gpu(max_active=512, prefilling=10, decoding=10)
    state = _state(gpus=[gpu])
    signals = compute_regime_signals(state)
    assert signals.contention_score_product < CONTENTION_SCORE_THRESHOLD


def test_priority_skew_and_min_conflict_queue_gate_a_active():
    waiting = [_req(0, priority=1.0), _req(1, priority=2.0)]
    state = _state(waiting=waiting)
    signals = compute_regime_signals(state)
    assert signals.priority_skew == pytest.approx(2.0)
    labels = compute_activity_labels(signals)
    assert labels.a_active is True

    # Single waiting request: no conflict possible even with skew.
    state_one = _state(waiting=[_req(0, priority=5.0)])
    signals_one = compute_regime_signals(state_one)
    labels_one = compute_activity_labels(signals_one)
    assert signals_one.queue_length < MIN_CONFLICT_QUEUE
    assert labels_one.a_active is False

    # Equal priorities: no conflict regardless of queue size.
    waiting_equal = [_req(0, priority=1.0), _req(1, priority=1.0)]
    state_equal = _state(waiting=waiting_equal)
    labels_equal = compute_activity_labels(compute_regime_signals(state_equal))
    assert labels_equal.a_active is False


# ---------------------------------------------------------------------------
# No leakage: labels never read post-outcome / hidden fields
# ---------------------------------------------------------------------------


def test_activity_label_computation_never_reads_family_or_scenario_identity():
    """compute_activity_labels/compute_regime_signals take only ObservableState
    (or RegimeSignals derived from it) -- structurally cannot access
    mechanism_family or scenario_id, which are never parameters."""
    import inspect

    sig_signals = inspect.signature(compute_regime_signals)
    sig_labels = inspect.signature(compute_activity_labels)
    assert list(sig_signals.parameters.keys()) == ["state"]
    assert list(sig_labels.parameters.keys()) == ["signals"]


def test_observable_request_has_no_actual_output_tokens_field():
    """Structural leakage guard: ObservableRequest (what regime signals are
    computed from) must not expose actual_output_tokens (ground truth,
    policy-hidden per Request's own docstring)."""
    assert not hasattr(_req(0), "actual_output_tokens")


# ---------------------------------------------------------------------------
# Temporal causality: TelemetryRecordingPolicy records BEFORE delegating,
# and only ever sees the state the simulator already builds pre-decision
# ---------------------------------------------------------------------------


class _ScriptedPolicy(BasePolicy):
    """Deterministic no-op policy for testing the wrapper in isolation."""

    name = "scripted_noop"

    def select_action(self, state: ObservableState) -> Action:
        return Action(admit={g.gpu_id: [] for g in state.gpu_states})


def test_telemetry_recording_policy_forwards_action_unmodified():
    wrapped = FIFOPolicy()
    telem = TelemetryRecordingPolicy(wrapped, "test::scenario", "FAMILY_A_FAIRNESS_STARVATION_V2")
    waiting = [_req(0), _req(1)]
    state = _state(waiting=waiting, gpus=[_gpu(max_active=5, max_kv=10000)], step=0)

    action_direct = FIFOPolicy().select_action(_state(waiting=waiting, gpus=[_gpu(max_active=5, max_kv=10000)], step=0))
    action_wrapped = telem.select_action(state)
    assert action_direct.admit == action_wrapped.admit


def test_telemetry_recording_policy_records_transitions_exactly():
    telem = TelemetryRecordingPolicy(_ScriptedPolicy(), "test::scenario", "FAMILY_C_KV_PRESSURE_V2",
                                      sample_stride_steps=1000)
    # Step 0: low KV -> c_active False.
    telem.select_action(_state(gpus=[_gpu(max_kv=1000, current_kv=100)], step=0))
    # Step 1: still False -> should NOT be recorded again by cadence (stride huge).
    telem.select_action(_state(gpus=[_gpu(max_kv=1000, current_kv=200)], step=1))
    # Step 2: crosses threshold -> True, must be recorded as a transition.
    telem.select_action(_state(gpus=[_gpu(max_kv=1000, current_kv=900)], step=2))
    # Step 3: still True -> not recorded again by cadence.
    telem.select_action(_state(gpus=[_gpu(max_kv=1000, current_kv=950)], step=3))
    # Step 4: drops back to False -> transition, must be recorded.
    telem.select_action(_state(gpus=[_gpu(max_kv=1000, current_kv=100)], step=4))

    recorded_steps = [r.step for r in telem.rows]
    assert recorded_steps[0] == 0  # forced first record
    assert 2 in recorded_steps  # True transition
    assert 4 in recorded_steps  # False transition
    assert telem.n_steps_observed == 5


def test_telemetry_row_never_touches_time_after_its_own_step():
    """Each recorded row's fields are computed strictly from the state
    passed to that single select_action call -- no row can be influenced
    by a later call. Verified by construction (compute_regime_signals
    takes one state snapshot, no mutable shared future-looking buffer)."""
    telem = TelemetryRecordingPolicy(_ScriptedPolicy(), "test::scenario", "FAMILY_A_FAIRNESS_STARVATION_V2",
                                      sample_stride_steps=1)
    telem.select_action(_state(waiting=[_req(0, priority=1.0)], step=0))
    first_signature = telem.rows[0].signals.to_dict()
    # A wildly different later state must not retroactively change row 0.
    telem.select_action(_state(waiting=[_req(0, priority=1.0), _req(1, priority=99.0)], step=1))
    assert telem.rows[0].signals.to_dict() == first_signature


# ---------------------------------------------------------------------------
# Built-artifact tests (skipped if telemetry hasn't been built locally)
# ---------------------------------------------------------------------------

pytestmark_skip = pytest.mark.skipif(
    not TELEMETRY_CSV.exists() or not MF_PSD_LONG.exists(),
    reason="online regime telemetry artifact or MF-PSD source not present locally",
)


@pytestmark_skip
def test_telemetry_columns_and_no_family_id_in_learnable_signals():
    with open(TELEMETRY_CSV, newline="") as f:
        header = next(csv.reader(f))
    assert header == list(TELEMETRY_COLUMNS)
    for col in TELEMETRY_LEARNABLE_SIGNAL_COLUMNS:
        assert col not in TELEMETRY_IDENTITY_COLUMNS
        assert "family" not in col.lower()
        assert "scenario" not in col.lower()


@pytestmark_skip
def test_telemetry_row_alignment_covers_all_176_scenarios():
    with open(TELEMETRY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    scenario_ids = {r["canonical_scenario_id"] for r in rows}
    with open(MF_PSD_LONG, newline="") as f:
        mf_psd_ids = {r["canonical_scenario_id"] for r in csv.DictReader(f)}
    assert scenario_ids == mf_psd_ids


@pytestmark_skip
def test_telemetry_first_recorded_step_per_scenario_has_empty_or_near_empty_queue():
    """Coarse leakage spot-check: at the first recorded step, no future
    arrival should already be reflected as a large waiting queue."""
    with open(TELEMETRY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    first_by_scenario: dict = {}
    for r in rows:
        sid = r["canonical_scenario_id"]
        step = int(r["step"])
        if sid not in first_by_scenario or step < first_by_scenario[sid][0]:
            first_by_scenario[sid] = (step, float(r["queue_length"]))
    assert all(q == 0.0 for _, q in first_by_scenario.values())


@pytestmark_skip
def test_mf_psd_and_family_c_reconstruction_sources_not_mutated_by_telemetry_build():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    for name, meta in manifest["sources_read"].items():
        path = REPO_ROOT / meta["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == meta["sha256"], f"{name} was mutated"


@pytestmark_skip
def test_no_cross_family_false_positives_for_any_activity_label():
    """Central empirical finding, guarded against silent regression: in
    this frozen replay, a_active never fires outside Family A, b_active_v2
    never fires outside Family B, c_active never fires outside Family C."""
    with open(TELEMETRY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        fam = r["mechanism_family"]
        if r["a_active"] == "True":
            assert fam == "FAMILY_A_FAIRNESS_STARVATION_V2"
        if r["b_active_v2"] == "True":
            assert fam == "FAMILY_B_PREFILL_DECODE_V2"
        if r["c_active"] == "True":
            assert fam == "FAMILY_C_KV_PRESSURE_V2"

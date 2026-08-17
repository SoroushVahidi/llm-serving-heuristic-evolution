"""Observable features for the KV-aware composition falsification v1.

No generator-label leakage. Only online-observable quantities from
ObservableRequest / ObservableState / ObservableGPUState that a real
scheduler could compute -- never scenario_id, seed, bulk_pressure,
urgent_arrival_phase, urgent_tightness, or parent-policy identity.

See docs/design/KV_COMPOSITION_FALSIFICATION_V1.md sections 1D-1E.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ..core.types import ObservableState, Request
from ..policies.kv_constrained_online import KVConstrainedOnlinePolicy
from ..policies.policy_library_v2_helpers import laxity_seconds
from ..policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA

# ---------------------------------------------------------------------------
# Explicit denylist
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURE_KEYS = frozenset({
    "scenario_id",
    "seed",
    "bulk_pressure",
    "urgent_arrival_phase",
    "urgent_tightness",
    "intended_winner",
    "intended",
    "winner",
    "generator_family",
    "generator_version",
    "token_length_source",
    "burstgpt_path",
    "allow_synthetic_tokens",
    "pair_id",
    "n_bulk",
    "n_urgent",
    "bulk_prompt_source",
    "bulk_output_source",
    "urgent_prompt_source",
    "urgent_output_source",
    "output_intervention",
    "stress_control_relationship",
    "kv_constrained_online",
    "least_laxity_first",
})

# Reference threshold shared with KVConstrainedOnlinePolicy's own default --
# used only to compute an *observable* per-request classification, never a
# generator label.
_URGENT_LAXITY_SECONDS = KVConstrainedOnlinePolicy().urgent_laxity_seconds
_STEP_SIZE = KVConstrainedOnlinePolicy().step_size

FEATURE_NAMES = (
    "n_queued_requests",
    "n_urgent_waiting",
    "fraction_urgent_waiting",
    "mean_slo_slack",
    "min_slo_slack",
    "mean_prompt_tokens",
    "kv_util_ratio",
    "n_active_requests",
)


def assert_no_hidden_leakage(feature_dict: Mapping[str, Any]) -> None:
    """Fail loudly if any forbidden key leaked into a feature dict."""
    bad = sorted(set(feature_dict) & FORBIDDEN_FEATURE_KEYS)
    if bad:
        raise ValueError(f"Hidden/generator features leaked into model inputs: {bad}")


# ===================================================================
# Scenario-level features (pre-run summary from a Request list)
# ===================================================================

def scenario_observable_features(requests: Sequence[Request]) -> Dict[str, float]:
    """Summarise a loaded trace using only online-observable fields.

    Used by the pre-run selector/hard-conditional rule (scenario-level).
    """
    if not requests:
        return {name: 0.0 for name in FEATURE_NAMES}

    prompts = np.array([r.prompt_tokens for r in requests], dtype=float)
    slacks = np.array(
        [r.slo_deadline - r.arrival_time for r in requests], dtype=float
    )
    n_urgent = sum(
        1 for r in requests
        if (r.slo_deadline - r.arrival_time) <= _URGENT_LAXITY_SECONDS
    )

    return {
        "n_queued_requests": float(len(requests)),
        "n_urgent_waiting": float(n_urgent),
        "fraction_urgent_waiting": float(n_urgent / len(requests)),
        "mean_slo_slack": float(np.mean(slacks)),
        "min_slo_slack": float(np.min(slacks)),
        "mean_prompt_tokens": float(np.mean(prompts)),
        "kv_util_ratio": 0.0,       # post-run; zero at scenario level
        "n_active_requests": 0.0,  # post-run; zero at scenario level
    }


def feature_vector(
    feature_dict: Mapping[str, float],
    names: Iterable[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Return a numpy feature vector aligned to FEATURE_NAMES."""
    return np.asarray([float(feature_dict.get(n, 0.0)) for n in names], dtype=float)


# ===================================================================
# Step-level features (computed from ObservableState at run-time)
# ===================================================================

def n_urgent_waiting(state: ObservableState) -> int:
    """Count of currently-waiting requests classified urgent, using
    KVConstrainedOnlinePolicy's own laxity_seconds()/urgent_laxity_seconds
    definition (no new mechanism -- this exact same threshold already
    governs the parent's urgent-override in _admit_filter/_score)."""
    return sum(
        1 for r in state.waiting_queue
        if laxity_seconds(r, state.time, _STEP_SIZE, DEFAULT_ALPHA, DEFAULT_BETA)
        <= _URGENT_LAXITY_SECONDS
    )


def step_features(state: ObservableState) -> Dict[str, float]:
    """Compute step-level observable features from the live scheduler state."""
    queue = state.waiting_queue
    if queue:
        prompts = [float(r.prompt_tokens) for r in queue]
        slacks = [
            r.slo_deadline - state.time - 0.001 * float(r.predicted_output_tokens)
            for r in queue
        ]
        n_urgent = n_urgent_waiting(state)
        n_queued = float(len(queue))
        mean_prompt = float(np.mean(prompts))
        mean_slack = float(np.mean(slacks))
        min_slack = float(np.min(slacks))
    else:
        n_urgent = 0
        n_queued = 0.0
        mean_prompt = 0.0
        mean_slack = 0.0
        min_slack = 0.0

    kv_util = 0.0
    n_active = 0.0
    if state.gpu_states:
        gpu = state.gpu_states[0]
        kv_util = float(gpu.current_kv_tokens) / max(gpu.max_kv_tokens, 1)
        n_active = float(len(gpu.active_request_ids))

    return {
        "n_queued_requests": n_queued,
        "n_urgent_waiting": float(n_urgent),
        "fraction_urgent_waiting": float(n_urgent / n_queued) if n_queued > 0 else 0.0,
        "mean_slo_slack": mean_slack,
        "min_slo_slack": min_slack,
        "mean_prompt_tokens": mean_prompt,
        "kv_util_ratio": kv_util,
        "n_active_requests": n_active,
    }


# ===================================================================
# Train-scenario feature builder
# ===================================================================

def build_scenario_feature_rows(request_lists: List[Sequence[Request]]) -> List[Dict[str, float]]:
    return [scenario_observable_features(reqs) for reqs in request_lists]

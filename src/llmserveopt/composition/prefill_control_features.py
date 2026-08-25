"""Observable features for Family B v2 PrefillControl composition.

No generator-label leakage. Only online-observable quantities from
ObservableRequest / ObservableState that a real scheduler could compute
without seeing scenario_id, seed, slo_emphasis, hog_count, late_pressure,
or intended winner.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ..core.types import ObservableRequest, ObservableState, Request

# ---------------------------------------------------------------------------
# Explicit denylist
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURE_KEYS = frozenset({
    "scenario_id",
    "seed",
    "slo_emphasis",
    "hog_count",
    "late_pressure",
    "intended_winner",
    "intended",
    "winner",
    "generator_family",
    "generator_version",
    "token_length_source",
    "burstgpt_path",
    "allow_synthetic_tokens",
    "pair_id",
    "max_active_sequences",
    "output_intervention",
    "arrival_shape",
    "n_total_jobs",
    "n_hog",
    "n_late",
    "hog_prompt_median",
    "late_prompt_median",
    "slack_hog_s",
    "slack_late_s",
    "step_token_budget",
    "tbt_slo_s",
    "late_start_s",
    "stress_control_relationship",
    "hog_ttft",
    "late_ttft",
    "family_b",
    "policies",
    "chunk_small",
    "chunk_large",
    "unlimited",
})

# Feature vector ordering -- kept stable for reproducibility.
FEATURE_NAMES = (
    "n_queued_requests",
    "fraction_prompts_gt_1024",
    "mean_prompt_tokens",
    "max_prompt_tokens",
    "mean_slo_slack",
    "min_slo_slack",
    "std_slo_slack",
    "fraction_urgent",
    "mean_predicted_output",
    "max_predicted_output",
    "n_decoding_active",
    "n_prefilling_active",
    "kv_util_ratio",
)


def assert_no_hidden_leakage(feature_dict: Mapping[str, Any]) -> None:
    """Fail loudly if any forbidden key leaked into the feature dict."""
    bad = sorted(set(feature_dict) & FORBIDDEN_FEATURE_KEYS)
    if bad:
        raise ValueError(
            f"Hidden/generator features leaked into model inputs: {bad}"
        )


# ===================================================================
# Scenario-level features (pre-run summary from Request list)
# ===================================================================

def scenario_observable_features(
    requests: Sequence[Request],
) -> Dict[str, float]:
    """Summarise a loaded trace using only online-observable fields.

    This is the feature vector used by the pre-run selector (scenario-level,
    not step-level).  A real scheduler would compute an analogous online
    approximation from the waiting-set queue.
    """
    if not requests:
        return {name: 0.0 for name in FEATURE_NAMES}

    prompts = np.array([r.prompt_tokens for r in requests], dtype=float)
    outputs = np.array([r.predicted_output_tokens for r in requests], dtype=float)
    arrivals = np.array([r.arrival_time for r in requests], dtype=float)
    slacks = np.array(
        [r.slo_deadline - r.arrival_time for r in requests], dtype=float
    )

    return {
        "n_queued_requests": float(len(requests)),
        "fraction_prompts_gt_1024": float(np.mean(prompts > 1024)),
        "mean_prompt_tokens": float(np.mean(prompts)),
        "max_prompt_tokens": float(np.max(prompts)),
        "mean_slo_slack": float(np.mean(slacks)),
        "min_slo_slack": float(np.min(slacks)),
        "std_slo_slack": float(np.std(slacks)),
        "fraction_urgent": float(np.mean(slacks <= 0.0)),
        "mean_predicted_output": float(np.mean(outputs)),
        "max_predicted_output": float(np.max(outputs)),
        "n_decoding_active": 0.0,   # post-run; zero at scenario level
        "n_prefilling_active": 0.0, # post-run; zero at scenario level
        "kv_util_ratio": 0.0,       # post-run; zero at scenario level
    }


def feature_vector(
    feature_dict: Mapping[str, float],
    names: Iterable[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Return a numpy feature vector aligned to FEATURE_NAMES."""
    feat_list = []
    for name in names:
        feat_list.append(float(feature_dict.get(name, 0.0)))
    return np.asarray(feat_list, dtype=float)


# ===================================================================
# Step-level features (computed from ObservableState at run-time)
# ===================================================================

def step_features(state: ObservableState) -> Dict[str, float]:
    """Compute step-level observable features from the live scheduler state.

    Used by the ContextualPrefillControl policy at every scheduling step
    so it can make a chunk-size decision without any generator labels.
    """
    queue = state.waiting_queue
    if queue:
        prompts = []
        outputs = []
        slack = []
        urgent = []
        for r in queue:
            prompts.append(float(r.prompt_tokens))
            outputs.append(float(r.predicted_output_tokens))
            s = r.slo_deadline - state.time - float(r.predicted_output_tokens) * 0.001
            slack.append(s)
            urgent.append(1 if s < 0.0 else 0)
        n_queued = float(len(queue))
        frac_gt_1024 = float(np.mean([p > 1024 for p in prompts]))
        mean_prompt = float(np.mean(prompts))
        max_prompt = float(np.max(prompts))
        mean_slack = float(np.mean(slack))
        min_slack = float(np.min(slack))
        std_slack = float(np.std(slack))
        frac_urgent = float(np.mean(urgent))
        mean_output = float(np.mean(outputs))
        max_output = float(np.max(outputs))
    else:
        n_queued = 0.0
        frac_gt_1024 = 0.0
        mean_prompt = 0.0
        max_prompt = 0.0
        mean_slack = 0.0
        min_slack = 0.0
        std_slack = 0.0
        frac_urgent = 0.0
        mean_output = 0.0
        max_output = 0.0

    n_decoding = 0.0
    n_prefilling = 0.0
    kv_util = 0.0
    if state.gpu_states:
        gpu = state.gpu_states[0]
        n_decoding = float(gpu.decoding_count)
        n_prefilling = float(gpu.prefilling_count)
        kv_util = float(gpu.current_kv_tokens) / max(gpu.max_kv_tokens, 1)

    return {
        "n_queued_requests": n_queued,
        "fraction_prompts_gt_1024": frac_gt_1024,
        "mean_prompt_tokens": mean_prompt,
        "max_prompt_tokens": max_prompt,
        "mean_slo_slack": mean_slack,
        "min_slo_slack": min_slack,
        "std_slo_slack": std_slack,
        "fraction_urgent": frac_urgent,
        "mean_predicted_output": mean_output,
        "max_predicted_output": max_output,
        "n_decoding_active": n_decoding,
        "n_prefilling_active": n_prefilling,
        "kv_util_ratio": kv_util,
    }


# ===================================================================
# Train-scenario feature builder
# ===================================================================

def build_scenario_feature_rows(
    request_lists: List[Sequence[Request]],
) -> List[Dict[str, float]]:
    """Build scenario-level feature rows for training a selector.

    Each scenario in `request_lists` is summarised into a single feature
    vector using `scenario_observable_features`, and the result is a list
    of dicts aligned to FEATURE_NAMES.
    """
    return [scenario_observable_features(reqs) for reqs in request_lists]

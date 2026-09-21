"""Observable features for ESTF/WFS composition (no generator-label leakage)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from ..core.types import Request
from ..policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA, predicted_service_proxy

# Explicit denylist — never used as model inputs.
FORBIDDEN_FEATURE_KEYS = frozenset(
    {
        "favored_tenant_size",
        "other_tenant_size",
        "target_utilization",
        "tenant_weight_skew",
        "prediction_noise_sigma",
        "seed",
        "scenario_id",
        "size_priority_alignment",
        "generator_family",
        "token_length_source",
        "burstgpt_path",
        "allow_synthetic_tokens",
        "pair_id",
        "max_active_sequences",
        "favored_slo_slack_s",
        "other_slo_slack_s",
        "n_total_jobs",
    }
)

FEATURE_NAMES = (
    "n_requests",
    "mean_prompt_tokens",
    "std_prompt_tokens",
    "mean_predicted_output",
    "std_predicted_output",
    "short_job_fraction",
    "mean_est_service",
    "std_est_service",
    "mean_priority",
    "std_priority",
    "priority_skew",
    "high_priority_fraction",
    "class_imbalance",
    "n_classes",
    "mean_slo_slack",
    "std_slo_slack",
    "mean_interarrival",
    "arrival_cv",
)


def assert_no_hidden_leakage(feature_dict: Mapping[str, Any]) -> None:
    bad = sorted(set(feature_dict) & FORBIDDEN_FEATURE_KEYS)
    if bad:
        raise ValueError(f"Hidden/generator features leaked into model inputs: {bad}")


def scenario_observable_features(
    requests: Sequence[Request],
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> Dict[str, float]:
    """Summarize a loaded request trace using only online-observable fields.

    These are the same quantities a scheduler could estimate from the current
    waiting set / recent history. Generator treatment labels are not used.
    """
    if not requests:
        return {name: 0.0 for name in FEATURE_NAMES}

    prompts = np.asarray([r.prompt_tokens for r in requests], dtype=float)
    outputs = np.asarray([r.predicted_output_tokens for r in requests], dtype=float)
    priorities = np.asarray([float(r.priority) for r in requests], dtype=float)
    arrivals = np.asarray([float(r.arrival_time) for r in requests], dtype=float)
    slacks = np.asarray(
        [float(r.slo_deadline) - float(r.arrival_time) for r in requests], dtype=float
    )
    est = np.asarray(
        [predicted_service_proxy(r, alpha=alpha, beta=beta) for r in requests],
        dtype=float,
    )
    med_out = float(np.median(outputs))
    short_frac = float(np.mean(outputs <= med_out))
    classes: Dict[str, int] = {}
    for r in requests:
        classes[r.class_id or "unknown"] = classes.get(r.class_id or "unknown", 0) + 1
    class_imbalance = max(classes.values()) / len(requests)
    order = np.argsort(arrivals)
    gaps = np.diff(arrivals[order])
    mean_gap = float(np.mean(gaps)) if len(gaps) else 0.0
    arrival_cv = float(np.std(gaps) / max(mean_gap, 1e-12)) if len(gaps) else 0.0
    pmin = float(np.min(priorities))
    pmax = float(np.max(priorities))
    feats = {
        "n_requests": float(len(requests)),
        "mean_prompt_tokens": float(np.mean(prompts)),
        "std_prompt_tokens": float(np.std(prompts)),
        "mean_predicted_output": float(np.mean(outputs)),
        "std_predicted_output": float(np.std(outputs)),
        "short_job_fraction": short_frac,
        "mean_est_service": float(np.mean(est)),
        "std_est_service": float(np.std(est)),
        "mean_priority": float(np.mean(priorities)),
        "std_priority": float(np.std(priorities)),
        "priority_skew": pmax / max(pmin, 1e-12),
        "high_priority_fraction": float(np.mean(priorities >= np.median(priorities))),
        "class_imbalance": float(class_imbalance),
        "n_classes": float(len(classes)),
        "mean_slo_slack": float(np.mean(slacks)),
        "std_slo_slack": float(np.std(slacks)),
        "mean_interarrival": mean_gap,
        "arrival_cv": arrival_cv,
    }
    assert_no_hidden_leakage(feats)
    return feats


def feature_vector(
    feature_dict: Mapping[str, float], names: Iterable[str] = FEATURE_NAMES
) -> np.ndarray:
    assert_no_hidden_leakage(feature_dict)
    return np.asarray([float(feature_dict[n]) for n in names], dtype=float)

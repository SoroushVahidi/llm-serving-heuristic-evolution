"""Shared cross-family context feature schema (SHARED_CORE_V1).

FEATURE-SCHEMA INVESTIGATION ARTIFACT. See
docs/audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md for
the full audit that motivates this module.

The existing MF-PSD v1 scenario table (`mf_psd.py`) carries 33 learnable
columns that are all family-prefixed (`feat_A__*`/`feat_B__*`/`feat_C__*`)
with explicit family-scoped missingness -- the Step-3 multi-family selector
experiment (`multifamily_contextual_selector_v1`) found this schema lets a
selector trivially identify family from missingness alone (100% CV accuracy),
and that this drives the observed LOFO-A collapse. This module instead
computes a small set of features that are:

  * defined by the SAME formula for every family (no family-prefixed
    columns, no family-scoped missingness);
  * computed purely from the shared `Request`/`GPUConfig` dataclasses
    (`core/types.py`) that every one of the three families' scenario
    templates already builds their `PolicySeparationScenario.requests` /
    `.gpu_configs` from -- this is a genuine, pre-existing structural
    invariant of the codebase, not a new assumption;
  * computed only from request/GPU fields a real online policy could see
    before or at admission time (arrival_time, prompt_tokens,
    predicted_output_tokens, slo_deadline, priority, class_id, and the
    static GPUConfig fields) -- `actual_output_tokens` is deliberately never
    read here, matching `Request`'s own documented contract that it is
    "hidden from online policies."

`compute_shared_context_features` is a pure function of
`(requests, gpu_configs)` -- it does not know, and cannot infer from its
inputs alone, which of the three families produced them.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Sequence, Tuple

from ..core.types import GPUConfig, Request

#: Canonical, ordered list of SHARED_CORE_V1 learnable feature names. Every
#: one of these is present (non-missing, non-family-prefixed) for every
#: scenario in every family -- see the audit doc section E/G.
SHARED_CORE_V1_FEATURES: Tuple[str, ...] = (
    # LOAD / QUEUE
    "n_requests",
    "window_span_s",
    "offered_rate_rps",
    # REQUEST SIZE
    "mean_prompt_tokens",
    "cv_prompt_tokens",
    "mean_predicted_output_tokens",
    "cv_predicted_output_tokens",
    "mean_predicted_total_tokens",
    # URGENCY / SLO
    "mean_slack_s",
    "min_slack_s",
    "frac_tight_slack",
    # PRIORITY / FAIRNESS
    "priority_cv",
    "n_distinct_request_classes",
    # RESOURCE PRESSURE
    "max_active_sequences",
    "max_kv_tokens",
    "token_footprint_per_kv",
    "concurrency_pressure",
)

SHARED_CORE_V1_VERSION = "shared_core_v1.0.0"


def _cv(values: Sequence[float]) -> float:
    """Coefficient of variation (population std / mean). 0.0 if mean is 0 or
    fewer than 2 values (a degenerate, not missing, value -- always defined)."""
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    std = statistics.pstdev(values)
    return std / mean


def compute_shared_context_features(
    requests: Sequence[Request], gpu_configs: Sequence[GPUConfig]
) -> Dict[str, float]:
    """Compute the SHARED_CORE_V1 feature vector for one scenario.

    Pure function of the shared `Request`/`GPUConfig` dataclasses -- reads no
    family identity, no scenario ID, no utility/outcome field, and no
    `actual_output_tokens` (policy-hidden ground truth). Deterministic:
    depends only on the (already-deterministic) contents of `requests` and
    `gpu_configs`.
    """
    if not requests:
        raise ValueError("compute_shared_context_features requires at least one request")
    if not gpu_configs:
        raise ValueError("compute_shared_context_features requires at least one gpu_config")

    n = len(requests)
    arrival_times = [r.arrival_time for r in requests]
    prompt_tokens = [float(r.prompt_tokens) for r in requests]
    pred_output_tokens = [float(r.predicted_output_tokens) for r in requests]
    total_tokens = [p + o for p, o in zip(prompt_tokens, pred_output_tokens)]
    slacks = [r.slo_deadline - r.arrival_time for r in requests]
    priorities = [float(r.priority) for r in requests]
    class_ids = {r.class_id for r in requests}

    window_span_s = max(arrival_times) - min(arrival_times)
    offered_rate_rps = (n - 1) / window_span_s if window_span_s > 0 else 0.0

    mean_prompt_tokens = statistics.fmean(prompt_tokens)
    mean_predicted_output_tokens = statistics.fmean(pred_output_tokens)
    mean_predicted_total_tokens = statistics.fmean(total_tokens)

    mean_slack_s = statistics.fmean(slacks)
    min_slack_s = min(slacks)
    median_slack_s = statistics.median(slacks)
    tight_threshold = 0.5 * median_slack_s
    frac_tight_slack = sum(1 for s in slacks if s < tight_threshold) / n

    priority_cv = _cv(priorities)

    max_active_sequences = statistics.fmean(g.max_active_sequences for g in gpu_configs)
    max_kv_tokens = statistics.fmean(g.max_kv_tokens for g in gpu_configs)

    token_footprint_per_kv = (
        (mean_predicted_total_tokens * n) / max_kv_tokens if max_kv_tokens > 0 else 0.0
    )
    concurrency_pressure = n / max_active_sequences if max_active_sequences > 0 else 0.0

    features: Dict[str, float] = {
        "n_requests": float(n),
        "window_span_s": float(window_span_s),
        "offered_rate_rps": float(offered_rate_rps),
        "mean_prompt_tokens": float(mean_prompt_tokens),
        "cv_prompt_tokens": float(_cv(prompt_tokens)),
        "mean_predicted_output_tokens": float(mean_predicted_output_tokens),
        "cv_predicted_output_tokens": float(_cv(pred_output_tokens)),
        "mean_predicted_total_tokens": float(mean_predicted_total_tokens),
        "mean_slack_s": float(mean_slack_s),
        "min_slack_s": float(min_slack_s),
        "frac_tight_slack": float(frac_tight_slack),
        "priority_cv": float(priority_cv),
        "n_distinct_request_classes": float(len(class_ids)),
        "max_active_sequences": float(max_active_sequences),
        "max_kv_tokens": float(max_kv_tokens),
        "token_footprint_per_kv": float(token_footprint_per_kv),
        "concurrency_pressure": float(concurrency_pressure),
    }

    for name in SHARED_CORE_V1_FEATURES:
        v = features[name]
        if not math.isfinite(v):
            raise ValueError(f"non-finite shared feature {name}={v!r}")

    assert set(features.keys()) == set(SHARED_CORE_V1_FEATURES)
    return features

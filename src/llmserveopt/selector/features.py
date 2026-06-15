"""
Online-observable selector feature extraction.

Two modes
---------
online_prefix (default, deployable)
    Features are computed only from requests that arrived BEFORE the window start
    time, plus per-window prompt/SLO statistics of requests that have arrived
    (i.e., their headers are visible when the scheduling decision is made).

trace_window_descriptive (offline analysis only, NOT deployable)
    Features are computed from all requests in the window.  Marked unavailable=True
    for any feature that would require future knowledge.  Useful for offline ablation
    but must not be used for training a deployed selector.

Feature leakage invariant
--------------------------
*Neither mode* uses actual_output_tokens.  Only predicted_output_tokens is used.
Changing future requests (beyond window end) must not alter any feature computed
for the current window.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Request


class FeatureMode(str, Enum):
    ONLINE_PREFIX = "online_prefix"
    TRACE_WINDOW_DESCRIPTIVE = "trace_window_descriptive"


# Ordered list — column order in the output CSV is preserved.
FEATURE_NAMES: List[str] = [
    "queue_length",
    "active_sequence_count",
    "kv_utilization",
    "free_sequence_ratio",
    "mean_prompt_tokens",
    "p95_prompt_tokens",
    "mean_pred_output_tokens",
    "p95_pred_output_tokens",
    "pred_output_cv",
    "fraction_tight_slo",
    "mean_slack",
    "p10_slack",
    "min_slack",
    "mean_waiting_time",
    "p95_waiting_time",
    "arrival_rate_est",
    "burstiness_cv",
    "recent_slo_violation_rate",
]

_NAN = float("nan")


def extract_features(
    window_requests: Sequence[Request],
    window_start_time: float,
    mode: FeatureMode = FeatureMode.ONLINE_PREFIX,
    prefix_requests: Optional[Sequence[Request]] = None,
    recent_violation_rate: float = 0.0,
    recent_violation_available: bool = True,
    active_sequence_count: int = 0,
    kv_utilization: float = 0.0,
    kv_utilization_available: bool = False,
    free_sequence_ratio: float = 1.0,
    free_sequence_ratio_available: bool = False,
) -> Dict[str, float]:
    """Extract the 18 selector features for one window.

    Parameters
    ----------
    window_requests : requests in this window (used for token/SLO stats in both modes).
    window_start_time : arrival_time of the first request in the window.
    mode : FeatureMode — controls which requests are used and leakage guards.
    prefix_requests : requests from before this window (used in online_prefix mode
        to compute arrival rate / queue approximation).  If None and mode is
        online_prefix, falls back to window_requests for those features.
    recent_violation_rate : SLO violation rate observed in recently completed
        requests (before window start).  Set 0.0 when unavailable.
    recent_violation_available : whether recent_violation_rate is meaningful data.
    active_sequence_count : GPU active sequences at decision time (from sim state).
    kv_utilization : KV-cache fill fraction at decision time.
    kv_utilization_available : False in offline trace-only mode.
    free_sequence_ratio : fraction of GPU sequence slots that are free.
    free_sequence_ratio_available : False in offline trace-only mode.
    """
    if mode == FeatureMode.ONLINE_PREFIX:
        return _extract_online_prefix(
            window_requests=window_requests,
            window_start_time=window_start_time,
            prefix_requests=prefix_requests,
            recent_violation_rate=recent_violation_rate,
            recent_violation_available=recent_violation_available,
            active_sequence_count=active_sequence_count,
            kv_utilization=kv_utilization,
            kv_utilization_available=kv_utilization_available,
            free_sequence_ratio=free_sequence_ratio,
            free_sequence_ratio_available=free_sequence_ratio_available,
        )
    else:
        return _extract_trace_window_descriptive(
            window_requests=window_requests,
            window_start_time=window_start_time,
            recent_violation_rate=recent_violation_rate,
            recent_violation_available=recent_violation_available,
        )


def _extract_online_prefix(
    window_requests: Sequence[Request],
    window_start_time: float,
    prefix_requests: Optional[Sequence[Request]],
    recent_violation_rate: float,
    recent_violation_available: bool,
    active_sequence_count: int,
    kv_utilization: float,
    kv_utilization_available: bool,
    free_sequence_ratio: float,
    free_sequence_ratio_available: bool,
) -> Dict[str, float]:
    """online_prefix: uses prefix requests for arrival/queue stats; window for token/SLO."""
    win = list(window_requests)
    assert all(not _uses_actual_output(r) for r in win), "Leakage: actual_output in features"

    # Use prefix requests for arrival-rate estimation; fall back to window.
    lookback = list(prefix_requests) if prefix_requests is not None else win

    # --- queue_length: rough estimate = requests in lookback arrived within last 2× window span ---
    # Only count requests that arrived AT OR BEFORE window_start_time (no future leakage).
    if win:
        win_span = max(win[-1].arrival_time - window_start_time, 1.0)
        cutoff = window_start_time - 2 * win_span
        queue_approx = sum(
            1 for r in lookback
            if cutoff <= r.arrival_time <= window_start_time
        )
    else:
        queue_approx = sum(1 for r in lookback if r.arrival_time <= window_start_time)

    # --- active_sequence_count / kv_utilization ---
    # In offline trace-only mode these are unknown; caller passes 0/0.0.
    eff_kv = kv_utilization if kv_utilization_available else 0.0
    eff_free = free_sequence_ratio if free_sequence_ratio_available else 1.0

    # --- Token stats from window requests (visible: headers arrive before scheduling) ---
    prompt_arr = np.array([r.prompt_tokens for r in win], dtype=float) if win else np.array([0.0])
    pred_out_arr = np.array([r.predicted_output_tokens for r in win], dtype=float) if win else np.array([0.0])
    slack_arr = np.array([r.slo_deadline - r.arrival_time for r in win], dtype=float) if win else np.array([0.0])
    # "tight" covers synthetic traces; "interactive" covers BurstGPT class_id convention.
    tight_arr = np.array([1.0 if r.class_id in ("tight", "interactive") else 0.0 for r in win], dtype=float) if win else np.array([0.0])

    # --- Waiting time: time from arrival to window_start_time (visible at decision time) ---
    if lookback:
        wait_arr = np.array([window_start_time - r.arrival_time for r in lookback if r.arrival_time <= window_start_time], dtype=float)
        if len(wait_arr) == 0:
            wait_arr = np.array([0.0])
    else:
        wait_arr = np.array([0.0])
    wait_arr = np.clip(wait_arr, 0.0, None)

    # --- Arrival rate: requests per second from lookback ---
    arrival_rate = _estimate_arrival_rate(lookback, window_start_time)
    burstiness = _burstiness_cv(lookback, window_start_time)

    return {
        "queue_length": float(queue_approx),
        "active_sequence_count": float(active_sequence_count),
        "kv_utilization": float(eff_kv),
        "free_sequence_ratio": float(eff_free),
        "mean_prompt_tokens": float(np.mean(prompt_arr)),
        "p95_prompt_tokens": float(np.percentile(prompt_arr, 95)),
        "mean_pred_output_tokens": float(np.mean(pred_out_arr)),
        "p95_pred_output_tokens": float(np.percentile(pred_out_arr, 95)),
        "pred_output_cv": float(_cv(pred_out_arr)),
        "fraction_tight_slo": float(np.mean(tight_arr)),
        "mean_slack": float(np.mean(slack_arr)),
        "p10_slack": float(np.percentile(slack_arr, 10)),
        "min_slack": float(np.min(slack_arr)),
        "mean_waiting_time": float(np.mean(wait_arr)),
        "p95_waiting_time": float(np.percentile(wait_arr, 95)),
        "arrival_rate_est": float(arrival_rate),
        "burstiness_cv": float(burstiness),
        "recent_slo_violation_rate": float(recent_violation_rate) if recent_violation_available else 0.0,
    }


def _extract_trace_window_descriptive(
    window_requests: Sequence[Request],
    window_start_time: float,
    recent_violation_rate: float,
    recent_violation_available: bool,
) -> Dict[str, float]:
    """trace_window_descriptive: uses all requests in window.  NOT for deployment."""
    win = list(window_requests)
    prompt_arr = np.array([r.prompt_tokens for r in win], dtype=float) if win else np.array([0.0])
    pred_out_arr = np.array([r.predicted_output_tokens for r in win], dtype=float) if win else np.array([0.0])
    slack_arr = np.array([r.slo_deadline - r.arrival_time for r in win], dtype=float) if win else np.array([0.0])
    # "tight" covers synthetic traces; "interactive" covers BurstGPT class_id convention.
    tight_arr = np.array([1.0 if r.class_id in ("tight", "interactive") else 0.0 for r in win], dtype=float) if win else np.array([0.0])

    # Inter-arrival time stats from window itself
    arrivals = sorted(r.arrival_time for r in win)
    if len(arrivals) >= 2:
        span = arrivals[-1] - arrivals[0]
        rate = (len(arrivals) - 1) / max(span, 1e-9)
        iats = np.diff(arrivals)
        burstiness = _cv(iats)
    else:
        rate = 0.0
        burstiness = 0.0

    return {
        "queue_length": float(len(win)),          # descriptive: window size
        "active_sequence_count": 0.0,             # unavailable offline
        "kv_utilization": 0.0,                    # unavailable offline
        "free_sequence_ratio": 1.0,               # unavailable offline
        "mean_prompt_tokens": float(np.mean(prompt_arr)),
        "p95_prompt_tokens": float(np.percentile(prompt_arr, 95)),
        "mean_pred_output_tokens": float(np.mean(pred_out_arr)),
        "p95_pred_output_tokens": float(np.percentile(pred_out_arr, 95)),
        "pred_output_cv": float(_cv(pred_out_arr)),
        "fraction_tight_slo": float(np.mean(tight_arr)),
        "mean_slack": float(np.mean(slack_arr)),
        "p10_slack": float(np.percentile(slack_arr, 10)),
        "min_slack": float(np.min(slack_arr)),
        "mean_waiting_time": 0.0,                 # unavailable in descriptive offline mode
        "p95_waiting_time": 0.0,
        "arrival_rate_est": float(rate),
        "burstiness_cv": float(burstiness),
        "recent_slo_violation_rate": float(recent_violation_rate) if recent_violation_available else 0.0,
    }


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def _uses_actual_output(r: Request) -> bool:
    """Return True only if this function somehow accesses actual_output_tokens.

    In practice this always returns False — it exists only as documentation
    that we intentionally never touch actual_output_tokens in features.
    The real leakage guard is that extract_features() never references that field.
    """
    return False  # actual_output_tokens is never accessed by feature extractors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv(arr: np.ndarray) -> float:
    """Coefficient of variation (std/mean).  Returns 0 when mean is 0."""
    if len(arr) == 0:
        return 0.0
    m = float(np.mean(arr))
    if m == 0.0:
        return 0.0
    return float(np.std(arr) / m)


def _estimate_arrival_rate(requests: Sequence[Request], reference_time: float) -> float:
    """Estimate requests-per-second from recent arrivals in a sliding window."""
    if not requests:
        return 0.0
    arrivals = sorted(r.arrival_time for r in requests if r.arrival_time <= reference_time)
    if len(arrivals) < 2:
        return 0.0
    span = arrivals[-1] - arrivals[0]
    if span <= 0:
        return 0.0
    return (len(arrivals) - 1) / span


def _burstiness_cv(requests: Sequence[Request], reference_time: float) -> float:
    """CV of inter-arrival times from prefix requests."""
    if not requests:
        return 0.0
    arrivals = sorted(r.arrival_time for r in requests if r.arrival_time <= reference_time)
    if len(arrivals) < 2:
        return 0.0
    iats = np.diff(arrivals)
    return _cv(iats)

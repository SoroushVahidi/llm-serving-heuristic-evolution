"""
Online-observable selector feature extraction.

Feature modes
-------------
causal (default for deployable selector evaluation)
    Features depend only on requests with ``arrival_time <= window_start_time``.
    Future arrivals inside the current window are excluded.  This is the only mode
    suitable for online/deployable selector claims.

offline_window_lookahead (offline diagnostic only, NOT deployable)
    Token/SLO statistics use all requests in the window, while arrival-rate and
    queue features still use prefix history up to ``window_start_time``.  This
    leaks future within-window arrivals and must not be used for deployable claims.

    The legacy alias ``online_prefix`` maps to this mode and is deprecated.

trace_window_descriptive (offline analysis only, NOT deployable)
    Descriptive statistics over the full window.  Useful for ablations only.

Feature leakage invariant
--------------------------
*No mode* uses actual_output_tokens, completed latency, oracle labels, or
post-hoc SLO outcomes.  Only predicted_output_tokens and request headers visible
at or before the decision time are used.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.types import Request


class FeatureMode(str, Enum):
    CAUSAL = "causal"
    OFFLINE_WINDOW_LOOKAHEAD = "offline_window_lookahead"
    TRACE_WINDOW_DESCRIPTIVE = "trace_window_descriptive"
    # Deprecated alias — preserved for backward-compatible configs.
    ONLINE_PREFIX = "online_prefix"


_FEATURE_MODE_ALIASES: Dict[str, FeatureMode] = {
    "online_prefix": FeatureMode.OFFLINE_WINDOW_LOOKAHEAD,
}


def parse_feature_mode(value: str | FeatureMode) -> FeatureMode:
    """Parse a config/CLI feature-mode string with legacy alias support."""
    if isinstance(value, FeatureMode):
        return value
    key = str(value).strip()
    if key in _FEATURE_MODE_ALIASES:
        return _FEATURE_MODE_ALIASES[key]
    return FeatureMode(key)


def feature_mode_is_deployable(mode: FeatureMode) -> bool:
    """Return True only for modes valid for deployable selector evaluation."""
    return mode == FeatureMode.CAUSAL


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


def extract_features(
    window_requests: Sequence[Request],
    window_start_time: float,
    mode: FeatureMode = FeatureMode.CAUSAL,
    prefix_requests: Optional[Sequence[Request]] = None,
    recent_violation_rate: float = 0.0,
    recent_violation_available: bool = True,
    active_sequence_count: int = 0,
    kv_utilization: float = 0.0,
    kv_utilization_available: bool = False,
    free_sequence_ratio: float = 1.0,
    free_sequence_ratio_available: bool = False,
) -> Dict[str, float]:
    """Extract the 18 selector features for one window."""
    resolved = parse_feature_mode(mode)
    if resolved == FeatureMode.CAUSAL:
        return _extract_causal(
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
    if resolved == FeatureMode.OFFLINE_WINDOW_LOOKAHEAD:
        return _extract_offline_window_lookahead(
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
    return _extract_trace_window_descriptive(
        window_requests=window_requests,
        window_start_time=window_start_time,
        recent_violation_rate=recent_violation_rate,
        recent_violation_available=recent_violation_available,
    )


def _observable_at_decision(
    window_requests: Sequence[Request],
    window_start_time: float,
    prefix_requests: Optional[Sequence[Request]],
) -> List[Request]:
    """Requests visible at window_start_time (no future within-window arrivals)."""
    observable: List[Request] = []
    if prefix_requests is not None:
        observable.extend(
            r for r in prefix_requests if r.arrival_time <= window_start_time
        )
    observable.extend(
        r for r in window_requests if r.arrival_time <= window_start_time
    )
    return observable


def _token_slo_stats(requests: Sequence[Request]) -> Dict[str, float]:
    reqs = list(requests)
    prompt_arr = np.array([r.prompt_tokens for r in reqs], dtype=float) if reqs else np.array([0.0])
    pred_out_arr = (
        np.array([r.predicted_output_tokens for r in reqs], dtype=float) if reqs else np.array([0.0])
    )
    slack_arr = (
        np.array([r.slo_deadline - r.arrival_time for r in reqs], dtype=float) if reqs else np.array([0.0])
    )
    tight_arr = (
        np.array(
            [1.0 if r.class_id in ("tight", "interactive") else 0.0 for r in reqs],
            dtype=float,
        )
        if reqs
        else np.array([0.0])
    )
    return {
        "mean_prompt_tokens": float(np.mean(prompt_arr)),
        "p95_prompt_tokens": float(np.percentile(prompt_arr, 95)),
        "mean_pred_output_tokens": float(np.mean(pred_out_arr)),
        "p95_pred_output_tokens": float(np.percentile(pred_out_arr, 95)),
        "pred_output_cv": float(_cv(pred_out_arr)),
        "fraction_tight_slo": float(np.mean(tight_arr)),
        "mean_slack": float(np.mean(slack_arr)),
        "p10_slack": float(np.percentile(slack_arr, 10)),
        "min_slack": float(np.min(slack_arr)),
    }


def _extract_causal(
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
    """causal: only requests with arrival_time <= window_start_time."""
    observable = _observable_at_decision(window_requests, window_start_time, prefix_requests)
    assert all(not _uses_actual_output(r) for r in observable), (
        "Leakage: actual_output in features"
    )

    lookback = observable

    if lookback:
        arrivals = [r.arrival_time for r in lookback if r.arrival_time <= window_start_time]
        if arrivals:
            win_span = max(max(arrivals) - window_start_time, 1.0)
            cutoff = window_start_time - 2 * win_span
            queue_approx = sum(
                1 for r in lookback if cutoff <= r.arrival_time <= window_start_time
            )
        else:
            queue_approx = 0
    else:
        queue_approx = 0

    eff_kv = kv_utilization if kv_utilization_available else 0.0
    eff_free = free_sequence_ratio if free_sequence_ratio_available else 1.0

    token_stats = _token_slo_stats(lookback)

    if lookback:
        wait_arr = np.array(
            [
                window_start_time - r.arrival_time
                for r in lookback
                if r.arrival_time <= window_start_time
            ],
            dtype=float,
        )
        if len(wait_arr) == 0:
            wait_arr = np.array([0.0])
    else:
        wait_arr = np.array([0.0])
    wait_arr = np.clip(wait_arr, 0.0, None)

    arrival_rate = _estimate_arrival_rate(lookback, window_start_time)
    burstiness = _burstiness_cv(lookback, window_start_time)

    return {
        "queue_length": float(queue_approx),
        "active_sequence_count": float(active_sequence_count),
        "kv_utilization": float(eff_kv),
        "free_sequence_ratio": float(eff_free),
        **token_stats,
        "mean_waiting_time": float(np.mean(wait_arr)),
        "p95_waiting_time": float(np.percentile(wait_arr, 95)),
        "arrival_rate_est": float(arrival_rate),
        "burstiness_cv": float(burstiness),
        "recent_slo_violation_rate": (
            float(recent_violation_rate) if recent_violation_available else 0.0
        ),
    }


def _extract_offline_window_lookahead(
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
    """offline_window_lookahead: full window token/SLO stats (NOT deployable)."""
    win = list(window_requests)
    assert all(not _uses_actual_output(r) for r in win), "Leakage: actual_output in features"

    lookback = list(prefix_requests) if prefix_requests is not None else win

    if win:
        win_span = max(win[-1].arrival_time - window_start_time, 1.0)
        cutoff = window_start_time - 2 * win_span
        queue_approx = sum(
            1 for r in lookback if cutoff <= r.arrival_time <= window_start_time
        )
    else:
        queue_approx = sum(1 for r in lookback if r.arrival_time <= window_start_time)

    eff_kv = kv_utilization if kv_utilization_available else 0.0
    eff_free = free_sequence_ratio if free_sequence_ratio_available else 1.0

    token_stats = _token_slo_stats(win)

    if lookback:
        wait_arr = np.array(
            [
                window_start_time - r.arrival_time
                for r in lookback
                if r.arrival_time <= window_start_time
            ],
            dtype=float,
        )
        if len(wait_arr) == 0:
            wait_arr = np.array([0.0])
    else:
        wait_arr = np.array([0.0])
    wait_arr = np.clip(wait_arr, 0.0, None)

    arrival_rate = _estimate_arrival_rate(lookback, window_start_time)
    burstiness = _burstiness_cv(lookback, window_start_time)

    return {
        "queue_length": float(queue_approx),
        "active_sequence_count": float(active_sequence_count),
        "kv_utilization": float(eff_kv),
        "free_sequence_ratio": float(eff_free),
        **token_stats,
        "mean_waiting_time": float(np.mean(wait_arr)),
        "p95_waiting_time": float(np.percentile(wait_arr, 95)),
        "arrival_rate_est": float(arrival_rate),
        "burstiness_cv": float(burstiness),
        "recent_slo_violation_rate": (
            float(recent_violation_rate) if recent_violation_available else 0.0
        ),
    }


def _extract_trace_window_descriptive(
    window_requests: Sequence[Request],
    window_start_time: float,
    recent_violation_rate: float,
    recent_violation_available: bool,
) -> Dict[str, float]:
    """trace_window_descriptive: uses all requests in window.  NOT for deployment."""
    win = list(window_requests)
    token_stats = _token_slo_stats(win)

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
        "queue_length": float(len(win)),
        "active_sequence_count": 0.0,
        "kv_utilization": 0.0,
        "free_sequence_ratio": 1.0,
        **token_stats,
        "mean_waiting_time": 0.0,
        "p95_waiting_time": 0.0,
        "arrival_rate_est": float(rate),
        "burstiness_cv": float(burstiness),
        "recent_slo_violation_rate": (
            float(recent_violation_rate) if recent_violation_available else 0.0
        ),
    }


def _uses_actual_output(r: Request) -> bool:
    """Document that feature extractors must never read actual_output_tokens."""
    return False


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

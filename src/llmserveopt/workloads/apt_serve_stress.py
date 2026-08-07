"""apt_serve_stress: Target and Counter workload generators designed to systematically
stress-test Apt-Serve's dual-tier cache mechanisms and explore headroom boundaries.
"""
from __future__ import annotations

import numpy as np
from typing import List
from ..core.types import Request


def generate_apt_serve_target_workload(
    seed: int = 2026,
    n_requests: int = 40,
    arrival_rate: float = 5.0,
    long_prompt_tokens: int = 600,   # Adjusted to safely fit within 1024-token (64-block) GPU caps
    short_prompt_tokens: int = 128,
    long_fraction: float = 0.2,
    tight_slack: float = 0.5,
    relaxed_slack: float = 15.0
) -> List[Request]:
    """Target generator designed from first principles to create legitimate Apt-Serve headroom.

    Cohort A (Long prompts, relaxed SLOs) occupy memory.
    Cohort B (Short prompts, urgent SLOs) arrive behind them.
    Apt-Serve can offload Cohort A to hidden cache, admit and complete Cohort B,
    and then restore Cohort A, meeting all SLOs. FIFO/EDF will thrash or violate.
    """
    rng = np.random.default_rng(seed)
    requests: List[Request] = []

    # Exponential inter-arrival times
    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=n_requests)
    arrival_times = np.cumsum(inter_arrivals)

    for i in range(n_requests):
        arrival_time = arrival_times[i]
        
        # Decide cohort
        is_long = rng.random() < long_fraction
        if is_long:
            prompt_tokens = long_prompt_tokens
            predicted_output = 32
            slo_deadline = arrival_time + relaxed_slack
            priority = 1.0
            class_id = "relaxed_long"
        else:
            prompt_tokens = short_prompt_tokens
            predicted_output = 16
            slo_deadline = arrival_time + tight_slack
            priority = 3.0
            class_id = "urgent_short"

        requests.append(Request(
            request_id=i + 1,
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            predicted_output_tokens=predicted_output,
            actual_output_tokens=predicted_output,  # exact prediction for simplicity
            slo_deadline=slo_deadline,
            priority=priority,
            class_id=class_id
        ))

    return requests


def generate_apt_serve_counter_workload(
    seed: int = 2026,
    n_requests: int = 40,
    scenario: str = "low_pressure",
    prompt_tokens: int = 128,
    slack: float = 5.0
) -> List[Request]:
    """Counter generator designed to bound or eliminate Apt-Serve's advantage.

    Scenarios:
    - "low_pressure": Memory is spacious, so tiering is never triggered. Simple policies are perfectly competitive.
    - "homogeneous": Every request has identical sizes and SLOs. No policy differentiation exists.
    - "adversarial_tight": SLOs are so tight that no policy has maneuvering room; all fail equally.
    """
    rng = np.random.default_rng(seed)
    requests: List[Request] = []

    arrival_rate = 5.0
    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=n_requests)
    arrival_times = np.cumsum(inter_arrivals)

    for i in range(n_requests):
        arrival_time = arrival_times[i]

        if scenario == "low_pressure":
            p_tokens = 32
            output = 16
            deadline = arrival_time + 10.0
            priority = 1.0
            class_id = "low_pressure"
        elif scenario == "homogeneous":
            p_tokens = prompt_tokens
            output = 16
            deadline = arrival_time + slack
            priority = 1.0
            class_id = "homogeneous"
        elif scenario == "adversarial_tight":
            p_tokens = 500 # Adjusted to safely fit in 1024-token budget
            output = 32
            deadline = arrival_time + 0.1 # virtually impossible
            priority = 1.0
            class_id = "adversarial_tight"
        else:
            raise ValueError(f"Unknown counter scenario: {scenario}")

        requests.append(Request(
            request_id=i + 1,
            arrival_time=arrival_time,
            prompt_tokens=p_tokens,
            predicted_output_tokens=output,
            actual_output_tokens=output,
            slo_deadline=deadline,
            priority=priority,
            class_id=class_id
        ))

    return requests


# ======================================================================
# Phase G: general-purpose regime workload generator.
#
# Scope note: this generator treats KV pressure, SLO heterogeneity, length
# heterogeneity, arrival dynamics, and cache-use structure as controlled
# *design-time* input knobs, not as guaranteed, precisely-calibrated
# runtime outcomes. "kv_pressure=high" means "arrival rate and prompt size
# scaled up to make contention more likely," not "peak simulator KV
# occupancy will measure at exactly X%." Realized pressure should be read
# from each run's completion_fraction / num_dropped in the output metrics,
# not assumed from the regime label alone. This mirrors this project's
# standing practice of not overclaiming instrumentation depth (see
# docs/PROJECT_MAP.md).
# ======================================================================

KV_PRESSURE_TIERS = ["low", "medium", "high", "near_capacity", "sustained_overload"]
SLO_PATTERNS = ["relaxed_homogeneous", "tight_homogeneous", "bimodal", "priority_correlated", "adversarial_mixed"]
LENGTH_PATTERNS = ["homogeneous", "prompt_heavy", "decode_heavy", "bimodal", "heavy_tail_prompt", "heavy_tail_output"]
ARRIVAL_PATTERNS = ["steady", "burst", "clustered_burst", "alternating", "spike"]
CACHE_USE_STRUCTURES = [
    "none", "kv_to_hidden_opportunity", "hidden_to_kv_opportunity",
    "long_relaxed_urgent_short", "recompute_avoidance", "thrash_risk",
]

_KV_PRESSURE_ARRIVAL_RATE = {
    "low": 2.0,
    "medium": 4.0,
    "high": 6.5,
    "near_capacity": 9.0,
    "sustained_overload": 13.0,
}


def _length_pair(rng: np.random.Generator, length_pattern: str) -> tuple[int, int]:
    """Return (prompt_tokens, predicted_output_tokens) for one request."""
    if length_pattern == "homogeneous":
        return 256, 64
    if length_pattern == "prompt_heavy":
        return 700, 32
    if length_pattern == "decode_heavy":
        return 64, 400
    if length_pattern == "bimodal":
        if rng.random() < 0.5:
            return 128, 16
        return 600, 128
    if length_pattern == "heavy_tail_prompt":
        # lognormal centered near 200, occasional very long prompts, capped
        # to stay inside the 1024-token (64-block) GPU budget used by Phase G.
        val = int(rng.lognormal(mean=5.2, sigma=0.6))
        return max(32, min(val, 900)), 32
    if length_pattern == "heavy_tail_output":
        val = int(rng.lognormal(mean=3.4, sigma=0.7))
        return 128, max(8, min(val, 350))
    raise ValueError(f"Unknown length_pattern: {length_pattern}")


def _arrival_times(rng: np.random.Generator, n_requests: int, arrival_pattern: str, base_rate: float) -> np.ndarray:
    """Return a sorted array of n_requests arrival times for the given pattern."""
    if arrival_pattern == "steady":
        inter = rng.exponential(1.0 / base_rate, size=n_requests)
        return np.cumsum(inter)

    if arrival_pattern == "burst":
        n_burst = int(n_requests * 0.6)
        n_tail = n_requests - n_burst
        burst = np.cumsum(rng.exponential(1.0 / (base_rate * 4.0), size=n_burst))
        tail_start = burst[-1] if n_burst > 0 else 0.0
        tail = tail_start + np.cumsum(rng.exponential(1.0 / (base_rate * 0.3), size=n_tail))
        return np.concatenate([burst, tail])

    if arrival_pattern == "clustered_burst":
        n_clusters = 4
        per_cluster = max(1, n_requests // n_clusters)
        times: List[float] = []
        cursor = 0.0
        remaining = n_requests
        for c in range(n_clusters):
            take = per_cluster if c < n_clusters - 1 else remaining
            take = min(take, remaining)
            cluster = cursor + np.cumsum(rng.exponential(1.0 / (base_rate * 3.0), size=take))
            times.extend(cluster.tolist())
            cursor = (cluster[-1] if take > 0 else cursor) + (2.0 / base_rate)  # quiet gap
            remaining -= take
        return np.array(sorted(times))

    if arrival_pattern == "alternating":
        window = max(1, n_requests // 6)
        times = []
        cursor = 0.0
        remaining = n_requests
        high = True
        while remaining > 0:
            take = min(window, remaining)
            rate = base_rate * 3.0 if high else base_rate * 0.5
            seg = cursor + np.cumsum(rng.exponential(1.0 / rate, size=take))
            times.extend(seg.tolist())
            cursor = seg[-1] if take > 0 else cursor
            remaining -= take
            high = not high
        return np.array(sorted(times))

    if arrival_pattern == "spike":
        n_spike = max(1, int(n_requests * 0.2))
        n_rest = n_requests - n_spike
        n_before = n_rest // 2
        n_after = n_rest - n_before
        before = np.cumsum(rng.exponential(1.0 / base_rate, size=n_before))
        spike_start = (before[-1] if n_before > 0 else 0.0) + 0.5
        spike = spike_start + np.cumsum(rng.exponential(1.0 / (base_rate * 8.0), size=n_spike))
        after_start = spike[-1] if n_spike > 0 else spike_start
        after = after_start + np.cumsum(rng.exponential(1.0 / base_rate, size=n_after))
        return np.concatenate([before, spike, after])

    raise ValueError(f"Unknown arrival_pattern: {arrival_pattern}")


def _slo_for(rng: np.random.Generator, arrival_time: float, priority_hint: float, slo_pattern: str) -> tuple[float, float, str]:
    """Return (slo_deadline, priority, class_id) for one request."""
    if slo_pattern == "relaxed_homogeneous":
        return arrival_time + 10.0, 1.0, "relaxed_homogeneous"
    if slo_pattern == "tight_homogeneous":
        return arrival_time + 0.5, 1.0, "tight_homogeneous"
    if slo_pattern == "bimodal":
        if rng.random() < 0.5:
            return arrival_time + 0.5, 3.0, "bimodal_tight"
        return arrival_time + 10.0, 1.0, "bimodal_relaxed"
    if slo_pattern == "priority_correlated":
        priority = float(rng.choice([1.0, 2.0, 3.0]))
        slack = {1.0: 8.0, 2.0: 2.0, 3.0: 0.4}[priority]
        return arrival_time + slack, priority, f"priority_{int(priority)}"
    if slo_pattern == "adversarial_mixed":
        if rng.random() < 0.3:
            return arrival_time + 0.1, 3.0, "adversarial_impossible"
        return arrival_time + 6.0, 1.0, "adversarial_lenient"
    raise ValueError(f"Unknown slo_pattern: {slo_pattern}")


def generate_apt_serve_regime_workload(
    seed: int,
    n_requests: int = 40,
    arrival_pattern: str = "steady",
    kv_pressure: str = "medium",
    slo_pattern: str = "relaxed_homogeneous",
    length_pattern: str = "homogeneous",
    cache_use_structure: str = "none",
) -> List[Request]:
    """General-purpose Phase G regime workload generator.

    Composes independently-controlled arrival-dynamics, KV-pressure,
    SLO-heterogeneity, length-heterogeneity, and cache-use-structure axes
    into a single deterministic trace. See the module-level scope note
    above for what "kv_pressure" does and does not guarantee.
    """
    if kv_pressure not in _KV_PRESSURE_ARRIVAL_RATE:
        raise ValueError(f"Unknown kv_pressure tier: {kv_pressure}")
    rng = np.random.default_rng(seed)
    base_rate = _KV_PRESSURE_ARRIVAL_RATE[kv_pressure]
    arrival_times = _arrival_times(rng, n_requests, arrival_pattern, base_rate)

    requests: List[Request] = []

    if cache_use_structure == "none":
        for i in range(n_requests):
            at = float(arrival_times[i])
            prompt_tokens, output_tokens = _length_pair(rng, length_pattern)
            deadline, priority, class_id = _slo_for(rng, at, 1.0, slo_pattern)
            requests.append(Request(
                request_id=i + 1, arrival_time=at, prompt_tokens=prompt_tokens,
                predicted_output_tokens=output_tokens, actual_output_tokens=output_tokens,
                slo_deadline=deadline, priority=priority, class_id=class_id,
            ))
        return requests

    # Cache-opportunity-focused regimes: a "cohort A" (long, relaxed,
    # arrives early, occupies memory) and "cohort B" (short, urgent,
    # arrives behind it) — generalizing generate_apt_serve_target_workload
    # across the length/SLO knobs above. thrash_risk narrows the gap
    # between the cohorts' slack values to encourage oscillation instead
    # of a clean one-shot offload/restore.
    #
    # IMPORTANT: scripts/apt_serve/fake_scheduler_worker.py's is_relaxed
    # heuristic falls back to `slo_deadline - timestamp > 5.0` for any
    # class_id other than the literal string "relaxed_long" (which only
    # generate_apt_serve_target_workload produces). relaxed_slack must
    # clear that 5.0 threshold or the hidden-tier mechanism never engages
    # at all for this cohort -- silently turning a "cache opportunity"
    # regime into a no-op. tight_slack must stay under it so the short
    # cohort is never mistaken for relaxed.
    long_fraction = 0.3
    if cache_use_structure == "thrash_risk":
        relaxed_slack = 6.0   # just above the 5.0 fallback threshold
        tight_slack = 4.5     # just below it -- narrow gap, both engaged
    else:
        relaxed_slack = 15.0
        tight_slack = 0.5

    for i in range(n_requests):
        at = float(arrival_times[i])
        is_long = rng.random() < long_fraction
        base_prompt, base_output = _length_pair(rng, length_pattern)
        if is_long:
            prompt_tokens = max(base_prompt, 500)
            output_tokens = max(16, min(base_output, 40))
            deadline = at + relaxed_slack
            priority = 1.0
            class_id = f"{cache_use_structure}_long"
        else:
            prompt_tokens = min(base_prompt, 200)
            output_tokens = max(8, min(base_output, 24))
            deadline = at + tight_slack
            priority = 3.0
            class_id = f"{cache_use_structure}_short"
        requests.append(Request(
            request_id=i + 1, arrival_time=at, prompt_tokens=prompt_tokens,
            predicted_output_tokens=output_tokens, actual_output_tokens=output_tokens,
            slo_deadline=deadline, priority=priority, class_id=class_id,
        ))
    return requests

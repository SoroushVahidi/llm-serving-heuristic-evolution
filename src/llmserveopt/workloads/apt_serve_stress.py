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

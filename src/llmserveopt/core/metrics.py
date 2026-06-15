"""
Metric computation from simulation results.

Phase 1.5 additions
-------------------
* TTFT (Time to First Token): first_token_time - arrival_time
* TPOT (Time Per Output Token): mean inter-token latency after first token
* prefill_delay: admission_time → first_token_time
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .types import CompletedRequest


@dataclass
class RunMetrics:
    policy_name: str
    workload_tag: str
    seed: int

    # Completion counts
    num_completed: int = 0
    num_dropped: int = 0
    num_slo_violated: int = 0

    # End-to-end latency (arrival → completion)
    mean_latency: float = float("nan")
    median_latency: float = float("nan")
    p95_latency: float = float("nan")
    p99_latency: float = float("nan")
    max_latency: float = float("nan")

    # Queuing delay (arrival → admission)
    mean_queuing_delay: float = float("nan")
    p95_queuing_delay: float = float("nan")

    # TTFT — Time To First Token (Phase 1.5; NaN when prefill not modelled)
    mean_ttft: float = float("nan")
    p95_ttft: float = float("nan")
    p99_ttft: float = float("nan")

    # TPOT — Time Per Output Token (Phase 1.5)
    mean_tpot: float = float("nan")
    p95_tpot: float = float("nan")

    # Prefill delay (admission → first token; Phase 1.5)
    mean_prefill_delay: float = float("nan")
    p95_prefill_delay: float = float("nan")

    # SLO
    slo_violation_rate: float = float("nan")

    # Weighted goodput — primary selector/evolution objective
    # = sum(priority_i * 1[met SLO_i]) / sum(priority_i)
    # Uses request.priority as weight (default 1.0 if priority == 0).
    weighted_goodput: float = float("nan")

    # Throughput
    request_throughput: float = float("nan")
    token_throughput: float = float("nan")

    # GPU utilization proxy
    mean_gpu_utilization: float = float("nan")
    total_gpu_busy_steps: int = 0
    mean_active_batch_size: float = float("nan")

    # Policy overhead
    total_policy_time_s: float = 0.0
    mean_policy_time_s: float = float("nan")

    # Simulation span
    wall_clock_s: float = float("nan")
    sim_duration: float = float("nan")


def compute_metrics(
    completed: List[CompletedRequest],
    dropped: List,
    sim_duration: float,
    gpu_utilization_history: List[float],
    active_batch_history: List[float],
    policy_name: str = "unknown",
    workload_tag: str = "unknown",
    seed: int = 0,
    policy_decision_times: Optional[List[float]] = None,
    wall_clock_s: float = float("nan"),
    idle_steps_skipped: int = 0,
) -> RunMetrics:
    m = RunMetrics(policy_name=policy_name, workload_tag=workload_tag, seed=seed)
    m.num_completed = len(completed)
    m.num_dropped = len(dropped)
    m.sim_duration = sim_duration
    m.wall_clock_s = wall_clock_s

    if completed:
        latencies  = np.array([c.latency       for c in completed], dtype=float)
        qdelays    = np.array([c.queuing_delay  for c in completed], dtype=float)
        violations = np.array([c.slo_violated   for c in completed], dtype=bool)

        m.mean_latency   = float(np.mean(latencies))
        m.median_latency = float(np.median(latencies))
        m.p95_latency    = float(np.percentile(latencies, 95))
        m.p99_latency    = float(np.percentile(latencies, 99))
        m.max_latency    = float(np.max(latencies))

        m.mean_queuing_delay = float(np.mean(qdelays))
        m.p95_queuing_delay  = float(np.percentile(qdelays, 95))

        m.num_slo_violated  = int(np.sum(violations))
        m.slo_violation_rate = float(np.mean(violations))

        # Weighted goodput: priority-weighted SLO-met rate.
        # Use priority as weight; fall back to 1.0 when priority is 0.
        weights = np.array(
            [c.request.priority if c.request.priority > 0 else 1.0 for c in completed],
            dtype=float,
        )
        met = (~violations).astype(float)
        total_weight = float(np.sum(weights))
        m.weighted_goodput = float(np.dot(weights, met) / total_weight) if total_weight > 0 else 0.0

        total_output_tokens = sum(c.request.actual_output_tokens for c in completed)
        if sim_duration > 0:
            m.request_throughput = m.num_completed / sim_duration
            m.token_throughput   = total_output_tokens / sim_duration

        # TTFT / TPOT — only defined when first_token_time was recorded
        ttfts = np.array([c.ttft for c in completed], dtype=float)
        tpots = np.array([c.tpot for c in completed], dtype=float)
        pfds  = np.array([c.prefill_delay for c in completed], dtype=float)

        valid_ttft = ttfts[~np.isnan(ttfts)]
        valid_tpot = tpots[~np.isnan(tpots)]
        valid_pfd  = pfds[~np.isnan(pfds)]

        if len(valid_ttft) > 0:
            m.mean_ttft = float(np.mean(valid_ttft))
            m.p95_ttft  = float(np.percentile(valid_ttft, 95))
            m.p99_ttft  = float(np.percentile(valid_ttft, 99))
        if len(valid_tpot) > 0:
            m.mean_tpot = float(np.mean(valid_tpot))
            m.p95_tpot  = float(np.percentile(valid_tpot, 95))
        if len(valid_pfd) > 0:
            m.mean_prefill_delay = float(np.mean(valid_pfd))
            m.p95_prefill_delay  = float(np.percentile(valid_pfd, 95))

    if gpu_utilization_history:
        total_steps = len(gpu_utilization_history) + idle_steps_skipped
        m.mean_gpu_utilization = (
            float(np.sum(gpu_utilization_history) / total_steps) if total_steps else float("nan")
        )
        m.total_gpu_busy_steps = int(np.sum(np.array(gpu_utilization_history) > 0))

    if active_batch_history:
        total_steps_b = len(active_batch_history) + idle_steps_skipped
        m.mean_active_batch_size = (
            float(np.sum(active_batch_history) / total_steps_b) if total_steps_b else float("nan")
        )

    if policy_decision_times:
        m.total_policy_time_s = sum(policy_decision_times)
        m.mean_policy_time_s  = m.total_policy_time_s / len(policy_decision_times)

    return m


def metrics_to_dict(m: RunMetrics) -> Dict:
    return {
        "policy":                   m.policy_name,
        "workload":                  m.workload_tag,
        "seed":                      m.seed,
        "num_completed":             m.num_completed,
        "num_dropped":               m.num_dropped,
        "num_slo_violated":          m.num_slo_violated,
        "mean_latency":              _fmt(m.mean_latency),
        "median_latency":            _fmt(m.median_latency),
        "p95_latency":               _fmt(m.p95_latency),
        "p99_latency":               _fmt(m.p99_latency),
        "max_latency":               _fmt(m.max_latency),
        "mean_queuing_delay":        _fmt(m.mean_queuing_delay),
        "p95_queuing_delay":         _fmt(m.p95_queuing_delay),
        "mean_ttft":                 _fmt(m.mean_ttft),
        "p95_ttft":                  _fmt(m.p95_ttft),
        "p99_ttft":                  _fmt(m.p99_ttft),
        "mean_tpot":                 _fmt(m.mean_tpot),
        "p95_tpot":                  _fmt(m.p95_tpot),
        "mean_prefill_delay":        _fmt(m.mean_prefill_delay),
        "p95_prefill_delay":         _fmt(m.p95_prefill_delay),
        "slo_violation_rate":        _fmt(m.slo_violation_rate),
        "weighted_goodput":          _fmt(m.weighted_goodput),
        "request_throughput":        _fmt(m.request_throughput),
        "token_throughput":          _fmt(m.token_throughput),
        "mean_gpu_utilization":      _fmt(m.mean_gpu_utilization),
        "total_gpu_busy_steps":      m.total_gpu_busy_steps,
        "mean_active_batch_size":    _fmt(m.mean_active_batch_size),
        "total_policy_time_s":       _fmt(m.total_policy_time_s),
        "mean_policy_time_s":        _fmt(m.mean_policy_time_s),
        "sim_duration":              _fmt(m.sim_duration),
        "wall_clock_s":              _fmt(m.wall_clock_s),
    }


def _fmt(v: float) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(v, 6) if isinstance(v, float) else v

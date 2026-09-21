"""Shared request/GPU-config construction helpers for Policy Separation
Dataset v1 templates. Mirrors the house style of
selector/dataset_v2/frontier_workload_families.py's `_req` helper and
selector/dataset_v2/scenario_redesign.py's `scarcity_gpu`/`service_model`
helpers — kept local to this package because PSD scenarios need explicit
per-request control (deterministic arrival offsets, hand-set deadlines,
deliberate prediction error) that the distributional generators in
workloads/synthetic.py are not built for.
"""
from __future__ import annotations

from typing import Optional

from ..core.types import GPUConfig, Request

#: Large slack used whenever a template does not care about SLO pressure
#: (keeps slo_violation_rate ~0 so it doesn't confound non-EDF families).
NO_PRESSURE_SLACK = 1_000.0


def req(
    request_id: int,
    arrival_time: float,
    prompt_tokens: int,
    predicted_output_tokens: int,
    *,
    actual_output_tokens: Optional[int] = None,
    slo_deadline: Optional[float] = None,
    priority: float = 1.0,
    class_id: str = "default",
) -> Request:
    """Build one Request with sane, documented defaults.

    `actual_output_tokens` defaults to `predicted_output_tokens` (accurate
    prediction) unless a template deliberately wants prediction error —
    at which point the divergence is the mechanism under test, not
    leakage (policies only ever observe `predicted_output_tokens`).
    `slo_deadline` defaults to `arrival_time + NO_PRESSURE_SLACK` unless a
    template deliberately wants deadline pressure.
    """
    actual = predicted_output_tokens if actual_output_tokens is None else actual_output_tokens
    deadline = arrival_time + NO_PRESSURE_SLACK if slo_deadline is None else slo_deadline
    return Request(
        request_id=request_id,
        arrival_time=arrival_time,
        prompt_tokens=prompt_tokens,
        predicted_output_tokens=predicted_output_tokens,
        actual_output_tokens=actual,
        slo_deadline=deadline,
        priority=priority,
        class_id=class_id,
    )


def generous_gpu(
    *,
    max_active_sequences: int = 32,
    max_batch_tokens: int = 32,
    max_kv_tokens: int = 200_000,
) -> GPUConfig:
    """A GPU with capacity far above any template's demand, so admission
    ordering/priority — not resource scarcity — is the thing separating
    policies. Used by the FCFS/SJF/EDF/fairness families."""
    return GPUConfig(
        gpu_id=0,
        max_active_sequences=max_active_sequences,
        max_batch_tokens=max_batch_tokens,
        max_kv_tokens=max_kv_tokens,
    )


def kv_scarce_gpu(
    *,
    max_kv_tokens: int,
    max_active_sequences: int = 16,
    max_batch_tokens: int = 16,
) -> GPUConfig:
    """A GPU whose KV capacity is the binding constraint — used by the
    cache/KV-aware family so `kv_constrained_online`'s admission-reserve
    logic actually has pressure to react to."""
    return GPUConfig(
        gpu_id=0,
        max_active_sequences=max_active_sequences,
        max_batch_tokens=max_batch_tokens,
        max_kv_tokens=max_kv_tokens,
    )


def apt_serve_dual_tier_gpu(
    *,
    max_kv_tokens: int = 20_000,
    hidden_cache_capacity_blocks: int = 4_000,
    cache_switch_latency: float = 0.05,
    hidden_restore_latency: float = 0.02,
    max_active_sequences: int = 16,
    max_batch_tokens: int = 16,
) -> GPUConfig:
    """A GPU with Apt-Serve's dual-tier hidden cache enabled — the only
    genuine cache-transition-cost mechanism in the simulator (see design
    doc section 4, family E)."""
    return GPUConfig(
        gpu_id=0,
        max_active_sequences=max_active_sequences,
        max_batch_tokens=max_batch_tokens,
        max_kv_tokens=max_kv_tokens,
        hybrid_cache_enabled=True,
        hidden_cache_capacity_blocks=hidden_cache_capacity_blocks,
        cache_switch_latency=cache_switch_latency,
        hidden_restore_latency=hidden_restore_latency,
    )

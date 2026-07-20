"""
Selector Dataset v2 -- calibrated targeted pilot (Option B scope).

Follow-up to docs/selector_v2_faithful_baseline_scope_audit.md's
`SELECTOR_SCOPE_DECISION = OPTION B`: train Selector v2 over the 8
historical monolithic policies only; the 3 faithful external baselines
(`vllm_faithful`, `sarathi_faithful`, `vllm_chunked_prefill_faithful`) are
never selector actions here -- they are evaluated separately, later, per
`docs/external_baseline_integration.md`'s Protocol C.

This module provides the generation building blocks; the adaptive
retained-window loop, checkpointing, and CLI live in
scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py.

SLO calibration: uniformly reference-ServiceModel-calibrated (see
slo_calibration.py) for EVERY window (synthetic families A-E and
real-trace family F alike) via `calibrate_window_e2e`, at a single
policy-independent reference service model -- never the obsolete fixed
`slo_deadline=1000.0`, and never derived from any policy's own achieved
latency. Real-trace windows go through `transform_requests` first for
time-scale/burst/prediction-noise diversity (its own `slo_scale` output is
irrelevant and always overwritten immediately after by
`calibrate_window_e2e`, kept only for compatibility with `transform_requests`'s
signature).

Newer-time-slice provenance: BurstGPT/Azure processed JSONL files are
chronologically sorted by `arrival_time` (verified at authoring time, not
assumed). Each file is partitioned into a HISTORICAL pool (first
`1 - OOD_RESERVED_FRACTION` of rows, the range the prior 910-window pilot's
seeds 0-9 already sampled from) and an OOD_RESERVED pool (the last
`OOD_RESERVED_FRACTION` of rows -- genuinely later, non-overlapping,
disjoint by row-index construction, not by seed-collision luck). No
calendar date exists in these sources beyond relative trace time, so "newer"
here means "later in the trace's own relative time axis," exactly as
instructed when no calendar date is available.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...core.types import GPUConfig, Request
from ...simulator.service_model import ServiceModel
from ...workloads.trace_io_extended import load_extended_jsonl  # noqa: F401 (re-exported)
from .builder import run_candidate_policy_on_window
from .frontier_workload_families import FAMILY_GENERATORS
from .schema import PolicyOutcomeVector
from .scenario_redesign import (
    DISCRIMINATIVE_POOL,
    REPRESENTATIVE_POOL,
    scarcity_gpu,
    service_model as _service_model_factory,
    transform_requests,
)
from .slo_calibration import calibrate_window_e2e

#: SELECTOR_SCOPE_DECISION = OPTION B (docs/selector_v2_faithful_baseline_scope_audit.md).
#: Exactly the 8 historical monolithic policies the 910-window SLO-calibrated
#: search already showed real, robust, oracle-headroom-backed ANWG
#: specialization for. Faithful baselines are deliberately excluded --
#: never add them here without a new, separately-documented scope decision.
CANDIDATE_POLICIES: Tuple[str, ...] = (
    "fifo",
    "edf",
    "scorpio_style_slo_guard",
    "admission_control",
    "weighted_shortest_processing",
    "estimated_service_time_first",
    "best_fit",
    "multi_bin_batching",
)

#: Excluded by the Option B scope decision -- external evaluation-only,
#: never a selector action in this pilot. Listed for documentation /
#: assertion purposes only, never instantiated by this module.
EXCLUDED_FAITHFUL_BASELINES: Tuple[str, ...] = (
    "vllm_faithful",
    "vllm_chunked_prefill_faithful",
    "sarathi_faithful",
)

#: Cross-topology reference baselines, out of scope for this monolithic
#: pilot entirely (see docs/external_baseline_integration.md Protocol C).
EXCLUDED_CROSS_TOPOLOGY_BASELINES: Tuple[str, ...] = (
    "distserve_faithful",
    "tetriinfer_paper_reimplementation",
    "llumnix_faithful",
)

REAL_TRACE_BASE_FILES: Tuple[Tuple[str, str, str], ...] = (
    ("burstgpt_scaled_moderate", "burstgpt", "data/processed/burstgpt/burstgpt_scaled_moderate_10k.jsonl"),
    ("burstgpt_scaled_high", "burstgpt", "data/processed/burstgpt/burstgpt_scaled_high_10k.jsonl"),
    ("azure_2023_code", "azure_llm_2023", "data/processed/azure/azure_llm_2023_code.jsonl"),
    ("azure_2023_conv", "azure_llm_2023", "data/processed/azure/azure_llm_2023_conv.jsonl"),
)

#: Last 15% of each chronologically-sorted trace file, reserved exclusively
#: for OOD_TEST -- disjoint by row-index construction from the HISTORICAL
#: pool the prior pilot's seeds 0-9 sampled from (verified: those seeds'
#: offsets covered up to row ~8980/9229, ~8063/8819, ~17131/19366 -- all
#: below this cut for their respective files, but the reservation below is
#: enforced structurally, not by re-checking that fact).
OOD_RESERVED_FRACTION = 0.15

REAL_TRACE_MAX_REQUESTS = 144

# name, pool_source(REPRESENTATIVE/DISCRIMINATIVE), bottleneck_class, time_scale, noise, bias, burst_amp
REAL_TRACE_TRANSFORMS: Tuple[Tuple[str, str, str, float, float, float, float], ...] = (
    ("representative", REPRESENTATIVE_POOL, "real_trace_representative", 1.0, 0.0, 1.0, 1.0),
    ("compressed_tight", DISCRIMINATIVE_POOL, "admission_pressure", 0.08, 0.2, 1.0, 1.0),
    ("burst_kv", DISCRIMINATIVE_POOL, "kv_pressure", 0.12, 0.25, 1.0, 5.0),
    ("noise_underpredict", DISCRIMINATIVE_POOL, "prediction_noise", 0.12, 0.7, 0.55, 2.5),
)

HISTORICAL_POOL = "historical"
OOD_RESERVED_POOL = "ood_reserved"


def _execution_service_model(budget: int, chunk: int) -> ServiceModel:
    """Shared execution service model for all 8 CANDIDATE_POLICIES -- unlike
    the faithful-baseline frontier search, none of these 8 need a
    policy-specific `decode_first` override (that override existed solely
    for `vllm_chunked_prefill_faithful`, which is out of scope here)."""
    return ServiceModel(
        enable_prefill_modeling=True, decode_first=True,
        enable_decode_prefill_contention=True,
        step_token_budget=budget, max_prefill_chunk_tokens=chunk,
    )


def _reference_service_model(budget: int, chunk: int) -> ServiceModel:
    """Policy-independent reference model used ONLY for SLO calibration
    (slo_calibration.reference_latency) -- never used to execute a
    simulation. Mirrors selector_v2_slo_calibrated_frontier_search.py's
    `_default_service_model(prefill=True, ...)` usage exactly."""
    return _service_model_factory(prefill=True, step_token_budget=budget, max_prefill_chunk_tokens=chunk)


@dataclass
class CandidateWindow:
    """One not-yet-evaluated window: requests already time/scale-transformed
    but NOT yet SLO-calibrated (calibration is applied uniformly by the
    caller so every window -- synthetic or real-trace -- goes through the
    exact same code path)."""
    group_key: str               # leakage-safe split group (ancestor-aware)
    dataset_family: str          # "controlled_stress" | "real_trace"
    source_trace: str            # "synthetic" | "burstgpt" | "azure_llm_2023"
    shape: str
    requests: List[Request]
    budget: int
    chunk: int
    max_kv_tokens: int
    max_active_sequences: int
    time_slice_pool: str         # "synthetic" | "historical" | "ood_reserved"
    time_slice_row_range: Optional[Tuple[int, int]] = None
    time_slice_arrival_range: Optional[Tuple[float, float]] = None
    request_plan_ancestor_id: str = ""


def synthetic_candidate_window(shape: str, rng: random.Random, window_idx: int) -> CandidateWindow:
    w = FAMILY_GENERATORS[shape](rng)
    return CandidateWindow(
        group_key=f"synthetic__{shape}__{window_idx}",
        dataset_family="controlled_stress",
        source_trace="synthetic",
        shape=w["shape"],
        requests=w["requests"],
        budget=w["budget"], chunk=w["chunk"],
        max_kv_tokens=w["max_kv_tokens"], max_active_sequences=w["max_active_sequences"],
        time_slice_pool="synthetic",
        request_plan_ancestor_id=f"synthetic__{shape}",
    )


def pool_row_range(n_rows: int, pool: str) -> Tuple[int, int]:
    """Public: the [lo, hi) row-index range of `pool` within a file of
    `n_rows` chronologically-sorted rows -- exposed for provenance
    reporting as well as internal slicing."""
    cut = int(round(n_rows * (1.0 - OOD_RESERVED_FRACTION)))
    if pool == HISTORICAL_POOL:
        return 0, cut
    if pool == OOD_RESERVED_POOL:
        return cut, n_rows
    raise ValueError(f"Unknown pool: {pool!r}")


def _load_pool_slice(
    path: Path, pool: str, seed: int, max_requests: int = REAL_TRACE_MAX_REQUESTS,
) -> Tuple[List[Request], Tuple[int, int]]:
    reqs, _metadata = load_extended_jsonl(path)
    n = len(reqs)
    lo, hi = pool_row_range(n, pool)
    pool_size = hi - lo
    if pool_size <= max_requests:
        return reqs[lo:hi], (lo, hi)
    start = lo + (seed * 7919) % (pool_size - max_requests + 1)
    end = start + max_requests
    return reqs[start:end], (start, end)


def real_trace_candidate_window(
    root: Path, base_name: str, source: str, rel_path: str,
    transform_name: str, pool: str, seed: int,
) -> Optional[CandidateWindow]:
    path = root / rel_path
    if not path.exists():
        return None
    transform = next((t for t in REAL_TRACE_TRANSFORMS if t[0] == transform_name), None)
    if transform is None:
        raise ValueError(f"Unknown real-trace transform: {transform_name!r}")
    _name, scenario_pool, bottleneck_class, time_scale, noise, bias, burst_amp = transform

    raw_slice, row_range = _load_pool_slice(path, pool, seed)
    if not raw_slice:
        return None
    transformed = transform_requests(
        raw_slice, time_scale=time_scale, slo_scale=1.0,  # slo_scale is a no-op: overwritten by calibrate_window_e2e
        prediction_noise_rel=noise, prediction_bias=bias, burst_amplification=burst_amp, seed=seed,
    )
    if not transformed:
        return None
    arrival_range = (raw_slice[0].arrival_time, raw_slice[-1].arrival_time)

    gpu = scarcity_gpu(max_active_sequences=8, max_batch_tokens=8, max_kv_tokens=7000)[0]
    budget, chunk = 512, 512
    return CandidateWindow(
        group_key=f"real_trace__{base_name}__{transform_name}__{pool}",
        dataset_family="real_trace",
        source_trace=source,
        shape=f"real_trace_stress__{base_name}__{transform_name}",
        requests=transformed,
        budget=budget, chunk=chunk,
        max_kv_tokens=gpu.max_kv_tokens, max_active_sequences=gpu.max_active_sequences,
        time_slice_pool=pool,
        time_slice_row_range=row_range,
        time_slice_arrival_range=arrival_range,
        request_plan_ancestor_id=f"real_trace__{base_name}",
    )


def all_real_trace_combinations() -> List[Tuple[str, str, str, str, str]]:
    """Every (base_name, source, rel_path, transform_name, pool) combination
    -- 4 base files x 4 transforms x 2 pools = 32 groups."""
    combos = []
    for base_name, source, rel_path in REAL_TRACE_BASE_FILES:
        for transform_name, *_rest in REAL_TRACE_TRANSFORMS:
            for pool in (HISTORICAL_POOL, OOD_RESERVED_POOL):
                combos.append((base_name, source, rel_path, transform_name, pool))
    return combos


def calibrate_candidate_window(window: CandidateWindow, multiplier: float) -> List[Request]:
    ref_model = _reference_service_model(window.budget, window.chunk)
    return calibrate_window_e2e(window.requests, ref_model, multiplier)


def run_window_all_candidates(
    window: CandidateWindow, calibrated_requests: List[Request], seed: int, workload_tag: str,
    drain_steps: int = 20_000,
) -> Optional[List[PolicyOutcomeVector]]:
    """Runs every one of the 8 CANDIDATE_POLICIES on `calibrated_requests`.
    Returns None (window NOT retained) unless ALL 8 succeed -- this pilot's
    retention criterion, so every retained window's stored outcome vector is
    genuinely complete, never partially populated."""
    gpu_configs = [GPUConfig(
        0, max_active_sequences=window.max_active_sequences,
        max_batch_tokens=1_000_000, max_kv_tokens=window.max_kv_tokens,
    )]
    sm = _execution_service_model(window.budget, window.chunk)
    outcomes: List[PolicyOutcomeVector] = []
    for pname in CANDIDATE_POLICIES:
        try:
            outcome = run_candidate_policy_on_window(
                pname, calibrated_requests, gpu_configs, sm,
                workload_tag=workload_tag, seed=seed, drain_steps=drain_steps,
            )
        except Exception:
            return None
        outcomes.append(outcome)
    return outcomes

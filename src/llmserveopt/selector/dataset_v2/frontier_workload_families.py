"""Contention-frontier workload families beyond hog-plus-runner.

See docs/selector_v2_slo_calibrated_frontier_search.md section 5. The
prior frontier search (docs/selector_v2_contention_frontier_search.md)
covered only two shapes (`admission_reorder`, `hog_runner_staggered`,
kept here as families A and D respectively); this module adds four more
so admission-order/KV-pressure interactions are probed more broadly
before deciding whether any of them specializes under a calibrated SLO.

Every generator returns raw, UNCALIBRATED requests (`slo_deadline` left
at a throwaway placeholder) -- callers must run them through
`slo_calibration.calibrate_window_e2e` before evaluating any policy.
Family F (real-trace stress) is not generated here: it reuses
`scenario_redesign.local_real_trace_stress_specs`/`transform_requests`
directly, which already preserves BurstGPT/Azure provenance and already
supports an `slo_scale` multiplier -- see the search script for how it is
combined with the same calibration multiplier grid used for A-E.
"""
from __future__ import annotations

import random
from typing import Dict, List

from ...core.types import Request

UNCALIBRATED_PLACEHOLDER_SLACK = 1.0  # overwritten by slo_calibration before any policy runs


def _req(rid: int, arrival: float, prompt: int, output: int) -> Request:
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=arrival + UNCALIBRATED_PLACEHOLDER_SLACK, priority=1.0, class_id="frontier",
    )


def family_a_same_arrival_heterogeneous_cluster(rng: random.Random) -> Dict:
    """A: 2-6 requests, ALL at t=0.0 exactly, sizes drawn log-uniform over
    a wide range, with request-ID order INDEPENDENT of size order (ids
    0..n-1 assigned before shuffling sizes) so admission-order-reordering
    policies and strict-FCFS-tie-break policies are forced to disagree
    about who goes first."""
    n = rng.randint(2, 6)
    sizes = [int(round(rng.uniform(200, 15000))) for _ in range(n)]
    rng.shuffle(sizes)
    outputs = [rng.choice([1, 5, 20]) for _ in range(n)]
    budget = rng.choice([300, 512, 800, 1200, 2048])
    reqs = [_req(i, 0.0, sizes[i], outputs[i]) for i in range(n)]
    return dict(shape="same_arrival_heterogeneous_cluster", requests=reqs,
                budget=budget, chunk=512, max_kv_tokens=200_000, max_active_sequences=64,
                n_requests=n, size_range=(min(sizes), max(sizes)))


def family_b_closely_spaced_heterogeneous_cluster(rng: random.Random) -> Dict:
    """B: like A, but arrivals spread over a FEW step_sizes (not exactly
    simultaneous) with large/small requests explicitly interleaved in
    arrival order (not grouped) -- tests whether A's divergence needs
    exact simultaneity or survives near-simultaneity too."""
    n = rng.randint(3, 8)
    sizes = sorted([int(round(rng.uniform(200, 15000))) for _ in range(n)])
    # Interleave: largest, smallest, 2nd-largest, 2nd-smallest, ...
    interleaved = []
    lo, hi = 0, len(sizes) - 1
    while lo <= hi:
        interleaved.append(sizes[hi])
        hi -= 1
        if lo <= hi:
            interleaved.append(sizes[lo])
            lo += 1
    step_size = 0.001
    gap = rng.choice([1, 2, 3]) * step_size
    outputs = [rng.choice([1, 5, 20]) for _ in range(n)]
    budget = rng.choice([300, 512, 800, 1200, 2048])
    reqs = [_req(i, i * gap, interleaved[i], outputs[i]) for i in range(n)]
    return dict(shape="closely_spaced_heterogeneous_cluster", requests=reqs,
                budget=budget, chunk=512, max_kv_tokens=200_000, max_active_sequences=64,
                n_requests=n, arrival_gap=gap)


def family_c_admission_reorder_boundary(rng: random.Random) -> Dict:
    """C: sizes deliberately straddle the chunk/budget boundary -- some
    requests fit in exactly 1 chunk, some need 2, some need 3+ -- probing
    whether admission-order reordering interacts with WHERE a request
    sits relative to the chunking boundary (not just its raw size rank)."""
    chunk = 512
    budget = chunk + rng.choice([0, 64, 256])
    boundary_multiples = [1, 1, 2, 2, 3]  # weighted toward 1-2 chunk requests
    n = rng.randint(3, 6)
    sizes = []
    for _ in range(n):
        k = rng.choice(boundary_multiples)
        jitter = rng.randint(-40, 40)
        sizes.append(max(1, k * chunk + jitter))
    rng.shuffle(sizes)
    outputs = [rng.choice([1, 5, 20]) for _ in range(n)]
    reqs = [_req(i, 0.0, sizes[i], outputs[i]) for i in range(n)]
    return dict(shape="admission_reorder_boundary", requests=reqs,
                budget=budget, chunk=chunk, max_kv_tokens=200_000, max_active_sequences=64,
                n_requests=n, chunk_multiples=[round(s / chunk, 2) for s in sizes])


def family_d_long_prefill_overlap(rng: random.Random) -> Dict:
    """D: the original hog-plus-staggered-runners shape (root cause A --
    self-limiting per docs/selector_v2_contention_frontier_search.md),
    kept for continuity/comparison at the same scale as before."""
    n_hogs = rng.randint(1, 3)
    hog_prompt = rng.choice([2000, 4000, 8000, 12000, 20000])
    n_runners = rng.randint(2, 40)
    runner_output = rng.choice([5, 10, 20, 40, 80])
    budget = 512 + rng.choice([1, 2, 3, 5, 8, 16, 32, 64])
    staggered = rng.random() < 0.6
    reqs = [_req(i, 0.0, hog_prompt, 1) for i in range(n_hogs)]
    for i in range(n_runners):
        arrival = 0.001 * (i + 1) if staggered else rng.choice([0.001, 0.002, 0.005])
        reqs.append(_req(n_hogs + i, arrival, rng.randint(1, 40), runner_output))
    return dict(shape="long_prefill_overlap", requests=reqs,
                budget=budget, chunk=512, max_kv_tokens=200_000, max_active_sequences=64,
                n_hogs=n_hogs, hog_prompt=hog_prompt, n_runners=n_runners, staggered=staggered)


def family_e_kv_pressure_admission_order(rng: random.Random) -> Dict:
    """E: family A/C's heterogeneous-size, same-arrival-time construction
    combined with KV capacity tight enough to also force ADMISSION
    queueing (not just execution-budget contention) -- probes whether
    memory-capacity queueing interacts with or masks the admission-order-
    reordering mechanism."""
    n = rng.randint(3, 6)
    sizes = [int(round(rng.uniform(200, 8000))) for _ in range(n)]
    rng.shuffle(sizes)
    outputs = [rng.choice([1, 5, 20]) for _ in range(n)]
    budget = rng.choice([300, 512, 800])
    total_prompt = sum(sizes)
    kv_headroom = rng.uniform(1.05, 1.5)
    max_kv_tokens = int(total_prompt * kv_headroom)
    reqs = [_req(i, 0.0, sizes[i], outputs[i]) for i in range(n)]
    return dict(shape="kv_pressure_admission_order", requests=reqs,
                budget=budget, chunk=512, max_kv_tokens=max_kv_tokens, max_active_sequences=64,
                n_requests=n, kv_headroom=round(kv_headroom, 3))


FAMILY_GENERATORS = {
    "A_same_arrival_heterogeneous_cluster": family_a_same_arrival_heterogeneous_cluster,
    "B_closely_spaced_heterogeneous_cluster": family_b_closely_spaced_heterogeneous_cluster,
    "C_admission_reorder_boundary": family_c_admission_reorder_boundary,
    "D_long_prefill_overlap": family_d_long_prefill_overlap,
    "E_kv_pressure_admission_order": family_e_kv_pressure_admission_order,
}

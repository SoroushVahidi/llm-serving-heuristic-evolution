"""Boundary-refinement scenario families for the SECOND policy-separation
compute experiment (policy_separation_boundary_refinement_v1).

This module does NOT reinvent the three-case diagnostic's mechanisms; it
refines two of its generators around the specific transitions job 1170116
(docs/audits/policy_separation_three_case_v1_20260810.md) found:

  Study A: reuses `templates_three_case.case1_fcfs_convoy` unmodified (that
  function already grew a backward-compatible `max_active_sequences`
  parameter for this experiment) with a finer arrival-offset grid, to find
  exactly where the FCFS-vs-size-aware choice disappears.

  Study B: a new prediction-inversion generator, `case2_prediction_inversion_boundary`,
  identical in mechanism to `templates_three_case.case2_sjf_prediction_inversion`
  but parameterized by an explicit numeric `target_utilization` instead of a
  two-bucket moderate/high string, so load can be swept on a fine calibrated
  grid. The original function is left untouched (still used by the
  completed job 1170116 config for reproducibility) rather than generalized
  in place, since threading a numeric override through its "moderate"/"high"
  string API would change that function's public signature.

  Study C reuses `templates_three_case.case3_edf_overload` /
  `generate_case3_grid` completely unmodified -- the requested "slack"
  dimension (feasible/borderline/tight) is deliberately NOT implemented as a
  third independent generator axis. `overload_factor` already parameterizes
  deadline slack directly (window_s is inversely proportional to it -- see
  that function's docstring), so a separate slack knob would double-count
  the same mechanism and confound the two axes. Slack tier is instead
  derived post-hoc from overload_factor by the aggregation step (see
  scripts/run_policy_separation_boundary_refinement.py::slack_tier) and
  reported as a grouping label, not a scenario parameter. This is a
  documented scope decision, not an oversight.

Every scenario built here still uses only `ObservableRequest`-visible
fields (`predicted_output_tokens`, `slo_deadline`, arrival order) for any
quantity a policy can act on; `actual_output_tokens` is exposed only for
decode length / post-hoc metrics, exactly as in templates_three_case.py.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import GPUConfig
from .builders import req
from .schema import PolicySeparationScenario
from .templates_three_case import (
    CASE1_ACTIVE_SEQUENCES,
    CASE2_ACTIVE_SEQUENCES,
    CASE2_DEADLINE_FLOOR_S,
    CASE2_DEADLINE_MULTIPLIER,
    CASE2_HETEROGENEITY_BOUNDS,
    CASE2_N_JOBS,
    CASE2_STEP_SIZE,
    _kendall_tau_stat,
    case1_fcfs_convoy,
)

GENERATOR_VERSION = "boundary_refinement_v1"

STEP_SIZE = CASE2_STEP_SIZE  # == ServiceModel default step_size (0.001s); shared across studies


# ---------------------------------------------------------------------------
# Study A: FCFS convoy boundary refinement (grid construction only -- the
# per-scenario generator is templates_three_case.case1_fcfs_convoy, reused
# as-is, including its now-optional max_active_sequences parameter).
# ---------------------------------------------------------------------------

def generate_case1_boundary_grid(
    ratios: List[int],
    short_counts: List[int],
    offsets: List[float],
    seeds: List[int],
    max_active_sequences_values: List[int] = (CASE1_ACTIVE_SEQUENCES,),
) -> List[PolicySeparationScenario]:
    """Full (ratio, short_count, offset, seed, role) x max_active_sequences
    cross. Callers that only want the primary mas=1 grid should pass the
    default `max_active_sequences_values=(1,)`; the mas>1 sub-study adds
    extra `mas` values on top (see the boundary-refinement config, which
    restricts the >1 sweep to a single representative (ratio, short_count)
    to keep the added cost small, per the task's "only if inexpensive"
    guidance)."""
    scenarios = []
    for mas in max_active_sequences_values:
        for ratio in ratios:
            for n_short in short_counts:
                for offset in offsets:
                    for seed in seeds:
                        for role in ("stress", "control"):
                            scenarios.append(
                                case1_fcfs_convoy(ratio, n_short, offset, seed, role, max_active_sequences=mas)
                            )
    return scenarios


# ---------------------------------------------------------------------------
# Study B: prediction-inversion decision-boundary refinement
# ---------------------------------------------------------------------------

CASE2B_HYPOTHESIS = (
    "Refines templates_three_case.CASE2_HYPOTHESIS's moderate/high load "
    "bucketing into a calibrated numeric target_utilization grid spanning "
    "clearly-nonbinding to strongly-binding, crossed with a finer "
    "inversion_fraction grid, to locate the (load, inversion_fraction) "
    "boundary at which estimated_service_time_first/weighted_shortest_processing/"
    "shortest_output_first's advantage over fifo crosses zero, rather than "
    "only confirming that a crossing exists somewhere between the two "
    "original buckets."
)


def _spearman_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via Pearson correlation of ranks (no scipy
    dependency, consistent with this package's existing Kendall-tau helper).
    Average ranks are used for ties. Returns nan for degenerate input
    (n<2 or a constant array)."""
    n = len(a)
    if n < 2:
        return float("nan")

    def _avg_ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        sorted_x = x[order]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            avg_rank = (i + j) / 2.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    ra, rb = _avg_ranks(np.asarray(a, dtype=float)), _avg_ranks(np.asarray(b, dtype=float))
    std_a, std_b = np.std(ra), np.std(rb)
    if std_a == 0 or std_b == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def case2_prediction_inversion_boundary(
    target_utilization: float,
    heterogeneity: str,
    inversion_fraction: float,
    seed: int,
) -> PolicySeparationScenario:
    """One prediction-inversion boundary-refinement cell: same construction
    as case2_sjf_prediction_inversion (same job count, capacity, deadline
    formula) but load is specified directly as a numeric target utilization
    instead of a "moderate"/"high" bucket, so Study B can sweep a fine,
    capacity-derived load grid instead of only two points.

    role is "control" at inversion_fraction==0.0 (accurate prediction) and
    "stress" otherwise, matching case2_sjf_prediction_inversion's
    convention; pair_id groups by (heterogeneity, target_utilization) so
    every inversion level at a fixed (heterogeneity, load) can be compared
    back to that pair's inversion_fraction==0.0 control.
    """
    if heterogeneity not in CASE2_HETEROGENEITY_BOUNDS:
        raise ValueError(f"unknown heterogeneity {heterogeneity!r}")
    if target_utilization <= 0.0:
        raise ValueError(f"target_utilization must be > 0, got {target_utilization!r}")

    rng = np.random.default_rng(seed)
    low, high = CASE2_HETEROGENEITY_BOUNDS[heterogeneity]
    actual_out = rng.integers(low, high, size=CASE2_N_JOBS)
    mean_service_s = float(np.mean(actual_out)) * STEP_SIZE
    capacity_per_s = CASE2_ACTIVE_SEQUENCES / mean_service_s
    rate = target_utilization * capacity_per_s
    duration = CASE2_N_JOBS / rate
    arrivals = np.sort(rng.uniform(0.0, duration, size=CASE2_N_JOBS))
    prompts = rng.integers(50, 300, size=CASE2_N_JOBS)

    predicted_out = actual_out.copy()
    order = np.argsort(actual_out)
    n_pairs = int(round(inversion_fraction * CASE2_N_JOBS / 2))
    for k in range(n_pairs):
        lo_idx = order[k]
        hi_idx = order[CASE2_N_JOBS - 1 - k]
        predicted_out[lo_idx], predicted_out[hi_idx] = predicted_out[hi_idx], predicted_out[lo_idx]

    rank_agreement_kendall_tau = _kendall_tau_stat(actual_out.astype(float), predicted_out.astype(float))
    rank_agreement_spearman = _spearman_stat(actual_out.astype(float), predicted_out.astype(float))

    # Flat per-cell deadline slack (see templates_three_case's identical
    # design note): keeps the mechanism isolated to scheduling ORDER, not
    # deadline redistribution, across inversion_fraction at fixed
    # (heterogeneity, load, seed).
    flat_deadline_slack_s = max(
        CASE2_DEADLINE_FLOOR_S,
        CASE2_DEADLINE_MULTIPLIER * float(np.mean(predicted_out)) * STEP_SIZE,
    )

    requests = []
    for i in range(CASE2_N_JOBS):
        arrival_i = float(arrivals[i])
        predicted_i = int(max(1, predicted_out[i]))
        requests.append(
            req(
                i, arrival_i, int(prompts[i]), predicted_i,
                actual_output_tokens=int(max(1, actual_out[i])),
                slo_deadline=arrival_i + flat_deadline_slack_s,
                class_id=heterogeneity,
            )
        )

    role = "control" if inversion_fraction == 0.0 else "stress"
    params = dict(
        inversion_fraction=inversion_fraction,
        heterogeneity=heterogeneity,
        target_utilization=target_utilization,
        seed=seed,
        role=role,
        rank_agreement_kendall_tau=rank_agreement_kendall_tau,
        rank_agreement_spearman=rank_agreement_spearman,
    )
    pair_id = f"prediction_inversion_boundary.{heterogeneity}.util{target_utilization}"
    scenario_id = (
        f"case2b_prediction_inversion_boundary.{role}.{heterogeneity}."
        f"util{target_utilization}.inv{inversion_fraction}.s{seed}"
    )

    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family="prediction_inversion_boundary",
        template_name="prediction_inversion_boundary_grid",
        generator_version=GENERATOR_VERSION,
        seed=seed,
        params=params,
        requests=tuple(requests),
        gpu_configs=(GPUConfig(
            gpu_id=0,
            max_active_sequences=CASE2_ACTIVE_SEQUENCES,
            max_batch_tokens=CASE2_ACTIVE_SEQUENCES,
            max_kv_tokens=200_000,
        ),),
        target_policy_family="B_sjf_size_aware",
        target_mechanism="size_prediction_rank_inversion_decision_boundary",
        expected_qualitative_hypothesis=CASE2B_HYPOTHESIS,
        stress_control_relationship=role,
        pair_id=pair_id,
        changed_parameters=("inversion_fraction",),
    )


def generate_case2_boundary_grid(
    target_utilizations: List[float],
    heterogeneity_levels: List[str],
    inversion_fractions: List[float],
    seeds: List[int],
) -> List[PolicySeparationScenario]:
    scenarios = []
    for heterogeneity in heterogeneity_levels:
        for target_utilization in target_utilizations:
            for inversion_fraction in inversion_fractions:
                for seed in seeds:
                    scenarios.append(
                        case2_prediction_inversion_boundary(
                            target_utilization, heterogeneity, inversion_fraction, seed
                        )
                    )
    return scenarios

"""Three-case diagnostic scenario families for the Policy Separation Dataset
v1's FIRST compute experiment.

Scope note: this module implements only the three narrow, theory-grounded
mechanisms named by this task -- FCFS convoy/HOL blocking (case 1), SJF
size-prediction inversion (case 2), and EDF unsalvageable overload (case 3)
-- NOT the full 25-template/5-family Phase 1 corpus described in
docs/design/POLICY_SEPARATION_DATASET_V1.md. It reuses that document's
schema (`PolicySeparationScenario`) and builder helper (`req`) unmodified
rather than inventing a parallel schema, since this experiment is
understood to be the first stage of the same effort. Each case uses its
own GPUConfig (not builders.generous_gpu()) because the mandated primary
metric (arrival_normalized_weighted_goodput) requires genuine admission
contention and deadline pressure to be sensitive to scheduling order at
all -- see the per-case capacity/deadline comments below.

Every scenario is deterministic given (case, cell parameters, seed, role).
No template reads `Request.actual_output_tokens` when constructing anything
a policy can observe (`predicted_output_tokens`, `slo_deadline`, arrival
order) -- case 2's prediction inversion deliberately diverges
`predicted_output_tokens` from `actual_output_tokens`; this is the
mechanism under test, not leakage, since policies only ever see
`ObservableRequest.predicted_output_tokens`.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.types import GPUConfig, Request
from .builders import req
from .schema import PolicySeparationScenario

GENERATOR_VERSION = "three_case_v1"


def _slo_deadline(arrival: float, predicted_output_tokens: int, step_size: float, k: float, floor_s: float) -> float:
    """SLO deadline derived only from a policy-visible quantity
    (predicted_output_tokens, already exposed via ObservableRequest) so
    every template in this module can give the mandated primary metric
    (arrival_normalized_weighted_goodput) genuine sensitivity to
    scheduling/admission order without deriving anything a policy can see
    from actual_output_tokens (which would be oracle leakage)."""
    predicted_service_s = predicted_output_tokens * step_size
    return arrival + max(floor_s, k * predicted_service_s)


# ---------------------------------------------------------------------------
# Case 1: FCFS convoy / head-of-line blocking
# ---------------------------------------------------------------------------

CASE1_SHORT_BASE_OUTPUT = 20
CASE1_SHORT_PROMPT = 50
CASE1_LONG_PROMPT = 200
CASE1_STEP_SIZE = 0.001  # matches ServiceModel default; used only for deadline calibration
CASE1_DEADLINE_MULTIPLIER = 10.0
CASE1_DEADLINE_FLOOR_S = 0.05
# Forces strict single-slot serialization so a convoy effect can exist at
# all -- matches the existing fifo_counter_head_of_line_blocking catalog
# entry's simulator_requirements (configs/stress_tests/
# algorithm_stress_test_catalog.yaml: max_active_sequences=1), reused for
# consistency rather than re-derived. generous_gpu() (32 slots) would let
# every request in this grid's smaller cells run concurrently with no
# admission contention at all -- not a convoy test.
CASE1_ACTIVE_SEQUENCES = 1

CASE1_HYPOTHESIS = (
    "Size-aware scheduling (estimated_service_time_first / "
    "weighted_shortest_processing) materially outperforms fifo's "
    "arrival_normalized_weighted_goodput when one very large request "
    "arrives immediately before a burst of many small requests (the "
    "convoy/HOL-blocking regime), because FIFO admits the long job first "
    "and blocks the short burst behind it while size-aware policies admit "
    "the short burst first. This gap should shrink materially in the "
    "paired control, which uses the identical multiset of requests but "
    "reverses arrival order (short burst first, long job last)."
)


def case1_fcfs_convoy(
    ratio: int,
    n_short: int,
    offset: float,
    seed: int,
    role: str,
) -> PolicySeparationScenario:
    """One FCFS convoy stress cell or its reversed-order control.

    role="stress": long request arrives at t=0 with the lowest request_id;
    short burst arrives at t=offset (offset>0.0 => strictly after the long
    job is already queued/admitted; a tiny offset still gives a genuine
    near-synchronous regime). role="control": SAME multiset of requests,
    order reversed -- short burst arrives at t=0 with the lowest
    request_ids; long request arrives at t=offset with the highest
    request_id. Request-id assignment (not just arrival_time) is
    deliberately role-dependent: the simulator's own trace loader
    (Simulator.load_trace) does a stable sort by arrival_time only, so ties
    preserve input-list order -- reversing request_id order is what makes
    the pairing meaningful even in the offset~0 near-synchronous regime,
    not only the offset>0 regime. changed_parameters records this as
    "arrival_order" since request_id here is purely an ordering label, not
    a workload property.
    """
    if role not in ("stress", "control"):
        raise ValueError(f"role must be 'stress' or 'control', got {role!r}")

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.9, 1.1, size=n_short + 1)
    short_out = np.maximum(1, np.round(CASE1_SHORT_BASE_OUTPUT * jitter[:n_short])).astype(int)
    long_out = max(1, int(round(CASE1_SHORT_BASE_OUTPUT * ratio * jitter[n_short])))

    if role == "stress":
        long_arrival, short_arrival = 0.0, offset
        long_id, short_ids = 0, list(range(1, n_short + 1))
    else:
        long_arrival, short_arrival = offset, 0.0
        long_id, short_ids = n_short, list(range(0, n_short))

    requests: List[Request] = [
        req(
            long_id, long_arrival, CASE1_LONG_PROMPT, long_out,
            slo_deadline=_slo_deadline(long_arrival, long_out, CASE1_STEP_SIZE,
                                        CASE1_DEADLINE_MULTIPLIER, CASE1_DEADLINE_FLOOR_S),
            class_id="long",
        )
    ]
    for i in range(n_short):
        so = int(short_out[i])
        requests.append(
            req(
                short_ids[i], short_arrival, CASE1_SHORT_PROMPT, so,
                slo_deadline=_slo_deadline(short_arrival, so, CASE1_STEP_SIZE,
                                            CASE1_DEADLINE_MULTIPLIER, CASE1_DEADLINE_FLOOR_S),
                class_id="short",
            )
        )
    requests.sort(key=lambda r: (r.arrival_time, r.request_id))

    params = dict(ratio=ratio, n_short=n_short, offset=offset, seed=seed, role=role)
    pair_id = f"fcfs_convoy.ratio{ratio}.nshort{n_short}.offset{offset}"
    scenario_id = f"case1_fcfs_convoy.{role}.ratio{ratio}.nshort{n_short}.offset{offset}.s{seed}"

    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family="fcfs_convoy",
        template_name="convoy_long_first" if role == "stress" else "convoy_short_first_control",
        generator_version=GENERATOR_VERSION,
        seed=seed,
        params=params,
        requests=tuple(requests),
        gpu_configs=(GPUConfig(
            gpu_id=0,
            max_active_sequences=CASE1_ACTIVE_SEQUENCES,
            max_batch_tokens=CASE1_ACTIVE_SEQUENCES,
            max_kv_tokens=200_000,
        ),),
        target_policy_family="A_fcfs_vs_size_aware",
        target_mechanism="convoy_head_of_line_blocking",
        expected_qualitative_hypothesis=CASE1_HYPOTHESIS,
        stress_control_relationship=role,
        pair_id=pair_id,
        changed_parameters=("arrival_order",),
    )


def generate_case1_grid(
    ratios: List[int], short_counts: List[int], offsets: List[float], seeds: List[int],
) -> List[PolicySeparationScenario]:
    scenarios = []
    for ratio in ratios:
        for n_short in short_counts:
            for offset in offsets:
                for seed in seeds:
                    for role in ("stress", "control"):
                        scenarios.append(case1_fcfs_convoy(ratio, n_short, offset, seed, role))
    return scenarios


# ---------------------------------------------------------------------------
# Case 2: SJF / size-prediction inversion
# ---------------------------------------------------------------------------

CASE2_HETEROGENEITY_BOUNDS = {
    "moderate": (50, 200),
    "strong": (30, 600),
}
#: Calibrated empirically (not analytically derived): with 8 slots and
#: 50-85% utilization, an M/G/8-style queue's own averaging keeps queueing
#: delay too low relative to service time for any deadline multiplier in a
#: sane range to bind -- ANWG pinned at 1.0 for every policy regardless of
#: inversion_fraction. 3 slots reintroduces enough queueing pressure at the
#: same utilization targets for scheduling order to matter.
CASE2_ACTIVE_SEQUENCES = 3
CASE2_N_JOBS = 60
CASE2_STEP_SIZE = 0.001  # matches ServiceModel default; used only for capacity calibration
CASE2_DEADLINE_MULTIPLIER = 1.5
CASE2_DEADLINE_FLOOR_S = 0.02

CASE2_HYPOTHESIS = (
    "Size-aware policies (estimated_service_time_first, "
    "weighted_shortest_processing, shortest_output_first) improve "
    "arrival_normalized_weighted_goodput over fifo when predicted_output_tokens "
    "preserves the true service-size ranking (inversion_fraction=0), and this "
    "advantage shrinks or reverses as an increasing fraction of "
    "predicted-vs-actual rankings are deliberately inverted, because these "
    "policies schedule purely off predicted_output_tokens. fifo (and, if "
    "included, aging_priority) should be comparatively insensitive to "
    "prediction inversion since neither reads predicted_output_tokens for "
    "ordering."
)


def _kendall_tau_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Simple O(n^2) Kendall tau-a (concordant-discordant)/total, no ties
    correction beyond the natural tie handling -- fine for the small n used
    here and avoids a scipy dependency in the hot scenario-construction path."""
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa == 0 or sb == 0:
                continue
            if sa == sb:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan")
    return (concordant - discordant) / total


def case2_sjf_prediction_inversion(
    inversion_fraction: float,
    heterogeneity: str,
    load: str,
    seed: int,
) -> PolicySeparationScenario:
    """One SJF prediction-inversion cell.

    role is "control" when inversion_fraction==0.0 (accurate prediction,
    the positive case) and "stress" otherwise (adversarial rank inversion),
    all sharing one pair_id per (heterogeneity, load) so the runner can
    compare the accurate-prediction baseline against each inversion level.
    """
    if heterogeneity not in CASE2_HETEROGENEITY_BOUNDS:
        raise ValueError(f"unknown heterogeneity {heterogeneity!r}")
    if load not in ("moderate", "high"):
        raise ValueError(f"unknown load {load!r}")

    rng = np.random.default_rng(seed)
    low, high = CASE2_HETEROGENEITY_BOUNDS[heterogeneity]
    actual_out = rng.integers(low, high, size=CASE2_N_JOBS)
    mean_service_s = float(np.mean(actual_out)) * CASE2_STEP_SIZE
    capacity_per_s = CASE2_ACTIVE_SEQUENCES / mean_service_s
    target_util = 0.5 if load == "moderate" else 0.85
    rate = target_util * capacity_per_s
    duration = CASE2_N_JOBS / rate
    arrivals = np.sort(rng.uniform(0.0, duration, size=CASE2_N_JOBS))
    prompts = rng.integers(50, 300, size=CASE2_N_JOBS)

    # predicted_output_tokens starts equal to actual_output_tokens (accurate
    # prediction), then a fraction of the true-size rank order is inverted
    # by swapping predicted values between symmetric top/bottom ranks --
    # actual_output_tokens (ground truth, used only for simulator decode
    # length and post-hoc metrics) is never touched.
    predicted_out = actual_out.copy()
    order = np.argsort(actual_out)  # ascending true-size rank
    n_pairs = int(round(inversion_fraction * CASE2_N_JOBS / 2))
    for k in range(n_pairs):
        lo_idx = order[k]
        hi_idx = order[CASE2_N_JOBS - 1 - k]
        predicted_out[lo_idx], predicted_out[hi_idx] = predicted_out[hi_idx], predicted_out[lo_idx]

    rank_agreement_kendall_tau = _kendall_tau_stat(
        actual_out.astype(float), predicted_out.astype(float)
    )

    # Deadline slack is a FLAT per-cell constant (mean predicted_out * step_size
    # * multiplier), not per-request -- inversion only permutes which request
    # owns which predicted_output_tokens value, so the mean (and therefore this
    # deadline) is identical across every inversion_fraction at fixed
    # (heterogeneity, load, seed). This deliberately isolates the mechanism
    # under test to scheduling ORDER: if deadlines were instead derived from
    # each request's OWN (invertible) predicted_output_tokens, fifo's results
    # would drift with inversion_fraction too (via deadline redistribution)
    # even though fifo never reads predicted_output_tokens for ordering,
    # confounding the very comparison this template exists to make.
    flat_deadline_slack_s = max(
        CASE2_DEADLINE_FLOOR_S,
        CASE2_DEADLINE_MULTIPLIER * float(np.mean(predicted_out)) * CASE2_STEP_SIZE,
    )

    requests: List[Request] = []
    for i in range(CASE2_N_JOBS):
        arrival_i = float(arrivals[i])
        predicted_i = int(max(1, predicted_out[i]))
        requests.append(
            req(
                i,
                arrival_i,
                int(prompts[i]),
                predicted_i,
                actual_output_tokens=int(max(1, actual_out[i])),
                slo_deadline=arrival_i + flat_deadline_slack_s,
                class_id=heterogeneity,
            )
        )

    role = "control" if inversion_fraction == 0.0 else "stress"
    params = dict(
        inversion_fraction=inversion_fraction,
        heterogeneity=heterogeneity,
        load=load,
        seed=seed,
        role=role,
        rank_agreement_kendall_tau=rank_agreement_kendall_tau,
        target_utilization=target_util,
    )
    pair_id = f"sjf_inversion.{heterogeneity}.{load}"
    scenario_id = (
        f"case2_sjf_inversion.{role}.{heterogeneity}.{load}."
        f"inv{inversion_fraction}.s{seed}"
    )

    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family="sjf_prediction_inversion",
        template_name="prediction_inversion_grid",
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
        target_mechanism="size_prediction_rank_inversion",
        expected_qualitative_hypothesis=CASE2_HYPOTHESIS,
        stress_control_relationship=role,
        pair_id=pair_id,
        changed_parameters=("inversion_fraction",),
    )


def generate_case2_grid(
    inversion_fractions: List[float], heterogeneity_levels: List[str], load_levels: List[str],
    seeds: List[int],
) -> List[PolicySeparationScenario]:
    scenarios = []
    for heterogeneity in heterogeneity_levels:
        for load in load_levels:
            for inversion_fraction in inversion_fractions:
                for seed in seeds:
                    scenarios.append(
                        case2_sjf_prediction_inversion(inversion_fraction, heterogeneity, load, seed)
                    )
    return scenarios


# ---------------------------------------------------------------------------
# Case 3: EDF unsalvageable overload
# ---------------------------------------------------------------------------

CASE3_ACTIVE_SEQUENCES = 4
CASE3_N_JOBS = 30
CASE3_NORMAL_OUTPUT = 150
CASE3_IMPOSSIBLE_OUTPUT_MULTIPLIER = 3.0
CASE3_IMPOSSIBLE_EPSILON = 0.05  # deadline slack for "impossible" jobs (seconds)
CASE3_CONTROL_LOOSEN_FACTOR = 6.0
CASE3_STEP_SIZE = 0.001

CASE3_HYPOTHESIS = (
    "Pure deadline ordering (edf, least_laxity_first) performs competitively "
    "with admission/overload-aware policies (scorpio_style_slo_guard, "
    "admission_control) in the feasible control (deadlines loosened so total "
    "demand fits available capacity), but under unsalvageable overload -- a "
    "burst where some earliest-deadline jobs cannot meet their deadline "
    "regardless of scheduling AND consume enough service time that servicing "
    "them blindly starves salvageable jobs -- admission/overload-aware "
    "policies sometimes preserve more arrival_normalized_weighted_goodput "
    "than pure EDF/LLF, which keep servicing the doomed earliest-deadline "
    "jobs. This ordering is not assumed in advance; it is what the run is "
    "checking."
)


def case3_edf_overload(
    overload_factor: float,
    fraction_impossible: float,
    seed: int,
    role: str,
) -> PolicySeparationScenario:
    """One EDF unsalvageable-overload stress cell or its loosened-deadline
    control.

    Capacity calibration (see module docstring / design notes): with
    max_active_sequences == max_batch_tokens == C and the default
    (non-prefill-modeled) ServiceModel, each active slot advances exactly
    one decode token per step of CASE3_STEP_SIZE seconds, so C jobs of mean
    output length L complete in aggregate at rate C / (L * step_size)
    jobs/sec. overload_factor = required_service_seconds / (C * window),
    solved for `window` given the fixed job set below.
    """
    if role not in ("stress", "control"):
        raise ValueError(f"role must be 'stress' or 'control', got {role!r}")

    rng = np.random.default_rng(seed)
    n_impossible = int(round(fraction_impossible * CASE3_N_JOBS))
    n_normal = CASE3_N_JOBS - n_impossible

    normal_jitter = rng.uniform(0.85, 1.15, size=n_normal)
    normal_out = np.maximum(1, np.round(CASE3_NORMAL_OUTPUT * normal_jitter)).astype(int)
    impossible_out = np.maximum(
        1,
        np.round(CASE3_NORMAL_OUTPUT * CASE3_IMPOSSIBLE_OUTPUT_MULTIPLIER
                  * rng.uniform(0.85, 1.15, size=n_impossible)),
    ).astype(int)

    total_required_s = float(np.sum(normal_out) + np.sum(impossible_out)) * CASE3_STEP_SIZE
    window = total_required_s / (CASE3_ACTIVE_SEQUENCES * overload_factor)
    if role == "control":
        window *= CASE3_CONTROL_LOOSEN_FACTOR

    arrival_jitter = rng.uniform(0.0, 0.01, size=CASE3_N_JOBS)

    requests: List[Request] = []
    rid = 0
    for i in range(n_normal):
        arrival = float(arrival_jitter[rid])
        deadline_span = rng.uniform(0.3 * window, window)
        requests.append(
            req(
                rid, arrival, 50, int(normal_out[i]),
                slo_deadline=arrival + deadline_span,
                class_id="normal",
            )
        )
        rid += 1
    for i in range(n_impossible):
        arrival = float(arrival_jitter[rid])
        epsilon = CASE3_IMPOSSIBLE_EPSILON * (CASE3_CONTROL_LOOSEN_FACTOR if role == "control" else 1.0)
        requests.append(
            req(
                rid, arrival, 50, int(impossible_out[i]),
                slo_deadline=arrival + epsilon,
                class_id="impossible",
            )
        )
        rid += 1
    requests.sort(key=lambda r: (r.arrival_time, r.request_id))

    params = dict(
        overload_factor=overload_factor,
        fraction_impossible=fraction_impossible,
        seed=seed,
        role=role,
        window_s=window,
        n_normal=n_normal,
        n_impossible=n_impossible,
        total_required_service_s=total_required_s,
    )
    pair_id = f"edf_overload.of{overload_factor}.fi{fraction_impossible}"
    scenario_id = (
        f"case3_edf_overload.{role}.of{overload_factor}.fi{fraction_impossible}.s{seed}"
    )

    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family="edf_unsalvageable_overload",
        template_name="unsalvageable_overload" if role == "stress" else "loosened_deadline_control",
        generator_version=GENERATOR_VERSION,
        seed=seed,
        params=params,
        requests=tuple(requests),
        gpu_configs=(GPUConfig(
            gpu_id=0,
            max_active_sequences=CASE3_ACTIVE_SEQUENCES,
            max_batch_tokens=CASE3_ACTIVE_SEQUENCES,
            max_kv_tokens=200_000,
        ),),
        target_policy_family="C_edf_deadline_aware",
        target_mechanism="unsalvageable_overload_vs_admission_awareness",
        expected_qualitative_hypothesis=CASE3_HYPOTHESIS,
        stress_control_relationship=role,
        pair_id=pair_id,
        changed_parameters=("slo_deadline",),
    )


def generate_case3_grid(
    overload_factors: List[float], fraction_impossible_levels: List[float], seeds: List[int],
) -> List[PolicySeparationScenario]:
    scenarios = []
    for overload_factor in overload_factors:
        for fraction_impossible in fraction_impossible_levels:
            for seed in seeds:
                for role in ("stress", "control"):
                    scenarios.append(
                        case3_edf_overload(overload_factor, fraction_impossible, seed, role)
                    )
    return scenarios

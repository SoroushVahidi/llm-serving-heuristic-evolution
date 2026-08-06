"""Workload generators for the Algorithm Stress-Test Library.

One function per `stress_test_id` in
`configs/stress_tests/algorithm_stress_test_catalog.yaml` (catalog
entries not backed by a runnable generator here are explicitly stubbed
with `NotImplementedError` and a pointer to why -- see the vLLM-LTR/PARS
domain-shift counter-cases, which require a new offline scoring pass
against the real checkpoints, out of scope for this pass).

Every generator returns a `List[Request]` sorted by arrival time. Per the
catalog's `forbidden_oracle_inputs` field (always `[actual_output_tokens]`
here), `predicted_output_tokens` is the ONLY length signal any policy
under test may read — `actual_output_tokens` is set to genuine ground
truth (used for the simulator's own decode-length modeling and for
computing this project's usual hidden metrics), and misprediction bias is
introduced by making `predicted_output_tokens` diverge from
`actual_output_tokens`, never the reverse. This mirrors the existing
pattern in `src/llmserveopt/workloads/synthetic.py`'s `prediction_noise`
and `baselines/vtc/fairness_workloads.py`'s tenant-mapping convention —
new code here, not a modification of either.

Does not touch `benchmarks/canonical_suite/`, any `baselines/vtc/**`
file, or CC5/CC6 config/core files — see
`docs/research/algorithm_stress_tests/CONCURRENCY_SAFETY_20260805.md`'s
exclusion list.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from llmserveopt.core.types import Request
from llmserveopt.workloads.distributions import (
    bursty_arrivals,
    lognormal_tokens,
    poisson_arrivals,
    uniform_tokens,
)

_LOOSE_SLO = 1000.0


def _mk(rid: int, arrival: float, prompt: int, predicted_out: int, actual_out: int,
        slo_deadline: float = None, priority: float = 1.0, class_id: str = "default") -> Request:
    predicted_out = max(1, int(predicted_out))
    actual_out = max(1, int(actual_out))
    return Request(
        request_id=rid,
        arrival_time=float(arrival),
        prompt_tokens=max(1, int(prompt)),
        predicted_output_tokens=predicted_out,
        actual_output_tokens=actual_out,
        slo_deadline=float(slo_deadline) if slo_deadline is not None else float(arrival) + _LOOSE_SLO,
        priority=priority,
        class_id=class_id,
    )


def _sorted(reqs: List[Request]) -> List[Request]:
    return sorted(reqs, key=lambda r: r.arrival_time)


# ----------------------------------------------------------------------
# 1. FIFO / FCFS
# ----------------------------------------------------------------------

def fifo_target_homogeneous_low_contention(smoke: bool = False, seed: int = 0) -> List[Request]:
    n = 40 if smoke else 200
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=4.0, duration=n / 4.0)
    prompts = lognormal_tokens(rng, len(arrivals), mean=200, sigma=0.3, low=50, high=800)
    outputs = lognormal_tokens(rng, len(arrivals), mean=150, sigma=0.2, low=80, high=250)
    reqs = [
        _mk(i, a, p, o, o)
        for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))
    ]
    return _sorted(reqs)


def fifo_counter_head_of_line_blocking(smoke: bool = False, seed: int = 0) -> List[Request]:
    """All requests arrive at exactly t=0 (within the same first
    simulator step, step_size=0.001s default) so every policy sees the
    SAME simultaneous waiting_queue at the first admission decision --
    the only way ordering-vs-arrival-time genuinely diverges. A long
    request arriving strictly before any short one (as an earlier draft
    of this generator did) gets admitted before the short ones even
    exist, making every policy behave identically regardless of ordering
    -- not a HOL-blocking test at all, just a degenerate single-candidate
    admission. Fixed here; see docs/research/algorithm_stress_tests/
    STRESS_TEST_VALIDATION_20260805.md for the diagnosis."""
    n_long = 1 if smoke else 2
    n_short = 10 if smoke else 40
    reqs = []
    rid = 0
    for _ in range(n_long):
        reqs.append(_mk(rid, 0.0, 200, 2000, 2000, class_id="long"))
        rid += 1
    for i in range(n_short):
        reqs.append(_mk(rid, 0.0, 50, 20, 20, class_id="short"))
        rid += 1
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 2. SOF (shortest_output_first)
# ----------------------------------------------------------------------

def sof_target_mixed_lengths_accurate_prediction(smoke: bool = False, seed: int = 1) -> List[Request]:
    n = 40 if smoke else 200
    rng = np.random.default_rng(seed)
    long_fraction = 0.2
    arrivals = poisson_arrivals(rng, rate=6.0, duration=n / 6.0)
    reqs = []
    for i, a in enumerate(arrivals):
        is_long = rng.random() < long_fraction
        out = int(rng.normal(400, 40)) if is_long else int(rng.normal(20, 3))
        out = max(5, out)
        reqs.append(_mk(i, a, 150, out, out, class_id="long" if is_long else "short"))
    return _sorted(reqs)


def sof_counter_long_job_starvation(smoke: bool = False, seed: int = 1) -> List[Request]:
    """Calibrated empirically (see STRESS_TEST_VALIDATION_20260805.md):
    with a short request's own service time only ~0.015-0.02s (15-token
    output, 1 slot), the short-arrival rate must EXCEED the short-job
    SERVICE rate (~1/0.015s =~ 65/s) for a short backlog to remain
    persistently non-empty -- an 8/s arrival rate (this generator's first
    draft) is far below that service rate, so the short queue drains to
    empty between arrivals almost immediately, handing the long job an
    easy, early admission opportunity (measured: max_queuing_delay=0.045s
    regardless of how many short requests eventually arrive -- because
    almost none of them are ever concurrently queued together). Rate=100/s
    keeps at least one short request waiting essentially continuously."""
    n_short = 60 if smoke else 300
    rng = np.random.default_rng(seed)
    reqs = [_mk(0, 0.0, 150, 3000, 3000, class_id="long")]
    for i in range(3):
        reqs.append(_mk(i + 1, 0.0, 50, 15, 15, class_id="short"))
    arrivals = poisson_arrivals(rng, rate=100.0, duration=n_short / 100.0)
    for i, a in enumerate(arrivals[:n_short]):
        reqs.append(_mk(i + 4, a, 50, 15, 15, class_id="short"))
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 3. ESTF (estimated_service_time_first)
# ----------------------------------------------------------------------

def estf_target_accurate_alpha_beta_estimate(smoke: bool = False, seed: int = 2) -> List[Request]:
    n = 40 if smoke else 200
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=6.0, duration=n / 6.0)
    prompts = rng.integers(50, 1500, size=len(arrivals))
    outputs = rng.integers(20, 400, size=len(arrivals))
    reqs = [_mk(i, a, p, o, o) for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))]
    return _sorted(reqs)


def estf_counter_reasoning_prompt_length_misprediction(smoke: bool = False, seed: int = 2) -> List[Request]:
    n = 40 if smoke else 150
    misprediction_fraction = 0.15
    understate_factor = 8.0
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=6.0, duration=n / 6.0)
    prompts = rng.integers(50, 800, size=len(arrivals))
    actual_outputs = rng.integers(100, 600, size=len(arrivals))
    reqs = []
    for i, (a, p, actual) in enumerate(zip(arrivals, prompts, actual_outputs)):
        is_mispredicted = rng.random() < misprediction_fraction
        predicted = max(5, int(actual / understate_factor)) if is_mispredicted else int(actual)
        reqs.append(_mk(i, a, p, predicted, actual, class_id="reasoning" if is_mispredicted else "normal"))
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 4. WSP (weighted_shortest_processing)
# ----------------------------------------------------------------------

def wsp_target_priority_length_balance(smoke: bool = False, seed: int = 3) -> List[Request]:
    n = 40 if smoke else 200
    rng = np.random.default_rng(seed)
    priority_classes = [1.0, 3.0, 5.0]
    arrivals = poisson_arrivals(rng, rate=6.0, duration=n / 6.0)
    reqs = []
    for i, a in enumerate(arrivals):
        pr = rng.choice(priority_classes)
        prompt = int(rng.integers(50, 800))
        out = int(rng.integers(20, 400))
        reqs.append(_mk(i, a, prompt, out, out, priority=float(pr)))
    return _sorted(reqs)


def wsp_counter_priority_service_time_conflict(smoke: bool = False, seed: int = 3) -> List[Request]:
    """As in sof_counter_long_job_starvation: seed a few low-priority
    short requests at t=0 alongside the high-priority long ones so the
    ranking conflict is genuinely contested from the first admission
    decision, not trivially resolved by one side arriving alone."""
    n_low = 30 if smoke else 100
    rng = np.random.default_rng(seed)
    reqs = []
    rid = 0
    for _ in range(5):
        reqs.append(_mk(rid, 0.0, 200, 2000, 2000, priority=5.0, class_id="high_priority_long"))
        rid += 1
    for _ in range(3):
        reqs.append(_mk(rid, 0.0, 30, 10, 10, priority=1.0, class_id="low_priority_short"))
        rid += 1
    arrivals = poisson_arrivals(rng, rate=8.0, duration=n_low / 8.0)
    for a in arrivals[:n_low]:
        reqs.append(_mk(rid, a, 30, 10, 10, priority=1.0, class_id="low_priority_short"))
        rid += 1
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 5. EDF
# ----------------------------------------------------------------------

def edf_target_feasible_heterogeneous_deadlines(smoke: bool = False, seed: int = 4) -> List[Request]:
    n = 40 if smoke else 200
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=4.0, duration=n / 4.0)
    reqs = []
    for i, a in enumerate(arrivals):
        slack = rng.uniform(2.0, 10.0)
        out = int(rng.integers(20, 200))
        reqs.append(_mk(i, a, 100, out, out, slo_deadline=a + slack))
    return _sorted(reqs)


def edf_counter_domino_effect_transient_overload(smoke: bool = False, seed: int = 4) -> List[Request]:
    """Demand must exceed capacity by a wide, sustained margin -- a mild
    overload just delays a few requests without cascading; the domino
    effect specifically needs the queue to never recover within the
    burst window. max_active_sequences=2 (catalog override) x tight
    slack x a burst well above combined service capacity."""
    n = 40 if smoke else 150
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=40.0, duration=n / 40.0)[:n]
    reqs = []
    for i, a in enumerate(arrivals):
        slack = rng.uniform(0.1, 0.3)
        out = int(rng.integers(100, 300))
        reqs.append(_mk(i, a, 100, out, out, slo_deadline=a + slack))
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 6. LLF (least_laxity_first)
# ----------------------------------------------------------------------

def llf_target_service_time_heterogeneity(smoke: bool = False, seed: int = 5) -> List[Request]:
    n = 40 if smoke else 200
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=4.0, duration=n / 4.0)
    reqs = []
    for i, a in enumerate(arrivals):
        out = int(rng.integers(20, 800))
        slack = 5.0
        reqs.append(_mk(i, a, 100, out, out, slo_deadline=a + slack))
    return _sorted(reqs)


def llf_counter_laxity_instability_under_prediction_error(smoke: bool = False, seed: int = 5) -> List[Request]:
    n = 40 if smoke else 150
    misprediction_fraction = 0.2
    understate_factor = 6.0
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=4.0, duration=n / 4.0)
    reqs = []
    for i, a in enumerate(arrivals):
        actual = int(rng.integers(100, 600))
        is_mispredicted = rng.random() < misprediction_fraction
        predicted = max(5, int(actual / understate_factor)) if is_mispredicted else actual
        reqs.append(_mk(i, a, 100, predicted, actual, slo_deadline=a + 2.0,
                         class_id="mispredicted" if is_mispredicted else "normal"))
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 7. Priority scheduling (aging_priority)
# ----------------------------------------------------------------------

def priority_target_bounded_high_priority_load(smoke: bool = False, seed: int = 6) -> List[Request]:
    n = 40 if smoke else 200
    high_priority_fraction = 0.2
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=4.0, duration=n / 4.0)
    reqs = []
    for i, a in enumerate(arrivals):
        is_hp = rng.random() < high_priority_fraction
        out = int(rng.integers(20, 200))
        reqs.append(_mk(i, a, 100, out, out, priority=5.0 if is_hp else 1.0,
                         class_id="high" if is_hp else "low"))
    return _sorted(reqs)


def priority_counter_continuous_high_priority_starves_low(smoke: bool = False, seed: int = 6) -> List[Request]:
    """Calibrated empirically (see docs/research/algorithm_stress_tests/
    STRESS_TEST_VALIDATION_20260805.md): the high-priority arrival rate
    must exceed max_active_sequences=2's service capacity by a WIDE
    margin (not just marginally) for a GROWING (not merely persistent)
    backlog to form -- only then does a low-priority request arriving
    mid-run reliably find genuine high-priority competition still waiting
    ahead of it, rather than a lucky momentarily-empty queue. At rate=40
    (service capacity ~20/s for 100-token outputs) over a 20s window,
    low-priority mean queuing delay is ~3x high-priority's (30.4s vs.
    10.0s) -- real, substantial, but NOT dramatic 30x-style starvation;
    the acceptance gate reflects this measured magnitude honestly rather
    than an untested guess."""
    duration = 8.0 if smoke else 20.0
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=40.0, duration=duration)
    reqs = []
    rid = 0
    for a in arrivals:
        out = int(rng.integers(80, 120))
        reqs.append(_mk(rid, a, 50, out, out, priority=5.0, class_id="high"))
        rid += 1
    n_low = 10
    for i in range(n_low):
        arrival = (i + 0.5) * duration / n_low
        reqs.append(_mk(rid, arrival, 50, 100, 100, priority=1.0, class_id="low"))
        rid += 1
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 8. SCORPIO-style SLO guard
# ----------------------------------------------------------------------

def scorpio_target_overload_selective_admission(smoke: bool = False, seed: int = 7) -> List[Request]:
    """Genuine, sustained overload is required: arrival rate must exceed
    what max_active_sequences=2 (catalog override) can serve, with slack
    tight enough that naive policies (FIFO/EDF, no admission control)
    accumulate real SLO violations for SCORPIO's selective admission to
    improve upon."""
    n = 60 if smoke else 300
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=30.0, duration=n / 30.0)
    reqs = []
    for i, a in enumerate(arrivals):
        slack = rng.uniform(0.15, 0.6)
        out = int(rng.integers(50, 250))
        reqs.append(_mk(i, a, 100, out, out, slo_deadline=a + slack, priority=rng.uniform(1.0, 5.0)))
    return _sorted(reqs)


def scorpio_counter_false_rejection_near_threshold(smoke: bool = False, seed: int = 7) -> List[Request]:
    """Sustained overload (as in scorpio_target_overload_selective_admission)
    plus systematically pessimistic predicted_output_tokens (scaled up
    from true length) for every request -- pushing genuinely-feasible
    requests to look infeasible to a laxity/decode-pressure-driven
    admission gate."""
    n = 40 if smoke else 200
    prediction_pessimism_factor = 1.8
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=25.0, duration=n / 25.0)
    reqs = []
    for i, a in enumerate(arrivals):
        actual_out = int(rng.integers(50, 200))
        slack = rng.uniform(0.2, 0.7)
        predicted_out = int(actual_out * prediction_pessimism_factor)
        reqs.append(_mk(i, a, 100, predicted_out, actual_out, slo_deadline=a + slack))
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 9. Regression ANWG selector
# ----------------------------------------------------------------------

def selector_target_in_distribution_regime(smoke: bool = False, seed: int = 8) -> List[Request]:
    """In-distribution: moderate, stable arrival rate and length spread --
    matches the general shape of this project's own training-window
    generation (see selector/dataset_v2/) without importing it directly
    (kept self-contained per this task's isolation requirement)."""
    n = 60 if smoke else 300
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=8.0, duration=n / 8.0)
    prompts = lognormal_tokens(rng, len(arrivals), mean=200, sigma=0.5, low=16, high=1024)
    outputs = lognormal_tokens(rng, len(arrivals), mean=150, sigma=0.5, low=8, high=512)
    reqs = [_mk(i, a, p, o, o, slo_deadline=a + rng.uniform(1.0, 5.0))
            for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))]
    return _sorted(reqs)


def selector_counter_out_of_distribution_regime_shift(smoke: bool = False, seed: int = 8) -> List[Request]:
    """OOD: a mid-run regime transition from low-contention/short-output
    to high-contention/long-output, a combination this project's own
    training windows are unlikely to have sampled contiguously."""
    n = 60 if smoke else 300
    rng = np.random.default_rng(seed)
    half = n // 2
    duration_each = (n / 8.0) / 2
    early_arr = poisson_arrivals(rng, rate=2.0, duration=duration_each)[:half]
    late_arr = duration_each + poisson_arrivals(rng, rate=25.0, duration=duration_each)[: n - half]
    reqs = []
    rid = 0
    for a in early_arr:
        out = int(rng.integers(8, 40))
        reqs.append(_mk(rid, a, 50, out, out, slo_deadline=a + 5.0, class_id="early_low_contention"))
        rid += 1
    for a in late_arr:
        out = int(rng.integers(400, 1200))
        reqs.append(_mk(rid, a, 800, out, out, slo_deadline=a + 1.0, class_id="late_high_contention"))
        rid += 1
    return _sorted(reqs)


# ----------------------------------------------------------------------
# 10-11. vLLM-LTR / PARS -- offline-scored, stubbed
# ----------------------------------------------------------------------

def vllm_ltr_target_predictive_prompt_semantics(smoke: bool = False, seed: int = 9) -> List[Request]:
    raise NotImplementedError(
        "vLLM-LTR is offline-scored against real prompt text via baselines/vllm_ltr/adapter/ "
        "(see that module) -- this generator would need to reuse the existing WildChat-control "
        "offline score cache, not synthesize new Request objects with a fabricated score. "
        "Out of scope for this pass; see the catalog entry's structural caveat."
    )


def vllm_ltr_counter_reasoning_domain_shift(smoke: bool = False, seed: int = 9) -> List[Request]:
    raise NotImplementedError(
        "Requires a NEW offline scoring pass against the real vLLM-LTR checkpoint on a "
        "reasoning-heavy prompt corpus that does not yet exist in this project -- "
        "real_system_followup_required=true in the catalog entry. Not attempted this pass."
    )


def pars_target_alpaca_style_instruction_prompts(smoke: bool = False, seed: int = 10) -> List[Request]:
    raise NotImplementedError(
        "PARS is offline-scored against real prompt text via baselines/pars/adapter/ -- "
        "same structural constraint as vllm_ltr_target_predictive_prompt_semantics."
    )


def pars_counter_reasoning_domain_shift(smoke: bool = False, seed: int = 10) -> List[Request]:
    raise NotImplementedError(
        "Requires a NEW offline scoring pass against the real PARS checkpoint on a "
        "reasoning-heavy prompt corpus -- real_system_followup_required=true. Not attempted."
    )


# ----------------------------------------------------------------------
# 12. Sarathi-Serve
# ----------------------------------------------------------------------
# Prompt-bucket target-token conventions mirror
# src/llmserveopt/real_llm/calibration_common.py's PROMPT_BUCKET_TARGET_TOKENS
# (short=100, medium=512, long=2048) -- the exact buckets used by the real
# Wulver GPU validation this section reproduces the shape of
# (scripts/run_sarathi_gpu_smoke_and_validation.py's make_scenarios(),
# docs/wulver_sarathi_vllm_repeated_validation.md). Smoke-scale generators
# below reproduce the EXACT request count / arrival-offset / bucket /
# output-target shape of the N=5-trial-validated real-hardware scenario;
# full-scale generators scale the same shape up for statistical power --
# an extrapolation beyond the literally-validated N, disclosed as such in
# the corresponding catalog entries (see
# docs/audits/sarathi_stress_test_catalog_completion_20260805.md).
# Requires simulator_requirements.enable_prefill_modeling=true (and, for
# the two target/counter mechanism entries that depend on decode
# protection, enable_decode_prefill_contention=true) to make Sarathi's
# chunked-prefill/stall-free behavior observable at all -- see
# docs/decode_prefill_contention_execution_model.md. This section does
# not touch sarathi_faithful.py itself.

_SHORT_TOKENS, _MEDIUM_TOKENS, _LONG_TOKENS = 100, 512, 2048


def sarathi_counter_long_prompt_moderate_output(smoke: bool = False, seed: int = 11) -> List[Request]:
    """Mirrors Wulver scenario A (sarathi_long_prompt_moderate_output /
    mistral_match_long_prompt_moderate_output): all requests are
    long-bucket prompts arriving simultaneously, moderate output. Real
    hardware result: robust vLLM E2E win, 5/5 trials, mean diff -0.2555s,
    95% CI [-0.298, -0.213] (job pair 1111988/1111989)."""
    n = 4 if smoke else 16
    reqs = [_mk(i, 0.0, _LONG_TOKENS, 256, 256, class_id="long") for i in range(n)]
    return _sorted(reqs)


def sarathi_target_active_decode_plus_arriving_prefill(smoke: bool = False, seed: int = 12) -> List[Request]:
    """Mirrors Wulver scenario B (sarathi_active_decode_plus_arriving_prefill):
    medium-bucket requests arrive at t=0 (already decoding by the time the
    long prefills below arrive), THEN long-bucket, short-output requests
    arrive staggered at t=3.0+1.0*i -- the exact fixture shape
    docs/decode_prefill_contention_execution_model.md identified as
    untested (a decode-phase request arriving strictly BEFORE a
    still-to-arrive long prefill on the same GPU). Real hardware result:
    ROBUST Sarathi E2E win, 5/5 trials, mean diff +1.0172s, 95% CI
    [0.990, 1.036] (job pair 1111988/1111989) -- the strongest real-hardware
    evidence for Sarathi's stall-free decode-protection claim in this
    project."""
    n_each = 4 if smoke else 16
    reqs = [_mk(i, 0.0, _MEDIUM_TOKENS, 256, 256, class_id="medium_decoding")
            for i in range(n_each)]
    reqs += [_mk(n_each + i, 3.0 + 1.0 * i, _LONG_TOKENS, 64, 64, class_id="long_arriving_prefill")
             for i in range(n_each)]
    return _sorted(reqs)


def sarathi_counter_prefill_heavy_burst(smoke: bool = False, seed: int = 13) -> List[Request]:
    """Mirrors Wulver scenario C (sarathi_prefill_heavy_burst): a burst of
    long-bucket prompts with SHORT output (prefill-dominated cost, decode
    barely matters), all arriving at t=0. Real hardware result: robust
    vLLM E2E win, 5/5 trials, mean diff -0.1466s, 95% CI [-0.157, -0.137]."""
    n = 6 if smoke else 24
    reqs = [_mk(i, 0.0, _LONG_TOKENS, 32, 32, class_id="long_prefill_heavy") for i in range(n)]
    return _sorted(reqs)


def sarathi_counter_mixed_prompt_lengths(smoke: bool = False, seed: int = 14) -> List[Request]:
    """Mirrors Wulver scenario D (sarathi_mixed_prompt_lengths): short/
    medium/long buckets cycled evenly, moderate output, all at t=0. Real
    hardware result: robust vLLM E2E win, 5/5 trials, mean diff -0.2052s,
    95% CI [-0.257, -0.161]."""
    n = 6 if smoke else 24
    buckets = [_SHORT_TOKENS, _MEDIUM_TOKENS, _LONG_TOKENS]
    labels = ["short", "medium", "long"]
    reqs = [_mk(i, 0.0, buckets[i % 3], 64, 64, class_id=labels[i % 3]) for i in range(n)]
    return _sorted(reqs)


def sarathi_target_kv_pressure(smoke: bool = False, seed: int = 15) -> List[Request]:
    """Mirrors Wulver scenario E (sarathi_matched_vllm_kv_pressure): long
    context + long decode at concurrency 12, all at t=0 -- matched to vLLM
    jobs 1111541/1111545's stress_kv_pressure shape. Real hardware result:
    ROBUST Sarathi E2E win, 5/5 trials, mean diff +0.8360s, 95% CI
    [0.769, 0.903]."""
    n = 12 if smoke else 36
    reqs = [_mk(i, 0.0, _LONG_TOKENS, 768, 768, class_id="long_kv_pressure") for i in range(n)]
    return _sorted(reqs)


def sarathi_counter_short_prompt_decode_dominated_regime(smoke: bool = False, seed: int = 16) -> List[Request]:
    """HYPOTHESIZED_ADVERSARIAL_REGIME, not derived from any single Wulver
    trial or paper-stated result: prefill is already trivially cheap
    (short-bucket prompts), so chunked-prefill's own fixed per-chunk
    scheduling overhead has little prefill cost left to amortize against,
    while decode volume dominates. Motivated by the general chunk-size-
    sensitivity finding in the literature review (technical review's
    "below C=128 the attention re-overhead kills throughput" framing,
    docs/audits/sarathi_official_artifact_audit_20260805.md section 4) but
    not a direct reproduction of any cited experiment -- an internally
    constructed hypothesis, labeled conservatively in the catalog."""
    n = 20 if smoke else 80
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(rng, rate=8.0, duration=n / 8.0)
    reqs = []
    for i, a in enumerate(arrivals):
        out = int(rng.integers(600, 1200))
        reqs.append(_mk(i, a, _SHORT_TOKENS, out, out, class_id="short_decode_dominated"))
    return _sorted(reqs)


def sarathi_counter_long_context_attention_recompute(smoke: bool = False, seed: int = 17) -> List[Request]:
    raise NotImplementedError(
        "Long-context (>=32K token) quadratic attention-recompute cost is NOT "
        "REPRESENTABLE in this simulator's timing model -- it has no attention-cost "
        "scaling term at all (confirmed in docs/audits/sarathi_official_artifact_audit_20260805.md "
        "section 6, 'Simulator compatibility'). Executing this generator would silently "
        "produce a flat-per-token-cost result that says nothing about the real "
        "quadratic-regime critique the paper itself sidesteps (prompts <=13K tokens "
        "tested). Catalog entry is spec-only, matching the vLLM-LTR/PARS "
        "out-of-scope pattern. Not attempted this pass."
    )


GENERATORS = {
    "fifo_target_homogeneous_low_contention": fifo_target_homogeneous_low_contention,
    "fifo_counter_head_of_line_blocking": fifo_counter_head_of_line_blocking,
    "sof_target_mixed_lengths_accurate_prediction": sof_target_mixed_lengths_accurate_prediction,
    "sof_counter_long_job_starvation": sof_counter_long_job_starvation,
    "estf_target_accurate_alpha_beta_estimate": estf_target_accurate_alpha_beta_estimate,
    "estf_counter_reasoning_prompt_length_misprediction": estf_counter_reasoning_prompt_length_misprediction,
    "wsp_target_priority_length_balance": wsp_target_priority_length_balance,
    "wsp_counter_priority_service_time_conflict": wsp_counter_priority_service_time_conflict,
    "edf_target_feasible_heterogeneous_deadlines": edf_target_feasible_heterogeneous_deadlines,
    "edf_counter_domino_effect_transient_overload": edf_counter_domino_effect_transient_overload,
    "llf_target_service_time_heterogeneity": llf_target_service_time_heterogeneity,
    "llf_counter_laxity_instability_under_prediction_error": llf_counter_laxity_instability_under_prediction_error,
    "priority_target_bounded_high_priority_load": priority_target_bounded_high_priority_load,
    "priority_counter_continuous_high_priority_starves_low": priority_counter_continuous_high_priority_starves_low,
    "scorpio_target_overload_selective_admission": scorpio_target_overload_selective_admission,
    "scorpio_counter_false_rejection_near_threshold": scorpio_counter_false_rejection_near_threshold,
    "selector_target_in_distribution_regime": selector_target_in_distribution_regime,
    "selector_counter_out_of_distribution_regime_shift": selector_counter_out_of_distribution_regime_shift,
    "vllm_ltr_target_predictive_prompt_semantics": vllm_ltr_target_predictive_prompt_semantics,
    "vllm_ltr_counter_reasoning_domain_shift": vllm_ltr_counter_reasoning_domain_shift,
    "pars_target_alpaca_style_instruction_prompts": pars_target_alpaca_style_instruction_prompts,
    "pars_counter_reasoning_domain_shift": pars_counter_reasoning_domain_shift,
    "sarathi_counter_long_prompt_moderate_output": sarathi_counter_long_prompt_moderate_output,
    "sarathi_target_active_decode_plus_arriving_prefill": sarathi_target_active_decode_plus_arriving_prefill,
    "sarathi_counter_prefill_heavy_burst": sarathi_counter_prefill_heavy_burst,
    "sarathi_counter_mixed_prompt_lengths": sarathi_counter_mixed_prompt_lengths,
    "sarathi_target_kv_pressure": sarathi_target_kv_pressure,
    "sarathi_counter_short_prompt_decode_dominated_regime": sarathi_counter_short_prompt_decode_dominated_regime,
    "sarathi_counter_long_context_attention_recompute": sarathi_counter_long_context_attention_recompute,
}

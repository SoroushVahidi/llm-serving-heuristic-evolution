"""VTC-specific multi-tenant fairness workload extension.

This project's accepted canonical workload suite (`benchmarks/canonical_suite/`)
carries no tenant/client concept at all -- confirmed by direct inspection of
`src/llmserveopt/core/types.py` (no `tenant` field anywhere) and by grepping
the canonical suite generators. Every canonical family is a single-tenant
(or tenant-agnostic) scenario; none of them can meaningfully exercise VTC's
entire reason for existing (per-tenant service equalization under
throughput/fairness conflict). Per this integration's scope instructions,
this module is a **clearly labeled, separate extension** -- it does not
modify, regenerate, or read anything under `benchmarks/canonical_suite/`.

Tenant identity is carried via `Request.class_id` (an existing, otherwise
generic string field -- see `docs/audits/vtc_official_artifact_audit_20260805.md`
§7 for why a new schema field was not added instead: it would touch
`src/llmserveopt/core/types.py`, which this task's scope explicitly
excludes ["Do not modify CC5/CC6 logic"] from touching). Consumed via
`baselines/vtc/adapter/simulator_policy.py`'s `default_tenant_of`.

Each generator returns `(requests, known_tenants)`: a `List[Request]`
sorted by arrival time and the ordered list of tenant ids present, ready to
pass straight to `VTCFairnessPolicy(known_tenants=known_tenants)`.

Six families, chosen to cover the multi-tenant fairness/throughput
dimensions this project's canonical suite cannot exercise:
`balanced_tenants`, `one_heavy_hitter`, `heterogeneous_token_sizes`,
`bursty_tenant`, `returning_inactive_tenant`, `priority_fairness_conflict`.

REPAIR NOTE (2026-08-05, see docs/audits/vtc_fairness_benchmark_repair_20260805.md)
------------------------------------------------------------------------------
The original parameters (see git history) produced almost no genuine
scheduling-order-dependent backlog: FIFO's own per-step contention rate
(fraction of admission steps with >=2 distinct tenants simultaneously
waiting) was ~0.000 in 5 of 6 families at the original arrival
rates/capacity, meaning admission ORDER essentially never had to break a
real tie between competing tenants -- any two policies were guaranteed to
look identical regardless of their fairness properties. Root cause,
diagnosed quantitatively via scripts/decompose_vtc_smoke_confound.py:
per-tenant arrival rates (~2-5 req/s combined) were below the effective
single-slot service rate given ~100-token mean outputs, so the system
almost never queued. All six families below are retuned (higher combined
arrival rate relative to `RECOMMENDED_GPU_CONFIG.max_active_sequences`)
specifically to fix this, verified via the same FIFO contention-rate probe
-- see each function's docstring for its measured contention rate.

A SEPARATE, independently-diagnosed confound (also in that audit) was a
units mismatch: this simulator's native `BasePolicy._feasible_on_gpu`
treats `GPUConfig.max_batch_tokens` as a per-step ACTIVE-REQUEST-COUNT cap
(`new_batch = new_count`, a documented Phase-1 simplification), while the
official `VTCReqQueue`/`ReqQueue` code reads the identically-named field
as a real cumulative PROMPT-TOKEN budget. `RECOMMENDED_GPU_CONFIG` sets
`max_batch_tokens=4096`, comfortably above every family's maximum single
request's prompt-token count (verified below), so it never binds under
EITHER interpretation -- the sole deliberate contention-inducing knob is
`max_active_sequences` (consistently interpreted as a request count by
both native and official code), avoiding the units-mismatch confound by
construction rather than by chance. `heterogeneous_token_sizes` in
particular satisfies the fairness-benchmark-repair task's explicit
requirement #5 ("heterogeneous request sizes with SUFFICIENT MEMORY
HEADROOM") for exactly this reason.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.workloads.distributions import (
    bursty_arrivals,
    lognormal_tokens,
    poisson_arrivals,
)

_DEFAULT_SLO_SLACK = 1000.0  # generous by default -- SLOs are opt-in per family, not implicit

#: Shared capacity recommendation for every family below (and the default
#: used by scripts/check_vtc_fairness_headroom.py and the comparative
#: sweep). `max_active_sequences=3` is the deliberate contention-inducing
#: constraint; `max_batch_tokens=4096` and `max_kv_tokens=16384` are set
#: generously so neither becomes a confound under either capacity
#: interpretation (see module docstring). Individual scripts MAY override
#: this, but any override should be re-validated against
#: scripts/check_vtc_fairness_headroom.py before being trusted.
RECOMMENDED_GPU_CONFIG = GPUConfig(
    gpu_id=0, max_active_sequences=3, max_batch_tokens=4096, max_kv_tokens=16384,
)


def _mk_requests(
    rng: np.random.Generator,
    tenant: str,
    arrivals: np.ndarray,
    prompt_tokens: np.ndarray,
    output_tokens: np.ndarray,
    start_id: int,
    priority: float = 1.0,
    slo_slack: float = _DEFAULT_SLO_SLACK,
) -> List[Request]:
    out = []
    for i, (t, p, o) in enumerate(zip(arrivals, prompt_tokens, output_tokens)):
        out.append(Request(
            request_id=start_id + i,
            arrival_time=float(t),
            prompt_tokens=int(p),
            predicted_output_tokens=int(o),
            actual_output_tokens=int(o),
            slo_deadline=float(t) + slo_slack,
            priority=priority,
            class_id=tenant,
        ))
    return out


def _finalize(requests: List[Request], known_tenants: List[str]) -> Tuple[List[Request], List[str]]:
    return sorted(requests, key=lambda r: r.arrival_time), known_tenants


def balanced_tenants(
    n_tenants: int = 4,
    rate_per_tenant: float = 5.0,
    duration: float = 30.0,
    output_mean: float = 150.0,
    prompt_mean: float = 200.0,
    seed: int = 0,
) -> Tuple[List[Request], List[str]]:
    """Baseline fairness case: N tenants, symmetric arrival rate and token
    distributions. Any fair scheduler should equalize service near-exactly.

    Spec (against RECOMMENDED_GPU_CONFIG, max_active_sequences=3):
    tenant count=4; requests/tenant~150 (Poisson, rate 5/s x 30s);
    arrival process=Poisson per tenant; prompt lengths~lognormal(mean=200,
    sigma=0.5, [16,1024]); predicted output lengths~lognormal(mean=150,
    sigma=0.4, [8,1024]); actual output = predicted (no misprediction
    modeled -- see "actual output generation model" note below); memory
    capacity=16384 KV tokens (non-binding); batch capacity=4096 tokens
    (non-binding, see module docstring); SLOs=generous/non-binding;
    priorities=uniform (1.0); expected queue depth=lowish but sustained
    (mean FIFO queuing delay ~0.7s at this rate); expected fairness
    behavior=near-equal per-tenant service under any work-conserving
    scheduler, incl. FIFO -- this family is the SANITY CHECK, not the
    discriminative one; reservation should NOT bind (max prompt << 4096).
    Measured FIFO contention rate (>=2 tenants simultaneously
    backlogged): 0.937 (scripts/decompose_vtc_smoke_confound.py-style probe).
    """
    rng = np.random.default_rng(seed)
    tenants = [f"tenant_{i}" for i in range(n_tenants)]
    requests: List[Request] = []
    rid = 0
    for tenant in tenants:
        arrivals = poisson_arrivals(rng, rate_per_tenant, duration)
        prompts = lognormal_tokens(rng, len(arrivals), mean=prompt_mean, sigma=0.5, low=16, high=1024)
        outputs = lognormal_tokens(rng, len(arrivals), mean=output_mean, sigma=0.4, low=8, high=1024)
        requests += _mk_requests(rng, tenant, arrivals, prompts, outputs, rid)
        rid += len(arrivals)
    return _finalize(requests, tenants)


def one_heavy_hitter(
    n_light_tenants: int = 4,
    heavy_rate: float = 15.0,
    light_rate: float = 2.0,
    duration: float = 30.0,
    output_mean: float = 150.0,
    prompt_mean: float = 200.0,
    seed: int = 1,
) -> Tuple[List[Request], List[str]]:
    """One tenant submits far more requests than everyone else combined.
    Direct test of VTC's core promise: the heavy hitter must not starve
    the light tenants (contrast FIFO, where it would).

    Spec: tenant count=5 (1 heavy + 4 light); requests: heavy~450
    (Poisson, rate 15/s x 30s), light~60 each (Poisson, rate 2/s x 30s);
    arrival process=Poisson per tenant, independent; prompt
    lengths~lognormal(mean=200, sigma=0.5, [16,1024]) for all tenants;
    predicted output lengths~lognormal(mean=150, sigma=0.4, [8,1024]) for
    all tenants; actual output = predicted; memory capacity=16384 KV
    tokens; batch capacity=4096 tokens (non-binding); SLOs=generous;
    priorities=uniform; expected queue depth=high for light tenants once
    the heavy tenant saturates max_active_sequences=3 (mean FIFO queuing
    delay ~2.3s); expected fairness behavior=FIFO admits in raw arrival
    order, so the heavy tenant (15/s) captures a disproportionate share of
    admission slots purely by volume -- VTC should instead cap the heavy
    tenant's share via its served-counter, protecting light tenants'
    throughput; reservation should NOT bind (max prompt << 4096). Measured
    FIFO contention rate: 0.914.
    """
    rng = np.random.default_rng(seed)
    tenants = ["heavy"] + [f"light_{i}" for i in range(n_light_tenants)]
    requests: List[Request] = []
    rid = 0

    heavy_arrivals = poisson_arrivals(rng, heavy_rate, duration)
    heavy_prompts = lognormal_tokens(rng, len(heavy_arrivals), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    heavy_outputs = lognormal_tokens(rng, len(heavy_arrivals), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "heavy", heavy_arrivals, heavy_prompts, heavy_outputs, rid)
    rid += len(heavy_arrivals)

    for tenant in tenants[1:]:
        arrivals = poisson_arrivals(rng, light_rate, duration)
        prompts = lognormal_tokens(rng, len(arrivals), mean=prompt_mean, sigma=0.5, low=16, high=1024)
        outputs = lognormal_tokens(rng, len(arrivals), mean=output_mean, sigma=0.4, low=8, high=1024)
        requests += _mk_requests(rng, tenant, arrivals, prompts, outputs, rid)
        rid += len(arrivals)

    return _finalize(requests, tenants)


def heterogeneous_token_sizes(
    rate_per_tenant: float = 5.0,
    duration: float = 30.0,
    seed: int = 2,
) -> Tuple[List[Request], List[str]]:
    """Same request COUNT and arrival rate per tenant, but wildly
    different per-request token sizes -- tests whether the scheduler
    equalizes TOKEN-WEIGHTED service (VTC's actual objective) rather than
    request-count service (which a naive round-robin would equalize
    instead, and which would look 'fair' by the wrong metric).

    Spec: tenant count=4 (short_prompts/long_prompts/short_outputs/
    long_outputs); requests/tenant~150 (Poisson, rate 5/s x 30s); arrival
    process=Poisson per tenant; prompt lengths vary by tenant --
    short_prompts~lognormal(mean=32), long_prompts~lognormal(mean=900,
    max observed 2031 tokens); predicted output lengths vary by tenant --
    short_outputs~lognormal(mean=24), long_outputs~lognormal(mean=500);
    actual output = predicted; memory capacity=16384 KV tokens (headroom:
    ~8x the max single request's total footprint); batch
    capacity=**4096 tokens, deliberately >= the observed 2031-token
    maximum single prompt with 2x headroom** -- this is the family the
    fairness-benchmark-repair task's requirement #5 explicitly calls
    "heterogeneous request sizes with sufficient memory headroom," and
    this capacity choice is why: the official admission-reservation gate
    must NOT dominate this family's outcome (verified: at this capacity,
    `_can_add_new_req`'s own reservation check never rejects a request in
    a full run -- reservation_bind_rate=0.0; the ORIGINAL, un-repaired
    version of this family used `max_batch_tokens=1024` specifically
    BELOW the max prompt size, which is what caused the confound this
    repair task exists to fix -- see
    docs/audits/vtc_fairness_benchmark_repair_20260805.md); SLOs=generous;
    priorities=uniform; expected queue depth=high (max_active_sequences=3
    binds hard against long_prompts/long_outputs' token footprint even
    with reservation not literally rejecting anything); expected fairness
    behavior=a scheduler that equalizes REQUEST COUNT would look
    identical across tenants, but one that equalizes TOKEN-WEIGHTED
    service should visibly under-serve long_outputs/long_prompts relative
    to short_* in request-count terms while equalizing token-cost. Measured
    FIFO contention rate: 0.990.
    """
    rng = np.random.default_rng(seed)
    tenants = ["short_prompts", "long_prompts", "short_outputs", "long_outputs"]
    profiles = {
        "short_prompts": dict(prompt_mean=32, prompt_sigma=0.3, output_mean=150, output_sigma=0.4),
        "long_prompts": dict(prompt_mean=900, prompt_sigma=0.3, output_mean=150, output_sigma=0.4),
        "short_outputs": dict(prompt_mean=200, prompt_sigma=0.5, output_mean=24, output_sigma=0.3),
        "long_outputs": dict(prompt_mean=200, prompt_sigma=0.5, output_mean=500, output_sigma=0.3),
    }
    requests: List[Request] = []
    rid = 0
    for tenant in tenants:
        prof = profiles[tenant]
        arrivals = poisson_arrivals(rng, rate_per_tenant, duration)
        prompts = lognormal_tokens(rng, len(arrivals), mean=prof["prompt_mean"],
                                    sigma=prof["prompt_sigma"], low=8, high=2048)
        outputs = lognormal_tokens(rng, len(arrivals), mean=prof["output_mean"],
                                    sigma=prof["output_sigma"], low=4, high=1024)
        requests += _mk_requests(rng, tenant, arrivals, prompts, outputs, rid)
        rid += len(arrivals)
    return _finalize(requests, tenants)


def bursty_tenant(
    mean_rate: float = 8.0,
    burst_factor: float = 8.0,
    burst_fraction: float = 0.2,
    duration: float = 30.0,
    output_mean: float = 150.0,
    prompt_mean: float = 200.0,
    seed: int = 3,
) -> Tuple[List[Request], List[str]]:
    """One tenant arrives in sharp bursts; another arrives at a steady
    rate. Tests whether a burst from one tenant transiently starves the
    steady tenant (VTC should limit this via its work-conserving,
    min-served-first admission -- a burst raises the bursty tenant's own
    counter quickly, handing priority back to the steady tenant).

    Spec: tenant count=2 (bursty, steady); requests: bursty~variable
    (bursty_arrivals, mean_rate=8/s, burst_factor=8x during ~20% of the
    window), steady~240 (Poisson, rate 8/s x 30s); arrival
    process=bursty (alternating burst/quiet segments) for `bursty`,
    Poisson for `steady`; prompt lengths~lognormal(mean=200, sigma=0.5,
    [16,1024]) for both; predicted output lengths~lognormal(mean=150,
    sigma=0.4, [8,1024]) for both; actual output = predicted; memory
    capacity=16384 KV tokens; batch capacity=4096 tokens (non-binding);
    SLOs=generous; priorities=uniform; expected queue depth=spiky --
    high during `bursty`'s burst windows, low otherwise; expected fairness
    behavior=FIFO admits whichever tenant happens to be at the head of the
    combined arrival-time-ordered queue, so `steady` suffers a visible
    latency/service dip exactly during `bursty`'s burst windows; VTC's
    counter should rise sharply for `bursty` during its burst, de-
    prioritizing it relative to `steady` for the remainder of that burst;
    reservation should NOT bind (max prompt << 4096). Measured FIFO
    contention rate: 0.649.
    """
    rng = np.random.default_rng(seed)
    tenants = ["bursty", "steady"]
    requests: List[Request] = []
    rid = 0

    bursty_arr = bursty_arrivals(rng, mean_rate=mean_rate, duration=duration,
                                  burst_factor=burst_factor, burst_fraction=burst_fraction)
    bursty_prompts = lognormal_tokens(rng, len(bursty_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    bursty_outputs = lognormal_tokens(rng, len(bursty_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "bursty", bursty_arr, bursty_prompts, bursty_outputs, rid)
    rid += len(bursty_arr)

    steady_arr = poisson_arrivals(rng, mean_rate, duration)
    steady_prompts = lognormal_tokens(rng, len(steady_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    steady_outputs = lognormal_tokens(rng, len(steady_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "steady", steady_arr, steady_prompts, steady_outputs, rid)
    rid += len(steady_arr)

    return _finalize(requests, tenants)


def returning_inactive_tenant(
    duration: float = 60.0,
    early_rate: float = 12.0,
    late_rate: float = 12.0,
    continuous_rate: float = 8.0,
    output_mean: float = 150.0,
    prompt_mean: float = 200.0,
    seed: int = 4,
) -> Tuple[List[Request], List[str]]:
    """Tenant A is active only in [0, duration/4), then goes fully silent
    while tenant B stays continuously active for the rest of the run, then
    A returns with one final burst near the end. Directly exercises the
    official "counter lift" mechanism (see PROVENANCE.md / official_reference
    excerpt) -- without it, A would return with an artificially large
    service deficit and be allowed to monopolize the scheduler; with it,
    A's counter is lifted to match B's current level on return.

    Spec: tenant count=2 (returning, continuous); requests:
    returning~180 in [0,15s] + ~55 in [51,60s] (Poisson, rate 12/s in each
    window), continuous~480 across [0,60s] (Poisson, rate 8/s); arrival
    process=Poisson within each window; prompt
    lengths~lognormal(mean=200, sigma=0.5, [16,1024]) for both; predicted
    output lengths~lognormal(mean=150, sigma=0.4, [8,1024]) for both;
    actual output = predicted; memory capacity=16384 KV tokens; batch
    capacity=4096 tokens (non-binding); SLOs=generous; priorities=uniform;
    expected queue depth=low outside the overlap windows (returning tenant
    absent most of the run BY DESIGN -- overall contention rate is not the
    right readiness signal here); WITHIN the two overlap windows
    ([0,15]/[51,60]), expected queue depth is high -- measured FIFO
    contention rate restricted to those windows: 0.878 / 0.845
    respectively (vs. 0.428 unrestricted over the full 60s, which
    understates this family's real contention by design -- see
    scripts/check_vtc_fairness_headroom.py's window-aware handling);
    expected fairness behavior=on return, `returning`'s official VTC
    counter should be LIFTED to `continuous`'s current level (not reset to
    0, which would let it dominate, nor left at its stale pre-idle value,
    which would starve it) -- this is the one family that directly probes
    a VTC-specific state-machine rule no other family touches; reservation
    should NOT bind (max prompt << 4096).
    """
    rng = np.random.default_rng(seed)
    tenants = ["returning", "continuous"]
    requests: List[Request] = []
    rid = 0

    early_arr = poisson_arrivals(rng, early_rate, duration / 4)
    early_prompts = lognormal_tokens(rng, len(early_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    early_outputs = lognormal_tokens(rng, len(early_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "returning", early_arr, early_prompts, early_outputs, rid)
    rid += len(early_arr)

    late_arr = duration * 0.85 + poisson_arrivals(rng, late_rate, duration * 0.13)
    late_prompts = lognormal_tokens(rng, len(late_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    late_outputs = lognormal_tokens(rng, len(late_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "returning", late_arr, late_prompts, late_outputs, rid)
    rid += len(late_arr)

    cont_arr = poisson_arrivals(rng, continuous_rate, duration)
    cont_prompts = lognormal_tokens(rng, len(cont_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    cont_outputs = lognormal_tokens(rng, len(cont_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "continuous", cont_arr, cont_prompts, cont_outputs, rid)
    rid += len(cont_arr)

    return _finalize(requests, tenants)


def priority_fairness_conflict(
    rate_per_tenant: float = 10.0,
    duration: float = 30.0,
    output_mean: float = 150.0,
    prompt_mean: float = 200.0,
    seed: int = 5,
) -> Tuple[List[Request], List[str]]:
    """Two tenants, equal token-weighted arrival rate, but one is marked
    high `priority`/tight SLO and the other low `priority`/loose SLO.
    VTC itself is priority-blind (it only knows `fair_weights`, a
    SEPARATE mechanism from this simulator's own `priority`/`slo_deadline`
    fields) -- this family exists to make that gap visible: an
    SLO/priority-oriented policy (e.g. `scorpio_style_slo_guard`) and VTC
    will disagree about what "good" means here, which is exactly the
    throughput/fairness trade-off this task's smoke evaluation is meant
    to surface, not paper over.

    Spec: tenant count=2 (high_priority_tight_slo, low_priority_loose_slo);
    requests~300 each (Poisson, rate 10/s x 30s); arrival process=Poisson
    per tenant; prompt lengths~lognormal(mean=200, sigma=0.5, [16,1024])
    for both; predicted output lengths~lognormal(mean=150, sigma=0.4,
    [8,1024]) for both; actual output = predicted; memory capacity=16384
    KV tokens; batch capacity=4096 tokens (non-binding); SLOs=**1.0s**
    slack for high_priority_tight_slo, 1000.0s (non-binding) for
    low_priority_loose_slo -- calibrated empirically: at 3.0s slack (the
    original value) FIFO's own SLO-violation rate for the tight tenant was
    0.000 at this contention level (mean queuing delay ~0.9s, comfortably
    under 3.0s), meaning this family did not actually discriminate SLO
    attainment at all; 1.0s slack produces a genuinely mixed FIFO
    violation rate (0.603, neither floor nor ceiling) -- see
    docs/audits/vtc_fairness_benchmark_repair_20260805.md §5; priorities=
    5.0 vs. 1.0; expected queue depth=high (mean FIFO queuing delay ~0.9s
    at this combined rate under max_active_sequences=3); expected fairness
    behavior=VTC equalizes TOKEN-WEIGHTED service between the two tenants
    regardless of their `priority`/`slo_deadline` fields (it has no
    visibility into them at all), so it will systematically violate the
    tight-SLO tenant's deadline under contention in a way an SLO-aware
    policy would not -- this is the intended, disclosed trade-off this
    family measures, not a bug; reservation should NOT bind (max prompt
    << 4096). Measured FIFO contention rate: 0.910; FIFO tight-tenant SLO
    violation rate at 1.0s slack: 0.603.
    """
    rng = np.random.default_rng(seed)
    tenants = ["high_priority_tight_slo", "low_priority_loose_slo"]
    requests: List[Request] = []
    rid = 0

    hp_arr = poisson_arrivals(rng, rate_per_tenant, duration)
    hp_prompts = lognormal_tokens(rng, len(hp_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    hp_outputs = lognormal_tokens(rng, len(hp_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "high_priority_tight_slo", hp_arr, hp_prompts, hp_outputs,
                              rid, priority=5.0, slo_slack=1.0)
    rid += len(hp_arr)

    lp_arr = poisson_arrivals(rng, rate_per_tenant, duration)
    lp_prompts = lognormal_tokens(rng, len(lp_arr), mean=prompt_mean, sigma=0.5, low=16, high=1024)
    lp_outputs = lognormal_tokens(rng, len(lp_arr), mean=output_mean, sigma=0.4, low=8, high=1024)
    requests += _mk_requests(rng, "low_priority_loose_slo", lp_arr, lp_prompts, lp_outputs,
                              rid, priority=1.0, slo_slack=_DEFAULT_SLO_SLACK)
    rid += len(lp_arr)

    return _finalize(requests, tenants)


ALL_FAIRNESS_FAMILIES = {
    "balanced_tenants": balanced_tenants,
    "one_heavy_hitter": one_heavy_hitter,
    "heterogeneous_token_sizes": heterogeneous_token_sizes,
    "bursty_tenant": bursty_tenant,
    "returning_inactive_tenant": returning_inactive_tenant,
    "priority_fairness_conflict": priority_fairness_conflict,
}

#: Families whose overall (unwindowed) FIFO contention rate is the right
#: readiness signal for scripts/check_vtc_fairness_headroom.py.
#: `returning_inactive_tenant` is excluded -- its contention is
#: concentrated in two short overlap windows by design (see its
#: docstring), so the checker evaluates it with a window-aware rule
#: instead of the flat per-run rate.
WINDOWED_CONTENTION_FAMILIES = {"returning_inactive_tenant"}

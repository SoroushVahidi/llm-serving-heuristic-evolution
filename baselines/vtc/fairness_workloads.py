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

_DEFAULT_SLO_SLACK = 1000.0  # generous -- these workloads probe fairness, not SLO attainment


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
    requests_per_tenant: int = 15,
    duration: float = 30.0,
    arrival_rate_per_tenant: float = 2.0,
    seed: int = 0,
) -> Tuple[List[Request], List[str]]:
    """Baseline fairness case: N tenants, symmetric arrival rate and token
    distributions. Any fair scheduler should equalize service near-exactly;
    this is the sanity-check family, not a discriminative one."""
    rng = np.random.default_rng(seed)
    tenants = [f"tenant_{i}" for i in range(n_tenants)]
    requests: List[Request] = []
    rid = 0
    for tenant in tenants:
        arrivals = poisson_arrivals(rng, arrival_rate_per_tenant, duration)[:requests_per_tenant]
        prompts = lognormal_tokens(rng, len(arrivals), mean=200, sigma=0.5, low=16, high=1024)
        outputs = lognormal_tokens(rng, len(arrivals), mean=100, sigma=0.5, low=8, high=512)
        requests += _mk_requests(rng, tenant, arrivals, prompts, outputs, rid)
        rid += len(arrivals)
    return _finalize(requests, tenants)


def one_heavy_hitter(
    n_light_tenants: int = 4,
    heavy_request_count: int = 60,
    light_request_count: int = 6,
    duration: float = 30.0,
    seed: int = 1,
) -> Tuple[List[Request], List[str]]:
    """One tenant submits far more requests than everyone else combined.
    Direct test of VTC's core promise: the heavy hitter must not starve
    the light tenants (contrast FIFO, where it would)."""
    rng = np.random.default_rng(seed)
    tenants = ["heavy"] + [f"light_{i}" for i in range(n_light_tenants)]
    requests: List[Request] = []
    rid = 0

    heavy_arrivals = poisson_arrivals(rng, heavy_request_count / duration, duration)[:heavy_request_count]
    heavy_prompts = lognormal_tokens(rng, len(heavy_arrivals), mean=200, sigma=0.5, low=16, high=1024)
    heavy_outputs = lognormal_tokens(rng, len(heavy_arrivals), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "heavy", heavy_arrivals, heavy_prompts, heavy_outputs, rid)
    rid += len(heavy_arrivals)

    for tenant in tenants[1:]:
        arrivals = poisson_arrivals(rng, light_request_count / duration, duration)[:light_request_count]
        prompts = lognormal_tokens(rng, len(arrivals), mean=200, sigma=0.5, low=16, high=1024)
        outputs = lognormal_tokens(rng, len(arrivals), mean=100, sigma=0.5, low=8, high=512)
        requests += _mk_requests(rng, tenant, arrivals, prompts, outputs, rid)
        rid += len(arrivals)

    return _finalize(requests, tenants)


def heterogeneous_token_sizes(
    duration: float = 30.0,
    seed: int = 2,
) -> Tuple[List[Request], List[str]]:
    """Same request COUNT and arrival rate per tenant, but wildly
    different per-request token sizes -- tests whether the scheduler
    equalizes TOKEN-WEIGHTED service (VTC's actual objective) rather than
    request-count service (which a naive round-robin would equalize
    instead, and which would look 'fair' by the wrong metric)."""
    rng = np.random.default_rng(seed)
    tenants = ["short_prompts", "long_prompts", "short_outputs", "long_outputs"]
    profiles = {
        "short_prompts": dict(prompt_mean=32, prompt_sigma=0.3, output_mean=100, output_sigma=0.5),
        "long_prompts": dict(prompt_mean=900, prompt_sigma=0.3, output_mean=100, output_sigma=0.5),
        "short_outputs": dict(prompt_mean=200, prompt_sigma=0.5, output_mean=16, output_sigma=0.3),
        "long_outputs": dict(prompt_mean=200, prompt_sigma=0.5, output_mean=500, output_sigma=0.3),
    }
    requests: List[Request] = []
    rid = 0
    n_per_tenant = 15
    for tenant in tenants:
        prof = profiles[tenant]
        arrivals = poisson_arrivals(rng, n_per_tenant / duration, duration)[:n_per_tenant]
        prompts = lognormal_tokens(rng, len(arrivals), mean=prof["prompt_mean"],
                                    sigma=prof["prompt_sigma"], low=8, high=2048)
        outputs = lognormal_tokens(rng, len(arrivals), mean=prof["output_mean"],
                                    sigma=prof["output_sigma"], low=4, high=1024)
        requests += _mk_requests(rng, tenant, arrivals, prompts, outputs, rid)
        rid += len(arrivals)
    return _finalize(requests, tenants)


def bursty_tenant(
    duration: float = 30.0,
    seed: int = 3,
) -> Tuple[List[Request], List[str]]:
    """One tenant arrives in sharp bursts; another arrives at a steady
    rate. Tests whether a burst from one tenant transiently starves the
    steady tenant (VTC should limit this via its work-conserving,
    min-served-first admission -- a burst raises the bursty tenant's own
    counter quickly, handing priority back to the steady tenant)."""
    rng = np.random.default_rng(seed)
    tenants = ["bursty", "steady"]
    requests: List[Request] = []
    rid = 0

    bursty_arr = bursty_arrivals(rng, mean_rate=2.0, duration=duration, burst_factor=8.0, burst_fraction=0.15)
    bursty_prompts = lognormal_tokens(rng, len(bursty_arr), mean=200, sigma=0.5, low=16, high=1024)
    bursty_outputs = lognormal_tokens(rng, len(bursty_arr), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "bursty", bursty_arr, bursty_prompts, bursty_outputs, rid)
    rid += len(bursty_arr)

    steady_arr = poisson_arrivals(rng, 2.0, duration)
    steady_prompts = lognormal_tokens(rng, len(steady_arr), mean=200, sigma=0.5, low=16, high=1024)
    steady_outputs = lognormal_tokens(rng, len(steady_arr), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "steady", steady_arr, steady_prompts, steady_outputs, rid)
    rid += len(steady_arr)

    return _finalize(requests, tenants)


def returning_inactive_tenant(
    duration: float = 60.0,
    seed: int = 4,
) -> Tuple[List[Request], List[str]]:
    """Tenant A is active only in [0, duration/4), then goes fully silent
    while tenant B stays continuously active for the rest of the run, then
    A returns with one final burst near the end. Directly exercises the
    official "counter lift" mechanism (see PROVENANCE.md / official_reference
    excerpt) -- without it, A would return with an artificially large
    service deficit and be allowed to monopolize the scheduler; with it,
    A's counter is lifted to match B's current level on return."""
    rng = np.random.default_rng(seed)
    tenants = ["returning", "continuous"]
    requests: List[Request] = []
    rid = 0

    early_arr = poisson_arrivals(rng, 3.0, duration / 4)
    early_prompts = lognormal_tokens(rng, len(early_arr), mean=200, sigma=0.5, low=16, high=1024)
    early_outputs = lognormal_tokens(rng, len(early_arr), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "returning", early_arr, early_prompts, early_outputs, rid)
    rid += len(early_arr)

    late_arr = duration * 0.9 + poisson_arrivals(rng, 3.0, duration * 0.08)
    late_prompts = lognormal_tokens(rng, len(late_arr), mean=200, sigma=0.5, low=16, high=1024)
    late_outputs = lognormal_tokens(rng, len(late_arr), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "returning", late_arr, late_prompts, late_outputs, rid)
    rid += len(late_arr)

    cont_arr = poisson_arrivals(rng, 2.5, duration)
    cont_prompts = lognormal_tokens(rng, len(cont_arr), mean=200, sigma=0.5, low=16, high=1024)
    cont_outputs = lognormal_tokens(rng, len(cont_arr), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "continuous", cont_arr, cont_prompts, cont_outputs, rid)
    rid += len(cont_arr)

    return _finalize(requests, tenants)


def priority_fairness_conflict(
    duration: float = 30.0,
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
    to surface, not paper over."""
    rng = np.random.default_rng(seed)
    tenants = ["high_priority_tight_slo", "low_priority_loose_slo"]
    requests: List[Request] = []
    rid = 0

    hp_arr = poisson_arrivals(rng, 2.0, duration)
    hp_prompts = lognormal_tokens(rng, len(hp_arr), mean=200, sigma=0.5, low=16, high=1024)
    hp_outputs = lognormal_tokens(rng, len(hp_arr), mean=100, sigma=0.5, low=8, high=512)
    requests += _mk_requests(rng, "high_priority_tight_slo", hp_arr, hp_prompts, hp_outputs,
                              rid, priority=5.0, slo_slack=3.0)
    rid += len(hp_arr)

    lp_arr = poisson_arrivals(rng, 2.0, duration)
    lp_prompts = lognormal_tokens(rng, len(lp_arr), mean=200, sigma=0.5, low=16, high=1024)
    lp_outputs = lognormal_tokens(rng, len(lp_arr), mean=100, sigma=0.5, low=8, high=512)
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

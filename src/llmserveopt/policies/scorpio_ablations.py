"""
SCORPIO ablation policy variants.

Each variant disables ONE component of ScorpioStyleSloGuardPolicy so that
component's contribution to overall WG can be isolated.

IMPORTANT: These are ablation-only policies.
  - NOT registered in registry.py
  - NOT in SELECTOR_CANDIDATE_NAMES
  - NOT in BASELINE_NAMES
  - Used exclusively by Phase 2B.14 metric audit / ablation script.
"""
from __future__ import annotations

from .scoring import DEFAULT_ALPHA, DEFAULT_BETA
from .scorpio_style_slo_guard import ScorpioStyleSloGuardPolicy

# Shared defaults matching configs/phase2b13 baseline.
_BASE = dict(
    step_size=0.001,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    laxity_threshold=0.0,
    ttft_slack_threshold=0.0,
    kv_utilization_threshold=0.65,
    decode_pressure_threshold=0.70,
    queue_overload_factor=3.0,
    admission_budget_refill=2.0,
    admission_budget_max=4.0,
    admission_cost=1.0,
    priority_weight=1.0,
    age_bonus=0.05,
    decode_penalty_weight=0.35,
    long_decode_token_threshold=256,
)


def _ablation(name: str, **overrides) -> ScorpioStyleSloGuardPolicy:
    p = ScorpioStyleSloGuardPolicy(**{**_BASE, **overrides})
    p.name = name
    return p


# ---------------------------------------------------------------------------
# Ablation factories — each returns a fresh instance with one component off.
# ---------------------------------------------------------------------------

def scorpio_no_rejection() -> ScorpioStyleSloGuardPolicy:
    """Disable all pre-filters and guard triggers: accept all requests."""
    return _ablation(
        "scorpio_no_rejection",
        laxity_threshold=-1000.0,
        ttft_slack_threshold=-1000.0,
        kv_utilization_threshold=2.0,
        decode_pressure_threshold=2.0,
        queue_overload_factor=1e9,
        admission_budget_max=1e9,
    )


def scorpio_deadline_only() -> ScorpioStyleSloGuardPolicy:
    """Laxity pre-filter only; no KV/decode guard, no credit budget throttle."""
    return _ablation(
        "scorpio_deadline_only",
        kv_utilization_threshold=2.0,
        decode_pressure_threshold=2.0,
        queue_overload_factor=1e9,
        admission_budget_max=1e9,
    )


def scorpio_no_kv_guard() -> ScorpioStyleSloGuardPolicy:
    """Disable KV-utilization threshold guard trigger (all other guards active)."""
    return _ablation(
        "scorpio_no_kv_guard",
        kv_utilization_threshold=2.0,
    )


def scorpio_no_credit_budget() -> ScorpioStyleSloGuardPolicy:
    """Unlimited admission budget: no per-step throttling."""
    return _ablation(
        "scorpio_no_credit_budget",
        admission_budget_max=1e9,
    )


def scorpio_no_laxity_filter() -> ScorpioStyleSloGuardPolicy:
    """Skip the laxity/TTFT pre-filter: all requests are candidates."""
    return _ablation(
        "scorpio_no_laxity_filter",
        laxity_threshold=-1000.0,
        ttft_slack_threshold=-1000.0,
    )


def scorpio_no_priority_weight() -> ScorpioStyleSloGuardPolicy:
    """Priority weight = 0: composite score ignores request priority."""
    return _ablation(
        "scorpio_no_priority_weight",
        priority_weight=0.0,
    )


def scorpio_no_age_bonus() -> ScorpioStyleSloGuardPolicy:
    """Age bonus = 0: composite score ignores how long request waited."""
    return _ablation(
        "scorpio_no_age_bonus",
        age_bonus=0.0,
    )


def scorpio_no_decode_penalty() -> ScorpioStyleSloGuardPolicy:
    """Decode penalty weight = 0: long-decode requests not penalized under KV pressure."""
    return _ablation(
        "scorpio_no_decode_penalty",
        decode_penalty_weight=0.0,
    )


# Registry of ablation factories (name → callable returning policy instance).
ABLATION_FACTORIES = {
    "scorpio_no_rejection":     scorpio_no_rejection,
    "scorpio_deadline_only":    scorpio_deadline_only,
    "scorpio_no_kv_guard":      scorpio_no_kv_guard,
    "scorpio_no_credit_budget": scorpio_no_credit_budget,
    "scorpio_no_laxity_filter": scorpio_no_laxity_filter,
    "scorpio_no_priority_weight": scorpio_no_priority_weight,
    "scorpio_no_age_bonus":     scorpio_no_age_bonus,
    "scorpio_no_decode_penalty": scorpio_no_decode_penalty,
}

ABLATION_NAMES = list(ABLATION_FACTORIES.keys())


def make_ablation(name: str) -> ScorpioStyleSloGuardPolicy:
    if name not in ABLATION_FACTORIES:
        raise KeyError(f"Unknown ablation '{name}'. Available: {ABLATION_NAMES}")
    return ABLATION_FACTORIES[name]()

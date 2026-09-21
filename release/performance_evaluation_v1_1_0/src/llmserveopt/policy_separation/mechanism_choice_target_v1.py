"""Mechanism-choice target formulas (feasibility investigation only).

See docs/audits/mechanism_choice_target_feasibility_v1_20260817.md. This
module implements ONLY the pure target-computation formulas used by that
audit's diagnostics (`scripts/analyze_mechanism_choice_target_feasibility_v1.py`)
-- it is deliberately NOT a "ready to train a selector on" artifact, because
the audit's own finding is `MECHANISM_TARGET_NO_GO` (the KV contrast is
confounded, not a genuine mechanism-relevance signal). It exists so the
audit's central negative finding is itself reproducible and unit-tested,
not so a future step can import it as a trained-selector input without
re-reading the audit's caveats.

Every function here reads only the six native ANWG columns of the frozen,
already-built dense unified utility matrix
(`experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`)
-- no policy is re-run, no scenario is regenerated, and `mechanism_family`
is never read by the target formula itself (only used afterwards, by the
diagnostics script, to check target-vs-family agreement).
"""
from __future__ import annotations

from typing import Dict, Tuple

EPS_DEFAULT = 0.01

#: The two canonical anchor policies whose ANWG gap defines each proposed
#: mechanism's "practical utility advantage" contrast. See audit doc secs.
#: 1-3 for why RANKING's pair (WFS vs ESTF) is structurally different from
#: CHUNK/KV's pairs (both genuinely bracket "mechanism on" vs "mechanism
#: off"; RANKING brackets "which ranking heuristic", since no ranking-
#: neutral policy exists in the canonical six-policy set).
MECHANISM_POLICY_PAIRS: Dict[str, Tuple[str, str]] = {
    "ranking": ("anwg__weighted_fair_share", "anwg__estimated_service_time_first"),
    "chunk": ("anwg__chunked_prefill_small", "anwg__full_prefill"),
    "kv": ("anwg__kv_constrained_online", "anwg__least_laxity_first"),
}

MECHANISMS: Tuple[str, ...] = tuple(MECHANISM_POLICY_PAIRS.keys())

NATIVE_MECHANISM_BY_FAMILY: Dict[str, str] = {
    "FAMILY_A_FAIRNESS_STARVATION_V2": "ranking",
    "FAMILY_B_PREFILL_DECODE_V2": "chunk",
    "FAMILY_C_KV_PRESSURE_V2": "kv",
}


def compute_mechanism_gains(anwg_row: Dict[str, float]) -> Dict[str, float]:
    """`mechanism_gain_m = |ANWG(policy_1) - ANWG(policy_2)|` for each of the
    three mechanisms' native anchor pair, on one scenario's 6-policy ANWG
    row. Pure function; reads only the 6 `anwg__*` values."""
    return {
        mech: abs(anwg_row[a] - anwg_row[b]) for mech, (a, b) in MECHANISM_POLICY_PAIRS.items()
    }


def classify_target(gains: Dict[str, float], eps: float = EPS_DEFAULT) -> Tuple[str, float, float]:
    """Return (top_mechanism, top_gain, margin_over_second_best). Ties
    broken alphabetically (stable, documented, matches the six-policy
    selector's own tie-break convention)."""
    ordered = sorted(gains.items(), key=lambda kv: (-kv[1], kv[0]))
    top_mech, top_gain = ordered[0]
    second_gain = ordered[1][1]
    return top_mech, top_gain, top_gain - second_gain


def classify_target_with_abstention(gains: Dict[str, float], eps: float = EPS_DEFAULT) -> str:
    """4-way target: `no_clear_mechanism` if the top gain does not clear
    `eps` (no mechanism offers a practically meaningful utility advantage)."""
    top_mech, top_gain, _ = classify_target(gains, eps=eps)
    return "no_clear_mechanism" if top_gain <= eps else top_mech

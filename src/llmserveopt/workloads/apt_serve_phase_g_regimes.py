"""apt_serve_phase_g_regimes: curated regime catalog for the Apt-Serve Phase G
comparative sweep.

Not a full factorial over (kv_pressure x slo_pattern x length_pattern x
arrival_pattern x cache_use_structure) -- that space has 5*5*6*5*6 = 4500
cells and is not tractable in one overnight run. Instead this is a curated,
diagonal-plus-targeted design: every dimension's every value appears in at
least one regime, cache-use-structure opportunities are cross-checked
against the pressure tiers most likely to matter, and a few axes get a
denser sweep (SLO, length, arrival) holding the others at a representative
"medium/high pressure, cache opportunity present" setting.

Each entry is a plain dict (not a dataclass) so it round-trips to/from JSON
without extra machinery, matching the run manifest's needs.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .apt_serve_stress import (
    ARRIVAL_PATTERNS,
    CACHE_USE_STRUCTURES,
    KV_PRESSURE_TIERS,
    LENGTH_PATTERNS,
    SLO_PATTERNS,
)

N_REQUESTS_DEFAULT = 36


def _regime(
    regime_id: str,
    kv_pressure: str,
    slo_pattern: str,
    length_pattern: str,
    arrival_pattern: str,
    cache_use_structure: str,
    n_requests: int = N_REQUESTS_DEFAULT,
) -> Dict[str, Any]:
    return {
        "regime_id": regime_id,
        "kv_pressure": kv_pressure,
        "slo_pattern": slo_pattern,
        "length_pattern": length_pattern,
        "arrival_pattern": arrival_pattern,
        "cache_use_structure": cache_use_structure,
        "n_requests": n_requests,
    }


def build_regime_catalog() -> List[Dict[str, Any]]:
    regimes: List[Dict[str, Any]] = []

    # (1) Pressure-tier baselines, cache_use="none" -- establishes each KV
    # pressure tier's behavior with no tiering opportunity at all, so any
    # apparent Apt-Serve advantage elsewhere can be compared against a
    # same-pressure control where the mechanism has nothing to do.
    for i, tier in enumerate(KV_PRESSURE_TIERS):
        regimes.append(_regime(
            f"pressure_{tier}_baseline",
            kv_pressure=tier,
            slo_pattern=SLO_PATTERNS[i % len(SLO_PATTERNS)],
            length_pattern=LENGTH_PATTERNS[i % len(LENGTH_PATTERNS)],
            arrival_pattern=ARRIVAL_PATTERNS[i % len(ARRIVAL_PATTERNS)],
            cache_use_structure="none",
        ))

    # (2) Cache-use-structure focus: every opportunity/risk hint crossed
    # with the three pressure tiers most likely to actually engage the
    # hidden-tier mechanism (medium/high/near_capacity -- "low" has no
    # pressure to relieve, "sustained_overload" may drop requests before
    # tiering can help).
    for hint in CACHE_USE_STRUCTURES:
        if hint == "none":
            continue
        for tier in ("medium", "high", "near_capacity"):
            regimes.append(_regime(
                f"cacheuse_{hint}_{tier}",
                kv_pressure=tier,
                slo_pattern="bimodal",
                length_pattern="bimodal",
                arrival_pattern="steady",
                cache_use_structure=hint,
            ))

    # (3) SLO-heterogeneity focus, holding cache_use at the generator's
    # primary "opportunity" hint and length at bimodal, across two pressure
    # tiers.
    for pattern in SLO_PATTERNS:
        for tier in ("medium", "high"):
            regimes.append(_regime(
                f"slo_{pattern}_{tier}",
                kv_pressure=tier,
                slo_pattern=pattern,
                length_pattern="bimodal",
                arrival_pattern="steady",
                cache_use_structure="kv_to_hidden_opportunity",
            ))

    # (4) Length-heterogeneity focus.
    for pattern in LENGTH_PATTERNS:
        regimes.append(_regime(
            f"length_{pattern}",
            kv_pressure="high",
            slo_pattern="bimodal",
            length_pattern=pattern,
            arrival_pattern="steady",
            cache_use_structure="hidden_to_kv_opportunity",
        ))

    # (5) Arrival-dynamics focus.
    for pattern in ARRIVAL_PATTERNS:
        regimes.append(_regime(
            f"arrival_{pattern}",
            kv_pressure="high",
            slo_pattern="bimodal",
            length_pattern="bimodal",
            arrival_pattern=pattern,
            cache_use_structure="long_relaxed_urgent_short",
        ))

    # De-duplicate by regime_id (defensive; the construction above should
    # already be unique) and return in a stable order.
    seen = set()
    unique_regimes = []
    for r in regimes:
        if r["regime_id"] in seen:
            raise ValueError(f"Duplicate regime_id: {r['regime_id']}")
        seen.add(r["regime_id"])
        unique_regimes.append(r)
    return unique_regimes


REGIME_CATALOG: List[Dict[str, Any]] = build_regime_catalog()

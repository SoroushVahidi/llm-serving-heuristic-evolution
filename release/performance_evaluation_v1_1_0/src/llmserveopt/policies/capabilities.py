"""Typed capability audit for scheduling policies used in composition.

This module answers, for any policy name known to the composition layer,
the questions downstream code needs before combining outputs:

* Does this policy expose comparable per-request scores?
* Does it expose only a deterministic rank over the waiting queue?
* Does it include admission (accept/reject) logic distinct from ranking?
* Is it backed by a native Python implementation, the verified heuristic
  DSL, or both?

It is deliberately declarative metadata, not a new runtime protocol: the
existing adapters (``rank_with_named_expert`` in ``composition.py``,
``score_with_named_expert`` in ``score_aggregation.py``) already do the
real work of extracting comparable outputs. ``PolicyCapabilities`` lets
composition code validate a request ("can expert X participate in score
aggregation?") with a clear error instead of silently degrading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Every name understood by rank_with_named_expert() in composition.py.
RANK_CAPABLE_EXPERTS: frozenset[str] = frozenset({
    "fifo",
    "edf",
    "shortest_output_first",
    "shortest_prompt_first",
    "weighted_shortest_processing",
    "estimated_service_time_first",
    "least_laxity_first",
    "slo_slack_score",
    "aging_priority",
    "kv_constrained_online",
    "adaptive_chunked_prefill",
    "scorpio_style_slo_guard",
})

# Subset of the above that also expose a single, well-defined comparable
# scalar score (higher = more preferred to schedule first) rather than only
# a multi-key sort order. See score_aggregation.score_with_named_expert().
SCORE_CAPABLE_EXPERTS: frozenset[str] = frozenset({
    "fifo",
    "edf",
    "shortest_output_first",
    "shortest_prompt_first",
    "weighted_shortest_processing",
    "estimated_service_time_first",
    "least_laxity_first",
    "slo_slack_score",
})

# Experts whose native behavior includes admission/rejection logic
# (a request can be excluded from the ranking entirely, not just ranked
# low), distinct from pure priority ranking.
ADMISSION_CAPABLE_EXPERTS: frozenset[str] = frozenset({
    "scorpio_style_slo_guard",
})

# Policies with a documented genome (heuristics/DSL) mapping; see
# structural_synthesis.map_policy_to_genome and
# docs/current/POLICY_GENOME_COVERAGE_AUDIT.md. Mirrors mapping_status values
# used there: "EXACT", "APPROXIMATE", or absent (unmapped/UNSUPPORTED).
DSL_MAPPING_STATUS: Mapping[str, str] = {
    "fifo": "EXACT",
    "edf": "EXACT",
    "shortest_output_first": "EXACT",
    "shortest_prompt_first": "EXACT",
    "first_fit": "EXACT",
    "orca_style": "EXACT",
    "slo_slack_score": "EXACT",
    "weighted_shortest_processing": "EXACT",
    "least_laxity_first": "EXACT",
    "estimated_service_time_first": "EXACT",
    "admission_control": "APPROXIMATE",
    "aging_priority": "APPROXIMATE",
    "scorpio_style_slo_guard": "APPROXIMATE",
    "kv_constrained_online": "APPROXIMATE",
    "adaptive_chunked_prefill": "APPROXIMATE",
    "sola_style_state_aware": "APPROXIMATE",
    "flow_control_stability": "APPROXIMATE",
    "weighted_fair_share": "APPROXIMATE",
    "multi_bin_batching": "APPROXIMATE",
}


class CapabilityError(ValueError):
    """Raised when composition code requests an unsupported capability."""


@dataclass(frozen=True)
class PolicyCapabilities:
    name: str
    supports_scores: bool
    supports_ranks: bool
    supports_admission: bool
    dsl_mapping_status: str = "UNMAPPED"
    notes: str = ""


def capabilities_for(name: str) -> PolicyCapabilities:
    """Return the typed capability record for a composition-layer expert name."""
    supports_ranks = name in RANK_CAPABLE_EXPERTS
    supports_scores = name in SCORE_CAPABLE_EXPERTS
    supports_admission = name in ADMISSION_CAPABLE_EXPERTS
    status = DSL_MAPPING_STATUS.get(name, "UNMAPPED")
    notes = ""
    if not supports_ranks:
        notes = "not registered with rank_with_named_expert; unknown to the composition layer"
    elif not supports_scores:
        notes = "ranking only; native sort key is not a single comparable scalar"
    return PolicyCapabilities(
        name=name,
        supports_scores=supports_scores,
        supports_ranks=supports_ranks,
        supports_admission=supports_admission,
        dsl_mapping_status=status,
        notes=notes,
    )


def require_score_capable(names: list[str]) -> None:
    """Raise CapabilityError listing every name that cannot supply scores."""
    unsupported = [n for n in names if n not in SCORE_CAPABLE_EXPERTS]
    if unsupported:
        raise CapabilityError(
            f"Experts {unsupported!r} do not expose comparable scores; "
            f"score-capable experts are {sorted(SCORE_CAPABLE_EXPERTS)!r}"
        )


def require_rank_capable(names: list[str]) -> None:
    unsupported = [n for n in names if n not in RANK_CAPABLE_EXPERTS]
    if unsupported:
        raise CapabilityError(
            f"Experts {unsupported!r} do not expose comparable ranks; "
            f"rank-capable experts are {sorted(RANK_CAPABLE_EXPERTS)!r}"
        )

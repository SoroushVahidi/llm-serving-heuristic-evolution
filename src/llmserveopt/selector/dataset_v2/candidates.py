"""
Unified candidate-policy resolution for Selector Dataset v2, spanning both
the historical registry (`policies/registry.py`) and the five new external
baselines (`policies/external_baselines_registry.py`). See
docs/selector_dataset_v2.md §10 for the full inclusion-criterion
discussion this module implements.

Neither `registry.py`'s `BASELINE_NAMES`/`SELECTOR_CANDIDATE_NAMES` nor
`selector/candidates.py`'s `SELECTOR_CANDIDATES` (the historical, v1
selector candidate list) are modified by anything here -- this module only
READS from both registries to build a NEW, v2-specific candidate list.
"""
from __future__ import annotations

from typing import List

from ...policies.base import BasePolicy
from ...policies.external_baselines_registry import (
    EXTERNAL_BASELINE_NAMES,
    TopologyClass,
    get_external_baseline_spec,
    make_external_baseline,
)
from ...policies.registry import BASELINE_NAMES, make_policy

#: Monolithic-topology-compatible external baselines (see
#: docs/external_baseline_integration.md §1 -- exactly `vllm_faithful`/
#: `sarathi_faithful`, verified against the registry, not assumed).
MONOLITHIC_EXTERNAL_BASELINES: List[str] = [
    name for name in EXTERNAL_BASELINE_NAMES
    if get_external_baseline_spec(name).topology_class == TopologyClass.MONOLITHIC
]

#: A curated subset of the historical 20-policy portfolio included in the
#: Selector Dataset v2 monolithic candidate pool. NOT the full 20 -- see
#: the module docstring / docs/selector_dataset_v2.md §10 for why an
#: unmanageable pairwise-runtime Cartesian product over all 20 + 2 new
#: baselines is avoided during pilot-scale iteration, and why THESE
#: specific seven were chosen (not an arbitrary subset):
#:   - fifo: simplest possible reference point.
#:   - edf, orca_style, slo_slack_score: each independently documented
#:     (Phase 2B.16, see docs/research_status.md) as OUTPERFORMING
#:     `scorpio_style_slo_guard` as a fixed baseline on fresh held-out
#:     data -- i.e. genuine, evidenced competitiveness, not a guess.
#:   - scorpio_style_slo_guard: the historical strongest all-around
#:     performer across Phase 2B.11-2B.16 -- must remain in the pool as
#:     the standard "hard to beat" comparison point.
#:   - admission_control: mechanistically distinct (explicit admission
#:     gating, not just ordering) from every other policy in this subset.
#:   - weighted_shortest_processing: service-time-aware, distinct
#:     scheduling principle from the deadline/SLO-aware entries above.
STRONG_HISTORICAL_MONOLITHIC_POLICIES: List[str] = [
    "fifo",
    "edf",
    "scorpio_style_slo_guard",
    "orca_style",
    "slo_slack_score",
    "admission_control",
    "weighted_shortest_processing",
]

# Verified at import time, not just at test time (mirrors
# selector/candidates.py's own established convention).
for _name in STRONG_HISTORICAL_MONOLITHIC_POLICIES:
    assert _name in BASELINE_NAMES, (
        f"'{_name}' is not in registry.py's BASELINE_NAMES -- "
        f"STRONG_HISTORICAL_MONOLITHIC_POLICIES is stale."
    )


def monolithic_candidate_policies() -> List[str]:
    """The Selector Dataset v2 Phase-1 (monolithic) candidate pool:
    every monolithic-compatible external baseline plus the curated strong
    historical subset. See module docstring for the inclusion criterion."""
    return list(MONOLITHIC_EXTERNAL_BASELINES) + list(STRONG_HISTORICAL_MONOLITHIC_POLICIES)


def is_policy_compatible_with_topology(policy_name: str, topology_class: str) -> bool:
    """Return whether ``policy_name`` is a legitimate within-class candidate."""
    if policy_name in EXTERNAL_BASELINE_NAMES:
        return get_external_baseline_spec(policy_name).topology_class.value == topology_class
    if policy_name in BASELINE_NAMES:
        return topology_class == TopologyClass.MONOLITHIC.value
    return False


def candidate_policies_for_topology(topology_class: str) -> List[str]:
    """Topology-specific v2 candidate policy set.

    Phase 1 is intentionally monolithic-only. Disaggregated and migratory
    topology rows can be represented by the schema, but are not promoted to
    selector-training candidate sets until there are enough comparable policies.
    """
    if topology_class == TopologyClass.MONOLITHIC.value:
        return monolithic_candidate_policies()
    if topology_class == TopologyClass.DISAGGREGATED_PREFILL_DECODE.value:
        return [
            name for name in EXTERNAL_BASELINE_NAMES
            if is_policy_compatible_with_topology(name, topology_class)
        ]
    if topology_class == TopologyClass.MULTI_INSTANCE_MIGRATORY.value:
        return [
            name for name in EXTERNAL_BASELINE_NAMES
            if is_policy_compatible_with_topology(name, topology_class)
        ]
    raise ValueError(f"Unknown topology_class: {topology_class!r}")


def fidelity_class_of(policy_name: str) -> str:
    """'faithful' | 'paper_reimplementation' for an external baseline,
    'historical' for anything in registry.py's BASELINE_NAMES."""
    if policy_name in EXTERNAL_BASELINE_NAMES:
        return get_external_baseline_spec(policy_name).fidelity_class.value
    if policy_name in BASELINE_NAMES:
        return "historical"
    raise KeyError(f"'{policy_name}' is neither an external baseline nor a historical BASELINE_NAMES entry.")


def make_candidate_policy(policy_name: str, seed: int = 0, **kwargs) -> BasePolicy:
    """Instantiate any Selector Dataset v2 candidate by name, resolving
    transparently across both registries."""
    if policy_name in EXTERNAL_BASELINE_NAMES:
        return make_external_baseline(policy_name, **kwargs)
    return make_policy(policy_name, seed=seed, **kwargs)

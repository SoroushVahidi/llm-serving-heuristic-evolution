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

This module defines THREE different policy sets. Do not conflate them --
each answers a different question:

- ``BASELINE_NAMES`` (imported from ``policies.registry``, 20 policies):
  the full historical/internal policy portfolio. Not v2-specific.
- ``MONOLITHIC_EXTERNAL_BASELINES`` + ``STRONG_HISTORICAL_MONOLITHIC_POLICIES``
  (via ``monolithic_candidate_policies()``, 3 + 11 = 14 policies): a
  BROADER, general-purpose diagnostic/exploration pool used by earlier
  (pre-Option-B) Selector Dataset v2 pilots and scripts
  (``build_selector_dataset_v2_pilot.py``,
  ``build_selector_dataset_v2_redesigned_pilot.py``). Still useful for
  historical reproducibility and broader diagnostic exploration -- NOT
  removed, NOT deprecated, just not the current trainable action space.
  Also exposed under the clearer alias ``MONOLITHIC_DIAGNOSTIC_POLICY_POOL``.
- ``SELECTOR_V2_OPTION_B_POLICIES`` (8 policies): the CURRENT, approved,
  canonical Selector v2 trainable action space, per the Option B scope
  decision in ``docs/selector_v2_faithful_baseline_scope_audit.md``. This
  is a strict subset of ``STRONG_HISTORICAL_MONOLITHIC_POLICIES`` above
  (it excludes ``shortest_output_first``, ``orca_style``, and
  ``slo_slack_score``, and excludes all faithful external baselines
  entirely). ``selector/dataset_v2/calibrated_targeted_pilot.py`` imports
  THIS constant -- it does not duplicate the policy list.

When in doubt about "what should the Selector v2 candidate set be today,"
the answer is ``SELECTOR_V2_OPTION_B_POLICIES``, not
``monolithic_candidate_policies()``.
"""
from __future__ import annotations

from typing import List, Tuple

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
#:   - shortest_output_first, estimated_service_time_first: included for the
#:     redesigned bottleneck pilot because the first v2 pilot showed that KV
#:     and decode-heavy regimes need explicit length-specialist comparators;
#:     both are online deployable and monolithic-compatible.
#:   - best_fit, multi_bin_batching: included for resource-scarcity regimes,
#:     where placement and batching can be materially different from SLO-rank
#:     policies.
STRONG_HISTORICAL_MONOLITHIC_POLICIES: List[str] = [
    "fifo",
    "edf",
    "shortest_output_first",
    "scorpio_style_slo_guard",
    "orca_style",
    "slo_slack_score",
    "admission_control",
    "weighted_shortest_processing",
    "estimated_service_time_first",
    "best_fit",
    "multi_bin_batching",
]

#: Clearer alias for the constant above: a BROADER diagnostic/exploration
#: pool (11 historical policies here + 3 monolithic external baselines via
#: `monolithic_candidate_policies()` = 14 total), NOT the current Option B
#: trainable action space (8 policies, `SELECTOR_V2_OPTION_B_POLICIES`
#: below). Retained for historical reproducibility of the pre-Option-B
#: pilots and for future broader diagnostic exploration if needed -- not
#: deprecated, just narrower in scope than its name might suggest at a
#: glance.
MONOLITHIC_DIAGNOSTIC_POLICY_POOL: List[str] = STRONG_HISTORICAL_MONOLITHIC_POLICIES

# Verified at import time, not just at test time (mirrors
# selector/candidates.py's own established convention).
for _name in STRONG_HISTORICAL_MONOLITHIC_POLICIES:
    assert _name in BASELINE_NAMES, (
        f"'{_name}' is not in registry.py's BASELINE_NAMES -- "
        f"STRONG_HISTORICAL_MONOLITHIC_POLICIES is stale."
    )


def monolithic_candidate_policies() -> List[str]:
    """The Selector Dataset v2 Phase-1 (monolithic) DIAGNOSTIC candidate
    pool: every monolithic-compatible external baseline plus the curated
    broader historical subset (14 total). See module docstring for the
    inclusion criterion.

    This is NOT the current Selector v2 trainable action space -- for that,
    use `SELECTOR_V2_OPTION_B_POLICIES` (8 policies) instead."""
    return list(MONOLITHIC_EXTERNAL_BASELINES) + list(STRONG_HISTORICAL_MONOLITHIC_POLICIES)


#: THE current, approved Selector v2 trainable action space (Option B, per
#: docs/selector_v2_faithful_baseline_scope_audit.md). Exactly these 8
#: historical-monolithic policies -- confirmed, across 1,511
#: window-evaluations, to have real, robust, oracle-headroom-backed ANWG
#: specialization. Faithful external baselines are deliberately excluded:
#: never add one here without a new, separately-documented scope decision.
#: `selector/dataset_v2/calibrated_targeted_pilot.py` imports this constant
#: directly rather than duplicating the list.
SELECTOR_V2_OPTION_B_POLICIES: Tuple[str, ...] = (
    "fifo",
    "edf",
    "scorpio_style_slo_guard",
    "admission_control",
    "weighted_shortest_processing",
    "estimated_service_time_first",
    "best_fit",
    "multi_bin_batching",
)

# Verified at import time: every Option B policy must be a real historical
# policy, and none may be an external baseline (faithful baselines are
# evaluation-only, never selector actions -- see module docstring).
for _name in SELECTOR_V2_OPTION_B_POLICIES:
    assert _name in BASELINE_NAMES, (
        f"'{_name}' is not in registry.py's BASELINE_NAMES -- "
        f"SELECTOR_V2_OPTION_B_POLICIES is stale."
    )
    assert _name not in EXTERNAL_BASELINE_NAMES, (
        f"'{_name}' is an external baseline -- SELECTOR_V2_OPTION_B_POLICIES "
        f"must contain only historical policies (faithful baselines are "
        f"evaluation-only per the Option B scope decision)."
    )


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

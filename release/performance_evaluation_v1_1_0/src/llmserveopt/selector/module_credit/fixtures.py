"""Deterministic small fixtures for module-credit tests and smoke reports."""
from __future__ import annotations

from typing import Any

from ...policies.registry import POLICY_LIBRARY_V2_NAMES
from ...policies.structural_synthesis import map_policy_to_genome
from ..suitability.encoders import structural_features


def synthetic_intervention_fixture() -> list[dict[str, Any]]:
    """Return deterministic intervention rows with real policy/module names.

    This fixture is not Wolverine data.  It is deliberately small and
    hand-constructed to exercise target construction, EDF state conditioning,
    ranking, and gate behavior.
    """
    policies = ["edf", "weighted_shortest_processing", "scorpio_style_slo_guard", "fifo", "least_laxity_first"]
    state_specs = [
        ("s_urgent_heterogeneous", "TRAIN", {"feat_queue_length": 6.0, "feat_pred_output_cv": 0.9, "feat_prompt_cv": 0.5, "feat_minimum_slack": 0.01}, {"edf": 0.92, "weighted_shortest_processing": 0.74, "scorpio_style_slo_guard": 0.62, "fifo": 0.55, "least_laxity_first": 0.70}),
        ("s_long_homogeneous", "TRAIN", {"feat_queue_length": 3.0, "feat_pred_output_cv": 0.02, "feat_prompt_cv": 0.02, "feat_minimum_slack": 0.06}, {"edf": 0.22, "weighted_shortest_processing": 0.45, "scorpio_style_slo_guard": 0.86, "fifo": 0.21, "least_laxity_first": 0.28}),
        ("s_mixed_validation", "VALIDATION", {"feat_queue_length": 4.0, "feat_pred_output_cv": 0.45, "feat_prompt_cv": 0.25, "feat_minimum_slack": 0.03}, {"edf": 0.61, "weighted_shortest_processing": 0.58, "scorpio_style_slo_guard": 0.66, "fifo": 0.50, "least_laxity_first": 0.57}),
        ("s_test_urgent", "ID_TEST", {"feat_queue_length": 7.0, "feat_pred_output_cv": 0.8, "feat_prompt_cv": 0.45, "feat_minimum_slack": 0.012}, {"edf": 0.90, "weighted_shortest_processing": 0.76, "scorpio_style_slo_guard": 0.68, "fifo": 0.54, "least_laxity_first": 0.72}),
        ("s_test_long", "ID_TEST", {"feat_queue_length": 2.0, "feat_pred_output_cv": 0.03, "feat_prompt_cv": 0.03, "feat_minimum_slack": 0.055}, {"edf": 0.25, "weighted_shortest_processing": 0.48, "scorpio_style_slo_guard": 0.84, "fifo": 0.24, "least_laxity_first": 0.30}),
    ]
    module_types = ["priority_rule", "admission_rule", "kv_guard"]
    rows: list[dict[str, Any]] = []
    for state_id, split, state_features, rewards in state_specs:
        library_best = max(rewards.values())
        for base in ("fifo", "weighted_shortest_processing"):
            for donor in policies:
                if donor == base:
                    continue
                for module_type in module_types:
                    compat = _compatible(donor, module_type)
                    gain = _synthetic_gain(state_features, donor, module_type, compat)
                    base_reward = rewards[base]
                    child_reward = max(0.0, min(1.05, base_reward + gain))
                    rows.append({
                        "state_id": state_id,
                        "state_features": dict(state_features),
                        "base_policy": base,
                        "donor_policy": donor,
                        "module_type": module_type,
                        "base_reward": base_reward,
                        "donor_reward": rewards[donor],
                        "intervention_reward": child_reward,
                        "library_best_reward": library_best,
                        "compatibility_metadata": {
                            "compatible": float(compat),
                            "structural_distance": abs(POLICY_LIBRARY_V2_NAMES.index(base) - POLICY_LIBRARY_V2_NAMES.index(donor)),
                        },
                        "source": "synthetic_module_credit_fixture",
                        "trace_family": "synthetic",
                        "temporal_block": state_id,
                        "split": split,
                        "split_group_key": state_id,
                        "seed": 20260722,
                    })
    return rows


def _compatible(donor: str, module_type: str) -> bool:
    module = getattr(map_policy_to_genome(donor), module_type, None)
    return module is not None and module.status != "UNSUPPORTED"


def _synthetic_gain(state_features: dict[str, float], donor: str, module_type: str, compatible: bool) -> float:
    if not compatible:
        return -0.02
    heterogeneous = state_features["feat_pred_output_cv"] > 0.4 and state_features["feat_minimum_slack"] < 0.03
    long_homogeneous = state_features["feat_pred_output_cv"] < 0.1 and state_features["feat_minimum_slack"] > 0.04
    if donor == "edf" and module_type == "priority_rule":
        return 0.20 if heterogeneous else -0.08
    if donor == "scorpio_style_slo_guard" and module_type in {"admission_rule", "kv_guard"}:
        return 0.18 if long_homogeneous else 0.02
    if donor == "weighted_shortest_processing" and module_type == "priority_rule":
        return 0.08
    if donor == "least_laxity_first" and module_type == "priority_rule":
        return 0.06 if heterogeneous else 0.01
    return 0.0

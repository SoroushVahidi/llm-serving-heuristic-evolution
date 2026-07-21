#!/usr/bin/env python
"""Small correctness smoke for the policy-composition harness.

This is not a scientific performance experiment.  It instantiates the five
planned treatment families where the repository can do so without waiting for
the running frontier/library workflows, checks leakage-safe fixed-policy
selection on a completed policy-vector CSV, and evaluates actions on tiny
observable states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.composition import instantiate_policy_for_treatment
from llmserveopt.selector.composition_experiment import (
    assert_no_split_group_leakage,
    check_upstream_readiness,
    load_policy_vectors_csv,
    select_best_fixed_policy_from_development,
    validate_treatment_selection_does_not_use_heldout,
)


DEFAULT_COMPLETED_VECTORS = Path(
    "/mmfs1/project/ikoutis/sv96/llmserveopt-data/"
    "selector_v2_overnight_20260720T235405/combined/full_policy_vectors.csv"
)
DEFAULT_FRONTIER_ROOT = Path(
    "/mmfs1/project/ikoutis/sv96/llmserveopt-data/"
    "policy_frontier_cartography_20260721T154408Z"
)
DEFAULT_LIBRARY_ROOT = Path(
    "/mmfs1/project/ikoutis/sv96/llmserveopt-data/"
    "policy_library_v2_expanded_20260721T171933Z"
)


def _req(request_id: int, prompt: int, output: int, deadline: float, arrival: float = 0.0) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=output,
        slo_deadline=deadline,
        priority=1.0,
        class_id="medium",
    )


def _state() -> ObservableState:
    gpu = ObservableGPUState(
        gpu_id=0,
        max_active_sequences=2,
        max_batch_tokens=128,
        max_kv_tokens=4096,
        active_request_ids=[],
        active_requests_info=[],
        current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    return ObservableState(
        time=5.0,
        waiting_queue=[
            _req(1, 512, 512, 25.0),
            _req(2, 32, 64, 25.0),
            _req(3, 1024, 32, 4.0),
        ],
        gpu_states=[gpu],
        completed_count=0,
        step=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-vectors", type=Path, default=DEFAULT_COMPLETED_VECTORS)
    parser.add_argument("--frontier-root", type=Path, default=DEFAULT_FRONTIER_ROOT)
    parser.add_argument("--policy-library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_policy_vectors_csv(args.policy_vectors)
    assert_no_split_group_leakage(rows)
    validate_treatment_selection_does_not_use_heldout(["TRAIN", "VALIDATION"])
    best_fixed, dev_means = select_best_fixed_policy_from_development(
        rows,
        development_splits=("TRAIN", "VALIDATION", "ROBUST_DEV"),
    )

    treatments = {
        "A_best_fixed_dev_selected": "best_fixed_placeholder",
        "B_discrete_selector_placeholder": "weighted_shortest_processing",
        "C_static_rank_ensemble": "static_rank_ensemble",
        "D_contextual_rank_ensemble": "contextual_rank_ensemble",
        "E_component_wise": "component_wise_scorpio_wsp",
    }
    treatment_actions: dict[str, dict] = {}
    for label, treatment in treatments.items():
        policy = instantiate_policy_for_treatment(treatment)
        action = policy.select_action(_state())
        treatment_actions[label] = {
            "treatment_factory_key": treatment,
            "admit": action.admit,
            "fallback_used": bool(getattr(policy, "decision_logs", []) and policy.decision_logs[-1].fallback_used),
        }

    readiness = check_upstream_readiness(args.frontier_root, args.policy_library_root)
    result = {
        "status": "PASS",
        "scientific_claim": "correctness smoke only",
        "best_fixed_selected_from_development": best_fixed,
        "best_fixed_development_mean": dev_means.get(best_fixed),
        "upstream_ready_for_full_experiment": readiness.ready,
        "frontier_final_report_exists": readiness.frontier_final_report_exists,
        "policy_library_final_report_exists": readiness.policy_library_final_report_exists,
        "treatments": treatment_actions,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

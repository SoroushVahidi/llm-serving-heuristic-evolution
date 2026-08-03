"""CC3 example heuristic DSL documents exercising the compositional
constructs added on top of the CC2 primitive registry.

Mirrors the convention in examples.py: each builder function's output is
also saved as a JSON file in configs/heuristics/examples/ (see
scripts/verify_heuristic_dsl.py for the manual verification CLI). These
serve as:
  - unit-test fixtures for the CC3 verifier/compiler/policy stack,
  - reference templates for each of the 8 CC3 required DSL constructs.
"""
from __future__ import annotations

from typing import Any, Dict


def edf_primitive() -> Dict[str, Any]:
    """Construct 1: EDF-style single primitive reference.

    Equivalent to examples.edf_like(), but expressed via a named CC2
    primitive reference rather than a raw req.deadline_urgency var.
    """
    return {
        "name": "edf_primitive",
        "description": "Earliest-deadline-first via a single named primitive reference.",
        "tie_breaker": "earliest_deadline",
        "default": {
            "request_score": {"op": "neg", "args": [{"primitive": "deadline_urgency"}]},
        },
    }


def weighted_deadline_length_ranking() -> Dict[str, Any]:
    """Construct 2: weighted deadline-plus-length ranking (bounded weighted sum)."""
    return {
        "name": "weighted_deadline_length_ranking",
        "description": (
            "Weighted combination of laxity urgency and shortest predicted output; "
            "both terms are named primitive references, not raw vars."
        ),
        "tie_breaker": "earliest_deadline",
        "default": {
            "request_score": {
                "op": "weighted_sum",
                "terms": [
                    [{"primitive": "laxity_urgency"}, 0.7],
                    [{"op": "neg", "args": [{"primitive": "predicted_output_length"}]}, 0.3],
                ],
            },
        },
    }


def sparse_topk_ranking_mixture() -> Dict[str, Any]:
    """Construct 3: sparse top-k ranking mixture over 3 primitive terms, k=2."""
    return {
        "name": "sparse_topk_ranking_mixture",
        "description": (
            "Selects the 2 largest-|weight| terms among laxity urgency, priority, "
            "and negative queue age each step; deterministic on weight ties (by term index)."
        ),
        "tie_breaker": "highest_priority",
        "default": {
            "request_score": {
                "op": "topk_mixture",
                "k": 2,
                "terms": [
                    [{"primitive": "laxity_urgency"}, 1.0],
                    [{"primitive": "priority"}, 0.5],
                    [{"op": "neg", "args": [{"primitive": "queue_age"}]}, 0.2],
                ],
            },
        },
    }


def conditional_kv_pressure_branch() -> Dict[str, Any]:
    """Construct 4: conditional branch keyed on a system-level RESOURCE_GUARD
    primitive (KV pressure), switching ranking strategy under high pressure."""
    return {
        "name": "conditional_kv_pressure_branch",
        "description": (
            "Under high system KV pressure, rank by shortest prompt (protect KV "
            "budget); otherwise rank by laxity urgency."
        ),
        "tie_breaker": "earliest_deadline",
        "regimes": [
            {
                "condition": {
                    "op": "sub",
                    "args": [{"primitive": "system_kv_pressure"}, {"const": 0.8}],
                },
                "request_score": {"op": "neg", "args": [{"primitive": "prompt_length"}]},
            }
        ],
        "default": {
            "request_score": {"primitive": "laxity_urgency"},
        },
    }


def admission_gate_with_fallback() -> Dict[str, Any]:
    """Construct 5: boolean admission gate (primitive_gate) with declared
    safe-fallback behavior when the gate rejects every candidate."""
    return {
        "name": "admission_gate_with_fallback",
        "description": (
            "Admits only requests that still fit their deadline (laxity_gate); "
            "if the gate rejects every candidate this step, the whole step "
            "delegates to the declared fifo_like fallback instead of stalling."
        ),
        "tie_breaker": "earliest_deadline",
        "fallback": {"policy": "fifo_like"},
        "on_no_admits": "safe_fallback",
        "default": {
            "request_score": {"primitive": "laxity_urgency"},
            "admission_condition": {
                "primitive_gate": "laxity_gate",
                "params": {"laxity_threshold": 0.0},
            },
        },
    }


def placement_score_composition() -> Dict[str, Any]:
    """Construct 6: GPU-placement composed from PLACEMENT-family primitives,
    kept separate from the request-ranking expression."""
    return {
        "name": "placement_score_composition",
        "description": (
            "Ranks requests by laxity urgency (RANKING family) but places each "
            "admitted request on the GPU with the smallest composite "
            "(projected_gpu_load, kv_pressure, gpu_id) placement key."
        ),
        "tie_breaker": "earliest_deadline",
        "placement": {
            "keys": [
                {"name": "projected_gpu_load"},
                {"name": "kv_pressure"},
            ],
        },
        "default": {
            "request_score": {"primitive": "laxity_urgency"},
        },
    }


def bounded_external_parameter_example() -> Dict[str, Any]:
    """Construct 7: a declared, bounded external parameter used as a regime
    threshold (not yet driven by a contextual predictor -- CC5+ scope)."""
    return {
        "name": "bounded_external_parameter_example",
        "description": (
            "Declares a bounded 'kv_weight' parameter (default 0.5) used as the "
            "KV-utilization regime-switch threshold; CC3 only supports "
            "declaration + default/override resolution, not learned adaptation."
        ),
        "tie_breaker": "earliest_deadline",
        "parameters": [
            {"name": "kv_weight", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
        ],
        "regimes": [
            {
                "condition": {
                    "op": "sub",
                    "args": [{"var": "sys.kv_utilization"}, {"param": "kv_weight"}],
                },
                "request_score": {"op": "neg", "args": [{"primitive": "prompt_length"}]},
            }
        ],
        "default": {
            "request_score": {"primitive": "laxity_urgency"},
        },
    }


ALL_PRIMITIVE_COMPOSITION_EXAMPLES = (
    edf_primitive,
    weighted_deadline_length_ranking,
    sparse_topk_ranking_mixture,
    conditional_kv_pressure_branch,
    admission_gate_with_fallback,
    placement_score_composition,
    bounded_external_parameter_example,
)

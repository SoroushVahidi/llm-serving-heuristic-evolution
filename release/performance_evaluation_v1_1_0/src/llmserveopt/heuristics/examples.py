"""
Helpers to construct canonical example heuristic DSL documents programmatically.

These examples are also saved as JSON files in configs/heuristics/examples/.
They serve as:
  - unit-test fixtures
  - sanity checks that the verifier/compiler/policy stack works end-to-end
  - reference templates for LLM heuristic generation
"""
from __future__ import annotations

from typing import Any, Dict


def fifo_like() -> Dict[str, Any]:
    """Pure arrival-order scheduling (equivalent to FIFO)."""
    return {
        "name": "fifo_like",
        "description": "Admit in arrival order; score = −arrival_time so earliest arrived wins.",
        "tie_breaker": "arrival_order",
        "default": {
            "request_score": {
                "op": "neg",
                "args": [{"var": "req.waiting_time"}],
            }
        },
    }


def edf_like() -> Dict[str, Any]:
    """Earliest Deadline First — score = deadline urgency."""
    return {
        "name": "edf_like",
        "description": "Prioritise requests closest to their SLO deadline.",
        "tie_breaker": "earliest_deadline",
        "default": {
            "request_score": {"var": "req.deadline_urgency"},
        },
    }


def slo_kv_balanced() -> Dict[str, Any]:
    """Balance SLO urgency with KV-cache cost to avoid thrashing."""
    return {
        "name": "slo_kv_balanced",
        "description": (
            "Weighted combination of deadline urgency and inverse KV cost. "
            "Switch to pure urgency when queue is short (< 8 requests)."
        ),
        "tie_breaker": "earliest_deadline",
        "regimes": [
            {
                "condition": {
                    "op": "sub",
                    "args": [{"const": 8.0}, {"var": "sys.queue_length"}],
                },
                "request_score": {"var": "req.deadline_urgency"},
            }
        ],
        "default": {
            "request_score": {
                "op": "weighted_sum",
                "terms": [
                    [{"var": "req.deadline_urgency"}, 0.7],
                    [
                        {
                            "op": "div_safe",
                            "args": [{"const": 1.0}, {"var": "req.estimated_kv_cost"}],
                        },
                        0.3,
                    ],
                ],
            }
        },
    }


def throughput_oriented() -> Dict[str, Any]:
    """Shortest-Job-First variant that maximises token throughput.

    Uses Shortest Predicted Output (SPO): score = −predicted_output_tokens.
    Under high KV pressure, also penalise large prompts.
    """
    return {
        "name": "throughput_oriented",
        "description": "Shortest predicted output first; KV-pressure regime also penalises long prompts.",
        "tie_breaker": "shortest_output",
        "regimes": [
            {
                "condition": {
                    "op": "sub",
                    "args": [{"var": "sys.kv_utilization"}, {"const": 0.8}],
                },
                "request_score": {
                    "op": "neg",
                    "args": [{"var": "req.estimated_kv_cost"}],
                },
            }
        ],
        "default": {
            "request_score": {
                "op": "neg",
                "args": [{"var": "req.predicted_output_tokens"}],
            }
        },
    }

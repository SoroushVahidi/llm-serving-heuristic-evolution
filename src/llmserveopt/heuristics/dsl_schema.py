"""
DSL schema constants and allowed-set definitions.

This is the single source of truth for what is / is not permitted in a heuristic DSL.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Allowed variables
# ---------------------------------------------------------------------------

REQUEST_VARS: frozenset[str] = frozenset({
    "req.prompt_tokens",
    "req.predicted_output_tokens",
    "req.waiting_time",
    "req.deadline_slack",
    "req.deadline_urgency",
    "req.priority_weight",
    "req.estimated_prefill_cost",
    "req.estimated_decode_cost",
    "req.estimated_kv_cost",
})

SYSTEM_VARS: frozenset[str] = frozenset({
    "sys.queue_length",
    "sys.active_sequence_count",
    "sys.kv_utilization",
    "sys.free_sequence_ratio",
    "sys.token_budget_utilization",
    "sys.arrival_rate_est",
    "sys.burstiness_cv",
    "sys.recent_slo_violation_rate",
    "sys.slo_pressure",
})

BATCH_VARS: frozenset[str] = frozenset({
    "batch.size",
    "batch.sum_prompt_tokens",
    "batch.sum_predicted_output_tokens",
    "batch.mean_predicted_output_tokens",
    "batch.max_predicted_output_tokens",
    "batch.length_imbalance",
    "batch.sum_priority_weight",
    "batch.min_deadline_slack",
    "batch.deadline_risk",
    "batch.estimated_kv_cost",
    "batch.sum_request_score",
})

# All allowed variable names (union)
ALLOWED_VARS: frozenset[str] = REQUEST_VARS | SYSTEM_VARS | BATCH_VARS

# Explicitly forbidden variable names (sampled subset; verifier also checks substrings)
FORBIDDEN_VARS: frozenset[str] = frozenset({
    "actual_output_tokens",
    "req.actual_output_tokens",
    "future_arrivals",
    "future_queue_length",
    "future_slo_violations",
    "oracle_policy",
    "best_policy",
    "completion_time",
    "req.completion_time",
    "ground_truth_output",
    "hidden_output_tokens",
})

# Substring patterns — any variable containing these strings is forbidden.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "actual_output",
    "future_",
    "oracle_",
    "ground_truth",
    "hidden_",
    "completion_time",
)

# ---------------------------------------------------------------------------
# Allowed operations
# ---------------------------------------------------------------------------

ALLOWED_OPS: frozenset[str] = frozenset({
    "const",
    "var",
    "add",
    "sub",
    "mul",
    "div_safe",
    "min",
    "max",
    "clip",
    "abs",
    "sqrt_safe",
    "log1p_safe",
    "neg",
    "weighted_sum",
    "if_then_else",
})

# Forbidden operation names (would introduce side effects, randomness, or code exec)
FORBIDDEN_OPS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "import",
    "random",
    "choice",
    "sample",
    "shuffle",
    "print",
    "open",
    "write",
    "delete",
    "exit",
    "system",
    "lambda",
    "def",
    "class",
    "sleep",
    "time",
})

# ---------------------------------------------------------------------------
# Allowed tie-breaker tokens
# ---------------------------------------------------------------------------

ALLOWED_TIE_BREAKERS: frozenset[str] = frozenset({
    "earliest_deadline",
    "highest_priority",
    "shortest_output",
    "shortest_prompt",
    "arrival_order",
    "lowest_request_id",
    "lowest_kv_cost",
})

# ---------------------------------------------------------------------------
# Default limits (can be overridden per-heuristic)
# ---------------------------------------------------------------------------

DEFAULT_LIMITS: dict = {
    "max_batch_candidates": 64,
    "max_regimes": 3,
    "max_expression_depth": 6,
    "max_terms": 16,
    "max_nodes": 64,
    "max_constant": 1000.0,
    "min_constant": -1000.0,
}

# Required top-level fields
REQUIRED_FIELDS: tuple[str, ...] = ("name", "default", "tie_breaker")

# Required fields inside a rule block (default or regime)
REQUIRED_RULE_FIELDS: tuple[str, ...] = ("request_score",)

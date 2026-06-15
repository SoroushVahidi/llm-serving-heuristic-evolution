"""
Prompt templates for LLM heuristic generation and repair.

All prompts instruct the LLM to produce deterministic offline heuristics
in the restricted JSON DSL.  No LLM is called at runtime during scheduling.
"""
from __future__ import annotations

import json
from typing import List, Tuple

from ..heuristics.dsl_schema import (
    ALLOWED_OPS,
    ALLOWED_TIE_BREAKERS,
    ALLOWED_VARS,
    BATCH_VARS,
    DEFAULT_LIMITS,
    FORBIDDEN_SUBSTRINGS,
    REQUEST_VARS,
    SYSTEM_VARS,
)

# ---------------------------------------------------------------------------
# System prompt (shared for generation and repair)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert in LLM inference serving scheduling.
You design deterministic scheduling heuristics expressed in a restricted JSON DSL.

IMPORTANT: You are NOT scheduling live requests. You are designing a heuristic OFFLINE
that will be verified and then run deterministically by the simulator.
No LLM will be called during request scheduling at runtime.

Return EXACTLY one valid JSON object. Do not include Markdown, code fences, or explanation.
Output only the raw JSON."""

# ---------------------------------------------------------------------------
# Variable catalogue (inline in prompt)
# ---------------------------------------------------------------------------

def _var_catalogue() -> str:
    lines = ["REQUEST VARIABLES (per candidate request):"]
    for v in sorted(REQUEST_VARS):
        lines.append(f"  {v}")
    lines.append("")
    lines.append("SYSTEM VARIABLES (aggregate serving system state):")
    for v in sorted(SYSTEM_VARS):
        lines.append(f"  {v}")
    lines.append("")
    lines.append("BATCH VARIABLES (current batch being built this step):")
    for v in sorted(BATCH_VARS):
        lines.append(f"  {v}")
    return "\n".join(lines)


def _ops_catalogue() -> str:
    ops = sorted(ALLOWED_OPS - {"const", "var"})
    return ", ".join(ops)


def _forbidden_summary() -> str:
    lines = [
        "FORBIDDEN — do not use any variable containing these substrings:",
    ]
    for s in sorted(FORBIDDEN_SUBSTRINGS):
        lines.append(f"  {s}")
    lines.append("")
    lines.append("FORBIDDEN OPERATIONS: eval, exec, import, random, choice, sample, shuffle, lambda, def, class")
    return "\n".join(lines)


def _limits_summary() -> str:
    return (
        f"max_expression_depth={DEFAULT_LIMITS['max_expression_depth']}, "
        f"max_nodes={DEFAULT_LIMITS['max_nodes']}, "
        f"max_terms={DEFAULT_LIMITS['max_terms']}, "
        f"max_regimes={DEFAULT_LIMITS['max_regimes']}, "
        f"constants in [{DEFAULT_LIMITS['min_constant']}, {DEFAULT_LIMITS['max_constant']}]"
    )


# ---------------------------------------------------------------------------
# Generation user prompt
# ---------------------------------------------------------------------------

_GENERATION_USER_TEMPLATE = """Design a NEW scheduling heuristic for online LLM inference serving.

## Objective
Maximize priority-weighted SLO goodput:
  priority_weighted_slo_goodput = sum(priority_i * 1[completion_time_i <= deadline_i]) / sum(priority_i)

Tie-breaks (in order when goodput is tied):
  1. lower SLO violation rate
  2. lower p95 TTFT (time-to-first-token)
  3. lower p95 latency
  4. higher request throughput

## What to design
A DETERMINISTIC online heuristic with:
- request_score: score each candidate request (higher = admitted sooner)
- tie_breaker: one of {tie_breakers}
- optional regimes: conditional rules triggered by system state
- optional batch_score: score the accumulated batch
- optional admission_condition: filter out inadmissible requests

## DSL structure
{{
  "name": "my_heuristic_name",
  "description": "brief description",
  "tie_breaker": "one of: {tie_breakers_list}",
  "regimes": [                     // optional, max {max_regimes}
    {{
      "condition": <expression>,   // activates when expression > 0
      "request_score": <expression>
    }}
  ],
  "default": {{
    "request_score": <expression>,  // REQUIRED
    "batch_score": <expression>,    // optional
    "admission_condition": <expression>  // optional; > 0 means admit
  }}
}}

## Expression nodes
{{"const": <number>}}            — constant in [{min_const}, {max_const}]
{{"var": "<name>"}}              — variable from allowed list
{{"op": "add",    "args": [e1, e2]}}
{{"op": "sub",    "args": [e1, e2]}}
{{"op": "mul",    "args": [e1, e2]}}
{{"op": "div_safe", "args": [e1, e2]}}     — safe division (no zero div)
{{"op": "min",    "args": [e1, e2]}}
{{"op": "max",    "args": [e1, e2]}}
{{"op": "neg",    "args": [e1]}}
{{"op": "abs",    "args": [e1]}}
{{"op": "clip",   "args": [e1, lo, hi]}}
{{"op": "sqrt_safe", "args": [e1]}}
{{"op": "log1p_safe", "args": [e1]}}
{{"op": "weighted_sum", "terms": [[e1, w1], [e2, w2], ...]}}
{{"op": "if_then_else", "cond": e, "then": e, "else": e}}

## Allowed variables
{var_catalogue}

## Forbidden
{forbidden_summary}

## Complexity limits
{limits}

## Instructions
- Be creative. Do not just copy FIFO or EDF.
- Use regime conditions to handle different system states (high load, tight SLOs, etc.)
- The scoring must be deterministic — no random elements.
- Weights in weighted_sum must be numeric constants.
- Return ONLY the JSON object. No explanation. No Markdown. No code fences.
"""

def build_generation_messages() -> List[dict]:
    """Return messages list for heuristic generation."""
    user = _GENERATION_USER_TEMPLATE.format(
        tie_breakers=", ".join(sorted(ALLOWED_TIE_BREAKERS)),
        tie_breakers_list=", ".join(sorted(ALLOWED_TIE_BREAKERS)),
        max_regimes=DEFAULT_LIMITS["max_regimes"],
        min_const=DEFAULT_LIMITS["min_constant"],
        max_const=DEFAULT_LIMITS["max_constant"],
        var_catalogue=_var_catalogue(),
        forbidden_summary=_forbidden_summary(),
        limits=_limits_summary(),
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Repair user prompt
# ---------------------------------------------------------------------------

_REPAIR_USER_TEMPLATE = """The following heuristic DSL candidate FAILED verification.

## Invalid candidate
```json
{candidate_json}
```

## Verifier errors ({error_count} errors)
{error_list}

## Instructions
Minimally repair the candidate to fix ONLY the listed errors while preserving the intended scheduling idea.

Key rules:
- Do NOT use any variable containing: {forbidden_substrings}
- Allowed variables: {allowed_vars_snippet}
- Allowed operations: {ops}
- Constants must be in [{min_const}, {max_const}]
- tie_breaker must be one of: {tie_breakers}
- required fields: name, tie_breaker, default.request_score

Return ONLY the repaired JSON object. No explanation. No Markdown.
"""

def build_repair_messages(
    candidate: dict,
    errors: List[Tuple[str, str]],
) -> List[dict]:
    """Return messages list for a repair attempt."""
    error_lines = "\n".join(f"  [{code}] {msg}" for code, msg in errors)
    allowed_snippet = ", ".join(sorted(REQUEST_VARS)[:6]) + ", ... (see full list in generation prompt)"
    user = _REPAIR_USER_TEMPLATE.format(
        candidate_json=json.dumps(candidate, indent=2),
        error_count=len(errors),
        error_list=error_lines,
        forbidden_substrings=", ".join(sorted(FORBIDDEN_SUBSTRINGS)),
        allowed_vars_snippet=allowed_snippet,
        ops=_ops_catalogue(),
        min_const=DEFAULT_LIMITS["min_constant"],
        max_const=DEFAULT_LIMITS["max_constant"],
        tie_breakers=", ".join(sorted(ALLOWED_TIE_BREAKERS)),
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

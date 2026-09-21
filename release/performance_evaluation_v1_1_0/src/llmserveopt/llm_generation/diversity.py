"""
Candidate diversity controls for LLM heuristic search.

Provides:
  - DESIGN_TARGETS: named design emphases that steer prompt variation
  - build_targeted_messages(): generation prompt with design target injected
  - deduplicate_candidates(): remove exact-JSON duplicates by SHA256
"""
from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .prompt_templates import _SYSTEM_PROMPT, _GENERATION_USER_TEMPLATE, _var_catalogue, _forbidden_summary, _limits_summary
from ..heuristics.dsl_schema import ALLOWED_TIE_BREAKERS, DEFAULT_LIMITS


# ---------------------------------------------------------------------------
# Design targets
# ---------------------------------------------------------------------------

DESIGN_TARGETS: Dict[str, str] = {
    "slo_urgency": (
        "Focus on minimizing SLO violations. Prioritize requests with tight remaining "
        "slack (deadline - now). Penalize requests that are likely to miss their deadline "
        "even if admitted. Use req.deadline_slack, req.deadline_urgency, req.time_to_deadline."
    ),
    "kv_pressure": (
        "Focus on managing KV-cache pressure. Prioritize requests that fit efficiently "
        "in the current KV cache. Deprioritize large KV footprints when memory is tight. "
        "Use sys.kv_utilization, req.estimated_kv_cost, batch.kv_used, batch.kv_remaining."
    ),
    "throughput_oriented": (
        "Focus on maximizing request throughput. Prioritize short requests to minimize "
        "head-of-line blocking. Use req.prompt_tokens, req.predicted_output_tokens, "
        "req.estimated_decode_cost, and shortest-first tie-breaking."
    ),
    "prefill_heavy": (
        "Focus on requests with large prompt prefill costs. The system is dominated by "
        "prefill compute (long prompts, short outputs). Prioritize requests that do NOT "
        "block the batch with long prefills. Consider chunking effects. "
        "Use req.prompt_tokens, req.estimated_prefill_cost, sys.queue_length."
    ),
    "mixed_slo": (
        "Handle a mix of tight, medium, and loose SLO classes. Tight-SLO requests "
        "(req.priority_weight > 2.0) should be served urgently. Loose-SLO requests "
        "should fill slack. Balance req.deadline_urgency with req.priority_weight and "
        "req.time_to_deadline. Use if_then_else to handle different priority tiers."
    ),
    "noisy_prediction_robust": (
        "The system has noisy output-length predictions (relative noise ~35%). Design "
        "a heuristic that is robust to prediction errors. Avoid over-relying on "
        "req.predicted_output_tokens alone. Combine with req.prompt_tokens, "
        "req.deadline_urgency, and sys.kv_utilization for robustness. "
        "Use clip or min/max to bound the effect of outlier predictions."
    ),
    "balanced": (
        "Balance SLO compliance, KV-cache efficiency, and throughput simultaneously. "
        "Use weighted_sum with multiple objectives. Include a regime for high-load "
        "(sys.kv_utilization > 0.7) that shifts weight toward SLO urgency, and a "
        "default regime that balances throughput with SLO compliance. "
        "Use req.deadline_urgency, sys.kv_utilization, req.estimated_kv_cost."
    ),
}

DEFAULT_TARGET_CYCLE = list(DESIGN_TARGETS.keys())


# ---------------------------------------------------------------------------
# Targeted prompt construction
# ---------------------------------------------------------------------------

_TARGET_SUFFIX = """
## Design target: {target_name}
{target_description}

Specifically optimize for this target while still satisfying the primary objective
(priority_weighted_slo_goodput). The design target is a hint, not a constraint.
"""


def build_targeted_messages(design_target: Optional[str] = None) -> List[dict]:
    """Return generation messages with an optional design target injected."""
    target_suffix = ""
    if design_target and design_target in DESIGN_TARGETS:
        target_suffix = _TARGET_SUFFIX.format(
            target_name=design_target,
            target_description=DESIGN_TARGETS[design_target],
        )

    user = (_GENERATION_USER_TEMPLATE + target_suffix).format(
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
# Deduplication
# ---------------------------------------------------------------------------

def _canonical_sha256(candidate: Dict[str, Any]) -> str:
    """Compute SHA256 of canonically serialized candidate JSON."""
    raw = json.dumps(candidate, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def deduplicate_candidates(
    records: List[Dict[str, Any]],
    *,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove exact-duplicate candidates (by canonical SHA256 of JSON).

    Returns (unique_records, removed_records).
    """
    seen: Dict[str, str] = {}  # sha -> candidate_id
    unique = []
    removed = []
    for rec in records:
        cand = rec.get("candidate", {})
        sha = _canonical_sha256(cand)
        if sha in seen:
            if verbose:
                meta = rec.get("metadata", {})
                print(f"  [DEDUP] {meta.get('candidate_id', '?')} is duplicate of {seen[sha]}")
            removed.append(rec)
        else:
            seen[sha] = rec.get("metadata", {}).get("candidate_id", sha[:8])
            unique.append(rec)
    return unique, removed

"""Faithful reproduction of the official ranking/tie-break rule.

Verified against ``official_reference/scheduler_ranking_excerpt.md``
(pinned commit ``13bbf6ff3dab661791d41362551b089e5f77c91c``,
``vllm/core/scheduler.py::_get_ltr_ordered_requests``):

    return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped),
                        key=lambda req: -req.aux_model_score))

Two properties of this rule matter and are reproduced exactly here:

1. Descending score = highest priority (``-score`` as the sort key).
2. Python's ``sorted()`` is stable, so requests with equal scores keep their
   relative order from the input sequence -- no secondary key is applied.

Scope note: the official rule reorders ``waiting + running + swapped``
combined (i.e. it can reprioritize already-admitted, in-flight requests,
not just new admissions). This project's simulator only asks policies to
choose an *admission* order from ``state.waiting_queue`` each step (see
``src/llmserveopt/policies/base.py``); already-admitted requests are
advanced by the service model, not re-ordered by the policy, except through
the separate ``preempt``/``swap``/``migrate``/``hold_decode`` verbs other
faithful baselines use. This adapter therefore reproduces the ranking rule
over whatever request sequence it is given (typically
``state.waiting_queue``) rather than the three-queue concatenation --
applying it to running/swapped requests would require one of those other
verbs and is out of scope for this scaffold (see the audit doc).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, TypeVar

from .errors import MissingScoreError

T = TypeVar("T")


def order_by_ltr_score(requests: Sequence[T], scores: Dict[int, float], id_attr: str = "request_id") -> List[T]:
    """Sort ``requests`` by descending precomputed LTR score.

    ``scores`` must contain every request's id (looked up via
    ``getattr(request, id_attr)``) -- there is no fallback for a missing
    score; this function raises rather than silently treating a missing
    score as zero/lowest/highest priority, per the "no hand-written
    substitute score" constraint on this baseline.
    """
    missing = [getattr(r, id_attr) for r in requests if getattr(r, id_attr) not in scores]
    if missing:
        raise MissingScoreError(
            f"No precomputed LTR score for request id(s) {missing}. This "
            "adapter never substitutes a hand-written heuristic score -- "
            "precompute scores for every request with "
            "adapter.checkpoint_loader before ranking."
        )
    # Stable sort: ties keep their relative order from the input sequence,
    # matching Python's sorted() used by the official _get_ltr_ordered_requests.
    return sorted(requests, key=lambda r: -scores[getattr(r, id_attr)])

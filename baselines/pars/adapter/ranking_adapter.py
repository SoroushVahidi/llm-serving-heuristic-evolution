"""PARS's scheduling ranking rule.

Unlike vLLM-LTR (``baselines/vllm_ltr/adapter/ranking_adapter.py``), the
official PARS repository does not ship an integrated vLLM scheduler patch
-- its README explicitly describes the integration as narrative only:
"assign request priorities through the platform scheduler (for example,
vLLM's priority scheduler)". There is no pinned-commit scheduler excerpt to
reproduce exactly (the situation vLLM-LTR was in), so this ranking rule is
derived directly from the training objective's own mathematics rather than
copied from an official ranking function -- documented as an inference,
not assumed to be identical to whatever exact tie-break/priority-injection
code the authors' internal vLLM integration uses (which isn't public).

Verified derivation (see ``provenance.HIGHER_SCORE_MEANS_LONGER_PREDICTED_RESPONSE``
and its docstring): ``MarginRankingLoss(score_A, score_B, target)`` with
``target = +1`` when prompt_A's real response is LONGER trains the model so
that ``score_A > score_B`` is rewarded exactly when response_A is longer --
i.e. a HIGHER score predicts a LONGER response. For shortest-job-first-style
scheduling (approximate SJF, PARS's own stated goal -- "approximating
shortest-job-first style decisions"), the shortest-predicted-response
request must be admitted first: **ascending** score order, the mirror image
of vLLM-LTR's descending-score rule.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, TypeVar

from .errors import MissingScoreError

T = TypeVar("T")


def order_by_pars_score(requests: Sequence[T], scores: Dict[int, float], id_attr: str = "request_id") -> List[T]:
    """Sort ``requests`` by ASCENDING precomputed PARS score (lowest score
    = shortest predicted response = highest scheduling priority).

    ``scores`` must contain every request's id -- no fallback for a missing
    score; raises rather than silently substituting a heuristic score,
    mirroring vLLM-LTR's ``order_by_ltr_score`` convention exactly.
    """
    missing = [getattr(r, id_attr) for r in requests if getattr(r, id_attr) not in scores]
    if missing:
        raise MissingScoreError(
            f"No precomputed PARS score for request id(s) {missing}. This "
            "adapter never substitutes a hand-written heuristic score -- "
            "precompute scores for every request with "
            "adapter.checkpoint_loader before ranking."
        )
    # Stable sort: ties keep their relative order from the input sequence.
    return sorted(requests, key=lambda r: scores[getattr(r, id_attr)])

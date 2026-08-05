"""The three labeled comparison variants for the VTC fairness-benchmark
repair (docs/audits/vtc_fairness_benchmark_repair_20260805.md §3).

Do not conflate these with each other in results tables -- the whole
point of this module is that they are NOT interchangeable, and the
original smoke evaluation's mistake was comparing variant A against
plain, UNMATCHED FIFO (native ``BasePolicy._feasible_on_gpu`` gate) and
attributing the entire difference to fairness ordering.

- **A. Official VTC** (``vtc_fairness_reference``,
  ``baselines.vtc.adapter.simulator_policy.VTCFairnessPolicy`` with
  ``batch_token_budget_override=None``): VTC's real, unmodified fairness
  ordering, under the real, unmodified official admission-feasibility
  gate, fed this project's GPUConfig numbers exactly as originally
  designed. This is "official VTC" -- the thing actually being evaluated.

- **B. Matched-admission FIFO** (``matched_admission_fifo``,
  ``baselines.vtc.adapter.matched_admission_fifo_policy.MatchedAdmissionFIFOPolicy``):
  plain FCFS-by-arrival ordering, via the official, unmodified ``ReqQueue``
  base class VTCReqQueue itself subclasses -- so its admission gate is the
  EXACT SAME code as variant A's, not a hand-matched approximation. Any
  difference between A and B isolates the effect of fairness ORDERING
  alone, holding admission semantics fixed.

- **C. Fairness-isolation VTC** (``vtc_fairness_isolation``,
  ``VTCFairnessPolicy`` with a non-default ``batch_token_budget_override``):
  VTC's real, unmodified fairness ordering, under the real, unmodified
  admission gate, but fed a `batch_max_tokens` value scaled to reflect
  actual per-workload token sizes rather than this project's
  GPUConfig.max_batch_tokens (which every native policy's simplified
  ``_feasible_on_gpu`` treats as a request-COUNT cap, not a token budget
  -- see ``simulator_policy.VTCFairnessPolicy.__init__``'s
  ``batch_token_budget_override`` docstring for the full units-mismatch
  explanation). This is still 100% official VTC code, unmodified -- ONLY
  the numeric capacity argument differs from variant A. Never call this
  "official VTC" in a results table; always "fairness-isolation VTC" or
  equivalent, to avoid the exact confusion this repair task was created
  to resolve.

``FAIRNESS_ISOLATION_BATCH_TOKEN_BUDGET`` (2048) was chosen empirically:
it comfortably exceeds every fairness-extension workload family's maximum
single-request prompt size (1353 tokens, in ``heterogeneous_token_sizes``)
while remaining a real, binding constraint relative to
``max_active_sequences``-scale concurrent demand -- verified directly:
raising it to 4096 or 8192 produces no further change in any family's
completion fraction (see
docs/audits/vtc_fairness_benchmark_repair_20260805.md §3's calibration
table), confirming 2048 is not simply "unlimited capacity in disguise."
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from llmserveopt.policies.base import BasePolicy

from .matched_admission_fifo_policy import MatchedAdmissionFIFOPolicy
from .simulator_policy import VTCFairnessPolicy

FAIRNESS_ISOLATION_BATCH_TOKEN_BUDGET = 2048


def official_vtc(known_tenants: Sequence[str], **kwargs) -> VTCFairnessPolicy:
    """Variant A."""
    kwargs.pop("batch_token_budget_override", None)
    return VTCFairnessPolicy(known_tenants=known_tenants, batch_token_budget_override=None, **kwargs)


def matched_admission_fifo(**kwargs) -> MatchedAdmissionFIFOPolicy:
    """Variant B."""
    return MatchedAdmissionFIFOPolicy(**kwargs)


def fairness_isolation_vtc(
    known_tenants: Sequence[str],
    batch_token_budget: int = FAIRNESS_ISOLATION_BATCH_TOKEN_BUDGET,
    **kwargs,
) -> VTCFairnessPolicy:
    """Variant C."""
    kwargs.pop("batch_token_budget_override", None)
    return VTCFairnessPolicy(
        known_tenants=known_tenants, batch_token_budget_override=batch_token_budget, **kwargs
    )


VARIANT_FACTORIES: Dict[str, Callable[..., BasePolicy]] = {
    "official_vtc": official_vtc,
    "matched_admission_fifo": matched_admission_fifo,
    "fairness_isolation_vtc": fairness_isolation_vtc,
}

VARIANT_LABELS: Dict[str, str] = {
    "official_vtc": "A: Official VTC (ordering + official reservation gate, unmodified capacity)",
    "matched_admission_fifo": "B: Matched-admission FIFO (FCFS ordering + official reservation gate)",
    "fairness_isolation_vtc": "C: Fairness-isolation VTC (ordering + official gate, capacity rescaled)",
}

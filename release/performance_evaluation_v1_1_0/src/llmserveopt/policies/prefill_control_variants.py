"""Prefill/decode chunk-control mechanism variants for Policy Separation
Family B v1 (see docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md).

Structural mechanism family
---------------------------
The simulator's Phase 1.5 execution model (see
docs/decode_prefill_contention_execution_model.md) performs actual per-request
chunked prefill when ``ServiceModel.enable_prefill_modeling=True``:

* ``enable_decode_prefill_contention=True, decode_first=False`` runs the
  vLLM-v0.4.2-style shared per-step budget: decode and prefill consume one
  combined budget in FCFS-by-arrival order, so an earlier-arriving large
  prefill chunk can exhaust the budget and stall later-arriving decodes that
  step.
* ``enable_decode_prefill_contention=True, decode_first=True`` runs the
  Sarathi-style decode-protected model: active decodes unconditionally receive
  their budget first; prefill uses only the remainder.

Policies in this simulator choose admission only (Action.admit); the chunk
size is a ``ServiceModel`` configuration. Each mechanism variant is therefore
defined as the pair (admission policy, ServiceModel execution config) and is
exposed through ``make_prefill_decode_variants()`` which returns
``{name: (policy, service_model_kwargs)}``.

For the three fixed variants (A/B/C) the admission policy is identical greedy
arrival-ordered admission, so the *only* difference is the execution mechanism
-- that isolates the causal mechanism. Variant D (adaptive) additionally
deferrals long-prompt admission under decode pressure; it is a diagnostic
baseline only (NOT the synthesized child).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..core.action import Action
from ..core.types import ObservableRequest, ObservableState
from .base import BasePolicy
from .policy_library_v2_helpers import deterministic_place

#: Step budget used by default by the Family B v1 configs. Kept here as the
#: canonical reference so chunk budgets can be stated relative to it. The
#: actual value is carried by each scenario's service_model_kwargs.
DEFAULT_STEP_TOKEN_BUDGET = 512

#: Calibrated per-request per-step chunk caps (smoke 2026-08-16): with
#: step_token_budget=512, small=64 leaves crumbs for later tenants while
#: large=256 still often blocks; unlimited takes the full remaining budget.
DEFAULT_CHUNK_SMALL = 64
DEFAULT_CHUNK_LARGE = 256

#: Decode-priority uses the small chunk so later decode tenants can enter
#: service during a convoy; decode_first=True is the Sarathi-style guard
#: (near-twin of shared+small on many natural traces; diverges on the
#: early-prefill / late-already-decoding microbench).
DEFAULT_DECODE_PRIORITY_CHUNK = 64

#: Chunk cap large enough that the per-step cap never binds for any prompt
#: length the Family B v1 templates produce (<= ~16K prompt tokens).
UNLIMITED_PREFILL_CHUNK = 65536


def _arrival_rank(req: ObservableRequest) -> Tuple[float, int]:
    """Deterministic arrival-ordered ranking (FIFO-like), matching the
    neutral baseline used by every fixed mechanism variant."""
    return (req.arrival_time, req.request_id)


class GreedyArrivalPrefillControlPolicy(BasePolicy):
    """Greedy arrival-ordered admission (FIFO-like placement) shared by the
    fixed mechanism variants A/B/C. The mechanism difference lives entirely
    in the paired ``service_model_kwargs`` execution config."""

    name = "greedy_arrival_prefill_control"

    def select_action(self, state: ObservableState) -> Action:
        ranked = sorted(state.waiting_queue, key=_arrival_rank)
        return deterministic_place(state, ranked)


class AdaptivePrefillControlPolicy(BasePolicy):
    """Diagnostic (NOT the synthesized child): handcrafted admission-level
    prefill control. Long-prompt requests are deferred while decode pressure
    is at/above a threshold AND the request still has SLO slack; otherwise
    admission is greedy arrival-ordered. Uses only online observables.

    Execution runs under the shared-contention model with the same fixed
    small chunk as ``chunked_prefill_small``, so the only difference from
    that variant is the admission deferral."""

    name = "adaptive_prefill_control"

    def __init__(
        self,
        long_prompt_threshold: int = 2048,
        pressure_threshold: float = 0.25,
        slack_margin_s: float = 0.02,
    ) -> None:
        self.long_prompt_threshold = int(long_prompt_threshold)
        self.pressure_threshold = float(pressure_threshold)
        self.slack_margin_s = float(slack_margin_s)

    def _decode_pressure(self, state: ObservableState) -> float:
        if not state.gpu_states:
            return 0.0
        denom = sum(g.max_active_sequences for g in state.gpu_states)
        if denom <= 0:
            return 0.0
        return sum(g.decoding_count for g in state.gpu_states) / denom

    def _slack_s(self, req: ObservableRequest, now: float) -> float:
        return req.slo_deadline - now - req.predicted_output_tokens * 0.001

    def select_action(self, state: ObservableState) -> Action:
        now = state.time
        pressure = self._decode_pressure(state)
        defer = pressure >= self.pressure_threshold

        ranked = sorted(state.waiting_queue, key=_arrival_rank)
        admitted: list[ObservableRequest] = []

        def admit_filter(req: ObservableRequest, _gpu, _admitted: list[ObservableRequest]) -> bool:
            is_long = req.prompt_tokens >= self.long_prompt_threshold
            if not is_long:
                return True
            if not defer:
                return True
            # Long prefill + elevated decode pressure: defer only if the
            # request still has slack; a tight-deadline long prefill must be
            # admitted (its TTFT matters more than the TBT cost it imposes).
            return self._slack_s(req, now) <= self.slack_margin_s

        return deterministic_place(state, ranked, admit_filter=admit_filter)


def make_prefill_decode_variants(
    *,
    chunk_small: int = DEFAULT_CHUNK_SMALL,
    chunk_large: int = DEFAULT_CHUNK_LARGE,
    decode_priority_chunk: int = DEFAULT_DECODE_PRIORITY_CHUNK,
) -> Dict[str, Tuple[BasePolicy, Dict[str, Any]]]:
    """Return ``{variant_name: (policy, service_model_kwargs)}`` for the
    Family B v1 mechanism set.

    The returned ``service_model_kwargs`` are merged over each scenario's
    base service-model kwargs (step budget, contention flag, ...) by the
    pilot runner; only the mechanism-relevant keys are set here.

    Chunk budgets are capped by the step budget at execution time, so the
    calibration step must verify that the chosen values are genuinely
    distinct (a chunk >= leftover budget behaves like ``full``).
    """
    full = GreedyArrivalPrefillControlPolicy()
    full.name = "full_prefill"

    chunked_small = GreedyArrivalPrefillControlPolicy()
    chunked_small.name = "chunked_prefill_small"

    chunked_large = GreedyArrivalPrefillControlPolicy()
    chunked_large.name = "chunked_prefill_large"

    decode_priority = GreedyArrivalPrefillControlPolicy()
    decode_priority.name = "decode_priority_chunked"

    adaptive = AdaptivePrefillControlPolicy()

    return {
        "full_prefill": (
            full,
            {
                "max_prefill_chunk_tokens": UNLIMITED_PREFILL_CHUNK,
                "decode_first": False,
            },
        ),
        "chunked_prefill_small": (
            chunked_small,
            {
                "max_prefill_chunk_tokens": int(chunk_small),
                "decode_first": False,
            },
        ),
        "chunked_prefill_large": (
            chunked_large,
            {
                "max_prefill_chunk_tokens": int(chunk_large),
                "decode_first": False,
            },
        ),
        "decode_priority_chunked": (
            decode_priority,
            {
                "max_prefill_chunk_tokens": int(decode_priority_chunk),
                "decode_first": True,
            },
        ),
        "adaptive_prefill_control": (
            adaptive,
            {
                "max_prefill_chunk_tokens": int(chunk_small),
                "decode_first": False,
            },
        ),
    }


def make_prefill_decode_variants_v2(
    *,
    chunk_small: int = DEFAULT_CHUNK_SMALL,
) -> Dict[str, Tuple[BasePolicy, Dict[str, Any]]]:
    """Family B v2 mechanism set: the two v1 anchors only.

    ``chunked_prefill_large``, ``decode_priority_chunked``, and
    ``adaptive_prefill_control`` are intentionally omitted. The v1 audit and
    diagnosis tests show they are twins of the anchors under arrival-FCFS
    (large ≈ full; decode-priority ≡ small on clean traces; adaptive ≡ small).
    ``decode_first=True`` is a real GPU semantic, but it is not activatable
    from natural admission traces without injecting mid-flight decode state
    or changing FCFS semantics; v2 does not manufacture that separation.
    """
    full = GreedyArrivalPrefillControlPolicy()
    full.name = "full_prefill"
    chunked_small = GreedyArrivalPrefillControlPolicy()
    chunked_small.name = "chunked_prefill_small"
    return {
        "full_prefill": (
            full,
            {
                "max_prefill_chunk_tokens": UNLIMITED_PREFILL_CHUNK,
                "decode_first": False,
            },
        ),
        "chunked_prefill_small": (
            chunked_small,
            {
                "max_prefill_chunk_tokens": int(chunk_small),
                "decode_first": False,
            },
        ),
    }
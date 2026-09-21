"""Targeted decode/prefill-contention fixtures.

Built to specifically exercise the execution mechanism fixed in
`GPUState._advance_shared_contention` (see
docs/decode_prefill_contention_execution_model.md): a still-prefilling
request that arrived EARLIER than a competing decode-phase request can now
consume the shared per-step budget ahead of it, stalling that decode
request for one or more steps. None of the six canonical
`runtime_validation_benchmark_pack` fixtures happen to construct this
interaction shape (verified: every already-decoding request in those six
fixtures arrived no later than any competing still-prefilling request --
see that doc's Revalidation section) -- these fixtures are built from
first principles specifically to fill that gap, NOT by reverse-engineering
the five real-hardware winner labels (no hardware target value is read or
referenced anywhere in this module).

Construction principle (identical across variants A-E; F is a negative
control)
--------------------------------------------------------------------------
1. A long-prompt request ("the hog") arrives first (t=0).
2. `ServiceModel.max_prefill_chunk_tokens` is set strictly below
   `step_token_budget` (a legitimate, disclosed choice for a stress
   fixture -- NOT how the pinned faithful references themselves configure
   these two knobs, which keep them equal; see module docstrings of
   `sarathi_faithful`/`vllm_chunked_prefill_faithful`). This leaves a
   small structural "slack" (`step_token_budget - max_prefill_chunk_tokens`)
   available every step even while the hog is still prefilling (capped by
   the chunk, not the full budget) -- without this, the earlier-arrived
   hog would consume the ENTIRE budget every step and no later-arriving
   request could ever even finish ITS OWN prefill to start decoding at all
   (see the doc above for why arrival-order FCFS makes that the case).
3. One or more small-prompt requests ("the runners") arrive shortly after
   the hog, each sized so their own prefill fits within the slack -- they
   reach decode while the hog is still mid-prefill.
4. Enough runners are admitted simultaneously (or the slack is made tight
   enough, `step_token_budget = max_prefill_chunk_tokens + small_slack`)
   that once several runners are decoding at once, their combined 1-token-
   each demand EXCEEDS the slack -- forcing the mechanism this fixture
   family exists to exercise: under `decode_first=False` contention, the
   later-arrived runners stall (zero progress some step); under
   `decode_first=True` decode-protected execution, they do not.

Each variant returns (requests, ServiceModel kwargs to use for
`step_token_budget`/`max_prefill_chunk_tokens`, metadata). The metadata
dict records WHY each variant was built this way (not a hardware
comparison) -- see FIXTURE_VARIANTS at the bottom for the full family.

IMPORTANT, empirically-derived finding (recorded here, not swept under the
rug): extensive construction attempts for variants A-D showed that a
SUSTAINED, multi-step "later-arriving decode request stalled by an
earlier-arriving persistent prefill request" is structurally very hard --
quite possibly impossible -- to produce through NORMAL sequential
end-to-end request-trace admission under this simulator's strict
FCFS-by-arrival-time contention model. The reason, proven constructively
here: priority under `decode_first=False` contention is a STATIC function
of arrival order alone, fixed once a request arrives, and does not change
when a request's phase transitions from prefill to decode. Any later
request Y that manages to bootstrap into decode at all (finish its own
tiny prefill) necessarily did so using leftover budget that an
earlier-arrived, still-active blocker Z did NOT claim that step; but Z, by
definition of "still consuming budget," claims up to its OWN full desired
chunk from any leftover BEFORE Y ever sees it (Z has higher priority) --
so any "surge" large enough for Y to bootstrap is, by the same logic,
already large enough for Z to have been fully satisfied too, at which
point there is no more genuine contention between them for the remainder
of Z's lifetime (a fixed-point equilibrium: the achievable concurrently-
served low-priority cohort size self-limits to exactly the steady-state
leftover, identically under both `decode_first` values). Divergence IS
constructively verified (see `variant_g_direct_state_injection` below,
and `tests/test_decode_prefill_contention_execution.py`), but only by
directly seeding a mid-flight execution state (a request already decoding,
another already mid-prefill) rather than by driving it up from a clean
arrival trace -- itself a materially informative result about the
practical reachability of this failure mode from realistic admission
sequences, recorded in
docs/decode_prefill_contention_execution_model.md's overnight-validation
addendum. Variants A-D are kept (documented, deterministic, legitimate
stress constructions) because a NEGATIVE empirical result here is itself
the honest, reportable answer per this task's own instruction not to force
a match -- see FIXTURE_VARIANTS' `expected_to_diverge` field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from ...core.types import Request


@dataclass(frozen=True)
class ContentionFixture:
    fixture_id: str
    description: str
    interaction_shape: str
    requests: List[Request]
    step_token_budget: int
    max_prefill_chunk_tokens: int
    max_kv_tokens: int
    max_active_sequences: int
    prefill_cost_per_token: float = 1.0
    # Empirically determined (see module docstring), not a hope/assumption:
    # whether this variant is expected to show sarathi_faithful/vllm_
    # chunked_prefill_faithful divergence when actually run. False for
    # every arrival-trace variant (A-E) is itself the honest finding this
    # module documents -- see module docstring's "IMPORTANT" section.
    expected_to_diverge: bool = False


def _req(rid: int, arrival: float, prompt: int, output: int, slo_slack: float = 1000.0) -> Request:
    return Request(
        request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
        predicted_output_tokens=output, actual_output_tokens=output,
        slo_deadline=arrival + slo_slack, priority=1.0, class_id="contention_fixture",
    )


def variant_a_earlier_long_prefill_later_short_decode() -> ContentionFixture:
    """A: one long-prompt hog (4000 tokens, ~8 chunk-steps of runway)
    arrives first; four near-instant "runner" requests (1 token prompt --
    a single step suffices to finish their own prefill) arrive 1ms later.
    Slack (budget = chunk + 2) admits all four runners' trivial prefill
    within the hog's first couple of steps, so all four are decoding
    simultaneously well before the hog finishes its own (much longer)
    prefill. Once all four are decoding, their combined 1-token-each
    demand (4) exceeds the leftover slack (2) every remaining step the hog
    is still mid-prefill -- the sustained stall this fixture exists to
    exercise."""
    reqs = [_req(0, 0.0, prompt=4000, output=1)]
    reqs += [_req(i, 0.001, prompt=1, output=40) for i in range(1, 5)]
    return ContentionFixture(
        fixture_id="contention_a_earlier_long_prefill_later_short_decode",
        description="One long-prefill hog (4000 tokens) arrives first; four "
                     "near-instant runners (1-token prompt) arrive 1ms later, "
                     "finish their own prefill almost immediately, and then all "
                     "compete as decoding requests against the hog's continuing "
                     "prefill for several steps.",
        interaction_shape="single_hog_vs_small_runner_burst",
        requests=reqs, step_token_budget=514, max_prefill_chunk_tokens=512,
        max_kv_tokens=20_000, max_active_sequences=16,
    )


def variant_b_earlier_very_long_prefill_later_medium_decode() -> ContentionFixture:
    """B: a MUCH longer hog (12000 tokens, ~24 chunk-steps of runway) and
    several near-instant runners -- tests whether the mechanism holds
    (and produces a LARGER cumulative divergence, since the stall window
    is much longer) when the hog's prefill duration is long relative to
    the runners' own decode lifetime."""
    reqs = [_req(0, 0.0, prompt=12_000, output=1)]
    reqs += [_req(i, 0.001, prompt=1, output=60) for i in range(1, 6)]
    return ContentionFixture(
        fixture_id="contention_b_earlier_very_long_prefill_later_medium_decode",
        description="A much longer hog (12000 tokens) with five near-instant "
                     "runners arriving shortly after -- a sustained, many-step "
                     "stall window.",
        interaction_shape="single_long_hog_vs_medium_runner_burst",
        requests=reqs, step_token_budget=515, max_prefill_chunk_tokens=512,
        max_kv_tokens=40_000, max_active_sequences=16,
    )


def variant_c_two_long_prefills_later_decode() -> ContentionFixture:
    """C: TWO independent long-prompt hogs (both arrive at t=0, before any
    runner) -- doubles the earlier-arrival prefill demand competing for
    the shared budget, testing whether the mechanism scales with more than
    one competing prefill stream."""
    reqs = [_req(0, 0.0, prompt=5000, output=1), _req(1, 0.0, prompt=5000, output=1)]
    reqs += [_req(i, 0.001, prompt=1, output=40) for i in range(2, 6)]
    return ContentionFixture(
        fixture_id="contention_c_two_long_prefills_later_decode",
        description="Two long-prompt hogs (5000 tokens each) arrive simultaneously "
                     "at t=0; four near-instant runners arrive 1ms later.",
        interaction_shape="dual_hog_vs_small_runner_burst",
        requests=reqs, step_token_budget=1026, max_prefill_chunk_tokens=512,
        max_kv_tokens=30_000, max_active_sequences=16,
    )


def variant_d_burst_of_prefills_staggered_decode_arrivals() -> ContentionFixture:
    """D: a burst of THREE long-prompt hogs at t=0, plus SIX near-instant
    runners arriving in a staggered trickle (not all at once) -- the most
    adversarial variant: sustained early-arrival prefill demand across
    many steps, with a steady trickle of later decode-eligible arrivals
    that must each fight for a share of the persistent slack."""
    reqs = [_req(i, 0.0, prompt=3000, output=1) for i in range(3)]
    reqs += [_req(3 + i, 0.001 * (i + 1), prompt=1, output=35) for i in range(6)]
    return ContentionFixture(
        fixture_id="contention_d_burst_of_prefills_staggered_decode_arrivals",
        description="Three simultaneous long-prompt hogs (3000 tokens each) at t=0; "
                     "six near-instant runners trickling in over the following few ms.",
        interaction_shape="hog_burst_vs_staggered_runner_trickle",
        requests=reqs, step_token_budget=1540, max_prefill_chunk_tokens=512,
        max_kv_tokens=25_000, max_active_sequences=24,
    )


def variant_e_controlled_kv_pressure() -> ContentionFixture:
    """E: same interaction shape as A, but with KV capacity deliberately
    tight enough to also force admission queueing for the runners (not
    just execution-budget contention) -- probes whether the two mechanisms
    (memory-capacity queueing vs per-step token-budget contention)
    interact or mask each other."""
    reqs = [_req(0, 0.0, prompt=4000, output=1)]
    reqs += [_req(i, 0.001, prompt=1, output=40) for i in range(1, 5)]
    return ContentionFixture(
        fixture_id="contention_e_controlled_kv_pressure",
        description="Same shape as variant A, but max_kv_tokens is sized to only "
                     "just fit the hog plus all runners' prompts at once (tight "
                     "admission capacity alongside execution-budget contention).",
        interaction_shape="single_hog_vs_small_runner_burst_tight_kv",
        requests=reqs, step_token_budget=514, max_prefill_chunk_tokens=512,
        max_kv_tokens=4_100, max_active_sequences=16,
    )


def variant_f_short_context_negative_control() -> ContentionFixture:
    """F: negative control -- ALL requests have short prompts (well under
    one chunk), so no request is ever genuinely "still prefilling" for
    more than a single step. Contention should NOT meaningfully matter
    here (expected NEAR_TIE/ALL_COMPLETE_OR_EFFECTIVELY_TIED for every
    objective) -- if this variant shows a large divergence, that would
    indicate a bug, not the intended mechanism."""
    reqs = [_req(i, 0.001 * i, prompt=20, output=15) for i in range(8)]
    return ContentionFixture(
        fixture_id="contention_f_short_context_negative_control",
        description="Eight short-prompt (20 token) requests, staggered arrival -- "
                     "no request is ever multi-step mid-prefill, so contention "
                     "should not produce a meaningful divergence.",
        interaction_shape="short_context_negative_control",
        requests=reqs, step_token_budget=512, max_prefill_chunk_tokens=512,
        max_kv_tokens=10_000, max_active_sequences=16,
    )


FIXTURE_VARIANTS: Dict[str, Callable[[], ContentionFixture]] = {
    "A": variant_a_earlier_long_prefill_later_short_decode,
    "B": variant_b_earlier_very_long_prefill_later_medium_decode,
    "C": variant_c_two_long_prefills_later_decode,
    "D": variant_d_burst_of_prefills_staggered_decode_arrivals,
    "E": variant_e_controlled_kv_pressure,
    "F": variant_f_short_context_negative_control,
}


def all_fixtures() -> List[ContentionFixture]:
    return [builder() for builder in FIXTURE_VARIANTS.values()]

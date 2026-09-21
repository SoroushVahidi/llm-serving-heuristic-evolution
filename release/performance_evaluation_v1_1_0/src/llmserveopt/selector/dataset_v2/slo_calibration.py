"""Policy-independent SLO calibration for synthetic contention windows.

Context (see docs/selector_v2_contention_frontier_search.md and
docs/selector_v2_slo_calibrated_frontier_search.md): every ad-hoc
scenario generator built for the Selector v2 contention-validation work
so far (`contention_fixtures.py`'s `_req()` default, `run_selector_v2_
overnight_validation.py`'s `_random_window`, the first
`selector_v2_contention_frontier_search.py`) used a fixed
`slo_deadline=1000.0` placeholder -- deliberately loose so those tasks
could isolate "does the execution mechanism produce ANY divergence" from
"does that divergence matter for the SLO-gated objective". It answered
the first question; this module answers the second by deriving a
calibrated, per-request deadline BEFORE any policy runs.

Method chosen (of the three considered -- see the doc's "SLO calibration
method" section for why the other two were not): **reference-service-
model estimate**. `reference_latency()` computes each request's
UNCONTENDED service time directly from the `ServiceModel` alone --
prefill time (chunked at `max_prefill_chunk_tokens`, `step_size` seconds
per step) plus decode time (`predicted_output_tokens * step_size`) -- with
no policy, no simulator run, no contention, and no other request in the
window even considered. This is:

* Policy-independent by construction: the function signature never
  accepts a policy, RunMetrics, or CompletedRequest -- there is nothing
  to leak a label from, and no policy's own achieved/best/oracle latency
  is ever read.
* Deterministic and reproducible: a pure function of
  (Request, ServiceModel).
* Not "the best case ever achievable in this window", because it ignores
  admission queueing and inter-request contention entirely -- every
  candidate policy's ACTUAL latency in a contended window is expected to
  be >= this reference, often well above it. The calibration multiplier
  (see `CALIBRATION_MULTIPLIER_GRID`) controls how much slack above that
  floor is granted.

Rejected alternatives (documented, not silently dropped):

* "Neutral reference policy" (fixed-priority FIFO, run once, freeze its
  latency as the deadline): rejected because ANY policy run, even a
  "neutral" one, is influenced by the SAME contention mechanism under
  audit -- freezing its output as the deadline would make whichever
  OTHER policy happens to schedule closest to FIFO's own admission order
  look artificially favored. The reference-service-model estimate has no
  such coupling because it never schedules anything.
* Percentile-of-observed-latency (what
  `selector_v2_contention_frontier_slo_sensitivity.py` used, post-hoc,
  for the exploratory sensitivity sweep in the prior task): rejected as
  the PRODUCTION calibration method because it requires running every
  candidate policy first, which both couples the deadline to the
  candidates under test and cannot be computed before generating a
  window (chicken-and-egg for a generator that wants to calibrate SLOs at
  construction time). Still valid, and still used, purely as a
  diagnostic cross-check (see `docs/selector_v2_slo_calibrated_frontier_
  search.md`'s calibration-method comparison).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...core.types import Request
from ...simulator.service_model import ServiceModel

SLO_CALIBRATION_SCHEMA_VERSION = "v1_reference_service_model"

# Bounded diagnostic grid (section 4): multipliers applied to the
# reference-service-model estimate. Never selected by which policy wins
# at a given value -- see the calibration-grid diagnostic script for the
# criteria (avoid universal success/failure, maximize discriminative
# fraction as a secondary tiebreak only after the primary saturation
# criterion).
CALIBRATION_MULTIPLIER_GRID = (0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)


@dataclass(frozen=True)
class ReferenceLatencyEstimate:
    request_id: int
    reference_prefill_s: float
    reference_ttft_s: float          # == reference_prefill_s + one decode step
    reference_tpot_s: float          # step_size, uncontended
    reference_e2e_s: float           # prefill + full decode, uncontended


def reference_latency(request: Request, service_model: ServiceModel) -> ReferenceLatencyEstimate:
    """Uncontended service-time estimate for one request under
    `service_model` alone -- no policy, no other request, no queueing.
    See module docstring for why this is the calibration basis."""
    if service_model.enable_prefill_modeling:
        n_prefill_steps = service_model.prefill_steps(request.prompt_tokens)
    else:
        n_prefill_steps = 0
    reference_prefill_s = n_prefill_steps * service_model.step_size
    reference_tpot_s = service_model.step_size
    reference_ttft_s = reference_prefill_s + reference_tpot_s
    n_decode_tokens = max(1, request.predicted_output_tokens)
    reference_e2e_s = reference_prefill_s + n_decode_tokens * service_model.step_size
    return ReferenceLatencyEstimate(
        request_id=request.request_id,
        reference_prefill_s=reference_prefill_s,
        reference_ttft_s=reference_ttft_s,
        reference_tpot_s=reference_tpot_s,
        reference_e2e_s=reference_e2e_s,
    )


def calibrate_e2e_deadline(request: Request, service_model: ServiceModel, multiplier: float) -> float:
    """`request.arrival_time + multiplier * reference_e2e_s` -- the single
    E2E deadline this module's callers assign to a freshly-constructed
    `Request` (historical/single-SLO schema, `slo_deadline`)."""
    ref = reference_latency(request, service_model)
    return request.arrival_time + multiplier * ref.reference_e2e_s


def calibrate_dual_slo(
    request: Request, service_model: ServiceModel, multiplier: float,
) -> tuple[float, float]:
    """`(ttft_slo_s, tpot_slo_s)` -- the new dual-SLO schema (section 3).
    Not stored on `Request` (that dataclass keeps its historical single-
    deadline field unchanged, see `SLO_CALIBRATION_SCHEMA_VERSION` for the
    versioning note) -- consumed post-hoc against `CompletedRequest.ttft`/
    `.tpot`, the same pattern the prior task's SLO-sensitivity script used
    for the single-deadline case."""
    ref = reference_latency(request, service_model)
    return multiplier * ref.reference_ttft_s, multiplier * ref.reference_tpot_s


def calibrate_window_e2e(
    requests: Sequence[Request], service_model: ServiceModel, multiplier: float,
) -> list[Request]:
    """Returns a NEW list of `Request`s with `slo_deadline` replaced by
    the calibrated value -- does not mutate the input (Request is
    frozen-by-convention elsewhere in this codebase; rebuilt via
    `dataclasses.replace` to make that explicit)."""
    from dataclasses import replace as _replace
    return [
        _replace(r, slo_deadline=calibrate_e2e_deadline(r, service_model, multiplier))
        for r in requests
    ]

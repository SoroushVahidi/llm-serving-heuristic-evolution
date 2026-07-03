# Real-LLM v2 Latency Fit — Simulator Integration Plan

**Status: not yet wired into the simulator by default.** This document
explains how the v2 latency fit (`docs/real_llm_latency_model_v2.md`,
`experiments/real_llm/latency_model_fit_v2/`) *could* calibrate simulator
service-time assumptions, and what an optional config for it looks like. No
simulator code path currently reads any real-LLM hosted-API fit; this is a
plan and a data file, not an implementation.

## Why this is a separate question from GPU calibration

The simulator already has one opt-in calibrated service model:
`src/llmserveopt/simulator/calibrated_service_model.py`
(`service_model: type: calibrated` in YAML, selected via
`build_service_model_from_config()` in
`src/llmserveopt/simulator/service_model_factory.py`, default remains
`type: synthetic`). That model is backed by **local GPU measurements**
(`results/gpu_calibration/service_curves.json`) — per-token prefill/decode
step counts measured on hardware the simulator itself controls.

The v2 real-LLM fit is a different kind of measurement: **external,
black-box, wall-clock request latency** from a hosted API the simulator has
no control over (no visibility into Cohere's or Gemini's internal batching,
GPU allocation, or scheduling — see the "unsafe claims" section of
`docs/real_llm_latency_model_v2.md`). Plugging it into the simulator the
same way as GPU calibration would conflate two different things: "how fast
a serving engine's internals are" (what `ServiceModel`/`CalibratedServiceModel`
represent) vs. "how fast a request round-trip to someone else's hosted
endpoint looks from outside" (what this fit measures). Treating the latter
as a drop-in replacement for the former is exactly the "unsafe claim" this
project's own docs warn against.

## What a real integration would need (not done here)

If a future task wants an actual `service_model: type: real_llm_calibrated`
option, it would need to decide (and test) at least:

1. **What the simulator asks for vs. what this fit predicts.** The
   simulator's service model answers "how many steps does this request's
   prefill/decode take," a per-token/per-step question. This fit answers
   "what wall-clock TTFT/latency did a whole streamed request take,"
   collapsing prefill+decode+network+provider-side scheduling into one
   opaque number. Translating between the two isn't a unit conversion —
   it's a modeling choice (e.g., treat `ttft_seconds` as "prefill+queue
   time" and `(latency-ttft)` as "decode time," which is what
   `fit_latency_model_ttft_plus_decode()`/`provider_decode_rates.csv`
   already do, but that equivalence has not been validated against this
   simulator's own step semantics).
2. **Whether "provider-controlled scheduling" makes sense as a per-request
   fixed cost at all**, given the simulator's request stream has its own
   scheduler making batching/preemption decisions this fit's source data
   never experienced (a single external client hitting a hosted endpoint,
   not the simulator's internal multi-tenant scheduler).
3. **Provider/model selection** — which of Cohere's ~88 tok/s or Gemini's
   ~289 tok/s (both `overall`, not per-target — see the "not reliable"
   list in `docs/real_llm_latency_model_v2.md`) a given experiment should
   assume, and how that choice should be surfaced in results so it's never
   silently conflated with the synthetic/GPU-calibrated defaults.
4. **Tests and docs proving no existing experiment's results change** unless
   a config explicitly opts in — the same bar `calibrated_service_model.py`
   already meets (`service_model: type: synthetic` remains the untouched
   default; Phase 1.5 experiments are documented as fully reproducible
   under it).

None of this is implemented in this task. Given the "unsafe claims" caveat
above, doing it properly is a design decision worth its own task, not a
byproduct of a latency-fit task.

## What is provided now

`configs/real_llm_latency/cohere_gemini_v2_fit.yaml` — a **data-only**
reference config capturing the `overall` (n=108, most-reliable) per-provider
decode rates and TTFT/latency summary stats from this fit, in a structured,
documented form, so a future integration task doesn't need to re-derive them
from raw JSON. No loader reads this file today; `grep -r
real_llm_latency configs/` and `grep -r cohere_gemini_v2_fit src/` both
return nothing outside this doc — that is intentional, not an oversight.

## If/when this gets wired in

The safest starting point would mirror `calibrated_service_model.py`'s
pattern exactly: a new, additive `service_model: type: real_llm_calibrated`
option, never replacing `type: synthetic`'s status as the default, gated by
its own tests proving existing configs (`configs/default_simulator.yaml`
and friends) are byte-for-byte unaffected. Until that lands, use this fit
for offline analysis and simulator-assumption sanity-checking only (e.g.,
"is our synthetic decode rate in the right ballpark vs. a real 88-289
tok/s hosted range?"), not as a runtime dependency of any experiment.

## See also

- `docs/real_llm_latency_model_v2.md` — the fit this plan is for, including
  safe/unsafe claims.
- `src/llmserveopt/simulator/calibrated_service_model.py`,
  `src/llmserveopt/simulator/service_model_factory.py` — the existing
  opt-in calibrated-service-model pattern this plan follows.
- `configs/real_llm_latency/cohere_gemini_v2_fit.yaml` — the data-only
  reference config described above.

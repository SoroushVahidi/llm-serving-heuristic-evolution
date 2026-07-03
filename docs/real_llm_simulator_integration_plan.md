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

## Offline sanity-check results (2026-07-03)

`scripts/compare_simulator_to_real_llm_latency.py` evaluates the
simulator's own `ServiceModel`/`CalibratedServiceModel` timing formulas
directly (no simulation run, no live API calls) against the v2 fit, writing
`experiments/real_llm/simulator_latency_sanity_check/{summary.json,
summary.md, comparison_by_target_output_tokens.csv,
comparison_by_provider.csv}`. Headline numbers:

| Entity | Decode rate (tok/s) | Prefill/TTFT analogue (512-token prompt) |
|---|---|---|
| Cohere (hosted) | 88.5 | 0.246s (TTFT, incl. network+scheduling) |
| Gemini (hosted) | 288.9 | 0.674s (TTFT, incl. network+scheduling) |
| Simulator default (`type: synthetic`) | 1000.0 | 0.000s (no analogue — instant prefill) |
| Simulator calibrated, batch=1 (RTX 5060 Ti, Qwen2.5-0.5B) | 147.7 | 0.017s (local GPU compute only) |
| Simulator calibrated, batch=8 | 100.2 | 0.017s |

Findings:

- **How the fit relates to simulator assumptions:** the default synthetic
  model (1000 tok/s, constant, no batching/context effect) is 3.5-11.3x
  faster than either hosted provider and has literally no TTFT-analogue
  (prefill is instantaneous by default). The GPU-calibrated variant, despite
  being a real hardware measurement, is for a much smaller model (0.5B) than
  either hosted API and lands closer to Cohere's rate at high batch size,
  closer to neither provider cleanly at batch=1.
- **Why hosted latency is useful but not equivalent to controllable serving
  latency:** the hosted decode rates are a real, order-of-magnitude sanity
  check ("is 1000 tok/s a plausible number for *any* real model? not really
  — even Gemini's fast 289 tok/s is well under a third of that"), but they
  bundle a specific provider's model size, hardware, and serving
  optimizations that this simulator does not and should not try to
  reproduce exactly.
- **What can be calibrated now (as an order-of-magnitude reference, not an
  exact target):**
  - the *existence* of a nonzero, TTFT-like intercept in real latency
    (~0.1-0.2s in the v2 fit's `latency ~ ttft + ...` model) — a signal that
    "zero fixed per-request overhead" (the synthetic default) is an
    optimistic simplification, even though the specific hosted intercept
    value should not be copied in directly (see below);
  - the *effective decode rate* as a plausibility range (tens to a few
    hundred tokens/sec for real models, vs. the synthetic default's 1000);
  - the *qualitative fact* that output-length scaling is linear, which the
    simulator already assumes by construction — this is at least directionally
    consistent, not a contradiction to fix.
- **What cannot be calibrated from hosted APIs:**
  - internal batching — hosted providers never expose how many concurrent
    requests were actually co-scheduled on the same accelerator;
  - GPU scheduling / hardware allocation — entirely invisible from a client;
  - KV-cache paging — no provider surfaces memory-management behavior to
    the client;
  - admission/preemption behavior — a hosted API's queueing/admission
    policy toward *other* tenants' traffic is never observable from one
    client's request stream, unlike the simulator's own scheduler policies,
    which are exactly what this project studies.
- **Recommended next step:** a vLLM real-serving external-baseline pilot —
  running an actual open-weights model behind vLLM (a serving engine whose
  internals *are* inspectable, unlike Cohere/Gemini) would let a future task
  validate the simulator's batching/scheduling assumptions against
  something structurally comparable, not just a black-box wall-clock number.

## vLLM real-serving external-baseline pilot (2026-07-03)

Why vLLM specifically, continuing from the "recommended next step" above:
Cohere and Gemini are black boxes — no batching, scheduling, or KV-cache
behavior is ever observable from a client. vLLM is a serving engine this
project could actually run and inspect, making it structurally comparable
to what `ServiceModel`/`CalibratedServiceModel` represent, not just another
opaque wall-clock number. Concretely, a vLLM pilot could (in a future task)
scrape vLLM's own `/metrics` endpoint for **true internal batch size** —
something no hosted API can ever expose — and compare it directly against
the simulator's own batching decisions under an equivalent workload.

`scripts/run_vllm_serving_baseline_pilot.py` adds this capability, reusing
the same safe synthetic length-targeted prompts as the Cohere/Gemini v2
pilots (`calibration_common.build_length_targeted_prompt`) so results are
directly comparable to `docs/real_llm_latency_model_v2.md` once real
numbers exist. It never calls Cohere, Gemini, OpenAI, or Azure — its only
network target is a vLLM server (local subprocess or an already-running one
at `--server-url`, e.g. on an HPC node).

**Status: not yet run against real vLLM.** This repo's environment doesn't
have vLLM installed (CUDA 13.0 / PyTorch 2.12.0 is too new for vLLM's
prebuilt wheels — the same constraint `configs/gpu_calibration/
online_validation.yaml` already documents for GPU calibration). The
script's live-server code paths (`query_vllm_completion`,
`launch_local_vllm_server`) are written against vLLM's documented
OpenAI-compatible streaming-completions protocol, but have only been
validated against a small stdlib `http.server` fake that reproduces the
same response shape (`tests/test_run_vllm_serving_baseline_pilot.py`), not
a real vLLM instance. `experiments/real_llm/vllm_serving_baseline_pilot/`
currently contains a dry-run plan only (108 requests, same grid shape as
the Cohere/Gemini v2 pilots) — `run_status:
planned_only_vllm_not_installed` in its own `run_config.json`/
`reproducibility.md` says so explicitly.

**Paper-safe status, stated plainly:**
- The vLLM code path has been exercised in `--dry-run`, `--mock`, and
  against a fake local HTTP server standing in for vLLM's SSE protocol.
  It has **not** been exercised against a real vLLM server.
- The current environment cannot install/run vLLM directly — this is a
  CUDA/PyTorch compatibility constraint, not a choice, and not something
  worked around by installing packages into this repo's main environment.
- **Do not cite this as a completed real-serving experiment.** The correct
  characterization until a real server actually runs and completes
  requests is "vLLM scaffold only, awaiting a compatible runtime" — not
  "vLLM baseline pilot" or "validated against vLLM."

To actually run it once vLLM is available on compatible hardware:

```
# Query an already-running server (e.g. launched separately on an HPC node):
python scripts/run_vllm_serving_baseline_pilot.py \
  --allow-live-server --server-url http://<host>:8000 \
  --model Qwen/Qwen2.5-0.5B \
  --output-dir experiments/real_llm/vllm_serving_baseline_pilot_<timestamp>

# Or let the script launch a local server itself (requires `vllm` on PATH):
python scripts/run_vllm_serving_baseline_pilot.py \
  --allow-live-server --launch-server --port 8000 \
  --model Qwen/Qwen2.5-0.5B \
  --output-dir experiments/real_llm/vllm_serving_baseline_pilot_<timestamp>
```

Remaining assumptions to validate once a real run is possible: that the
OpenAI-compatible `/v1/completions` SSE response shape assumed here matches
the deployed vLLM version exactly (vLLM has changed its API surface across
releases), and that `stream_options.include_usage` is supported by that
version (older vLLM releases may not populate `usage` in streamed chunks,
in which case `output_tokens`/`prompt_tokens` would come back `None` and
should be backfilled from the tokenizer instead — not implemented here).

## What is provided now

- `configs/real_llm_latency/cohere_gemini_v2_fit.yaml` — a **data-only**
  reference config capturing the `overall` (n=108, most-reliable) per-provider
  decode rates and TTFT/latency summary stats from this fit, in a structured,
  documented form, so a future integration task doesn't need to re-derive them
  from raw JSON. No loader reads this file today; `grep -r
  real_llm_latency configs/` and `grep -r cohere_gemini_v2_fit src/` both
  return nothing outside this doc — that is intentional, not an oversight.
  (A per-provider split — `cohere_v2_fit.yaml` / `gemini_v2_fit.yaml` — was
  considered and rejected: the combined file already separates
  `providers.cohere`/`providers.gemini` cleanly, and
  `compare_simulator_to_real_llm_latency.py` reads it directly, so a split
  would only add two more files to keep in sync with no new capability.)
- `scripts/compare_simulator_to_real_llm_latency.py` — the offline
  comparison script described above. Reads the fitted config/model dir and
  the simulator's own service-model classes; makes no API calls and no
  simulator config changes.

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
- `scripts/run_vllm_serving_baseline_pilot.py`,
  `tests/test_run_vllm_serving_baseline_pilot.py`,
  `experiments/real_llm/vllm_serving_baseline_pilot/` — the vLLM
  external-baseline pilot described above (currently plan-only; vLLM is not
  installed in this environment).

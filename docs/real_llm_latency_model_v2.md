# Real-LLM Latency Model — v2 (Length-Targeted)

**Status: fitted from existing logs only. No live API calls were made to
produce this document or the fit it describes.**

Source data: the two completed v2 length-targeted pilots —

| Provider | Directory | Model |
|---|---|---|
| Cohere | `experiments/real_llm/cohere_v2_length_targeted_20260703T134447Z/` | `command-r7b-12-2024` |
| Gemini | `experiments/real_llm/gemini_v2_length_targeted_20260703T141723Z/` | `gemini-3.1-flash-lite` (Vertex AI) |

Both completed 108/108 requests with 0 failures. Full fit outputs (JSON,
Markdown, CSVs) live in `experiments/real_llm/latency_model_fit_v2/`,
produced by `scripts/fit_real_llm_latency_model_v2.py`. See
`docs/real_llm_v2_workload_proposal.md` for how the v2 workload was
designed and `docs/real_llm_cohere_gemini_comparison.md` for the pilots'
own black-box comparison writeup (v1-era, predates this fit).

## What was fitted

Three things, per provider and pooled (with a provider indicator):

1. **TTFT model**: `ttft_seconds ~ intercept + target_output_tokens
   (falls back to output_tokens only if a row has no target) +
   prompt_tokens + concurrency_level [+ provider indicator]`.
2. **Provider latency model**: `provider_request_latency_seconds ~
   intercept + ttft_seconds + output_tokens + prompt_tokens +
   concurrency_level [+ provider indicator]`.
3. **Decode-rate table**: `decode_seconds = latency - ttft ~ a *
   output_tokens + b`, fit once per provider overall (n=108) and once per
   (provider, target_output_tokens) group (n=36 each), so
   `effective_decode_rate_tokens_per_sec = 1/a` can be read off at each
   target length instead of only as one provider-wide average.

All three are plain OLS (`numpy.linalg.lstsq`), the same interpretable,
non-black-box approach `scripts/fit_real_llm_latency_model.py` (the v1
fit) already used — this is a calibration baseline, not a predictive
production model.

## How it was fitted

`scripts/fit_real_llm_latency_model_v2.py` reads each pilot directory's
`requests.jsonl` directly (via the shared `load_dataset()` loader in
`fit_real_llm_latency_model.py`, extended to also carry
`target_output_tokens` / `workload_version` / `reached_target_output_range`)
and uses **`provider_request_latency_seconds`** — never
`total_wall_time_seconds` — as the fitted "latency" value, so local
RPM-limiter wait time never enters any coefficient. `ttft_seconds` was
always measured from inside each provider's streaming call, so it needs no
correction either way. See
`tests/test_fit_real_llm_latency_model_v2.py::test_rate_limiter_wait_excluded_from_latency_used_in_fit`
for the regression test.

Outputs, all under `experiments/real_llm/latency_model_fit_v2/`:

- `latency_model_fit_v2.json` / `.md` — full fit results, human-readable.
- `latency_model_fit_v2.csv` — the 216-row record set the fits were computed from.
- `provider_decode_rates.csv` — decode rate by (provider, target_output_tokens).
- `residuals_by_provider.csv` — per-(provider, model_type) residual summary (mean/std/RMSE).
- `model_inputs_manifest.json` — input directories, request counts, error rates, git commit at fit time.

## Why v2 is better than v1

v1's source pilots (`docs/real_llm_cohere_gemini_comparison.md`) used a
prompt that asked for "one short sentence," so generated output length
stayed ~22-35 tokens **regardless of `max_tokens`**, for both providers.
That meant v1's `coef_output_tokens` / `effective_decode_rate_tokens_per_sec`
were fit over a range of essentially one output length — not a real
decode-rate measurement, just noise dressed up as a coefficient. v1's TTFT
model also used realized `output_tokens` as a feature, which is backwards:
TTFT is measured *before* decoding starts, so it cannot causally depend on
how many tokens are eventually generated.

v2 fixes both:

- `target_output_tokens` (64/128/256) is a real, varying,
  **known-in-advance** independent variable, so the TTFT model can
  correctly condition on the requested length rather than the realized one,
  and the decode-rate fit spans an actual 64-256 token range per provider.
- Both providers now show output length that scales with the target (mean
  achieved ratio 0.81-1.00x for Cohere, 0.91-1.04x for Gemini — see the
  fit's own "Output-token distribution by target length" table), so
  `effective_decode_rate_tokens_per_sec` reflects a real relationship for
  the first time in this project: Cohere ≈88 tokens/sec, Gemini ≈289
  tokens/sec (both `overall`, n=108, R²=0.89 / 0.83).

## Which metrics are reliable

**Reliable:**
- TTFT stats (mean/p50/p95/p99) per provider — always measured from inside
  the streaming call, immune to the RPM-wait artifact fixed in
  `docs/real_llm_cohere_gemini_comparison.md`.
- Provider latency stats per provider (same immunity).
- The **`overall`** per-provider decode rate (n=108, R²=0.83-0.89) — this
  is the only decode-rate number reliable enough to quote on its own.
- The provider latency model's `coef_ttft_seconds` and `coef_output_tokens`
  (R²=0.89 Cohere, 0.92 Gemini) — TTFT and output length are legitimately
  strong, well-identified predictors of total latency.
- The raw (model-free) latency-by-target crossover table — this is not a
  regression result, just grouped means, and is the most trustworthy
  single finding in this document (see below).

**Not reliable — flagged explicitly in the fit's own output:**
- Per-`target_output_tokens` decode-rate cells (n=36 each). Within a single
  target group, realized `output_tokens` barely varies (it clusters near
  its own target by construction), so the *slope* — literally the thing
  being estimated — is poorly identified. R² for these cells ranges
  0.00-0.49 (Cohere) and 0.00-0.47 (Gemini); several are flagged
  "⚠️ low, do not trust" directly in `provider_decode_rates.csv` /
  `latency_model_fit_v2.md`. Do not quote e.g. "Gemini decodes at 553
  tokens/sec at target=128" — that number is fit noise (R²=0.008).
- Both providers' standalone TTFT models have low R² (0.05 Cohere, 0.06
  Gemini) — this is a *correct*, not broken, result: it means TTFT genuinely
  doesn't vary much with prompt length, output target, or concurrency
  within a provider at this pilot's scale. The pooled TTFT model's much
  higher R² (0.55) comes almost entirely from the provider-indicator term
  (`coef_is_Gemini=+0.43s`) — i.e., "which provider" explains far more of
  TTFT's variance than any request parameter does.
- Any coefficient extrapolated outside this pilot's own grid (targets
  64/128/256, concurrency 1-8, 3 prompt buckets, 108 requests/provider).

## The target=64 crossover finding remains visible

Yes — confirmed directly from the raw per-target means (not a modeling
artifact):

| target | Cohere latency (s) | Gemini latency (s) | faster |
|---|---|---|---|
| 64  | 0.892 | 1.045 | **Cohere** |
| 128 | 1.843 | 1.112 | **Gemini** |
| 256 | 3.082 | 1.645 | **Gemini** |

Cohere's lower TTFT (~0.25s vs. Gemini's ~0.67s) wins at very short outputs;
Gemini's faster decode (~289 vs. ~88 tokens/sec, `overall` rate) overtakes
it by target=128 and pulls further ahead at target=256. This crossover was
first observed in the raw pilot comparison and is reproduced here from the
same underlying data, independent of any OLS assumption.

## How this should calibrate the simulator

See `docs/real_llm_simulator_integration_plan.md` for the full discussion.
Short version: **this fit is not yet wired into the simulator by default.**
An optional, data-only reference config
(`configs/real_llm_latency/cohere_gemini_v2_fit.yaml`) captures the
`overall` per-provider decode rates and TTFT/latency stats in a structured
form for future use, but no simulator code path currently reads it, and no
existing default was changed.

## Safe claims

- v2 successfully elicited length-dependent outputs from both providers
  (mean achieved ratio 0.81-1.04x target, vs. v1's flat ~23-32 tokens
  regardless of `max_tokens`).
- Provider latency increases with output length for both providers (Cohere
  0.89s→1.84s→3.08s, Gemini 1.04s→1.11s→1.65s across targets 64/128/256).
- Cohere and Gemini both completed the v2 grid with 0 failures (108/108
  each, 216/216 total).
- The fitted model provides a reproducible first calibration for hosted-LLM
  latency behavior — anyone can rerun
  `scripts/fit_real_llm_latency_model_v2.py` against the same logs and get
  the same numbers, with no API calls required.
- These hosted-API experiments validate latency/cost behavior but not
  control over provider-internal scheduling — both pilots are black-box
  client-observed measurements only.

## Unsafe claims

- **Do not claim this project's scheduler beats Cohere or Gemini.** Nothing
  here involves any scheduler of ours; it compares two providers' externally
  observed latency to each other.
- **Do not claim this fitted model generalizes to all models or providers.**
  It covers exactly two models (`command-r7b-12-2024`,
  `gemini-3.1-flash-lite`), one pilot each, 108 requests each, one time
  window, one region/client. A different Cohere or Gemini model, a
  different day, or a different account tier could easily produce different
  numbers.
- **Do not claim Azure, OpenAI, or Fireworks behavior without running
  them.** No data exists for these providers in this fit; the "pooled"
  model only ever has 2 providers.
- **Do not treat hosted-API provider latency as equivalent to controllable
  vLLM server internals.** This is an external, black-box measurement of a
  managed service under someone else's batching/scheduling/hardware
  allocation — it calibrates "what latency looks like from outside," not
  "how a serving engine's internals behave," which is what the simulator
  itself models.
- **Do not quote a per-target-length decode rate with R² < 0.5** (see the
  "not reliable" list above) as if it were a measurement; several of these
  cells are fit noise, not signal.

## See also

- `experiments/real_llm/latency_model_fit_v2/latency_model_fit_v2.md` — the
  full auto-generated fit report this document summarizes.
- `docs/real_llm_simulator_integration_plan.md` — how (and whether) to wire
  this into simulator service-time assumptions.
- `docs/real_llm_v2_workload_proposal.md` — the v2 workload design that made
  this fit possible.
- `docs/real_llm_cohere_gemini_comparison.md` — the v1-era pilot comparison
  and RPM-wait-artifact writeup this fit's methodology builds on.

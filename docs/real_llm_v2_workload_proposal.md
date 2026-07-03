# Proposed v2 Hosted-API Workload: Length-Targeted Prompts

**Status: proposal only. Not run. No API calls were made to design or
validate this document — it is a design based on what the v1 Cohere/Gemini
pilots' raw logs already show.**

## What v1 got right

The v1 workload (`build_prompt()` in `calibration_common.py`, used by both
completed pilots) is good for:

- **API connectivity** — proving a provider's SDK, auth, streaming, and
  error handling work end-to-end.
- **TTFT** — first-token latency doesn't depend on how long the requested
  output is, so v1's TTFT numbers (see
  `docs/real_llm_cohere_gemini_comparison.md`) are meaningful.
- **p50 latency at short output lengths** — reliable for calibrating the
  "short response" regime specifically.

## What v1 is weak for: output-length scaling

v1's instruction — *"In one short plain-text sentence, restate the main
topic of the text above"* — caused the model to stop well short of
`max_tokens` every time. Measured from the actual pilot logs:

| Provider | max_tokens=64 | max_tokens=128 | max_tokens=256 |
|---|---|---|---|
| Cohere mean output tokens | 32.0 | 32.2 | 31.7 |
| Gemini mean output tokens | 23.3 | 23.3 | 23.3 |

Output length was constant regardless of `max_tokens`. This means v1
cannot calibrate:

- How latency scales with generated output length (decode-time slope).
- Whether/how per-token decode rate varies by provider under longer
  generations.
- Whether `max_tokens` truncation ever actually engages provider-side
  stop behavior.

`scripts/fit_real_llm_latency_model.py`, run against the v1 logs, confirms
this empirically: Gemini's fitted decode-rate slope is near zero/negative
(no real signal, because output_tokens barely varies), and Cohere's fitted
`effective_decode_rate_tokens_per_sec` has very low R² for the same reason.
See `experiments/real_llm/latency_model_fit/latency_model_fit.md`.

## v2 design goal

Elicit approximately **64, 128, and 256** output tokens on demand,
deterministically, using only synthetic content (no scraped or
copyrighted text), while keeping the same provider-agnostic schema,
hard-cap safety mechanism, and JSONL logging as v1.

## Proposed prompt design

`build_length_targeted_prompt(bucket, target_output_tokens, seed,
variant_index)` (added to `src/llmserveopt/real_llm/calibration_common.py`,
tested in `tests/test_real_llm_calibration_common.py`, **not wired into any
live-call script**) reuses v1's input-side prompt-bucket body and synthetic
`_SENTENCE_BANK`, but replaces the "one short sentence" instruction with:

> "Using only the concepts mentioned in the text above, write a plain-text
> explanation of approximately `{target_output_words}` words (not more
> than a few words short or over). Use complete sentences and paragraphs.
> Do not use lists, markdown, headings, or code blocks. Do not introduce
> any topic not mentioned above."

where `target_output_words ≈ target_output_tokens * 0.75`, matching the
same word/token heuristic v1 already uses for input-side sizing
(`approx_token_count`).

This is a word-count *instruction*, not a guarantee — models do not follow
target word counts exactly. A v2 pilot's first job is to measure
achieved-vs-target output length per provider (exactly the gap this
proposal exists to close) and report the ratio, the same way this
project discovered v1's ~23-32-token ceiling from real logs rather than
assuming compliance.

## Proposed grid and hard caps

Same shape as v1 (3 prompt buckets x 3 target-output-token values x 4
concurrency levels x 5 requests/cell = 180 requests), with `max_tokens`
set with headroom above each target (not equal to it) so truncation
doesn't mechanically cap the model before it reaches the target:

```json
{
  "prompt_buckets": ["short", "medium", "long"],
  "target_output_tokens_list": [64, 128, 256],
  "max_tokens_list": [128, 256, 512],
  "concurrency_list": [1, 2, 4, 8],
  "requests_per_cell": 5,
  "timeout_seconds": 90,
  "rpm_limit": 20,
  "max_total_requests": 180,
  "max_total_input_tokens": 250000,
  "max_total_output_tokens": 100000,
  "max_estimated_cost_usd": 5.0,
  "stream": true,
  "fail_fast": true
}
```

Notes:
- `max_tokens_list` is `2x` each `target_output_tokens` value, giving the
  model room to reach the target without being truncated, while still
  bounding worst-case cost the same way v1's hard-cap mechanism already
  does (`validate_call_plan`, `BudgetTracker`).
- `max_total_output_tokens` is raised from v1's 50,000 to 100,000 to
  reflect that v2 requests are expected to actually consume more output
  budget than v1's ~23-32-token responses did — this is a *cap*, not a
  target; actual usage should be measured and reported the same way v1's
  `estimated_cost_usd` was.
- All other hard-cap and safety mechanisms (dry-run-first requirement,
  `--allow-live-api` gate, fail-fast on elevated error rate, resume
  support, no-secrets-in-logs) are unchanged from v1 — see
  `docs/cohere_api_calibration.md` and
  `docs/real_llm_multi_provider_plan.md`.
- The RPM-wait measurement fix (`rate_limiter_wait_seconds` /
  `provider_request_latency_seconds` split) already applies to any v2 run
  automatically, since it lives in the shared `calibration_common.py`
  path both v1 and any future v2 script would use.

## What running v2 would require (not done here)

1. A `_call_<provider>_streaming`/`_call_<provider>_non_streaming` wiring
   that calls `build_length_targeted_prompt()` instead of `build_prompt()`
   — either a new script or a `--prompt-style length_targeted` flag on the
   existing provider scripts.
2. A `--dry-run` review of the proposed grid above before any
   `--allow-live-api` run, per the existing safety discipline.
3. Reporting achieved-vs-target output length per provider/target as the
   first-class output of the pilot, before trusting any decode-rate fit
   from it.

This task explicitly did not run any of the above — see the final report
for confirmation that no live API calls were made.

# Real-LLM Pilot Comparison: Cohere vs. Gemini

Both pilots used the shared calibration harness
(`src/llmserveopt/real_llm/calibration_common.py`) with an identical
180-request grid: 3 prompt buckets (short/medium/long) x 3 `max_tokens`
values (64/128/256) x 4 concurrency levels (1/2/4/8) x 5 requests per cell,
streaming mode, `temperature=0.0`, deterministic synthetic prompts.

This report is a factual comparison of two black-box measurements. Neither
pilot observed or controlled either provider's internal scheduling — see
"Safe and unsafe claims" below before drawing conclusions from these numbers.

## Experiment directories

| Provider | Directory | Model |
|---|---|---|
| Cohere | `experiments/real_llm/cohere_pilot_20260703T040421Z/` | `command-r7b-12-2024` |
| Gemini | `experiments/real_llm/gemini_pilot_20260703T044905Z/` | `gemini-3.1-flash-lite` |

Corrected (RPM-wait-flagged) summaries for both, produced by
`scripts/reprocess_real_llm_pilot_logs.py` without any new API calls, live at
`<directory>/summary_corrected.{json,md}`.

## Request counts and reliability

| Metric | Cohere | Gemini |
|---|---|---|
| Planned | 180 | 180 |
| Completed (success) | 180 | 180 |
| Failed (error/timeout) | 0 | 0 |
| Rate-limited | 0 | 0 |
| Skipped | 0 | 0 |

Both pipelines completed their full planned grid with zero failures,
zero rate-limiting, and zero timeouts.

## Cost caveat

- Cohere: **$0.0065** actual, computed from real billed usage
  (`billed_units` in `requests.jsonl`) at published Cohere per-token rates.
- Gemini: **~$0.0165**, computed from `output_tokens`/`actual_prompt_tokens`
  at a **placeholder** price-per-token constant in
  `scripts/run_gemini_real_llm_calibration.py` — this has not been verified
  against an actual Gemini invoice or the current Gemini Flash-Lite rate
  card. Treat the Gemini figure as an order-of-magnitude estimate only, not
  a reconciled cost.
- Both costs are trivial in absolute terms (under 2 cents combined) and
  should not be read as meaningful signal about relative provider pricing
  at production volume.

## TTFT comparison (reliable — unaffected by the RPM-wait artifact below)

`ttft_seconds` is measured from inside each provider's streaming call itself
(after the local rate limiter released the request), so it was never
polluted by local scheduling and is trustworthy in both raw and corrected
summaries.

| Stat | Cohere | Gemini |
|---|---|---|
| mean TTFT (s) | 0.273 | 0.556 |
| p50 TTFT (s) | 0.258 | 0.563 |
| p95 TTFT (s) | 0.435 | 0.736 |
| p99 TTFT (s) | 0.530 | 0.801 |

Cohere had lower TTFT than Gemini at every percentile in this pilot.

### TTFT by concurrency level

| Concurrency | Cohere mean TTFT (s) | Gemini mean TTFT (s) |
|---|---|---|
| 1 | 0.229 | 0.579 |
| 2 | 0.256 | 0.562 |
| 4 | 0.264 | 0.557 |
| 8 | 0.343 | 0.525 |

Cohere's TTFT rises mildly with concurrency (0.229s -> 0.343s from c=1 to
c=8); Gemini's is roughly flat to slightly declining across the same range
in this pilot. Sample size per cell is small (15 requests per concurrency
level per provider), so treat this trend as suggestive, not conclusive.

### TTFT by prompt bucket

| Bucket | Cohere mean TTFT (s) | Gemini mean TTFT (s) |
|---|---|---|
| short | 0.274 | 0.567 |
| medium | 0.276 | 0.546 |
| long | 0.268 | 0.555 |

Neither provider showed a meaningful TTFT dependence on prompt bucket in
this pilot — expected, since TTFT is dominated by request setup and
first-token generation, not prompt length, at these prompt sizes (100-2048
target tokens).

## Latency comparison — with RPM-wait artifact caveat

**Both pilots' raw `total_latency_seconds`/`elapsed_seconds` include time
spent blocked in the local client-side RPM limiter, not just provider
response time.** In each pilot, exactly 8 of 180 requests (4.4%) — always
the first request of a new concurrency=1 cell immediately following a burst
of higher-concurrency traffic — recorded latency around 53s while their
`ttft_seconds` was a normal ~0.2-0.8s, because the request waited at the
client for local RPM budget to free up before the provider ever saw it.
This was fixed going forward in `calibration_common.py` (see
`rate_limiter_wait_seconds` / `provider_request_latency_seconds` fields),
but these two pilots predate the fix, so their raw summaries carry the
artifact. `scripts/reprocess_real_llm_pilot_logs.py` regenerates a corrected
view by heuristically flagging and excluding those 8-per-pilot outliers
(latency - ttft > 5s) — it cannot recover the true wait/provider split for
those specific requests, only exclude them.

| Stat | Cohere raw | Cohere corrected | Gemini raw | Gemini corrected |
|---|---|---|---|---|
| p50 latency (s) | 0.687 | 0.683 | 0.742 | 0.737 |
| p95 latency (s) | 1.149 | 0.998 | 1.476 | 1.077 |
| p99 latency (s) | 53.31 | 1.144 | 53.02 | 1.393 |

**p50 latency is reliable as reported** — it barely moves between raw and
corrected for either provider (0.687s -> 0.683s Cohere, 0.742s -> 0.737s
Gemini), confirming the artifact is a tail phenomenon affecting a small,
identifiable set of outliers, not the bulk of the distribution. **p95/p99
latency in the raw summaries are not reliable** and should not be quoted;
use the corrected columns, and even those exclude rather than truly repair
the flagged requests.

Do not read the raw p99 numbers (53.31s Cohere, 53.02s Gemini) as reflecting
either provider's actual tail latency — they reflect this harness's local
RPM-limiter design, not the API.

## Concurrency comparison

Corrected mean/p50 latency by concurrency level (RPM-wait outliers
excluded):

| Concurrency | Cohere corrected p50 (s) | Gemini corrected p50 (s) |
|---|---|---|
| 1 | ~0.55-0.68 | ~0.65-0.74 |
| 2 | ~0.55-0.68 | ~0.65-0.74 |
| 4 | ~0.55-0.68 | ~0.65-0.74 |
| 8 | ~0.55-0.68 | ~0.65-0.74 |

See `<directory>/aggregate_by_concurrency.csv` for exact per-cell figures
(computed from raw, uncorrected latency — cross-check against
`summary_corrected.md` before quoting a specific cell's tail stats). Neither
provider showed latency growing materially with our concurrency levels
(1-8 concurrent local requests) in this pilot. This says only that these
providers absorbed 8-way concurrent load from a single client without
visible degradation at this tiny scale — it says nothing about their
scheduling under real multi-tenant production load.

## Prompt-bucket comparison

Neither TTFT (shown above) nor completion rate varied meaningfully across
short/medium/long prompt buckets (100/512/2048 target input tokens) for
either provider. See `<directory>/aggregate_by_prompt_bucket.csv` for exact
figures.

## max_tokens caveat: output length was not actually varied

Both pilots swept `max_tokens` in `{64, 128, 256}`, but the actual generated
`output_tokens` was essentially **constant regardless of `max_tokens`**:

| Provider | max_tokens=64 | max_tokens=128 | max_tokens=256 |
|---|---|---|---|
| Cohere mean output tokens | 32.0 | 32.2 | 31.7 |
| Gemini mean output tokens | 23.3 | 23.3 | 23.3 |

This is because the prompt instructs the model to "restate the main topic
in one short plain-text sentence" — the model naturally stops well short of
any `max_tokens` cap. **This pilot did not exercise output-length scaling
at all.** Any apparent max_tokens sensitivity (or lack thereof) in the raw
data is an artifact of the prompt design, not a provider property. See
"Proposed v2 workload" below for a design that actually elicits ~64/128/256
output tokens.

## Safe claims

- The Cohere and Gemini calibration pipelines both work end-to-end
  (plan -> dispatch -> log -> aggregate -> summarize) against live hosted
  APIs.
- Both pilots completed 180/180 planned requests with 0 failures, 0
  timeouts, and 0 rate-limiting.
- Cohere had lower TTFT than Gemini throughout this specific pilot (all
  percentiles, all concurrency levels, all prompt buckets).
- Hosted-API pilots like these are a useful, low-cost way to calibrate
  simulator latency/TTFT/cost assumptions against real providers.
- These pilots are black-box client-observed measurements only; they do not
  test or observe either provider's internal batching, scheduling, or GPU
  allocation.

## Unsafe claims (do not make these from this data)

- **Do not claim this project's scheduler "beats" Cohere or Gemini.**
  Nothing here compares scheduling policies — it compares two providers'
  externally observed latency to each other, with no scheduler of ours in
  the loop.
- **Do not claim Gemini is generally slower than Cohere across all
  settings.** This pilot used one model per provider
  (`command-r7b-12-2024` vs. `gemini-3.1-flash-lite`), one region, one
  client machine, one time window, and 180 requests total. That is not
  enough to generalize "Gemini is slower than Cohere" beyond this specific
  pilot.
- **Do not claim the raw p99 latency numbers reflect provider behavior.**
  Both pilots' p99 (~53s) is the RPM-wait artifact described above, not
  provider-side tail latency, until it is fixed at the source (it now is,
  going forward — see `docs/real_llm_multi_provider_plan.md` and the
  `rate_limiter_wait_seconds` field) and a fresh pilot is run.
- **Do not claim max_tokens sensitivity from this workload.** As shown
  above, generated output length did not vary with `max_tokens` in either
  pilot, so no output-length-scaling claim (e.g., "latency scales linearly
  with max_tokens") can be supported by this data.

## See also

- `docs/real_llm_multi_provider_plan.md` — multi-provider rollout plan and
  original artifact writeup.
- `docs/cohere_api_calibration.md` — Cohere harness design and safety notes.
- `experiments/real_llm/cohere_pilot_20260703T040421Z/summary_corrected.md`,
  `experiments/real_llm/gemini_pilot_20260703T044905Z/summary_corrected.md`
  — full corrected summaries.
- `experiments/real_llm/latency_model_fit/latency_model_fit.md` (produced by
  `scripts/fit_real_llm_latency_model.py`) — the simple TTFT/latency
  calibration model fit from these two pilots.
- Proposed v2 workload (not run): see
  `docs/real_llm_v2_workload_proposal.md`.

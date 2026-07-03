# Real-LLM Multi-Provider Calibration Plan

Status as of this writing: **Cohere is the first completed live pilot**
(180/180 requests, $0.0065 actual cost — see
`experiments/real_llm/cohere_pilot_20260703T040421Z/`). Gemini/Vertex, Azure
OpenAI, and Fireworks currently have **dry-run/mock-only skeleton scripts**;
none of them can make a live API call yet.

This document explains the rollout order, why each provider gets a
different budget/pilot size, and the shared infrastructure that makes their
output directly comparable.

## Shared infrastructure

`src/llmserveopt/real_llm/calibration_common.py` holds everything that is
identical across providers: prompt-bucket construction, request-grid
expansion, hard-cap validation (pre-flight and runtime, concurrency-safe),
JSONL logging schema, aggregation (`aggregate_by_cell.csv`,
`aggregate_by_concurrency.csv`, `aggregate_by_prompt_bucket.csv`,
`summary.json`/`summary.md`), reproducibility metadata, and the anti-
overwrite/resume logic. This was extracted from the working Cohere
implementation (`scripts/run_cohere_api_calibration.py`) after its live
pilot completed successfully, and the Cohere script was refactored to use it
— verified byte-for-byte identical dry-run output and all 35 pre-existing
tests passing unchanged before and after.

Every provider script built on this module produces the same output files
in `--output-dir`: `requests.jsonl`, `summary.json`, `summary.md`,
`aggregate_by_cell.csv`, `aggregate_by_concurrency.csv`,
`aggregate_by_prompt_bucket.csv`, `manifest.json`, `run_config.json`,
`reproducibility.md`, `errors.jsonl` (plus `run.log` when launched via the
tmux `tee` pattern, and `git_diff.patch` if the repo was dirty). This is
enforced by `tests/test_real_llm_provider_skeletons.py`, which runs a
`--mock` pilot through every provider script (Cohere included) and asserts
identical `requests.jsonl` field sets and identical `aggregate_by_cell.csv`
columns.

`scripts/run_gemini_api_calibration.py` (the original Phase 2C.4
config-file-driven dry-run script) is **untouched** and remains available
for its own purpose. The new `scripts/run_gemini_real_llm_calibration.py` is
a separate script using the shared grid schema — kept separate rather than
merged, since unifying them would have required rewriting the original
script's config-file CLI and risked its own passing test suite
(`tests/test_gemini_api_calibration.py`) for no benefit to this plan.

## Provider status and rollout order

| Provider | Script | Status | Credit budget | Pilot size |
|---|---|---|---|---|
| **Cohere** | `scripts/run_cohere_api_calibration.py` | **Live pilot complete** | — | 180 requests, $0.0065 actual |
| **Gemini / Vertex AI** | `scripts/run_gemini_real_llm_calibration.py` | Dry-run/mock only | not credit-constrained | Recommended next: same 180-request grid as Cohere |
| **Azure OpenAI** | `scripts/run_azure_openai_api_calibration.py` | Dry-run/mock only | **$100 total** | Smaller: 54-request default grid (3 buckets × 2 max-token values × 3 concurrency × 3 repeats), tighter default caps |
| **Fireworks AI** | `scripts/run_fireworks_api_calibration.py` | Dry-run/mock only | **$50 total** | Tiny: same smaller default grid as Azure, tightest default cost cap ($1) |
| CloudRift | `src/llmserveopt/llm_generation/providers.py` (`CloudRiftProvider`) | Used for Phase 4 heuristic generation | — | Not part of this latency-calibration plan — see below |

### Why Gemini/Vertex is next

Gemini/Vertex has no known credit constraint recorded for this project (see
[api_provider_setup.md](api_provider_setup.md)), and Google's Gemini Flash
family is priced comparably to Cohere's Command R7B, so it can reuse the
same 180-request grid and hard caps as the completed Cohere pilot without
adjustment. It is the natural next replication target to see whether the
Cohere findings (e.g., TTFT ~0.27s mean, ~0.2–0.5s range, no rate-limiting
or errors at 20 RPM) generalize across providers, or are Cohere-specific.

### Why Azure OpenAI and Fireworks get smaller pilots

Azure OpenAI has a **$100 total credit** budget and Fireworks has a **$50
total credit** budget for this project. Both scripts' default hard caps
(`--max-total-requests 60`, `--max-total-output-tokens 16,000`,
`--max-estimated-cost-usd 2` for Azure / `$1` for Fireworks) are
deliberately tighter than Cohere's defaults, so a mistaken full-size run
cannot meaningfully dent either budget. Scale these up only after reviewing
a first small pilot's actual cost per request.

### Why CloudRift is out of scope for this plan

CloudRift is already used for Phase 4 offline LLM heuristic generation
(`src/llmserveopt/llm_generation/providers.py`), calling a large
general-purpose model (`Qwen/Qwen3.6-35B-A3B-FP8`) for *generating* candidate
scheduling heuristics — a different workload shape (long, thinking-model
outputs, `max_tokens >= 8000`) than this pilot's short bounded-latency
probes. Reusing it here would conflate two unrelated experiments and is not
planned.

## Sequencing rule: no parallel live runs

**Do not launch a new provider's live pilot while another provider's live
pilot is running, and do not launch any live run until the Cohere pilot's
artifacts have been reviewed.** This keeps:

- Cost attribution unambiguous (each provider's invoice maps to exactly one
  pilot run).
- RPM/concurrency measurements uncontaminated by unrelated network traffic
  from a second concurrent experiment on the same machine.
- Review effort focused: one pilot's results should inform whether the next
  pilot's grid/caps need adjustment (e.g., the Cohere pilot surfaced a
  measurement artifact — see below — worth fixing before repeating it
  elsewhere).

## Known measurement caveat from the Cohere pilot (applies to all providers)

The Cohere pilot's `total_latency_seconds`/`elapsed_seconds` fields include
time spent blocked in the RPM limiter's `acquire()` call, not just the
provider's response time. 8 of 180 requests (4.4%) — always the first
request of a new concurrency=1 cell right after a burst of higher-
concurrency traffic — showed `total_latency_seconds` around 53s while
`ttft_seconds` was ~0.2–0.3s, because the request had to wait ~53s for the
client-side RPM budget to free up before Cohere ever saw it. This inflates
tail-latency statistics (p99 latency was 53.3s vs. p95 of 1.15s) in a way
that has nothing to do with the provider. When reviewing any pilot's
`summary.md`, treat `mean_ttft_s`/percentile TTFT as the reliable
provider-latency signal, and treat extreme `total_latency_seconds` outliers
as suspects for RPM-wait inflation until cross-checked against `ttft_seconds`
for the same request. A future improvement would log queueing wait
separately from provider-observed latency.

## How to run a dry-run for each pending provider

```bash
# Gemini/Vertex — recommended next live target
python scripts/run_gemini_real_llm_calibration.py \
  --dry-run \
  --prompt-buckets short,medium,long \
  --max-tokens-list 64,128,256 \
  --concurrency-list 1,2,4,8 \
  --requests-per-cell 5 \
  --output-dir experiments/real_llm/gemini_pilot_DRYRUN

# Azure OpenAI — smaller pilot ($100 credit)
python scripts/run_azure_openai_api_calibration.py \
  --dry-run \
  --model gpt-4o-mini \
  --prompt-buckets short,medium,long \
  --max-tokens-list 64,128,256 \
  --concurrency-list 1,2,4 \
  --requests-per-cell 2 \
  --output-dir experiments/real_llm/azure_openai_pilot_DRYRUN

# Fireworks — tiny pilot ($50 credit)
python scripts/run_fireworks_api_calibration.py \
  --dry-run \
  --model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --prompt-buckets short,medium,long \
  --max-tokens-list 64,128,256 \
  --concurrency-list 1,2,4 \
  --requests-per-cell 2 \
  --output-dir experiments/real_llm/fireworks_pilot_DRYRUN
```

All three refuse `--allow-live-api` (without `--mock`) with a clear "live
mode is not yet implemented" error and exit code 6, since none of them has
a tested `build_client_fn`/`call_streaming_fn`/`call_non_streaming_fn`
implementation yet (see `live_implemented=False` in each script's `main()`).

## What "implementing live mode" for a new provider means

Following the Cohere script as the template
(`scripts/run_cohere_api_calibration.py`):

1. Add a `_build_client()` function that reads the provider's API key from
   its env var and constructs an SDK client.
2. Add `_call_<provider>_non_streaming(client, planned, timeout_s)` and
   `_call_<provider>_streaming(client, planned, timeout_s)` functions
   returning `{"text", "finish_reason", "prompt_tokens", "output_tokens",
   "ttft_seconds"}`.
3. Pass these to `cc.run_calibration_main(..., live_implemented=True,
   build_client_fn=..., call_streaming_fn=..., call_non_streaming_fn=...)`.
4. Write tests mirroring `tests/test_cohere_api_calibration.py` (dry-run,
   mock live execution, resume, fail-fast, budget caps, schema, no-secrets)
   before ever setting `live_implemented=True` for real.
5. Get a real, valid API key in the environment and run `--dry-run` first,
   then a tiny `--allow-live-api` pilot with conservative caps, reviewed
   before scaling up — same discipline as the Cohere pilot.

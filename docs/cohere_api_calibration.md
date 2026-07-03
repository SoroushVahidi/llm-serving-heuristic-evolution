# Cohere API Calibration — Real-LLM Latency/TTFT Pilot

`scripts/run_cohere_api_calibration.py` measures observed latency and
time-to-first-token (TTFT) from the real Cohere hosted Chat API across a grid
of `(prompt_bucket, max_tokens, concurrency_level)` cells, so the simulator's
timing assumptions can be checked against an external, non-simulated LLM
serving stack.

This document explains the goal, safety controls, and exact commands.
For the earlier one-shot connectivity check, see
[cohere_smoke_test.md](cohere_smoke_test.md); that script issues a single
request and is not this experiment.

## Goal, and what this is NOT

**Goal:** black-box latency/TTFT calibration. We record what a client
observes — request submitted → first streamed token → last token — across a
range of prompt lengths, output-length budgets, and client-side concurrency
levels, using the real Cohere API.

**This is not:**
- Control over or visibility into Cohere's internal scheduler, batching, or
  GPU allocation. We cannot see or influence how Cohere batches concurrent
  requests server-side; "concurrency_level" here means *how many requests our
  client has in flight at once*, not a guarantee about server-side batching.
- A production-representative workload. Prompts are synthetic, deterministic,
  and small (a 180-request pilot). Cohere may serve production traffic
  differently (different priority tier, different regional routing, etc.).
- A cost/pricing validation. The pricing table baked into the script
  (`_PRICE_PER_M_INPUT_USD` / `_PRICE_PER_M_OUTPUT_USD`) is an approximate
  placeholder for pre-flight cap-checking only — always verify against your
  actual Cohere invoice for anything beyond a tiny pilot.

Given that scope, safe wording is: *"We measured observed end-to-end latency
and time-to-first-token from the Cohere Chat API under a small synthetic
prompt/concurrency grid, as an external calibration point for the
simulator's timing model."* Do not claim this validates or reproduces
Cohere's internal scheduling policy.

## Design mirrors `run_gemini_api_calibration.py`

Same conventions as the existing Gemini calibration script
(`scripts/run_gemini_api_calibration.py`, see
[api_provider_setup.md](api_provider_setup.md)):
default-refuse unless `--dry-run` or `--allow-live-api` is passed, hard caps
validated before any call, `--mock` for network-free testing, and a
manifest/reproducibility trail written before execution.

Extensions specific to this script: per-cell concurrency (real thread pools,
not just a labeled field), streaming/TTFT measurement, a global RPM limiter,
runtime budget tracking (not just pre-flight), fail-fast on error rate or
repeated rate-limiting, and `--resume` for continuing an interrupted run.

## Prompt buckets

Prompts are generated deterministically from `--seed` + bucket name +
per-request variant index (see `build_prompt()`). They are synthetic,
generic sentences about LLM serving concepts — no copyrighted text.

| Bucket | Target input tokens (approx) |
|---|---|
| short | ~100 |
| medium | ~512 |
| long | ~2048 |

Each request appends a unique `(request variant SEED-BUCKET-INDEX)` tag
before the instruction line. This is deliberate: identical repeated prompts
can hit Cohere's server-side prompt cache (`cached_tokens` was observed
non-zero in preliminary smoke testing when a prompt was repeated
byte-for-byte), which would understate real per-request cost and possibly
latency. Varying the tag defeats that caching while keeping prompt length
and content otherwise fixed and reproducible.

## Running a dry-run

```bash
python scripts/run_cohere_api_calibration.py \
  --dry-run \
  --model command-r7b-12-2024 \
  --prompt-buckets short,medium,long \
  --max-tokens-list 64,128,256 \
  --concurrency-list 1,2,4,8 \
  --requests-per-cell 5 \
  --output-dir experiments/real_llm/cohere_pilot_DRYRUN
```

This writes `manifest.json`, `run_config.json`, `reproducibility.md`,
`summary.json`/`summary.md` describing the planned grid (180 requests for
the values above) and **never imports the `cohere` SDK's live client path or
makes any network call.**

## Running the live pilot

Review the hard caps below before running. Defaults are already
conservative; the explicit values shown mirror the first pilot design.

```bash
export COHERE_API_KEY=...   # never commit or print this

EXP_DIR="experiments/real_llm/cohere_pilot_$(date -u +%Y%m%dT%H%M%SZ)"

python scripts/run_cohere_api_calibration.py \
  --allow-live-api --stream \
  --model command-r7b-12-2024 \
  --prompt-buckets short,medium,long \
  --max-tokens-list 64,128,256 \
  --concurrency-list 1,2,4,8 \
  --requests-per-cell 5 \
  --timeout-seconds 90 --rpm-limit 20 \
  --max-total-requests 180 --max-total-input-tokens 250000 \
  --max-total-output-tokens 50000 --max-estimated-cost-usd 5 \
  --seed 20260703 --fail-fast \
  --output-dir "$EXP_DIR"
```

### Running inside tmux (recommended for anything beyond a trivial pilot)

```bash
tmux new-session -d -s cohere-real-llm-calibration "bash -lc '
  set -euo pipefail
  mkdir -p \"$EXP_DIR\"
  python scripts/run_cohere_api_calibration.py \
    --allow-live-api --stream \
    --model command-r7b-12-2024 \
    --prompt-buckets short,medium,long \
    --max-tokens-list 64,128,256 \
    --concurrency-list 1,2,4,8 \
    --requests-per-cell 5 \
    --timeout-seconds 90 --rpm-limit 20 \
    --max-total-requests 180 --max-total-input-tokens 250000 \
    --max-total-output-tokens 50000 --max-estimated-cost-usd 5 \
    --seed 20260703 --fail-fast \
    --output-dir \"$EXP_DIR\" \
    2>&1 | tee \"$EXP_DIR/run.log\"
'"
```

Monitor with:

```bash
tmux attach -t cohere-real-llm-calibration
tail -f "$EXP_DIR/run.log"
```

The session keeps running if your terminal/SSH connection drops.

## Resuming an interrupted run

```bash
python scripts/run_cohere_api_calibration.py \
  --allow-live-api --stream --resume \
  --model command-r7b-12-2024 \
  --prompt-buckets short,medium,long \
  --max-tokens-list 64,128,256 \
  --concurrency-list 1,2,4,8 \
  --requests-per-cell 5 \
  --seed 20260703 \
  --output-dir "$EXP_DIR"
```

`--resume` reads the existing `requests.jsonl`, skips any `request_id`
already logged with `status: success`, and continues everything else
(failed/timeout/rate-limited/not-yet-attempted). Without `--resume`, the
script refuses to run into a non-empty existing output directory (anti-
overwrite protection) — you must pass `--resume` or choose a fresh
`--output-dir`.

## Where outputs are saved, and what each file means

All paths are under `--output-dir` (e.g. `experiments/real_llm/cohere_pilot_.../`):

| File | Meaning |
|---|---|
| `run_config.json` | Exact resolved CLI configuration for this run. |
| `manifest.json` | Reproducibility metadata + planned request grid summary (git commit/branch/dirty, Python/SDK versions, host, env-var presence flags — never values, first 5 planned requests as a preview). |
| `reproducibility.md` | Human-readable version of the manifest. |
| `git_diff.patch` | Full `git diff` output, written only if the repo was dirty when the run started. |
| `requests.jsonl` | One JSON record per request attempt, written and flushed immediately after each request completes (see schema below). This is the source of truth for resume and aggregation. |
| `errors.jsonl` | Subset of `requests.jsonl` where `status` is `error`, `timeout`, or `rate_limited`. |
| `summary.json` / `summary.md` | Aggregated statistics: status counts, latency/TTFT percentiles, tokens/sec, estimated cost, total billed tokens. |
| `aggregate_by_cell.csv` | Per-`(prompt_bucket, max_tokens, concurrency_level)` cell statistics. |
| `aggregate_by_concurrency.csv` | Statistics grouped by concurrency level only. |
| `aggregate_by_prompt_bucket.csv` | Statistics grouped by prompt bucket only. |
| `run.log` | Full stdout/stderr of the run (only present when launched via the tmux `tee` pattern above). |

### `requests.jsonl` schema

Each line: `request_id`, `experiment_id`, `model`, `prompt_bucket`,
`intended_prompt_tokens`, `actual_prompt_tokens`, `max_tokens`,
`concurrency_level`, `request_index`, `start_time_iso`, `end_time_iso`,
`elapsed_seconds`, `ttft_seconds` (null unless `--stream`),
`total_latency_seconds`, `output_text_length_chars`, `output_tokens`,
`billed_units`, `finish_reason`, `status`
(`success`/`error`/`timeout`/`rate_limited`/`skipped`), `error_type`,
`error_message`, `retry_count`, `was_resumed`.

**Raw response text is never stored** — only its length in characters. The
API key is never written to any output file (enforced by an automated test
that greps every output file for the key value).

## Safety / cost controls

- **Refuses to run** unless `--dry-run` or `--allow-live-api` is passed.
- **Refuses live calls** without `COHERE_API_KEY` set (unless `--mock`).
- **Pre-flight hard-cap validation**: worst-case (every request hits
  `max_tokens`) input tokens, output tokens, request count, and estimated
  cost are checked against `--max-total-input-tokens`,
  `--max-total-output-tokens`, `--max-total-requests`,
  `--max-estimated-cost-usd` *before any request is sent*. Violating any cap
  aborts with no API calls made.
- **Runtime budget enforcement**: caps are re-checked continuously during
  execution (not just at plan time), using a reservation scheme that is safe
  under concurrency — each in-flight request holds its worst-case token
  reservation until it completes, so concurrent workers cannot jointly
  overshoot a cap before their results land. Once a cap would be exceeded,
  remaining requests are marked `status: skipped` rather than sent.
- **RPM limiter** (`--rpm-limit`, default 20): a global sliding-60-second-
  window limiter shared across all concurrency levels.
- **Fail-fast** (`--fail-fast`): aborts remaining requests (marking them
  `skipped`) if either (a) 3 consecutive rate-limited (429) responses occur,
  or (b) the overall error rate exceeds 10% after at least 10 attempts.
- **Per-request timeout** (`--timeout-seconds`) enforced via the Cohere SDK's
  `request_options.timeout_in_seconds`; the SDK's own automatic retries are
  disabled (`max_retries: 0`) so this script's own bounded retry/backoff
  (2 retries, 1s/2s backoff, only for retryable error classes) is the only
  retry behavior in effect, keeping `retry_count` in the log accurate.
- **Anti-overwrite**: refuses to start into a non-empty `--output-dir`
  unless `--resume` is passed.
- **No secrets in output**: the API key is read from the environment only;
  `manifest.json` records only `COHERE_API_KEY_present: true/false`, never
  the value. This is covered by an automated test
  (`test_api_key_never_written_to_output_files`).

## First pilot configuration (reference)

| Parameter | Value |
|---|---|
| Model | `command-r7b-12-2024` (cheapest current Command model) |
| Streaming | enabled |
| Prompt buckets | short (~100 tok), medium (~512 tok), long (~2048 tok) |
| Max output tokens | 64, 128, 256 |
| Concurrency levels | 1, 2, 4, 8 |
| Requests per cell | 5 |
| Total planned requests | 3 × 3 × 4 × 5 = **180** |
| Timeout | 90s |
| RPM limit | 20 |
| Max estimated cost | $5 |
| Max total output tokens | 50,000 |
| Max total input tokens | 250,000 |

Worst-case estimate for this grid (computed by `--dry-run`): ≈165K input
tokens, ≈27K output tokens, **≈$0.01 estimated cost** — all comfortably
inside the caps above. If you change the grid, lower the workload rather
than raising the caps, and re-run `--dry-run` first to confirm the new
worst case still fits.

## Testing

`tests/test_cohere_api_calibration.py` exercises the full script (dry-run,
mock live execution, resume, fail-fast, budget-cap enforcement, schema, and
secret-leakage checks) without any network access or real credentials.

```bash
python -m pytest tests/test_cohere_api_calibration.py -q
```

# Cohere API Smoke Test

`scripts/smoke_test_cohere_api.py` is a minimal connectivity/latency sanity
check for the Cohere API, ahead of the first real-LLM latency/calibration
experiment.

## What it does

- Loads `COHERE_API_KEY` from the environment and fails clearly if missing.
- Makes **one** minimal chat request to `command-r7b-12-2024` (the smallest/
  cheapest current Command model) with a short prompt and `max_tokens=5`.
- Prints only safe metadata: success/failure, model name, elapsed time,
  token counts, response length, and whether streaming/TTFT measurement is
  supported.
- Never prints the API key. Never prints the raw response text unless
  `--debug` is passed explicitly.

## Running it

```bash
# Required
export COHERE_API_KEY=...

# Non-streaming (default)
python scripts/smoke_test_cohere_api.py

# Streaming, with time-to-first-token (TTFT) measurement
python scripts/smoke_test_cohere_api.py --stream

# Override model / prompt / max_tokens
python scripts/smoke_test_cohere_api.py --model command-r7b-12-2024 --max-tokens 5
```

## Expected output

Non-streaming:

```
Cohere API smoke test
  requested_model: command-r7b-12-2024
  stream_mode:     False
  result:          SUCCESS
  model:           command-r7b-12-2024
  elapsed_s:       0.236
  prompt_tokens:   3.0
  completion_tokens: 4.0
  response_length: 11 chars
  streaming_supported: True
```

Streaming:

```
Cohere API smoke test
  requested_model: command-r7b-12-2024
  stream_mode:     True
  result:          SUCCESS
  model:           command-r7b-12-2024
  streaming:       True
  ttft_s:          0.196
  total_elapsed_s: 0.232
  finish_reason:   MAX_TOKENS
  prompt_tokens:   3.0
  completion_tokens: 4.0
  response_length: 11 chars
```

If `COHERE_API_KEY` is unset, the script exits with a clear error (exit
code 1) and never prompts for or exposes a key.

## Safety precautions

- The API key is read from the environment only — never hardcoded, never
  printed, never logged.
- Exactly one request is issued per invocation (two if `--stream` and
  non-streaming are both run), with `max_tokens=5` and a two-word prompt, to
  keep cost effectively zero.
- Raw response text is suppressed by default; pass `--debug` only when you
  need to inspect model output locally.

## Why this is only a sanity check

This script confirms that:

1. `COHERE_API_KEY` is valid and the account is reachable.
2. The `cohere` Python SDK (`ClientV2.chat` / `ClientV2.chat_stream`) is
   being called with the correct current v2 API shape — matching what
   `CohereProvider` in `src/llmserveopt/llm_generation/providers.py` already
   uses.
3. Streaming and TTFT measurement are available if the real-LLM calibration
   experiment needs per-token timing (analogous to
   `scripts/run_gpu_calibration.py` for local GPU calibration, or
   `scripts/run_gemini_api_calibration.py` for Gemini).

It does **not** exercise a representative prompt/output-length distribution,
concurrency, batching behavior, or cost budgeting — that is the scope of the
full API calibration experiment (see `docs/api_provider_setup.md` and the
`run_gemini_api_calibration.py` dry-run/live pattern, which should be
mirrored for Cohere once this smoke test confirms basic connectivity).

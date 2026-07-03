# Real-LLM Latency Calibration Model Fit — v2 (length-targeted)

Generated: 2026-07-03T14:56:28.563615+00:00
Git commit at fit time: `f2bf21a6d392717898fa707a40495756b6754832` (dirty: True)

This is a simple, interpretable OLS calibration baseline computed
entirely from existing pilot logs — **no live API calls were made**
to produce this fit. See docs/real_llm_latency_model_v2.md for the
full write-up, safe/unsafe claims, and simulator-calibration
guidance before using any coefficient here.

## Input experiment directories

| Provider | Model | Workload | Total records | Error rate | Reached target range |
|---|---|---|---|---|---|
| Cohere | `command-r7b-12-2024` | v2 | 108 | 0.0% | 94.4% |
| Gemini | `gemini-3.1-flash-lite` | v2 | 108 | 0.0% | 100.0% |

## Output-token distribution by target length

**Cohere** (`command-r7b-12-2024`)

| target | n_success | mean_output_tokens | mean_ratio | frac_reached |
|---|---|---|---|---|
| 64 | 36 | 51.97 | 0.812 | 83.3% |
| 128 | 36 | 127.56 | 0.997 | 100.0% |
| 256 | 36 | 241.58 | 0.944 | 100.0% |

**Gemini** (`gemini-3.1-flash-lite`)

| target | n_success | mean_output_tokens | mean_ratio | frac_reached |
|---|---|---|---|---|
| 64 | 36 | 66.67 | 1.042 | 100.0% |
| 128 | 36 | 118.67 | 0.927 | 100.0% |
| 256 | 36 | 232.67 | 0.909 | 100.0% |

## Cohere (command-r7b-12-2024)

- n records used in fit: 108
- TTFT stats (s): n=108, mean=0.2462, p50=0.2351, p95=0.3363, p99=0.3866
- Provider latency stats (s, excludes rate_limiter_wait_seconds): n=108, mean=1.9391, p50=1.8596, p95=3.7744, p99=4.0699

### TTFT model: `ttft ~ intercept + target_output_tokens(or output_tokens) + prompt_tokens + concurrency`
  n=108, R^2=0.0533
  intercept=0.221626
  coef_target_or_output_tokens=0.000079
  coef_prompt_tokens=-0.000002
  coef_concurrency_level=0.003843

### Provider latency model: `latency ~ intercept + ttft_seconds + output_tokens + prompt_tokens + concurrency`
  n=108, R^2=0.8903
  intercept=0.201332
  coef_ttft_seconds=0.647392
  coef_output_tokens=0.011335
  coef_prompt_tokens=0.000032
  coef_concurrency_level=-0.010827

## Gemini (gemini-3.1-flash-lite)

- n records used in fit: 108
- TTFT stats (s): n=108, mean=0.6741, p50=0.6006, p95=1.2678, p99=1.7924
- Provider latency stats (s, excludes rate_limiter_wait_seconds): n=108, mean=1.2675, p50=1.1127, p95=1.9757, p99=2.2172

### TTFT model: `ttft ~ intercept + target_output_tokens(or output_tokens) + prompt_tokens + concurrency`
  n=108, R^2=0.0630
  intercept=0.601084
  coef_target_or_output_tokens=0.000302
  coef_prompt_tokens=0.000079
  coef_concurrency_level=-0.010694

### Provider latency model: `latency ~ intercept + ttft_seconds + output_tokens + prompt_tokens + concurrency`
  n=108, R^2=0.9236
  intercept=0.181822
  coef_ttft_seconds=0.877818
  coef_output_tokens=0.003498
  coef_prompt_tokens=0.000003
  coef_concurrency_level=0.001055

## Pooled model (baseline provider: Cohere)

- providers: Cohere, Gemini
- n records: 216

### Pooled TTFT model (provider-indicator coefficients are the offset vs. baseline)
  n=216, R^2=0.5517
  intercept=0.197423
  coef_target_or_output_tokens=0.000190
  coef_prompt_tokens=0.000038
  coef_concurrency_level=-0.003425
  coef_is_Gemini=0.428250

### Pooled provider latency model
  n=216, R^2=0.7833
  intercept=0.623048
  coef_ttft_seconds=0.791559
  coef_output_tokens=0.007987
  coef_prompt_tokens=0.000025
  coef_concurrency_level=-0.005756
  coef_is_Gemini=-1.001741

## Whether the target=64 crossover finding remains visible

Raw (model-free) mean provider latency by (provider, target_output_tokens),
computed directly from the same records used to fit the models above —
independent of any regression assumption:

| provider | target | n | mean latency (s) | mean TTFT (s) | mean output tokens |
|---|---|---|---|---|---|
| Cohere | 64 | 36 | 0.8923 | 0.2376 | 51.97 |
| Cohere | 128 | 36 | 1.8431 | 0.2474 | 127.56 |
| Cohere | 256 | 36 | 3.0818 | 0.2537 | 241.58 |
| Gemini | 64 | 36 | 1.0448 | 0.6608 | 66.67 |
| Gemini | 128 | 36 | 1.1122 | 0.6488 | 118.67 |
| Gemini | 256 | 36 | 1.6455 | 0.7126 | 232.67 |

- target=64: **Cohere** faster (Cohere=0.892s, Gemini=1.045s, gap=0.152s)
- target=128: **Gemini** faster (Gemini=1.112s, Cohere=1.843s, gap=0.731s)
- target=256: **Gemini** faster (Gemini=1.645s, Cohere=3.082s, gap=1.436s)

## Decode-rate estimates by provider and target length

Fit as `decode_seconds = latency_seconds - ttft_seconds ~ a * output_tokens + b`
(same form as v1's `ttft_plus_decode`), per (provider, target_output_tokens) group,
so the rate is not averaged across target lengths.

**Caveat:** each per-target row has only n=36 and a narrow within-group
range of realized `output_tokens` (since actual output clusters near its
own target), which makes the *slope* (tokens/sec) hard to identify from a
single target group alone — low R^2 rows below (arbitrarily, R^2 < 0.5) should
not be read as a real decode-rate measurement, only the `overall` per-provider row (n=108, pooled across all three targets) is reasonably
well-identified. This is a real limitation of a 36-request-per-cell pilot,
not a bug in the fit.

| provider | target | n | tokens/sec | R^2 |
|---|---|---|---|---|
| Cohere | overall | 108 | 88.5 | 0.887 |
| Cohere | 64 | 36 | 94.9 | 0.487 ⚠️ low, do not trust |
| Cohere | 128 | 36 | 96.8 | 0.423 ⚠️ low, do not trust |
| Cohere | 256 | 36 | 106.2 | 0.231 ⚠️ low, do not trust |
| Gemini | overall | 108 | 288.9 | 0.833 |
| Gemini | 64 | 36 | n/a | 0.000 ⚠️ low, do not trust |
| Gemini | 128 | 36 | 553.1 | 0.008 ⚠️ low, do not trust |
| Gemini | 256 | 36 | 70.8 | 0.465 ⚠️ low, do not trust |

## Residual / caveat discussion

Per-(provider, model) residual summary for both fitted models
(`actual - predicted`, in seconds). A `rmse` well above the
provider's own TTFT/latency p50 (see stats above) means the linear
form is a poor fit for that provider/model and coefficients should
not be trusted for extrapolation beyond the pilot's own grid
(targets 64/128/256, concurrency 1-8, 3 prompt buckets).

| provider | model_type | n | mean_residual (s) | std_residual (s) | rmse (s) |
|---|---|---|---|---|---|
| Cohere | latency | 108 | 0.0000 | 0.3218 | 0.3218 |
| Cohere | ttft | 108 | 0.0000 | 0.0512 | 0.0512 |
| Gemini | latency | 108 | -0.0000 | 0.1022 | 0.1022 |
| Gemini | ttft | 108 | -0.0000 | 0.2662 | 0.2662 |


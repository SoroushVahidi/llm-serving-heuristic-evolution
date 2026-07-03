# Real-LLM Latency Calibration Model Fit

Generated: 2026-07-03T13:03:05.781842+00:00
Inputs: experiments/real_llm/cohere_pilot_20260703T040421Z, experiments/real_llm/gemini_pilot_20260703T044905Z
RPM-wait outliers excluded: True
Targets fit: both
Latency model form: ttft_plus_decode

This is a simple, interpretable OLS calibration baseline, not a
production-quality predictive model. See docs/real_llm_cohere_gemini_comparison.md for the pilots' full caveats before using these
coefficients to update simulator service-time assumptions.

**Output-token-scaling caveat:** the source pilots' prompts elicited
~22-35 output tokens regardless of `max_tokens` (see the max_tokens
caveat in docs/real_llm_cohere_gemini_comparison.md), so
`coef_output_tokens` / `effective_decode_rate_tokens_per_sec` below
are fit over a very narrow output-length range and should be treated
as low-confidence until refit on the proposed v2 workload (see
docs/real_llm_v2_workload_proposal.md), which is designed to
actually vary output length.

## Gemini (gemini-3.1-flash-lite)

- n records used: 172 (RPM-wait-flagged excluded: 0)
- TTFT stats (s): n=172, mean=0.5512, p50=0.5615, p95=0.7329, p99=0.7738
- Latency stats (s): n=172, mean=0.7738, p50=0.7374, p95=1.0768, p99=1.3926

### TTFT model: `ttft ~ intercept + prompt_tokens + output_tokens + concurrency`
  n=172, R^2=0.0447
  intercept=0.312940
  coef_prompt_tokens=0.000006
  coef_output_tokens=0.010940
  coef_concurrency_level=-0.005471

### Latency model (ttft_plus_decode)
  n=172, R^2=0.0009
  intercept=0.307633
  coef_output_tokens=-0.003649
  effective_decode_rate_tokens_per_sec=None
  decode_intercept_s=0.307633

## cohere (command-r7b-12-2024)

- n records used: 172 (RPM-wait-flagged excluded: 0)
- TTFT stats (s): n=172, mean=0.2745, p50=0.2596, p95=0.4353, p99=0.5318
- Latency stats (s): n=172, mean=0.7213, p50=0.6828, p95=0.9980, p99=1.1439

### TTFT model: `ttft ~ intercept + prompt_tokens + output_tokens + concurrency`
  n=172, R^2=0.2765
  intercept=0.179775
  coef_prompt_tokens=-0.000006
  coef_output_tokens=0.001228
  coef_concurrency_level=0.015662

### Latency model (ttft_plus_decode)
  n=172, R^2=0.0774
  intercept=-0.071625
  coef_output_tokens=0.016217
  effective_decode_rate_tokens_per_sec=61.6629
  decode_intercept_s=-0.071625

## Pooled model (baseline provider: Gemini)

- providers: Gemini, cohere
- n records: 344
- TTFT stats (s): n=344, mean=0.4129, p50=0.4118, p95=0.6862, p99=0.7565
- Latency stats (s): n=344, mean=0.7476, p50=0.7131, p95=1.0205, p99=1.2888

### Pooled TTFT model (provider indicator coefficients are the offset vs. baseline)
  n=344, R^2=0.7133
  intercept=0.443286
  coef_prompt_tokens=-0.000003
  coef_output_tokens=0.003900
  coef_concurrency_level=0.005061
  coef_is_cohere=-0.310492

### Pooled latency model (ttft_plus_decode)
  n=344, R^2=0.3534
  intercept=-0.318342
  coef_output_tokens=0.023637
  effective_decode_rate_tokens_per_sec=42.3065
  decode_intercept_s=-0.318342


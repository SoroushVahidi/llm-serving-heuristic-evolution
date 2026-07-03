# Simulator vs. Real-LLM (Cohere/Gemini v2) Latency Sanity Check

**Offline analysis only — no live API calls were made.** This compares
the simulator's own service-time formulas (evaluated directly, not via
a simulation run) against the fitted hosted-API latency model in
`docs/real_llm_latency_model_v2.md`. See
`docs/real_llm_simulator_integration_plan.md` for why these measure
different things and should not be equated.

- Fitted config: `/home/soroush/llm-serving-heuristic-evolution/configs/real_llm_latency/cohere_gemini_v2_fit.yaml`
- Fitted model dir: `/home/soroush/llm-serving-heuristic-evolution/experiments/real_llm/latency_model_fit_v2`
- GPU calibration file: `/home/soroush/llm-serving-heuristic-evolution/results/gpu_calibration/service_curves.json` (found)

## What decode tok/s does the simulator implicitly assume?

- **cohere** (hosted_api, `command-r7b-12-2024`): 88.5 tokens/sec (R^2=0.887)
- **gemini** (hosted_api, `gemini-3.1-flash-lite`): 288.9 tokens/sec (R^2=0.833)
- **simulator_synthetic_default** (simulator, `ServiceModel (enable_prefill_modeling=False)`): 1000.0 tokens/sec
- **simulator_calibrated_rtx5060ti_b1** (simulator, `CalibratedServiceModel (Qwen2.5-0.5B, RTX 5060 Ti)`): 147.7 tokens/sec
- **simulator_calibrated_rtx5060ti_b8** (simulator, `CalibratedServiceModel (Qwen2.5-0.5B, RTX 5060 Ti)`): 100.2 tokens/sec

## Is the simulator closer to Cohere v2, Gemini v2, or neither?

The default synthetic model's 1000.0 tokens/sec is numerically closest to **gemini**, but faster than both providers — see the ratio columns in `comparison_by_target_output_tokens.csv`. This is a coincidental numeric proximity, not evidence the synthetic model represents either provider's actual behavior (it was hand-tuned for Phase 1, not calibrated against any provider).

## Are simulator decode assumptions faster or slower than hosted measurements?

- Synthetic default is **11.3x faster** than cohere (1000.0 vs. 88.5 tok/s).
- Synthetic default is **3.5x faster** than gemini (1000.0 vs. 288.9 tok/s).

## Does simulator latency scale linearly with output length?

Yes, by construction in both simulator variants: `decode_time = output_tokens * per_token_time`, with zero intercept. Real hosted
latency is *also* well-approximated as linear in output_tokens
(R^2=0.89-0.92 per `docs/real_llm_latency_model_v2.md`) but with a
materially nonzero intercept (~0.1-0.2s, comparable in scale to TTFT)
that the simulator's zero-intercept formulas do not represent.

## Does simulator TTFT/prefill have any analogue to hosted TTFT?

The simulator's default (`enable_prefill_modeling=False`) has **no TTFT analogue at all** — prefill is instantaneous (computed value: 0.0000s). 
The GPU-calibrated variant does model prefill (~0.0170s at a 512-token prompt), but this is *local GPU compute time only* for a 0.5B model — one to two orders of magnitude smaller than hosted TTFT (cohere=0.246s, gemini=0.674s), because hosted TTFT also bundles network round-trip and provider-side admission/queueing that a local prefill-compute formula cannot represent.

## Which real-LLM quantities can safely calibrate the simulator?

- The `overall` per-provider decode rate (Cohere ~88.5 tok/s, Gemini
  ~289 tok/s) as an order-of-magnitude sanity check against the
  simulator's own decode-rate assumption — not a value to copy in
  directly, since it reflects a specific hosted model/provider, not
  a serving-engine property this simulator controls.
- The qualitative finding that provider latency is linear in output
  length with a nonzero intercept — worth checking the simulator's
  own service model reflects *some* fixed per-request overhead if a
  future task adds one, even though the specific hosted intercept
  value should not be copied in directly (see below).

## Which hosted measurements should NOT be used directly?

- Any hosted TTFT value, as a stand-in for simulator prefill time —
  it bundles network + provider-side admission/scheduling the
  simulator has no equivalent for and should not fabricate one for.
- Any per-target-length decode rate with R^2 < 0.5 (see
  `docs/real_llm_latency_model_v2.md`) — several of these are fit
  noise, not a real per-length effect.
- Hosted concurrency/prompt-bucket latency trends — both pilots showed
  no significant concurrency effect at 1-8 concurrent requests from a
  single client, which says nothing about how either provider (or this
  simulator) behaves under real multi-tenant load.

## Comparison tables

See `comparison_by_target_output_tokens.csv` and `comparison_by_provider.csv`
for the full numeric detail behind every claim above.


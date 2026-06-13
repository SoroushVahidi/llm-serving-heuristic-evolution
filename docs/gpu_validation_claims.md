# GPU Validation Claims: Safe vs Unsafe

## What the Calibration DOES Show

**Safe claims:**

1. "Prefill latency for Qwen2.5-0.5B on RTX 5060 Ti scales approximately linearly
   with prompt token count in the range [32, 2048] tokens."

2. "Decode throughput per token increases sub-linearly with batch size
   in the range [1, 8] for static batching."

3. "The fitted linear model predicts held-out prefill latencies with
   MAPE of X% on our specific hardware." (Replace X with actual value.)

4. "CalibratedServiceModel step counts differ from real GPU measurements
   by at most Y% for the held-out validation grid." (Replace Y with actual.)

5. "The calibration model fits in <2 GB VRAM on a 16 GB GPU."

---

## What the Calibration Does NOT Show

**Unsafe claims (do not make):**

1. ~~"These curves apply to other GPU models or architectures."~~
   Each GPU model has different compute-to-memory ratios.

2. ~~"These curves apply to other LLMs."~~
   Attention complexity and memory access patterns differ by model family.

3. ~~"The simulator accurately predicts production LLM serving latencies."~~
   HF Transformers static batching != vLLM continuous batching.

4. ~~"Prefill is perfectly linear for all sequence lengths."~~
   Beyond 2048 tokens, memory bandwidth limits and attention quadratic complexity
   may cause non-linear behavior not captured by the linear fit.

5. ~~"These measurements represent production-quality benchmarks."~~
   No server overhead, no network, no multi-tenant effects, no KV cache eviction.

---

## What the Calibration Is Good For

- Setting the `prefill_cost_per_token` parameter in `ServiceModel` to realistic values
- Comparing scheduling policies where relative differences matter
- Sanity-checking that simulator step counts are in the right order of magnitude
- Phase 2 experiments where you want a "realistic" service model baseline

## Required Caveats

Any results paper or report using `CalibratedServiceModel` must include:
- Hardware specification (RTX 5060 Ti, 16 GB)
- Model specification (Qwen2.5-0.5B, bfloat16)
- Backend specification (HF Transformers, static batching)
- Validation MAPE from `results/gpu_calibration/validation_report.json`

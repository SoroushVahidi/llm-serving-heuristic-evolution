# Calibration Backend Decision

## Decision: Hugging Face Transformers (HF Transformers) Backend

**Date:** 2026-06-10  
**Status:** Final for Phase 1.7B

---

## Why HF Transformers

vLLM is not installed, and CUDA 13.0 / PyTorch 2.12.0 is too new for existing vLLM
prebuilt wheels. Attempting to build vLLM from source is not feasible in the current
environment. HF Transformers is already available and compatible.

| Factor | HF Transformers | vLLM |
|---|---|---|
| Installed | Yes | No |
| CUDA 13.0 compatible | Yes | No (no wheels exist) |
| Continuous batching | No | Yes |
| KV cache management | Basic (generate API) | Advanced (PagedAttention) |
| Suitable for calibration | Yes | Would be preferred for serving emulation |

---

## Operating Mode

**Single-request and static-batch mode** using `model.generate()`.

- No continuous batching
- Requests are padded to equal length within a batch
- `use_cache=True` for realistic KV cache behavior
- Timing via CUDA events for sub-millisecond precision

---

## What is Measured

- **Prefill time**: `model.generate(prompt, max_new_tokens=1)` — forward pass on prompt + 1 decode step
- **Decode time per token**: `(total_generate_time - prefill_time) / (output_tokens - 1)`
- **Batch latency**: `model.generate([batch], max_new_tokens=N)` with left-padding

---

## Known Limitations

1. **No continuous batching**: HF Transformers uses static batching. Real LLM serving systems
   (vLLM, TensorRT-LLM) use continuous batching which amortizes KV cache overhead differently.
   The calibrated decode times will overestimate latency for mixed-length batch scenarios.

2. **Padding inflates batch measurements**: Left-padding to the longest sequence means shorter
   sequences do unnecessary computation. This is inherent to the HF Transformers API.

3. **Prefill + 1 decode mixed**: The "prefill time" measurement includes exactly one decode step.
   For prompts < 32 tokens, this can meaningfully inflate the measured prefill cost.

4. **No KV cache sharing**: Each generate() call starts with a fresh KV cache. Prefix caching
   effects seen in production vLLM deployments are not captured.

5. **Single GPU only**: No tensor parallelism. The RTX 5060 Ti has 16 GB VRAM which is
   sufficient for Qwen2.5-0.5B but not for larger models without quantization.

6. **Model-specific curves**: Curves fitted on Qwen2.5-0.5B may not transfer to other models
   even with a scaling factor, because attention patterns and memory bandwidth behavior differ.

---

## Acceptable Use Cases

- Calibrating the Phase 1.7B simulator step_size parameter
- Validating that simulator prefill step counts are in the right ballpark
- Phase 2 experiments using CalibratedServiceModel as an alternative to synthetic ServiceModel

---

## Not Suitable For

- Precise SLO prediction in production
- Modeling continuous-batching systems
- Multi-GPU or tensor-parallel configurations
- Models larger than ~7B without quantization on this hardware

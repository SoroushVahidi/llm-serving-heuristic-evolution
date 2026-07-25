# CalibratedServiceModel

## Overview

`CalibratedServiceModel` is an optional alternative to the synthetic `ServiceModel`.
It uses GPU-measured timing curves from `results/gpu_calibration/service_curves.json`
to predict simulator step counts more accurately.

The original `service_model.py` is NOT modified. All Phase 1.5 experiments remain
runnable without any changes.

---

## Config Format

In any simulator YAML config, add:

```yaml
service_model:
  type: calibrated
  calibration_file: results/gpu_calibration/service_curves.json
  step_size: 0.001                 # optional, default 0.001
  enable_prefill_modeling: true    # optional, default true
  max_prefill_steps: 10000         # optional, safety clamp
```

---

## Python API

```python
from llmserveopt.simulator.calibrated_service_model import (
    CalibratedServiceModel,
    load_calibrated_service_model_from_config,
)

# Direct instantiation
csm = CalibratedServiceModel(
    calibration_file="results/gpu_calibration/service_curves.json"
)

# Prefill steps for a 512-token prompt
steps = csm.compute_prefill_steps(prompt_tokens=512)

# Decode time per token at batch_size=4, context=768 tokens
decode_s = csm.compute_decode_step_time(batch_size=4, context_tokens=768)

# Load from YAML config dict
csm2 = load_calibrated_service_model_from_config({
    "type": "calibrated",
    "calibration_file": "results/gpu_calibration/service_curves.json",
})
```

---

## Interpolation Behavior

### In-range values (within calibration grid)

For prompt_tokens in [32, 2048], the linear fit gives good predictions:
```
prefill_time_s = a0 + a1 * prompt_tokens
steps = ceil(prefill_time_s / step_size)
```

### Out-of-range values

The model extrapolates linearly. For very large prompt_tokens (> 2048),
this may underestimate the true prefill time if GPU runs out of memory bandwidth.
A warning is printed and the result is clamped to `[1, max_prefill_steps]`.

For very small prompt_tokens (< 32), the constant term `a0` dominates
and the prediction is a reasonable approximation.

---

## Limitations

1. **Model-specific**: Curves are for Qwen2.5-0.5B on RTX 5060 Ti only.
2. **Decode wall-clock helper is offline-only today:**
   `compute_decode_step_time()` / `decode_time()` are used by calibration
   comparison scripts and tests. The discrete-event GPU step path in
   `simulator/gpu.py` does **not** call them; decode progress remains
   token-budget based. See
   `docs/current/KNOWN_SIMULATOR_HEURISTIC_GAPS.md`.
3. Prefill step counts from calibration *can* affect Phase-1.5
   `prefill_remaining` when `service_model.type: calibrated` is configured.
4. **Static batch assumption**: Fitted from HF Transformers generate(), not vLLM continuous batching.
5. **Linear fit**: May not capture non-linear effects at extreme batch sizes or sequence lengths.
6. **Single GPU**: No multi-GPU or tensor-parallel effects modeled.
7. **No quantization effects**: All measurements use bfloat16.

---

## Error Handling

If the calibration file is missing:
```
FileNotFoundError: Calibration file not found: results/gpu_calibration/service_curves.json
Run the calibration pipeline to generate it:
  python scripts/run_gpu_calibration.py --config configs/gpu_calibration/calibration_grid.yaml
  python scripts/fit_service_curves.py ...
```

# GPU Calibration Methodology

## Overview

Phase 1.7B calibrates the simulator service model against real GPU measurements.
The goal is to replace or supplement the synthetic `ServiceModel` with a
`CalibratedServiceModel` that uses measured GPU timing curves.

---

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- CUDA: 13.0, Driver: 580.142
- RAM: 56 GB, Disk: 595 GB free

## Software Backend

Hugging Face Transformers + PyTorch 2.12.0+cu130.
See `docs/calibration_backend_decision.md` for rationale.

## Calibration Model

**Qwen/Qwen2.5-0.5B** (Apache-2.0 license)

Reasons:
- 0.5B params × 2 bytes (bfloat16) ≈ 1 GB VRAM — trivially fits in 16 GB
- Fast grid traversal across many batch sizes and prompt lengths
- No gated access required
- Realistic multi-head attention for profiling

---

## Calibration Grid

### Training Grid (`configs/gpu_calibration/calibration_grid.yaml`)

| Dimension | Values |
|---|---|
| Prompt lengths (tokens) | 32, 128, 512, 1024, 2048 |
| Output lengths (tokens) | 16, 64, 128, 256 |
| Batch sizes | 1, 2, 4, 8 |
| Warmup runs | 2 |
| Measurement runs | 5 |

Total combinations: 5 × 4 × 4 = **80 grid points**.

### Validation Grid (`configs/gpu_calibration/validation_grid.yaml`)

Held-out grid NOT used during fitting:

| Dimension | Values |
|---|---|
| Prompt lengths | 64, 256, 768, 1536 |
| Output lengths | 32, 96, 192 |
| Batch sizes | 1, 3, 6 |
| Measurement runs | 3 |

Total: 4 × 3 × 3 = **36 held-out points**.

---

## Fitting Method

### Prefill Curve

```
prefill_time_s = a0 + a1 * prompt_tokens
```

Fitted with OLS (numpy lstsq) on all valid non-skipped rows.

### Decode Curve

```
decode_time_per_token_s = b0 + b1 * batch_size + b2 * context_tokens
```

where `context_tokens = prompt_tokens + output_tokens` (approximate KV cache size).

---

## Output Files

| File | Description |
|---|---|
| `results/gpu_calibration/raw_measurements.csv` | Raw timing data (all grid points × runs) |
| `results/gpu_calibration/service_curves.json` | Fitted curves + lookup tables |
| `results/gpu_calibration/fit_report.json` | Fit quality metrics |
| `results/gpu_calibration/validation_summary.csv` | Per-point validation errors |
| `results/gpu_calibration/validation_report.json` | Aggregate validation MAPE |
| `results/gpu_calibration/plots/` | All diagnostic plots |

---

## Validation Results

See `results/gpu_calibration/validation_report.json` for actual numbers.

Calibration is considered sufficient for Phase 2 if:
- Prefill MAPE < 20% on held-out points
- Decode MAPE < 20% on held-out points

These thresholds reflect that the simulator uses abstract step counts, not
nanosecond-accurate timing. A 20% error in prefill cost maps to approximately
a 20% error in queue delay estimates, which is acceptable for policy comparison.

---

## Limitations

- See `docs/calibration_backend_decision.md` for backend limitations
- Linear fit may underfit if GPU has non-linear throughput at very small or very large batches
- Measurements are snapshot values — GPU boost clocks cause run-to-run variance of 2–5%
- All measurements are for Qwen2.5-0.5B; curves do not transfer to other models

# Phase 1.7B: GPU Calibration Milestone

**Date:** 2026-06-10
**Completed:** 2026-06-10
**Status:** COMPLETE

---

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- CUDA 13.0, Driver 580.142
- PyTorch 2.12.0+cu130
- RAM: 56 GB, Disk: 595 GB free

## Backend

Hugging Face Transformers, static batching (no vLLM — CUDA 13 too new for prebuilt wheels).

## Model

Qwen/Qwen2.5-0.5B, bfloat16, device_map=auto.

## Calibration Grid

- Training: 5 prompt_lengths × 4 output_lengths × 4 batch_sizes = 80 combinations
- Validation: 4 × 3 × 3 = 36 held-out points

## Validated Results

Run with:
1. `scripts/run_gpu_calibration.py`
2. `scripts/fit_service_curves.py`
3. `scripts/validate_simulator_calibration.py`

Held-out validation results (`results/gpu_calibration/validation_report.json`):

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| Prefill MAPE (held-out) | **11.88%** | < 20% | ✓ |
| Decode MAPE (held-out) | **12.15%** | < 20% | ✓ |
| Prefill max error | 22.16% | — | — |
| Decode max error | 31.4% | — | — |
| `calibration_sufficient` | true | — | ✓ |

Fit quality notes:
- Prefill R² = 0.989 (excellent linear fit)
- Decode R² = 0.258 (weak linear fit; non-linear batch-size effects not captured)
  → Decode MAPE passes the 20% threshold but the fit is approximate.
  → Document as known limitation in any paper using CalibratedServiceModel.

Derived simulator parameters (at 512-token reference, batch=4):
- `prefill_cost_per_token`: 16.23
- `decode_steps_per_token`: 7.69
- Suggested `step_size`: 0.00631 s (default kept at 0.001 s for compatibility)

## Known Limitations

1. Static batching only (HF Transformers; vLLM unavailable on CUDA 13.0)
2. Linear decode fit (R²=0.26) may underestimate throughput degradation at
   large batch sizes (>8) or very short sequences
3. Curves specific to Qwen2.5-0.5B on RTX 5060 Ti; do not transfer to other models/GPUs
4. Single GPU only; no multi-GPU profiling
5. `CalibratedServiceModel` not yet wired into experiment runners at milestone time
   (fixed in Phase 1.7C)

## Files Created

- `src/llmserveopt/calibration/__init__.py`
- `src/llmserveopt/calibration/prompt_generator.py`
- `src/llmserveopt/calibration/measurement.py`
- `src/llmserveopt/calibration/benchmark_backend.py`
- `src/llmserveopt/calibration/curve_fitting.py`
- `src/llmserveopt/calibration/simulator_adapter.py`
- `src/llmserveopt/simulator/calibrated_service_model.py`
- `scripts/inspect_gpu_environment.py`
- `scripts/run_gpu_calibration.py`
- `scripts/fit_service_curves.py`
- `scripts/validate_simulator_calibration.py`
- `configs/gpu_calibration/model.yaml`
- `configs/gpu_calibration/calibration_grid.yaml`
- `configs/gpu_calibration/validation_grid.yaml`
- `tests/test_calibration_utils.py` (10 non-GPU unit tests)
- `tests/test_calibration_gpu.py` (3 GPU integration tests)
- `tests/fixtures/service_curves_fixture.json`

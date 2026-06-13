# Phase 1.7B: GPU Calibration Milestone

**Date:** 2026-06-10  
**Status:** In Progress

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

## Results

See `results/gpu_calibration/` after running:
1. `scripts/run_gpu_calibration.py`
2. `scripts/fit_service_curves.py`
3. `scripts/validate_simulator_calibration.py`

## Known Limitations

1. Static batching only (HF Transformers)
2. Linear fit may not capture non-linear effects at extreme parameters
3. Curves specific to Qwen2.5-0.5B on RTX 5060 Ti
4. Single GPU only

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

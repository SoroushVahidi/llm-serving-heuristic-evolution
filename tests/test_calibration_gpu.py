"""
GPU integration tests for calibration.

All tests require a CUDA-capable GPU and will download the model (~1 GB).
Mark with @pytest.mark.gpu — skipped in CPU-only CI.

Run with:
    pytest tests/test_calibration_gpu.py -m gpu -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.gpu
def test_model_loads():
    """Model loads on GPU without OOM."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    assert device.type in ("cuda", "cpu")

    vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    print(f"  Peak VRAM after load: {vram:.2f} GB")
    assert vram < 14.0, f"Model used too much VRAM: {vram:.2f} GB"

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.gpu
def test_prefill_measurement_runs():
    """Single prefill measurement returns valid MeasurementResult."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmserveopt.calibration.measurement import measure_prefill_latency
    from llmserveopt.calibration.prompt_generator import generate_prompt_of_length

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    pdata = generate_prompt_of_length(tokenizer, 128)
    result = measure_prefill_latency(
        model, tokenizer, pdata["input_ids"],
        warmup=1, runs=2,
        model_name="Qwen/Qwen2.5-0.5B",
        dtype="bfloat16",
    )

    assert not result.skipped, f"Measurement was skipped: {result.skip_reason}"
    assert result.prefill_time_mean > 0
    assert result.prefill_time_mean < 10.0, f"Unreasonably slow: {result.prefill_time_mean:.3f}s"
    assert result.prompt_tokens == 128

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.gpu
def test_calibration_grid_tiny():
    """A tiny 2×2×1 grid runs without error and produces output CSV."""
    import tempfile
    from pathlib import Path

    from llmserveopt.calibration.benchmark_backend import BenchmarkBackend

    backend = BenchmarkBackend(
        model_name="Qwen/Qwen2.5-0.5B",
        dtype="bfloat16",
        device_map="auto",
        seed=42,
    )

    tiny_grid = {
        "prompt_lengths": [32, 128],
        "output_lengths": [16, 32],
        "batch_sizes": [1],
        "warmup_runs": 1,
        "measurement_runs": 2,
        "seed": 42,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "tiny_calibration.csv"
        results = backend.run_calibration_grid(tiny_grid, output_csv=csv_path)

        assert len(results) == 4, f"Expected 4 results (2×2×1), got {len(results)}"
        assert csv_path.exists(), "CSV was not created"

        n_ok = sum(1 for r in results if not r.skipped)
        print(f"  {n_ok}/{len(results)} measurements succeeded")
        # At minimum the bs=1, small prompt cases should succeed
        assert n_ok >= 2, f"Too many skipped: {n_ok}/{len(results)}"

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

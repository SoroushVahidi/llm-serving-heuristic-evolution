#!/usr/bin/env python3
"""
Run the GPU calibration grid and save raw measurements.

Usage:
    python scripts/run_gpu_calibration.py \\
        --config configs/gpu_calibration/calibration_grid.yaml \\
        2>&1 | tee results/gpu_calibration/calibration_run.log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU calibration grid")
    parser.add_argument(
        "--config",
        default="configs/gpu_calibration/calibration_grid.yaml",
        help="Path to calibration_grid.yaml",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    config = load_yaml(config_path)
    model_config_path = ROOT / config["model_config"]
    model_config = load_yaml(model_config_path)

    grid_config = config["grid"]
    output_csv = ROOT / config["output"]["raw_measurements"]

    model_name = model_config["model"]["name"]
    dtype = model_config["model"]["dtype"]
    device_map = model_config["model"]["device_map"]
    trust_remote = model_config["model"].get("trust_remote_code", False)
    seed = grid_config.get("seed", 42)

    print(f"=== GPU Calibration Run ===")
    print(f"  model: {model_name}")
    print(f"  dtype: {dtype}")
    print(f"  device_map: {device_map}")
    print(f"  output: {output_csv}")
    print(f"  grid:")
    print(f"    prompt_lengths: {grid_config['prompt_lengths']}")
    print(f"    output_lengths: {grid_config['output_lengths']}")
    print(f"    batch_sizes: {grid_config['batch_sizes']}")
    print(f"    warmup_runs: {grid_config['warmup_runs']}")
    print(f"    measurement_runs: {grid_config['measurement_runs']}")

    from llmserveopt.calibration.benchmark_backend import BenchmarkBackend

    backend = BenchmarkBackend(
        model_name=model_name,
        dtype=dtype,
        device_map=device_map,
        seed=seed,
        trust_remote_code=trust_remote,
    )

    results = backend.run_calibration_grid(grid_config, output_csv=output_csv)

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'prompt':>8} {'output':>8} {'batch':>6} {'prefill_ms':>12} {'decode_us_tok':>14} {'mem_gb':>8} {'skip':>6}")
    print("-" * 70)
    for r in results:
        if r.skipped:
            print(f"{r.prompt_tokens:>8} {r.output_tokens:>8} {r.batch_size:>6} {'SKIPPED':>12} {r.skip_reason[:20]:>14} {'':>8} {'Y':>6}")
        else:
            print(
                f"{r.prompt_tokens:>8} {r.output_tokens:>8} {r.batch_size:>6} "
                f"{r.prefill_time_mean*1000:>12.2f} "
                f"{r.decode_time_per_token_mean*1e6:>14.1f} "
                f"{r.peak_memory_gb:>8.2f} "
                f"{'N':>6}"
            )

    n_ok = sum(1 for r in results if not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    print(f"\nTotal: {len(results)} measurements, {n_ok} succeeded, {n_skip} skipped")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
